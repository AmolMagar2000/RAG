"""LangGraph nodes. Each takes RAGState and returns a partial update.

Privacy model: employee names never reach reason/generate. Names are swapped for refs (E1, E2) in the
question and in the facts; the refs are mapped back to names only in the final guardrail step.
"""
import functools
import json
import re
from datetime import date
from typing import Any, Dict, List, Literal, Optional

from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from app.config import get_settings
from app.db import emigrant
from app.graph.state import RAGState
from app.guardrails.pii import mask_pii
from app.llm import router
from app.logging_utils.tracer import get_tracer
from app.retrieval import retriever

settings = get_settings()


# ----------------------------------------------------------------------------- schemas
class Classification(BaseModel):
    issue_type: Literal[
        "right_to_work", "record_keeping", "change_of_circumstances", "monitoring",
        "reporting_duties", "visa_sponsorship", "general_policy", "out_of_scope",
    ]
    confidence: float = Field(ge=0, le=1)
    scope: Literal["employee", "workforce", "policy_only"]
    employee_mentions: List[str]
    letter_date: Optional[str] = None
    sub_queries: List[str]


class Finding(BaseModel):
    statement: str
    basis: Literal["policy", "employee_data", "both"]
    policy_refs: List[str]
    employee_refs: List[str]
    severity: Literal["info", "warning", "critical"]


class Findings(BaseModel):
    findings: List[Finding]
    gaps: List[str]


class Action(BaseModel):
    title: str
    detail: str
    owner: str
    deadline: Optional[str] = None
    priority: Literal["low", "medium", "high"]
    policy_refs: List[str]


class ActionPlan(BaseModel):
    summary: str
    risk_level: Literal["low", "medium", "high"]
    actions: List[Action]
    open_questions: List[str]


# ----------------------------------------------------------------------------- helpers
def safe(fn):
    """Any unexpected error becomes an escalation (fail closed) instead of a 500."""
    @functools.wraps(fn)
    async def wrapper(state: RAGState) -> Dict[str, Any]:
        try:
            return await fn(state)
        except GraphBubbleUp:      # interrupt() must propagate
            raise
        except router.LLMUnavailable as exc:
            get_tracer(state["trace_id"]).event(fn.__name__, error="llm_unavailable", detail=str(exc))
            return {"escalation": "The AI providers are unavailable right now, so no guidance-backed answer can be produced."}
        except Exception as exc:  # noqa: BLE001
            get_tracer(state["trace_id"]).event(fn.__name__, error=type(exc).__name__, detail=str(exc))
            return {"escalation": f"Internal error in step '{fn.__name__}'. Please retry or contact support."}
    return wrapper


def _cite(c: Dict[str, Any]) -> str:
    return f"{c['source_doc']} §{c['section']} ({c['guidance_version']})"


def _policy_block(chunks: List[Dict[str, Any]]) -> str:
    return "\n\n".join(f"[{c['label']}] {_cite(c)}\n{c['text']}" for c in chunks)


def _match(mention: str, roster: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hits = process.extract(mention, [r["name"] for r in roster], scorer=fuzz.WRatio, limit=3)
    return [{"employee_id": roster[i]["id"], "name": n, "score": round(float(s), 1)} for n, s, i in hits]


def _finalize(state: RAGState, picks: Dict[str, int]) -> Dict[str, Any]:
    """Assign refs, build the name map, and rewrite the question so no names reach later LLM calls."""
    roster = {r["id"]: r["name"] for r in emigrant.list_employees(state["employer_id"])}  # tenant-scoped
    question, refs, resolved, pii_map = state["sanitized_question"], {}, [], {}
    for mention, emp_id in picks.items():
        if emp_id not in roster:           # a selection outside this employer's roster is ignored
            continue
        if emp_id not in refs:
            refs[emp_id] = f"E{len(refs) + 1}"
            resolved.append({"ref": refs[emp_id], "employee_id": emp_id})
            pii_map[refs[emp_id]] = roster[emp_id]
        for text in {mention, roster[emp_id]}:
            question = re.sub(re.escape(text), refs[emp_id], question, flags=re.I)
    update: Dict[str, Any] = {"resolved": resolved, "pii_map": pii_map, "llm_question": question, "ambiguous": []}
    if not resolved:
        update["escalation"] = "No matching employee was selected, so no employee-specific analysis was run."
    return update


# ----------------------------------------------------------------------------- nodes
@safe
async def sanitize(state: RAGState) -> Dict[str, Any]:
    text, n = mask_pii(state["question"])
    get_tracer(state["trace_id"]).event("sanitize", question=text, pii_masked=n, employer_id=state["employer_id"])
    return {"sanitized_question": text, "pii_masked_input": n, "today": date.today().isoformat(), "regen_count": 0}


SYSTEM_CLASSIFY = (
    "You triage questions from HR teams at UK visa-sponsoring employers about sponsor-licence compliance. "
    "Classify strictly. Do not answer the question. Never invent names."
)


@safe
async def classify(state: RAGState) -> Dict[str, Any]:
    prompt = (
        f"Today's date: {state['today']}\nQuestion: {state['sanitized_question']}\n\n"
        "Fields:\n"
        "- issue_type: one label from the schema; use out_of_scope if unrelated to sponsor compliance.\n"
        "- confidence: 0-1 in your classification.\n"
        "- scope: 'employee' if about specific named people, 'workforce' if about the whole workforce/alerts "
        "without naming anyone, else 'policy_only'.\n"
        "- employee_mentions: exact name strings as written in the question (empty if none).\n"
        "- letter_date: YYYY-MM-DD only if the question refers to a specific UKVI letter/guidance date, else null.\n"
        "- sub_queries: 1-3 short, self-contained search queries for the sponsor guidance (no employee names)."
    )
    res, meta = await router.complete_structured(Classification, SYSTEM_CLASSIFY, prompt, temperature=0.0)
    as_of = state.get("query_date") or res.letter_date or state["today"]
    try:
        date.fromisoformat(as_of[:10])
        as_of = as_of[:10]
    except ValueError:
        as_of = state["today"]
    get_tracer(state["trace_id"]).llm(
        "classify", meta, issue_type=res.issue_type, confidence=res.confidence, scope=res.scope,
        mentions=len(res.employee_mentions), as_of=as_of, sub_queries=res.sub_queries,
    )
    update: Dict[str, Any] = {
        "issue_type": res.issue_type, "confidence": res.confidence, "scope": res.scope,
        "employee_mentions": res.employee_mentions, "sub_queries": res.sub_queries, "as_of": as_of,
        "llm_question": state["sanitized_question"],
    }
    if res.issue_type == "out_of_scope":
        update["escalation"] = "This looks outside sponsor-compliance guidance, so it was not answered."
    elif res.confidence < settings.CLASSIFY_MIN_CONFIDENCE:
        update["escalation"] = f"I'm not confident enough (confidence {res.confidence:.2f}) in what is being asked. Please rephrase."
    return update


@safe
async def resolve_employees(state: RAGState) -> Dict[str, Any]:
    tr = get_tracer(state["trace_id"])
    mentions = state.get("employee_mentions") or []
    if not mentions:
        return {"escalation": "I couldn't tell which employee you mean. Please include the employee's name."}
    roster = emigrant.list_employees(state["employer_id"])
    picks: Dict[str, int] = {}
    ambiguous: List[Dict[str, Any]] = []
    for m in mentions:
        cands = _match(m, roster)
        strong = [c for c in cands if c["score"] >= settings.FUZZY_THRESHOLD]
        clear = len(strong) == 1 or (len(strong) > 1 and strong[0]["score"] - strong[1]["score"] >= settings.FUZZY_AMBIGUITY_GAP)
        if strong and clear:
            picks[m] = strong[0]["employee_id"]
        else:
            ambiguous.append({"mention": m, "candidates": (strong or [c for c in cands if c["score"] >= 50])[:3]})
    tr.event("resolve_employees", mentions=len(mentions), auto_resolved=len(picks), ambiguous=len(ambiguous))
    if ambiguous:
        return {"ambiguous": ambiguous, "picks": picks}
    return {**_finalize(state, picks), "picks": picks}


async def human_confirm(state: RAGState) -> Dict[str, Any]:
    answer = interrupt({"type": "confirm_employee", "ambiguous": state["ambiguous"]})
    tr = get_tracer(state["trace_id"])
    if answer.get("cancel"):
        tr.event("human_confirm", result="cancelled")
        return {"escalation": "Employee confirmation was cancelled; no analysis was run."}
    picks = dict(state.get("picks") or {})
    for mention, emp_id in (answer.get("selections") or {}).items():
        if emp_id is not None:
            picks[mention] = int(emp_id)
    tr.event("human_confirm", result="confirmed", selections=len(picks))
    return {**_finalize(state, picks), "picks": picks}


@safe
async def retrieve_policy(state: RAGState) -> Dict[str, Any]:
    queries = (state.get("sub_queries") or [])[:3] or [state["sanitized_question"]]
    res = await retriever.retrieve_many(queries, state["as_of"])
    get_tracer(state["trace_id"]).event(
        "retrieve_policy", queries=queries, as_of=state["as_of"], status=res.status, detail=res.detail,
        top_k=len(res.chunks),
        chunks=[{"id": c["id"], "dense": round(c["dense_score"], 3), "rerank": c.get("rerank_score"),
                 "section": c["section"], "version": c["guidance_version"], "preview": c["text"][:90]}
                for c in res.chunks],
    )
    if res.status == "unavailable":
        return {"escalation": "Policy search is unavailable right now, so I can't give guidance-backed advice."}
    if res.status == "no_match":
        return {"escalation": f"I couldn't find indexed guidance relevant to this question that was valid on {state['as_of']}. "
                              "I won't answer from general knowledge."}
    chunks = []
    for i, c in enumerate(res.chunks, 1):
        c.pop("vector", None)
        chunks.append({**c, "label": f"P{i}"})
    return {"policy_chunks": chunks}


@safe
async def retrieve_emigrant(state: RAGState) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    update: Dict[str, Any] = {}
    if state["scope"] == "employee":
        for r in state.get("resolved", []):
            f = emigrant.get_employee_facts(state["employer_id"], r["employee_id"])
            if f:
                facts[r["ref"]] = f
    elif state["scope"] == "workforce":
        pii_map, resolved = {}, []
        for i, row in enumerate(emigrant.get_workforce_alerts(state["employer_id"]), 1):
            ref = f"E{i}"
            facts[ref] = row["facts"]
            pii_map[ref] = row["name"]
            resolved.append({"ref": ref, "employee_id": row["employee_id"]})
        update.update({"pii_map": pii_map, "resolved": resolved})
    get_tracer(state["trace_id"]).event(
        "retrieve_emigrant", scope=state["scope"], source="read-only parameterised SQL scoped by employer_id",
        rows=len(facts), deterministic=True,
    )
    return {**update, "employee_facts": facts}


SYSTEM_REASON = (
    "You are a UK sponsor-compliance analyst. Use ONLY the policy excerpts and employee facts provided. "
    "Do not use outside knowledge, do not recompute dates or numbers, and do not show your reasoning. "
    "Output findings only. Every finding must cite policy excerpt labels (e.g. P1) it relies on and employee refs "
    "(e.g. E1) it concerns. If the excerpts do not cover something the question needs, list it under gaps."
)


def _context(state: RAGState) -> str:
    facts = json.dumps(state.get("employee_facts") or {}, indent=1) if state.get("employee_facts") else "(none)"
    return (
        f"QUESTION: {state['llm_question']}\nGUIDANCE AS OF: {state['as_of']}\n\n"
        f"POLICY EXCERPTS (valid on that date):\n{_policy_block(state['policy_chunks'])}\n\n"
        f"EMPLOYEE FACTS (computed by the system, authoritative):\n{facts}"
    )


@safe
async def reason(state: RAGState) -> Dict[str, Any]:
    res, meta = await router.complete_structured(Findings, SYSTEM_REASON, _context(state), temperature=0.2)
    get_tracer(state["trace_id"]).llm("reason", meta, findings_count=len(res.findings), gaps=len(res.gaps))
    return {"findings": res.model_dump()}


SYSTEM_GENERATE = (
    "You write the final compliance action plan for an HR team from the findings provided. Be concrete and brief. "
    "Every action MUST list at least one policy_refs label (e.g. P1) that supports it, and may only use labels that "
    "exist. Refer to employees only by their refs (E1, E2). Do not introduce facts, numbers or deadlines that are "
    "not in the policy excerpts or employee facts. Put anything unresolved in open_questions."
)


@safe
async def generate_action_plan(state: RAGState) -> Dict[str, Any]:
    prompt = f"{_context(state)}\n\nFINDINGS:\n{json.dumps(state['findings'], indent=1)}"
    if state.get("guardrail_feedback"):
        prompt += f"\n\nYOUR PREVIOUS ATTEMPT FAILED VALIDATION: {state['guardrail_feedback']}\nFix this."
    res, meta = await router.complete_structured(ActionPlan, SYSTEM_GENERATE, prompt, temperature=0.1)
    get_tracer(state["trace_id"]).llm(
        "generate", meta, actions=len(res.actions), citations=sorted({r for a in res.actions for r in a.policy_refs}),
    )
    return {"plan": res.model_dump()}


def _numbers(text: str) -> set:
    return {n for n in re.findall(r"\d+(?:\.\d+)?", text) if len(n.replace(".", "")) >= 2}


@safe
async def guardrail(state: RAGState) -> Dict[str, Any]:
    tr = get_tracer(state["trace_id"])
    plan, findings = state["plan"], state["findings"]
    chunks = {c["label"]: c for c in state["policy_chunks"]}
    pii_map = state.get("pii_map", {})
    problems: List[str] = []

    if not plan["actions"] and not plan["open_questions"]:
        problems.append("The plan has no actions and no open questions.")
    for a in plan["actions"]:
        if not a["policy_refs"]:
            problems.append(f"Action '{a['title']}' has no policy citation.")
        bad = [r for r in a["policy_refs"] if r not in chunks]
        if bad:
            problems.append(f"Action '{a['title']}' cites unknown labels {bad}; valid labels: {sorted(chunks)}.")
    all_text = json.dumps(plan) + json.dumps(findings)
    unknown = sorted({r for r in re.findall(r"\bE\d+\b", all_text) if r not in pii_map})
    if unknown:
        problems.append(f"Unknown employee refs used: {unknown}.")

    # Soft check: numbers in the output that appear nowhere in the context are surfaced as warnings.
    corpus = _policy_block(state["policy_chunks"]) + json.dumps(state.get("employee_facts") or {}) + state["llm_question"]
    ungrounded = sorted(_numbers(json.dumps(plan)) - _numbers(corpus))
    warnings = [f"Number(s) {ungrounded} in the plan are not present in the retrieved context - verify before acting."] if ungrounded else []

    if problems:
        attempt = state.get("regen_count", 0) + 1
        tr.event("guardrail", citation_check="fail", problems=problems, attempt=attempt)
        update: Dict[str, Any] = {"guardrail_ok": False, "guardrail_feedback": " ".join(problems), "regen_count": attempt}
        if attempt > settings.MAX_REGENERATIONS:
            update["escalation"] = "The generated answer failed citation checks after retries, so it was withheld."
        return update

    def render(text: str) -> str:
        return mask_pii(re.sub(r"\bE\d+\b", lambda m: pii_map.get(m.group(0), m.group(0)), text))[0]

    cited = {r for a in plan["actions"] for r in a["policy_refs"]} | {r for f in findings["findings"] for r in f["policy_refs"]}
    final = {
        "status": "ok", "trace_id": state["trace_id"], "as_of": state["as_of"], "issue_type": state["issue_type"],
        "summary": render(plan["summary"]), "risk_level": plan["risk_level"], "warnings": warnings,
        "actions": [
            {"title": render(a["title"]), "detail": render(a["detail"]), "owner": render(a["owner"]),
             "deadline": a["deadline"], "priority": a["priority"],
             "citations": [_cite(chunks[r]) for r in a["policy_refs"] if r in chunks]}
            for a in plan["actions"]
        ],
        "findings": [
            {"statement": render(f["statement"]), "severity": f["severity"], "basis": f["basis"],
             "citations": [_cite(chunks[r]) for r in f["policy_refs"] if r in chunks]}
            for f in findings["findings"]
        ],
        "open_questions": [render(q) for q in plan["open_questions"]], "gaps": [render(g) for g in findings["gaps"]],
        "sources": [
            {"label": c["label"], "citation": _cite(c), "url": c.get("url", ""), "valid_from": c["valid_from"],
             "text": c["text"], "dense_score": round(c["dense_score"], 3), "cited": c["label"] in cited}
            for c in state["policy_chunks"]
        ],
    }
    tr.event("guardrail", citation_check="pass", pii_masked_input=state.get("pii_masked_input", 0),
             risk_level=plan["risk_level"], warnings=len(warnings))
    return {"guardrail_ok": True, "final": final, "warnings": warnings}


async def escalate(state: RAGState) -> Dict[str, Any]:
    reason_text = state.get("escalation") or "Escalated for human review."
    get_tracer(state["trace_id"]).event("escalate", reason=reason_text)
    return {"final": {"status": "escalated", "reason": reason_text, "trace_id": state["trace_id"]}}
