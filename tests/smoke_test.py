"""Offline end-to-end smoke test: in-memory Qdrant, hash embeddings, stubbed LLM. No API keys needed.

Run:  python -m tests.smoke_test      (or: pytest -q tests)
"""
import asyncio
import hashlib
import math
import os
import re
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.update({
    "QDRANT_URL": "", "QDRANT_PATH": ":memory:", "EMBEDDING_DIM": "64", "MIN_DENSE_SCORE": "0.10",
    "DB_PATH": f"{_tmp}/emigrant.db", "LOG_DIR": f"{_tmp}/logs", "RAW_DIR": f"{_tmp}/raw",
})

from pathlib import Path  # noqa: E402

from app.db import emigrant  # noqa: E402
from app.graph import nodes  # noqa: E402
from app.graph.build import build_graph  # noqa: E402
from app.ingestion.ingest import ingest_manifest  # noqa: E402
from app.llm import embeddings, router  # noqa: E402
from app.pipeline import resume_query, run_query  # noqa: E402
from app.retrieval import qdrant_store, retriever  # noqa: E402

DIM = 64


def _vec(text: str):
    v = [0.0] * DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


async def fake_embed_docs(texts):
    return [_vec(t) for t in texts]


async def fake_embed_query(text):
    return _vec(text)


MENTIONS = {"value": ["Priya Sharma"], "scope": "employee", "issue": "change_of_circumstances", "conf": 0.92}


async def fake_llm(schema, system, prompt, *, temperature=0.0):
    meta = router.LLMMeta("fake", "fake-model", 100, 40, 5.0, 0.0001)
    name = schema.__name__
    labels = re.findall(r"\[(P\d+)\]", prompt)
    who = "E1" if "E1" in prompt else "the employer"
    if name == "Classification":
        return nodes.Classification(
            issue_type=MENTIONS["issue"], confidence=MENTIONS["conf"], scope=MENTIONS["scope"],
            employee_mentions=MENTIONS["value"], letter_date=None,
            sub_queries=["report change of work location working days"],
        ), meta
    if name == "Findings":
        return nodes.Findings(findings=[nodes.Finding(
            statement=f"{who} changing work location must be reported to the Home Office.", basis="both",
            policy_refs=labels[:1], employee_refs=["E1"] if who == "E1" else [], severity="warning")], gaps=[]), meta
    if name == "ActionPlan":
        return nodes.ActionPlan(summary=f"{who} needs a work location change report.", risk_level="medium", actions=[
            nodes.Action(title=f"Report location change for {who}", detail="Submit the report through the sponsorship system.",
                         owner="HR", deadline=None, priority="high", policy_refs=labels[:1])], open_questions=[]), meta
    raise AssertionError(name)


async def main():
    embeddings.embed_documents, embeddings.embed_query = fake_embed_docs, fake_embed_query
    router.complete_structured = fake_llm

    emigrant.init_db()
    await qdrant_store.ensure_collection()
    manifest = Path(__file__).resolve().parents[1] / "data/raw/manifest.json"
    print("ingested:", await ingest_manifest(manifest))
    docs = await qdrant_store.list_documents()
    assert len(docs) == 2 and sum(d["is_current"] for d in docs) == 1, docs

    # --- version-aware retrieval -------------------------------------------------
    q = "within how many working days must a change of work location be reported"
    old = await retriever.retrieve_many([q], "2024-06-01")
    new = await retriever.retrieve_many([q], "2024-12-01")
    assert old.status == "ok" and "10 working days of the change" in old.chunks[0]["text"], old.chunks[:1]
    assert new.status == "ok" and "7 working days of the change" in new.chunks[0]["text"], new.chunks[:1]
    before = await retriever.retrieve_many([q], "2023-01-01")
    assert before.status == "no_match", before   # nothing valid before the first version
    print("version filter OK")

    graph = build_graph()

    # --- happy path: resolved employee, names restored only in the final output ----
    res = await run_query(graph, "Priya Sharma is moving to Bradford, what must we report?", 1, "2024-12-01")
    assert res["status"] == "ok", res
    assert "Priya Sharma" in res["summary"] and "E1" not in res["summary"], res["summary"]
    assert res["actions"][0]["citations"] and res["sources"][0]["cited"]
    print("happy path OK:", res["summary"])

    # --- tenant isolation: same name exists in employer 2 but is a different person ----
    f1 = emigrant.get_employee_facts(1, 1)
    f2 = emigrant.get_employee_facts(2, 1)  # id 1 belongs to employer 1 only
    assert f1 and f2 is None
    print("tenant scoping OK")

    # --- ambiguous -> human confirm -> resume --------------------------------------
    MENTIONS["value"] = ["Priya Sh"]
    paused = await run_query(graph, "Priya Sh needs a location change report?", 1, "2024-12-01")
    assert paused["status"] == "needs_confirmation", paused
    cands = paused["request"]["ambiguous"][0]["candidates"]
    assert len(cands) >= 2, cands
    done = await resume_query(graph, paused["thread_id"], {"Priya Sh": cands[0]["employee_id"]})
    assert done["status"] == "ok", done
    # a selection from another tenant must be ignored
    paused = await run_query(graph, "Priya Sh needs a location change report?", 1, "2024-12-01")
    bad = await resume_query(graph, paused["thread_id"], {"Priya Sh": 9})  # id 9 = employer 2's Priya Sharma
    assert bad["status"] == "escalated", bad
    print("human-in-the-loop OK")

    # --- low confidence / out of scope / no guidance -> escalate -----------------------
    MENTIONS.update(value=[], scope="policy_only", conf=0.3)
    low = await run_query(graph, "hmm something", 1)
    assert low["status"] == "escalated" and "confident" in low["reason"], low
    MENTIONS.update(conf=0.9)
    retriever.settings.MIN_DENSE_SCORE = 0.95          # force the abstain path
    none = await run_query(graph, "zzzz qqqq xxxx", 1, "2024-12-01")
    retriever.settings.MIN_DENSE_SCORE = 0.10
    assert none["status"] == "escalated" and "couldn't find indexed guidance" in none["reason"], none
    print("escalation paths OK:", none["reason"])

    # --- citation guardrail: bad citation -> regenerate x2 -> escalate ---------------------
    orig = fake_llm

    async def bad_llm(schema, system, prompt, **kw):
        out, meta = await orig(schema, system, prompt, **kw)
        if schema.__name__ == "ActionPlan":
            out.actions[0].policy_refs = ["P99"]
        return out, meta

    router.complete_structured = bad_llm
    MENTIONS.update(scope="policy_only", conf=0.9)
    g = await run_query(graph, "how many working days to report a change of work location", 1, "2024-12-01")
    assert g["status"] == "escalated" and "citation" in g["reason"], g
    print("guardrail OK")

    from app.eval import ragas_runner
    router.complete_structured = fake_llm
    ev = await ragas_runner.retrieval_eval()
    print("retrieval eval:", {k: v for k, v in ev.items() if k != "rows"})
    await qdrant_store.close()
    print("\nALL SMOKE TESTS PASSED")


def test_pipeline():
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
