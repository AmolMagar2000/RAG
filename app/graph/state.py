from typing import Any, Dict, List, Optional, TypedDict


class RAGState(TypedDict, total=False):
    # input
    trace_id: str
    question: str
    employer_id: int
    query_date: Optional[str]
    # sanitize / classify
    today: str
    sanitized_question: str
    pii_masked_input: int
    issue_type: str
    confidence: float
    scope: str                       # employee | workforce | policy_only
    employee_mentions: List[str]
    sub_queries: List[str]
    as_of: str                       # date used for version filtering
    # employee resolution
    resolved: List[Dict[str, Any]]   # [{ref, employee_id}]
    ambiguous: List[Dict[str, Any]]  # [{mention, candidates:[{employee_id,name,score}]}]
    picks: Dict[str, int]            # mention -> employee_id (confirmed or auto-resolved)
    pii_map: Dict[str, str]          # E1 -> real name (never sent to an LLM)
    llm_question: str                # question with names replaced by refs
    # retrieval
    policy_chunks: List[Dict[str, Any]]
    employee_facts: Dict[str, Any]   # ref -> facts
    # reasoning / generation
    findings: Dict[str, Any]
    plan: Dict[str, Any]
    guardrail_ok: bool
    guardrail_feedback: str
    regen_count: int
    warnings: List[str]
    # outcome
    escalation: str
    final: Dict[str, Any]
