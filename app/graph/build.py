"""StateGraph assembly.

START -> sanitize -> classify -> [resolve_employees -> (human_confirm)] -> retrieve_policy
      -> retrieve_emigrant -> reason -> generate_action_plan -> guardrail -> END
Any step can route to `escalate` (low confidence, ambiguous/cancelled, no guidance, provider failure,
citation failure after retries). guardrail loops back to generate_action_plan up to MAX_REGENERATIONS.
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import RAGState


def _guard(next_node: str):
    return lambda s: "escalate" if s.get("escalation") else next_node


def _after_classify(s):
    if s.get("escalation"):
        return "escalate"
    return "resolve_employees" if s["scope"] == "employee" else "retrieve_policy"


def _after_resolve(s):
    if s.get("ambiguous"):
        return "human_confirm"
    return "escalate" if s.get("escalation") else "retrieve_policy"


def _after_retrieve_policy(s):
    if s.get("escalation"):
        return "escalate"
    return "reason" if s["scope"] == "policy_only" else "retrieve_emigrant"


def _after_guardrail(s):
    if s.get("escalation"):
        return "escalate"
    return END if s.get("guardrail_ok") else "generate_action_plan"


def build_graph(checkpointer=None):
    g = StateGraph(RAGState)
    for name in ("sanitize", "classify", "resolve_employees", "human_confirm", "retrieve_policy",
                 "retrieve_emigrant", "reason", "generate_action_plan", "guardrail", "escalate"):
        g.add_node(name, getattr(nodes, name))
    g.add_edge(START, "sanitize")
    g.add_edge("sanitize", "classify")
    g.add_conditional_edges("classify", _after_classify, ["escalate", "resolve_employees", "retrieve_policy"])
    g.add_conditional_edges("resolve_employees", _after_resolve, ["human_confirm", "escalate", "retrieve_policy"])
    g.add_conditional_edges("human_confirm", _guard("retrieve_policy"), ["escalate", "retrieve_policy"])
    g.add_conditional_edges("retrieve_policy", _after_retrieve_policy, ["escalate", "reason", "retrieve_emigrant"])
    g.add_conditional_edges("retrieve_emigrant", _guard("reason"), ["escalate", "reason"])
    g.add_conditional_edges("reason", _guard("generate_action_plan"), ["escalate", "generate_action_plan"])
    g.add_conditional_edges("generate_action_plan", _guard("guardrail"), ["escalate", "guardrail"])
    g.add_conditional_edges("guardrail", _after_guardrail, ["escalate", "generate_action_plan", END])
    g.add_edge("escalate", END)
    # MemorySaver is fine for dev; use a SQLite/Postgres checkpointer so paused runs survive restarts.
    return g.compile(checkpointer=checkpointer or MemorySaver())
