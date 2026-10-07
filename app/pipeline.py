"""Run / resume the graph and shape the API result."""
import uuid
from typing import Any, Dict, Optional

from app.logging_utils.tracer import get_tracer


async def _collect(graph, thread_id: str) -> Dict[str, Any]:
    cfg = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(cfg)
    if snap.next:  # paused on interrupt()
        payload = [i.value for t in snap.tasks for i in t.interrupts]
        return {"status": "needs_confirmation", "thread_id": thread_id, "trace_id": thread_id,
                "request": payload[0] if payload else {}}
    final = dict(snap.values.get("final") or {"status": "escalated", "reason": "No result produced.", "trace_id": thread_id})
    tracer = get_tracer(thread_id)
    final["run_summary"] = tracer.finish(final["status"])
    return final


async def run_query(graph, question: str, employer_id: int, query_date: Optional[str] = None) -> Dict[str, Any]:
    trace_id = uuid.uuid4().hex[:12]
    cfg = {"configurable": {"thread_id": trace_id}}
    await graph.ainvoke(
        {"trace_id": trace_id, "question": question, "employer_id": employer_id, "query_date": query_date}, cfg,
    )
    return await _collect(graph, trace_id)


async def resume_query(graph, thread_id: str, selections: Dict[str, Optional[int]], cancel: bool = False) -> Dict[str, Any]:
    from langgraph.types import Command

    cfg = {"configurable": {"thread_id": thread_id}}
    await graph.ainvoke(Command(resume={"selections": selections, "cancel": cancel}), cfg)
    return await _collect(graph, thread_id)
