"""Per-run trace: logs/run_<id>.jsonl (machine) + logs/run_<id>.txt (human). Strings are PII-masked."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.config import get_settings
from app.guardrails.pii import mask_pii

_registry: Dict[str, "Tracer"] = {}


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii(value)[0]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class Tracer:
    def __init__(self, trace_id: str):
        folder = Path(get_settings().LOG_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        self.trace_id = trace_id
        self.jsonl = folder / f"run_{trace_id}.jsonl"
        self.txt = folder / f"run_{trace_id}.txt"
        self.tokens = 0
        self.cost = 0.0
        self.llm_ms = 0.0
        self.t0 = time.perf_counter()

    def event(self, name: str, **fields: Any) -> None:
        rec = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "event": name, **_clean(fields)}
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        body = " ".join(f"{k}={json.dumps(v, default=str)}" for k, v in rec.items() if k not in ("ts", "event"))
        with open(self.txt, "a", encoding="utf-8") as f:
            f.write(f"[{rec['ts']}] {name:<18} {body}\n")

    def llm(self, node: str, meta: Any, **extra: Any) -> None:
        self.tokens += meta.input_tokens + meta.output_tokens
        self.cost += meta.cost_usd
        self.llm_ms += meta.latency_ms
        self.event(
            node, provider=meta.provider, model=meta.model, input_tokens=meta.input_tokens,
            output_tokens=meta.output_tokens, latency_ms=round(meta.latency_ms),
            fallback_reason=meta.fallback_reason, **extra,
        )

    def finish(self, status: str) -> Dict[str, Any]:
        summary = {
            "status": status, "total_tokens": self.tokens,
            "total_latency_ms": round((time.perf_counter() - self.t0) * 1000),
            "llm_latency_ms": round(self.llm_ms), "cost_usd": round(self.cost, 5),
        }
        self.event("run_summary", **summary)
        return summary


def get_tracer(trace_id: str) -> Tracer:
    if trace_id not in _registry:
        _registry[trace_id] = Tracer(trace_id)
    return _registry[trace_id]
