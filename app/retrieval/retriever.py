"""Hybrid retrieval: dense + BM25 (RRF) -> optional rerank -> abstain threshold.

Fail-closed: status is "ok" | "no_match" | "unavailable". Callers must never answer policy
questions from model memory when status != "ok".
"""
import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.llm import embeddings
from app.retrieval import qdrant_store, reranker, sparse

settings = get_settings()


@dataclass
class Retrieval:
    status: str
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    detail: str = ""


def _cosine(a: List[float], b: List[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


async def retrieve_many(
    queries: List[str], as_of: Any, top_k: Optional[int] = None, source_doc: Optional[str] = None
) -> Retrieval:
    k = top_k or settings.TOP_K
    queries = [q for q in dict.fromkeys(queries) if q.strip()]  # de-dupe, keep order
    try:
        dense_vecs = await asyncio.gather(*(embeddings.embed_query(q) for q in queries))
        batches = await asyncio.gather(*(
            qdrant_store.hybrid_search(dv, sparse.encode_query(q), as_of, settings.CANDIDATE_K, source_doc)
            for q, dv in zip(queries, dense_vecs)
        ))
    except Exception as exc:  # noqa: BLE001
        return Retrieval("unavailable", detail=f"{type(exc).__name__}: {exc}")

    merged: Dict[str, Dict[str, Any]] = {}
    for q, dv, hits in zip(queries, dense_vecs, batches):
        for h in hits:
            h["dense_score"] = _cosine(dv, h.pop("vector", None) or [])
            h["matched_query"] = q
            if h["id"] not in merged or h["dense_score"] > merged[h["id"]]["dense_score"]:
                merged[h["id"]] = h
    pool = list(merged.values())
    if not pool:
        return Retrieval("no_match", detail="no chunk valid on the requested date")

    scores = await reranker.rerank(queries[0], [c["text"] for c in pool])
    if scores is not None:
        for c, s in zip(pool, scores):
            c["rerank_score"] = s
        key, threshold = "rerank_score", settings.MIN_RERANK_SCORE
    else:
        key, threshold = "dense_score", settings.MIN_DENSE_SCORE
    pool.sort(key=lambda c: c[key], reverse=True)
    kept = [c for c in pool if c[key] >= threshold][:k]
    if not kept:
        return Retrieval("no_match", detail=f"best {key}={pool[0][key]:.2f} below threshold {threshold}")
    return Retrieval("ok", kept)
