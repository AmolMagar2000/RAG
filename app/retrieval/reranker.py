"""Optional cross-encoder reranker (RERANK_ENABLED=true). Needs: pip install sentence-transformers"""
import asyncio
from typing import List, Optional

from app.config import get_settings

settings = get_settings()
_model = None


async def rerank(query: str, texts: List[str]) -> Optional[List[float]]:
    global _model
    if not settings.RERANK_ENABLED or not texts:
        return None
    if _model is None:
        from sentence_transformers import CrossEncoder  # lazy: heavy import

        _model = CrossEncoder(settings.RERANKER_MODEL)  # single-label head -> sigmoid scores in 0..1
    scores = await asyncio.to_thread(_model.predict, [(query, t) for t in texts])
    return [float(s) for s in scores]
