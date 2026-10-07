"""Jina embeddings (jina-embeddings-v5-omni-small, 1024-d, normalised).

Documents and queries use different `task` values. Text is sent as {"text": ...} items (omni input format).
"""
import asyncio
from typing import List, Optional

import httpx

from app.config import get_settings

settings = get_settings()
_TRANSPORT: Optional[httpx.AsyncBaseTransport] = None   # tests inject a mock transport here


class EmbeddingError(RuntimeError):
    pass


async def _embed(texts: List[str], task: str) -> List[List[float]]:
    if not settings.JINA_API_KEY:
        raise EmbeddingError("JINA_API_KEY is not set")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {settings.JINA_API_KEY}"}
    out: List[List[float]] = []
    async with httpx.AsyncClient(timeout=60, transport=_TRANSPORT) as client:
        for start in range(0, len(texts), settings.JINA_BATCH_SIZE):
            batch = texts[start:start + settings.JINA_BATCH_SIZE]
            body = {"model": settings.JINA_MODEL, "task": task, "normalized": True,
                    "input": [{"text": t} for t in batch]}
            if settings.JINA_SEND_DIMENSIONS:
                body["dimensions"] = settings.EMBEDDING_DIM
            for attempt in range(4):
                resp = await client.post(settings.JINA_API_URL, headers=headers, json=body)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                break
            if resp.status_code != 200:
                raise EmbeddingError(f"Jina API {resp.status_code}: {resp.text[:300]}")
            rows = sorted(resp.json()["data"], key=lambda r: r["index"])
            vectors = [r["embedding"] for r in rows]
            if len(vectors) != len(batch):
                raise EmbeddingError(f"Jina returned {len(vectors)} embeddings for {len(batch)} inputs")
            if vectors and len(vectors[0]) != settings.EMBEDDING_DIM:
                raise EmbeddingError(
                    f"Jina returned {len(vectors[0])}-dim vectors but EMBEDDING_DIM={settings.EMBEDDING_DIM}. "
                    "Set EMBEDDING_DIM to match (and re-ingest), or set JINA_SEND_DIMENSIONS=true if the model supports it."
                )
            out.extend(vectors)
    return out


async def embed_documents(texts: List[str]) -> List[List[float]]:
    return await _embed(texts, settings.JINA_TASK_DOC)


async def embed_query(text: str) -> List[float]:
    return (await _embed([text], settings.JINA_TASK_QUERY))[0]
