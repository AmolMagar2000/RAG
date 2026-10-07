"""Qdrant access (qdrant-client). Embedded mode when QDRANT_URL is empty, server/cloud otherwise.

Collection: named dense vector + BM25-style sparse vector (IDF modifier). Every chunk carries
valid_from / valid_to so queries can be answered "as of" the date of a UKVI letter.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient, models as qm

from app.config import get_settings
from app.retrieval.sparse import Sparse

settings = get_settings()
DENSE, SPARSE = "dense", "bm25"
NO_END = "2999-12-31T00:00:00Z"
_client: Optional[AsyncQdrantClient] = None


def parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif hasattr(value, "year"):
        dt = datetime(value.year, value.month, value.day)
    else:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        if settings.QDRANT_URL:
            _client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)
        elif settings.QDRANT_PATH == ":memory:":
            _client = AsyncQdrantClient(location=":memory:")
        else:
            _client = AsyncQdrantClient(path=settings.QDRANT_PATH)
    return _client


async def close() -> None:
    global _client
    if _client is not None:
        await _client.close()
    _client = None


async def ensure_collection() -> None:
    c, name = client(), settings.QDRANT_COLLECTION
    if await c.collection_exists(name):
        info = await c.get_collection(name)
        vectors = info.config.params.vectors
        size = vectors[DENSE].size if isinstance(vectors, dict) and DENSE in vectors else None
        if size != settings.EMBEDDING_DIM:
            raise RuntimeError(
                f"Collection '{name}' has dense size {size}, but EMBEDDING_DIM={settings.EMBEDDING_DIM}. "
                "Delete the old collection (e.g. remove data/qdrant) or use a new QDRANT_COLLECTION, then re-ingest."
            )
        return
    await c.create_collection(
        name,
        vectors_config={DENSE: qm.VectorParams(size=settings.EMBEDDING_DIM, distance=qm.Distance.COSINE)},
        sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
    )
    for field, schema in (
        ("valid_from", qm.PayloadSchemaType.DATETIME), ("valid_to", qm.PayloadSchemaType.DATETIME),
        ("source_doc", qm.PayloadSchemaType.KEYWORD), ("guidance_version", qm.PayloadSchemaType.KEYWORD),
    ):
        await c.create_payload_index(name, field_name=field, field_schema=schema)  # no-op in embedded mode


def point_id(doc_hash: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ukvi:{doc_hash}:{chunk_index}"))  # idempotent re-ingestion


async def upsert_chunks(chunks: List[Dict[str, Any]], dense: List[List[float]], sparse: List[Sparse]) -> None:
    points = [
        qm.PointStruct(
            id=point_id(ch["doc_hash"], ch["chunk_index"]),
            vector={DENSE: d, SPARSE: qm.SparseVector(indices=s.indices, values=s.values)},
            payload=ch,
        )
        for ch, d, s in zip(chunks, dense, sparse)
    ]
    for start in range(0, len(points), 64):
        await client().upsert(settings.QDRANT_COLLECTION, points=points[start:start + 64], wait=True)


def _version_filter(as_of: datetime, source_doc: Optional[str]) -> qm.Filter:
    must = [
        qm.FieldCondition(key="valid_from", range=qm.DatetimeRange(lte=as_of)),
        qm.FieldCondition(key="valid_to", range=qm.DatetimeRange(gte=as_of)),
    ]
    if source_doc:
        must.append(qm.FieldCondition(key="source_doc", match=qm.MatchValue(value=source_doc)))
    return qm.Filter(must=must)


async def hybrid_search(
    dense: List[float], sparse: Sparse, as_of: Any, k: int, source_doc: Optional[str] = None
) -> List[Dict[str, Any]]:
    flt = _version_filter(parse_dt(as_of), source_doc)
    prefetch = [qm.Prefetch(query=dense, using=DENSE, filter=flt, limit=k)]
    if sparse.indices:  # all-stopword queries produce an empty sparse vector
        prefetch.append(qm.Prefetch(
            query=qm.SparseVector(indices=sparse.indices, values=sparse.values), using=SPARSE, filter=flt, limit=k,
        ))
    res = await client().query_points(
        settings.QDRANT_COLLECTION, prefetch=prefetch, query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=k, with_payload=True, with_vectors=[DENSE],
    )
    out = []
    for p in res.points:
        vec = p.vector.get(DENSE) if isinstance(p.vector, dict) else p.vector
        out.append({"id": str(p.id), "fused_score": float(p.score or 0.0), "vector": vec, **(p.payload or {})})
    return out


async def _scroll_all(flt: Optional[qm.Filter], fields: List[str]) -> List[Any]:
    out, offset = [], None
    while True:
        points, offset = await client().scroll(
            settings.QDRANT_COLLECTION, scroll_filter=flt, limit=500, offset=offset,
            with_payload=fields, with_vectors=False,
        )
        out.extend(points)
        if offset is None:
            return out


async def list_documents() -> List[Dict[str, Any]]:
    seen: Dict[tuple, Dict[str, Any]] = {}
    for p in await _scroll_all(None, ["source_doc", "guidance_version", "valid_from", "valid_to", "is_current"]):
        pl = p.payload or {}
        key = (pl.get("source_doc"), pl.get("guidance_version"))
        row = seen.setdefault(key, {**pl, "chunks": 0})
        row["chunks"] += 1
    return sorted(seen.values(), key=lambda r: (str(r.get("source_doc")), str(r.get("valid_from"))))


async def reconcile_versions(source_doc: str) -> None:
    """Order-independent versioning: each version is valid until the next version's valid_from."""
    flt = qm.Filter(must=[qm.FieldCondition(key="source_doc", match=qm.MatchValue(value=source_doc))])
    versions: Dict[str, str] = {}
    for p in await _scroll_all(flt, ["guidance_version", "valid_from"]):
        pl = p.payload or {}
        versions[pl["guidance_version"]] = pl["valid_from"]
    ordered = sorted(versions.items(), key=lambda kv: parse_dt(kv[1]))
    for i, (version, _) in enumerate(ordered):
        last = i == len(ordered) - 1
        valid_to = NO_END if last else iso(parse_dt(ordered[i + 1][1]) - timedelta(seconds=1))
        await client().set_payload(
            settings.QDRANT_COLLECTION,
            payload={"valid_to": valid_to, "is_current": last},
            points=qm.Filter(must=[
                qm.FieldCondition(key="source_doc", match=qm.MatchValue(value=source_doc)),
                qm.FieldCondition(key="guidance_version", match=qm.MatchValue(value=version)),
            ]),
            wait=True,
        )
