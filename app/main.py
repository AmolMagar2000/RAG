"""FastAPI app: /query /confirm /ingest /documents /employees /trace /eval /health + test UI."""
import re
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import emigrant
from app.graph.build import build_graph
from app.ingestion.ingest import ingest_file
from app.ingestion.loader import ALLOWED
from app.pipeline import resume_query, run_query
from app.retrieval import qdrant_store

settings = get_settings()
TEMPLATE = Path(__file__).parent / "templates" / "index.html"
MAX_UPLOAD = 15 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    emigrant.init_db()
    await qdrant_store.ensure_collection()
    app.state.graph = build_graph()
    yield
    await qdrant_store.close()


app = FastAPI(title="UKVI Compliance Co-Pilot", lifespan=lifespan)


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    employer_id: int = 1     # DEMO ONLY: derive from the authenticated session in production
    query_date: Optional[date] = None


class ConfirmIn(BaseModel):
    thread_id: str = Field(pattern=r"^[a-f0-9]{12}$")
    selections: Dict[str, Optional[int]] = {}
    cancel: bool = False


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(TEMPLATE)


@app.get("/health")
async def health():
    try:
        docs = await qdrant_store.list_documents()
        qdrant_ok = True
    except Exception:  # noqa: BLE001
        docs, qdrant_ok = [], False
    return {"qdrant": qdrant_ok, "indexed_versions": len(docs), "jina_key": bool(settings.JINA_API_KEY), "gemini_key": bool(settings.GEMINI_API_KEY),
            "groq_key": bool(settings.GROQ_API_KEY), "reranker": settings.RERANK_ENABLED}


@app.post("/query")
async def query(body: QueryIn):
    return await run_query(app.state.graph, body.question, body.employer_id, body.query_date.isoformat() if body.query_date else None)


@app.post("/confirm")
async def confirm(body: ConfirmIn):
    return await resume_query(app.state.graph, body.thread_id, body.selections, body.cancel)


@app.get("/employers")
async def employers():
    return emigrant.list_employers()


@app.get("/employees")
async def employees(employer_id: int = 1):
    return emigrant.list_employees(employer_id)


@app.get("/documents")
async def documents():
    return await qdrant_store.list_documents()


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...), source_doc: str = Form(...), guidance_version: str = Form(...),
    valid_from: date = Form(...), url: str = Form(""),
):
    name = Path(file.filename or "upload").name
    if Path(name).suffix.lower() not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type. Allowed: {sorted(ALLOWED)}")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File too large (15 MB max)")
    dest = Path(settings.RAW_DIR) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    try:
        return await ingest_file(dest, source_doc=source_doc, guidance_version=guidance_version,
                                 valid_from=valid_from.isoformat(), url=url)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Ingestion failed: {type(exc).__name__}: {exc}")


@app.get("/trace/{trace_id}", response_class=PlainTextResponse)
async def trace(trace_id: str):
    if not re.fullmatch(r"[a-f0-9]{12}", trace_id):
        raise HTTPException(400, "Bad trace id")
    path = Path(settings.LOG_DIR) / f"run_{trace_id}.txt"
    if not path.exists():
        raise HTTPException(404, "Trace not found")
    return path.read_text(encoding="utf-8")


@app.post("/eval")
async def run_eval(ragas: bool = False):
    from app.eval import ragas_runner

    out = {"retrieval": await ragas_runner.retrieval_eval()}
    if ragas:
        try:
            out["ragas"] = await ragas_runner.ragas_eval(app.state.graph)
        except Exception as exc:  # noqa: BLE001
            out["ragas"] = {"error": str(exc)}
    return JSONResponse(out)
