# UKVI Compliance Co-Pilot (RAG)

FastAPI + LangGraph + Qdrant (hybrid dense + BM25) + Jina embeddings (jina-embeddings-v5-omni-small, 1024-d) + Gemini 2.5 Flash for generation (Groq gpt-oss-120b fallback) + RAGAS, with a plain-HTML test UI.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add JINA_API_KEY (embeddings) and GEMINI_API_KEY and/or GROQ_API_KEY
python -m app.ingestion.ingest  # indexes the two SYNTHETIC sample guidance versions in data/raw
uvicorn app.main:app --reload   # open http://localhost:8000
```
Qdrant runs embedded (on disk at `data/qdrant`) when `QDRANT_URL` is empty. For a server: `docker run -p 6333:6333 qdrant/qdrant`, then set `QDRANT_URL=http://localhost:6333`.

**The sample documents are synthetic test data, not real UKVI guidance.** Replace them with official sponsor guidance (PDF/HTML via the UI "Documents" tab or `data/raw/manifest.json`).

**Changing embedding model/dimension?** The Qdrant collection is fixed to `EMBEDDING_DIM`. Delete `data/qdrant` (or set a new `QDRANT_COLLECTION`) and re-ingest; the app refuses to start on a size mismatch.

## Flow
```
sanitize -> classify -> resolve_employees -> (human_confirm) -> retrieve_policy -> retrieve_emigrant
         -> reason -> generate_action_plan -> guardrail -> END        (any step -> escalate)
```
| Step | What it does |
|---|---|
| sanitize | Masks emails, phones, NI/passport-like numbers before any LLM call |
| classify | LLM: issue type, confidence, scope (employee / workforce / policy-only), employee mentions, letter date, 1-3 sub-queries. Confidence < 0.60 -> escalate |
| resolve_employees | rapidfuzz >= 85 against the tenant's roster; ambiguous or unmatched -> `interrupt()` for human confirmation |
| retrieve_policy | Hybrid search filtered to chunks valid on the as-of date; abstain threshold; no match -> escalate (never answers from model memory) |
| retrieve_emigrant | Deterministic parameterised SQL, tenant-scoped; derived facts (days to expiry, overdue flags) computed in Python |
| reason / generate | Structured JSON via Pydantic. The LLM sees employee refs (E1, E2), never names |
| guardrail | Every action must cite valid chunk labels; unknown refs fail; failures regenerate up to 2x then escalate. Names restored only here |

## Endpoints
`POST /query` · `POST /confirm` · `POST /ingest` · `GET /documents` · `GET /employees` · `GET /trace/{id}` · `POST /eval[?ragas=true]` · `GET /health`

Each run writes `logs/run_<trace_id>.jsonl` and `.txt` (provider, tokens, latency, retrieved chunks, citations, estimated cost). Strings in logs are PII-masked.

## Versioning
Each chunk has `valid_from` / `valid_to`. Ingesting a new version of the same `source_doc` automatically closes the previous one (`reconcile_versions`, order-independent). Pass an as-of date in the UI (or mention a letter date) to query older guidance.

## Evaluation
```bash
python -m app.eval.ragas_runner          # hit@k, MRR, abstain accuracy (no LLM judge)
pip install -r requirements-eval.txt
python -m app.eval.ragas_runner --ragas  # + faithfulness, answer relevancy, context precision/recall
```
Edit `app/eval/golden_set.jsonl` (replace with 30-50 real cases). The RAGAS judge defaults to Groq so it differs from the Gemini generator.
**Tune `MIN_DENSE_SCORE` (or `MIN_RERANK_SCORE` with `RERANK_ENABLED=true`) on your golden set** until abstain accuracy and hit@k are both acceptable.

## Tests (offline, no API keys)
```bash
python -m tests.smoke_test && python -m tests.test_api
```
Covers ingestion, version filtering, hybrid retrieval, human-in-the-loop, tenant scoping, escalation paths, citation guardrail, API endpoints, and trace output, using in-memory Qdrant, hash embeddings and a stubbed LLM.

## Not verified here
Real Jina/Gemini/Groq calls (Jina task names `retrieval.passage`/`retrieval.query` for v5-omni, model names, `reasoning_effort`), RAGAS, the cross-encoder reranker, Qdrant server mode, and PDF extraction quality on real guidance were not run. Check them first with your keys.

## Before production
- Derive `employer_id` from the authenticated session (the UI dropdown is a demo).
- Use a persistent LangGraph checkpointer (SQLite/Postgres) instead of `MemorySaver`.
- Use paid-tier / no-training API terms (ideally a UK/EU region) before sending real employee data.
- Add auth, rate limiting, and log retention rules; replace the SQLite mock with your eMigrant data access.
