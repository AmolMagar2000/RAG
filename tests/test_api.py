"""API smoke test with the same offline stubs. Run: python -m tests.test_api"""
from tests.smoke_test import fake_embed_docs, fake_embed_query, fake_llm, MENTIONS  # sets env vars on import

from pathlib import Path

from fastapi.testclient import TestClient

from app.llm import embeddings, router


def main():
    embeddings.embed_documents, embeddings.embed_query = fake_embed_docs, fake_embed_query
    router.complete_structured = fake_llm
    from app.main import app

    raw = Path(__file__).resolve().parents[1] / "data/raw"
    with TestClient(app) as c:
        assert c.get("/").status_code == 200 and "Compliance Co-Pilot" in c.get("/").text
        assert c.get("/health").json()["qdrant"] is True
        assert len(c.get("/employees?employer_id=1").json()) == 6
        for f, v, d in (("sample_sponsor_guidance_v1.md", "S1", "2024-04-01"), ("sample_sponsor_guidance_v2.md", "S2", "2024-10-30")):
            r = c.post("/ingest", data={"source_doc": "SAMPLE", "guidance_version": v, "valid_from": d},
                       files={"file": (f, (raw / f).read_bytes())})
            assert r.status_code == 200, r.text
        assert c.post("/ingest", data={"source_doc": "x", "guidance_version": "x", "valid_from": "2024-01-01"},
                      files={"file": ("evil.exe", b"x")}).status_code == 400
        assert len(c.get("/documents").json()) == 2
        r = c.post("/query", json={"question": "Priya Sharma location change?", "employer_id": 1, "query_date": "2024-12-01"}).json()
        assert r["status"] == "ok" and "Priya Sharma" in r["summary"], r
        t = c.get(f"/trace/{r['trace_id']}")
        assert t.status_code == 200 and "run_summary" in t.text and "retrieve_policy" in t.text
        assert c.get("/trace/..%2F..%2Fetc").status_code in (400, 404)
        ev = c.post("/eval").json()
        assert ev["retrieval"]["hit_at_k"] == 1.0
        print(t.text)
    print("API TESTS PASSED")


if __name__ == "__main__":
    main()
