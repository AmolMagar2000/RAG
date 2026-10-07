"""Verifies the Jina request/response handling with a mock transport (no network). Run: python -m tests.test_jina"""
import asyncio
import json
import os

os.environ.update({"JINA_API_KEY": "test-key", "EMBEDDING_DIM": "4"})

import httpx  # noqa: E402

from app.llm import embeddings  # noqa: E402

seen = []


def handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    seen.append((request.headers["authorization"], body))
    # return out of order to prove we sort by index
    data = [{"index": i, "embedding": [float(i), 0.0, 0.0, 1.0]} for i in range(len(body["input"]))][::-1]
    return httpx.Response(200, json={"data": data})


async def main():
    embeddings._TRANSPORT = httpx.MockTransport(handler)
    vecs = await embeddings.embed_documents(["a", "b", "c"])
    assert [v[0] for v in vecs] == [0.0, 1.0, 2.0], vecs
    auth, body = seen[-1]
    assert auth == "Bearer test-key" and body["model"] == "jina-embeddings-v5-omni-small"
    assert body["task"] == "retrieval.passage" and body["normalized"] is True
    assert body["input"] == [{"text": "a"}, {"text": "b"}, {"text": "c"}] and "dimensions" not in body
    q = await embeddings.embed_query("hello")
    assert seen[-1][1]["task"] == "retrieval.query" and len(q) == 4

    embeddings.settings.EMBEDDING_DIM = 8       # dimension mismatch must fail loudly
    try:
        await embeddings.embed_query("x")
        raise AssertionError("expected EmbeddingError")
    except embeddings.EmbeddingError as e:
        assert "EMBEDDING_DIM" in str(e)
    print("JINA CLIENT TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
