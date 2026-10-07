"""Evaluation.

1. Retrieval metrics (hit@k, MRR, abstain accuracy): no LLM judge, fast, CI-friendly.
2. RAGAS (faithfulness, answer relevancy, context precision/recall): optional, needs requirements-eval.txt.
   The judge defaults to Groq so it differs from the Gemini generator (limits self-preference bias).

CLI: python -m app.eval.ragas_runner [--ragas]
"""
import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from app.config import get_settings
from app.retrieval import retriever

settings = get_settings()
GOLDEN = Path(__file__).with_name("golden_set.jsonl")


def load_golden() -> List[Dict[str, Any]]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


async def retrieval_eval() -> Dict[str, Any]:
    rows, hits, rr, abstain_ok, answerable = [], 0, 0.0, 0, 0
    golden = load_golden()
    for g in golden:
        res = await retriever.retrieve_many([g["question"]], g["query_date"])
        sections = [c["section"].split()[0] for c in res.chunks]
        if g["expect_abstain"]:
            ok = res.status != "ok"
            abstain_ok += ok
            rows.append({"q": g["question"], "expect": "abstain", "status": res.status, "pass": ok})
            continue
        answerable += 1
        rank = next((i + 1 for i, s in enumerate(sections) if s in g["expected_sections"]), None)
        hits += rank is not None
        rr += (1 / rank) if rank else 0.0
        rows.append({"q": g["question"], "as_of": g["query_date"], "sections": sections, "rank": rank, "pass": rank is not None})
    n_abstain = len(golden) - answerable
    return {
        "hit_at_k": round(hits / answerable, 3) if answerable else None,
        "mrr": round(rr / answerable, 3) if answerable else None,
        "abstain_accuracy": round(abstain_ok / n_abstain, 3) if n_abstain else None,
        "rows": rows,
    }


async def ragas_eval(graph) -> Dict[str, Any]:
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy
    except ImportError as exc:
        raise RuntimeError("RAGAS not installed: pip install -r requirements-eval.txt") from exc
    from app.pipeline import run_query

    samples = []
    for g in load_golden():
        if g["expect_abstain"]:
            continue
        out = await run_query(graph, g["question"], 1, g["query_date"])
        if out.get("status") != "ok":
            continue
        answer = out["summary"] + " " + " ".join(f"{a['title']}: {a['detail']}" for a in out["actions"])
        samples.append({"user_input": g["question"], "response": answer,
                        "retrieved_contexts": [s["text"] for s in out["sources"]], "reference": g["ground_truth"]})
    if not samples:
        return {"error": "No answerable samples produced an 'ok' result."}

    if settings.RAGAS_JUDGE == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        judge = ChatGoogleGenerativeAI(model=settings.GEMINI_MODEL, google_api_key=settings.GEMINI_API_KEY, temperature=0)
    else:
        from langchain_groq import ChatGroq
        judge = ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY, temperature=0)
    from langchain_core.embeddings import Embeddings
    from app.llm import embeddings as jina

    class JinaEmbeddings(Embeddings):          # RAGAS needs embeddings for answer relevancy; reuse Jina
        def embed_documents(self, texts):
            return asyncio.run(jina.embed_documents(texts))

        def embed_query(self, text):
            return asyncio.run(jina.embed_query(text))

    emb = JinaEmbeddings()

    result = await asyncio.to_thread(
        evaluate, EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=LangchainLLMWrapper(judge), embeddings=LangchainEmbeddingsWrapper(emb),
    )
    df = result.to_pandas()
    numeric = df.select_dtypes("number")
    return {"samples": len(samples), "scores": {c: round(float(numeric[c].mean()), 3) for c in numeric.columns}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ragas", action="store_true")
    args = ap.parse_args()

    async def _run():
        from app.retrieval import qdrant_store
        try:
            print(json.dumps(await retrieval_eval(), indent=2))
            if args.ragas:
                from app.db import emigrant
                from app.graph.build import build_graph
                emigrant.init_db()
                print(json.dumps(await ragas_eval(build_graph()), indent=2))
        finally:
            await qdrant_store.close()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
