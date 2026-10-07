"""Ingestion: load -> chunk -> embed (dense + sparse) -> upsert -> reconcile version validity.

CLI:  python -m app.ingestion.ingest --manifest data/raw/manifest.json
      python -m app.ingestion.ingest --dir data/raw/my_pdfs --valid-from 2026-01-01
"""
import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from app.ingestion.chunker import chunk_document
from app.ingestion.loader import ALLOWED, load_text
from app.llm import embeddings
from app.retrieval import qdrant_store, sparse


async def ingest_text(
    text: str, *, source_doc: str, guidance_version: str, valid_from: str, url: str = ""
) -> Dict[str, Any]:
    await qdrant_store.ensure_collection()
    doc_hash = hashlib.sha256(f"{source_doc}|{guidance_version}|{text}".encode()).hexdigest()[:16]
    valid_from_iso = qdrant_store.iso(qdrant_store.parse_dt(valid_from))
    chunks = chunk_document(text)
    if not chunks:
        raise ValueError("No text could be extracted from the document")
    payloads, embed_inputs = [], []
    for ch in chunks:
        payloads.append({
            "text": ch["text"], "source_doc": source_doc, "section": ch["section"], "section_path": ch["section_path"],
            "url": url, "guidance_version": guidance_version, "valid_from": valid_from_iso,
            "valid_to": qdrant_store.NO_END, "is_current": True, "chunk_index": ch["chunk_index"], "doc_hash": doc_hash,
        })
        # The section path is embedded WITH the text so both dense and BM25 see the context.
        embed_inputs.append(f"{source_doc} > {ch['section_path']}\n{ch['text']}")
    dense = await embeddings.embed_documents(embed_inputs)
    await qdrant_store.upsert_chunks(payloads, dense, [sparse.encode_document(t) for t in embed_inputs])
    await qdrant_store.reconcile_versions(source_doc)
    return {"source_doc": source_doc, "guidance_version": guidance_version, "chunks": len(chunks), "doc_hash": doc_hash}


async def ingest_file(path: Path, **meta: Any) -> Dict[str, Any]:
    return await ingest_text(load_text(path), **meta)


async def ingest_manifest(manifest: Path) -> list:
    entries = json.loads(manifest.read_text())
    results = []
    for e in sorted(entries, key=lambda x: x["valid_from"]):
        results.append(await ingest_file(
            manifest.parent / e["file"], source_doc=e["source_doc"], guidance_version=e["guidance_version"],
            valid_from=e["valid_from"], url=e.get("url", ""),
        ))
    return results


def pretty_name(stem: str) -> str:
    """'Sponsor-guidance-Part-3_v1' -> 'Sponsor guidance Part 3 v1'"""
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", stem)).strip()


async def ingest_dir(folder: Path, *, valid_from: str, version: str) -> list:
    """Ingest every supported file in a folder. Each file becomes its OWN document (source_doc = file name),
    so unrelated PDFs never supersede each other. One failing file does not stop the rest."""
    results = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() not in ALLOWED:
            continue
        try:
            results.append(await ingest_file(
                f, source_doc=pretty_name(f.stem), guidance_version=version, valid_from=valid_from,
            ))
        except Exception as exc:  # noqa: BLE001
            results.append({"file": f.name, "error": f"{type(exc).__name__}: {exc}"})
    return results


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None, help="JSON manifest with per-file metadata")
    ap.add_argument("--dir", default=None, help="folder of PDFs/HTML/MD/TXT to ingest in one go")
    ap.add_argument("--valid-from", default="2000-01-01", help="used with --dir (2000-01-01 = always valid)")
    ap.add_argument("--version", default="v1", help="guidance version label used with --dir")
    args = ap.parse_args(argv)

    async def _run():
        try:
            if args.dir:
                results = await ingest_dir(Path(args.dir), valid_from=args.valid_from, version=args.version)
            else:
                results = await ingest_manifest(Path(args.manifest or "data/raw/manifest.json"))
            for r in results:
                print(r)
        finally:
            await qdrant_store.close()

    asyncio.run(_run())


if __name__ == "__main__":
    main()