"""Section-aware chunker: split on headings, then pack paragraphs up to max_chars."""
import re
from typing import Dict, List

_MD = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_NUM = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+([A-Z][^\n]{2,100})$")


def _sentences(text: str) -> List[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def _pack(paragraphs: List[str], max_chars: int) -> List[str]:
    pieces: List[str] = []
    for para in paragraphs:
        pieces.extend([para] if len(para) <= max_chars else _sentences(para))
    chunks, buf = [], ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 2 > max_chars:
            chunks.append(buf)
            buf = ""
        buf = f"{buf}\n\n{piece}" if buf else piece
    if buf:
        chunks.append(buf)
    return chunks


def chunk_document(text: str, max_chars: int = 1100) -> List[Dict]:
    stack: List[str] = []
    sections: List[Dict] = []      # {path, label, lines}
    current = {"path": "", "label": "", "lines": []}

    def flush():
        if "".join(current["lines"]).strip():
            sections.append({**current, "lines": list(current["lines"])})

    for raw in text.splitlines():
        line = raw.strip()
        md, num = _MD.match(line), _NUM.match(line)
        if md or num:
            flush()
            if md:
                level, label = len(md.group(1)), md.group(2)
            else:
                level, label = num.group(1).count(".") + 1, f"{num.group(1)} {num.group(2)}"
            stack = stack[: level - 1] + [label]
            current = {"path": " > ".join(stack), "label": label, "lines": []}
        else:
            current["lines"].append(raw)
    flush()

    chunks: List[Dict] = []
    for sec in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(sec["lines"])) if p.strip()]
        for body in _pack(paragraphs, max_chars):
            chunks.append({
                "section": sec["label"], "section_path": sec["path"],
                "text": re.sub(r"[ \t]+", " ", body), "chunk_index": len(chunks),
            })
    return chunks
