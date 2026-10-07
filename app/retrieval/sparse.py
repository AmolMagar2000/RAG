"""Dependency-free BM25-style sparse encoder.

Qdrant's `modifier: idf` multiplies stored values by IDF at query time, so we only emit the
BM25 term-frequency component. Exact regulatory terms ("SOC 2136", "10 working days") survive
tokenisation, which is the point of hybrid search. Swap for fastembed `Qdrant/bm25` if preferred.
"""
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

_STOP = set(
    "a an the and or of to in on for with by at from as is are was were be been it this that these those "
    "if then than so such not no any all can could should would may might will shall do does did has have had".split()
)


@dataclass
class Sparse:
    indices: List[int]
    values: List[float]


def tokenize(text: str) -> List[str]:
    out = []
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOP:
            continue
        if not tok.isdigit():
            for suffix in ("ings", "ing", "edly", "ed", "es", "s"):
                if tok.endswith(suffix) and len(tok) - len(suffix) >= 4:
                    tok = tok[: -len(suffix)]
                    break
        out.append(tok)
    return out


def _index(token: str) -> int:
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "big") & 0x7FFFFFFF


def encode_document(text: str, k1: float = 1.2, b: float = 0.75, avgdl: float = 120.0) -> Sparse:
    tokens = tokenize(text)
    length = len(tokens) or 1
    weights: Dict[int, float] = {}
    for token, tf in Counter(tokens).items():
        idx = _index(token)
        weights[idx] = weights.get(idx, 0.0) + tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avgdl))
    return Sparse(list(weights.keys()), list(weights.values()))


def encode_query(text: str) -> Sparse:
    idxs = sorted({_index(t) for t in tokenize(text)})
    return Sparse(idxs, [1.0] * len(idxs))
