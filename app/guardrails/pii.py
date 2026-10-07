"""Regex PII masking. Applied to user input (before any LLM call), logs, and final output."""
import re
from typing import Tuple

_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("NI_NUMBER", re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b", re.I)),
    ("PHONE", re.compile(r"(?<!\d)(?:\+44\s?|0)\d(?:[\s-]?\d){8,10}(?!\d)")),
    ("PASSPORT", re.compile(r"(?<![\d-])\d{9}(?![\d-])")),
]


def mask_pii(text: str) -> Tuple[str, int]:
    count = 0
    for label, pattern in _PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        count += n
    return text, count
