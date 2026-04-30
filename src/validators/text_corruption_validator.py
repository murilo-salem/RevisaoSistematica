"""text_corruption_validator.py — Detect corrupted table/text artifacts."""

from __future__ import annotations

import re
from typing import List

from utils import ValidationFlag


_CORRUPTION_PATTERNS = [
    re.compile(r"/C\d+", re.IGNORECASE),
    re.compile(r"\bC\d+\s+C\b"),
    re.compile(r"\b\d+\s*/\s*\d+\s*/\s*\d+\b"),
    re.compile(r"\|\s*\|\s*\|"),
    re.compile(r"[\uFFFD]"),
]


def find_corruption_tokens(text: str) -> List[str]:
    hits: List[str] = []
    for pattern in _CORRUPTION_PATTERNS:
        for m in pattern.finditer(text):
            token = m.group(0)
            if token and token not in hits:
                hits.append(token)
    return hits


def validate_text(text: str, related_id: str) -> List[ValidationFlag]:
    tokens = find_corruption_tokens(text)
    if not tokens:
        return []

    severity = "high" if len(tokens) >= 3 else "medium"
    return [ValidationFlag(
        flag_type="text_corruption",
        severity=severity,
        message=f"Possível texto/tabela corrompido detectado: {', '.join(tokens[:6])}",
        related_ids=[related_id],
    )]


def sanitize_table_text(text: str) -> str:
    """Best-effort cleanup for known corruption patterns."""
    cleaned = text
    cleaned = re.sub(r"/C\d+", "", cleaned)
    cleaned = re.sub(r"\bC(\d+)\s+C\b", r"\1 C", cleaned)
    cleaned = re.sub(r"\|\s*\|\s*\|", "|", cleaned)
    cleaned = cleaned.replace("\uFFFD", "")
    return cleaned
