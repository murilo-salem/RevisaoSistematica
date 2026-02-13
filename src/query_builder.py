"""
query_builder.py — Generate a structured PICO question and a Boolean
search query from a free-text research topic.

The LLM is asked to return a JSON object so the output can be parsed
deterministically.  A fallback regex parser is included for robustness.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Tuple

from utils import PICOModel, call_llm, save_json

logger = logging.getLogger("systematic_review.query_builder")

# ------------------------------------------------------------------ #
#  PICO + Boolean query generation                                     #
# ------------------------------------------------------------------ #

_PICO_PROMPT = """\
You are a systematic-review methodologist.

Given the research topic below, produce:
1. A structured PICO question (Population, Intervention, Comparison, Outcome).
2. A Boolean search query suitable for PubMed (using AND, OR, and MeSH terms where appropriate).

Return ONLY a valid JSON object with these exact keys:
{{
  "population": "...",
  "intervention": "...",
  "comparison": "...",
  "outcome": "...",
  "query": "..."
}}

Research topic:
{topic}
"""


def _parse_llm_response(raw: str, topic: str) -> PICOModel:
    """Try to extract a PICOModel from the LLM response.

    First attempts ``json.loads``; falls back to regex extraction of
    key–value pairs if the JSON is malformed.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # Attempt 1 — direct JSON parse
    try:
        data = json.loads(cleaned)
        model = PICOModel(topic=topic, **{k: v for k, v in data.items() if k in PICOModel.model_fields})
        return model
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Attempt 2 — regex fallback
    fields: Dict[str, str] = {}
    for key in ("population", "intervention", "comparison", "outcome", "query"):
        match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', cleaned, re.IGNORECASE)
        if match:
            fields[key] = match.group(1)

    if fields:
        return PICOModel(topic=topic, **fields)

    # Attempt 3 — use entire response as the query
    logger.warning("Could not parse PICO JSON — using raw response as query")
    return PICOModel(
        topic=topic,
        population="",
        intervention="",
        comparison="",
        outcome="",
        query=cleaned[:500],
    )


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def build_query(topic: str, cfg: Dict[str, Any]) -> Tuple[PICOModel, str]:
    """Return ``(PICOModel, pubmed_query_string)`` for *topic*.

    The result is also persisted to ``data/processed/pico.json`` for the
    audit trail.
    """
    prompt = _PICO_PROMPT.format(topic=topic)
    logger.info("Generating PICO and search query for topic: %s", topic)

    raw = call_llm(prompt, cfg)
    pico = _parse_llm_response(raw, topic)

    # Persist for audit
    save_json(pico.model_dump(), cfg["paths"]["pico"])
    logger.info("PICO saved → %s", cfg["paths"]["pico"])
    logger.info("Boolean query: %s", pico.query)

    return pico, pico.query
