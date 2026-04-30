"""
extraction.py — Structured data extraction from included studies.

For each included study LLM is prompted to extract key variables
(study design, sample size, intervention, outcomes, effect sizes…).
Results are validated against a Pydantic schema and saved as JSON.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from tqdm import tqdm

from utils import ExtractionResult, StudyRecord, call_llm, save_json

logger = logging.getLogger("systematic_review.extraction")

# ------------------------------------------------------------------ #
#  Extraction prompt                                                   #
# ------------------------------------------------------------------ #

_EXTRACT_PROMPT = """\
You are a data-extraction specialist for systematic reviews.

From the study below, extract the following variables and return
ONLY a valid JSON object.  If a value is not available, use null.

{{
  "study_design": "RCT | cohort | case-control | cross-sectional | other",
  "sample_size": <int or null>,
  "population": "<description of population>",
  "intervention": "<intervention description>",
  "comparison": "<comparator or control>",
  "outcome": "<primary outcome>",
  "effect_size": <float or null>,
  "ci_lower": <float or null>,
  "ci_upper": <float or null>,
  "p_value": <float or null>,
  "notes": "<any relevant comments>",
  "study_scale": "laboratory | pilot | industrial | field | simulation | null",
  "geographic_scope": "<country or region where the study was conducted, or null>",
  "funding_source": "<funding body or grant, or null if not disclosed>",
  "conflict_of_interest": "<COI statement, or null if not disclosed>",
  "limitations": "<key limitations stated by the authors, or null>"
}}

### Extraction guidance
- **study_scale**: Infer from the methodology. If the study describes
  bench-scale experiments, use "laboratory". If it mentions scale-up
  or larger reactors, use "pilot" or "industrial". Field trials = "field".
  Computational/modeling work = "simulation".
- **geographic_scope**: Look for author affiliations, study sites, or
  geographic references in the text.
- **limitations**: Summarise in one sentence the main limitations
  acknowledged by the authors.

### Study
Title: {title}
Abstract: {abstract}
"""


def _parse_extraction(raw: str, pmid: str) -> ExtractionResult | None:
    """Parse the LLM extraction response into an ExtractionResult."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        data = json.loads(cleaned)
        data["pmid"] = pmid
        return ExtractionResult(**{
            k: v for k, v in data.items()
            if k in ExtractionResult.model_fields
        })
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Failed to parse extraction for PMID %s: %s", pmid, exc)
        return None


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def extract_data(
    studies: List[StudyRecord],
    cfg: Dict[str, Any],
) -> List[ExtractionResult]:
    """Extract structured data from each included study.

    Returns a list of validated ExtractionResult objects.  Results are
    also saved to ``data/processed/extracted_data.json``.
    """
    logger.info("Extracting data from %d included studies", len(studies))

    extracted: List[ExtractionResult] = []
    errors: List[Dict[str, str]] = []

    for study in tqdm(studies, desc="Extracting data"):
        prompt = _EXTRACT_PROMPT.format(
            title=study.title,
            abstract=study.abstract,
        )
        raw = call_llm(prompt, cfg)
        result = _parse_extraction(raw, study.pmid)

        if result is not None:
            extracted.append(result)
        else:
            errors.append({"pmid": study.pmid, "error": "parse_failure"})

    # Persist results
    save_json(
        [e.model_dump() for e in extracted],
        cfg["paths"]["extracted"],
    )

    # Persist errors if any
    if errors:
        save_json(errors, "data/processed/extraction_errors.json")
        logger.warning("%d extraction errors logged", len(errors))

    logger.info("Extraction complete: %d/%d successful", len(extracted), len(studies))
    return extracted
