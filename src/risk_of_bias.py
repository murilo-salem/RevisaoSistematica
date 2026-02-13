"""
risk_of_bias.py — Automated risk-of-bias assessment for included studies.

Uses LLM to evaluate each study across standard RoB domains:
  • Selection bias
  • Performance bias
  • Detection bias
  • Attrition bias
  • Reporting bias

Each domain receives a rating (low / unclear / high) with justification.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from tqdm import tqdm

from utils import (
    ExtractionResult,
    RiskOfBiasItem,
    RiskOfBiasResult,
    StudyRecord,
    call_llm,
    save_json,
)

logger = logging.getLogger("systematic_review.risk_of_bias")

# ------------------------------------------------------------------ #
#  RoB prompt                                                          #
# ------------------------------------------------------------------ #

_ROB_PROMPT = """\
You are a methodological quality assessor for systematic reviews.

Evaluate the study below for risk of bias across these five domains.
For each domain assign a rating (low, unclear, or high) and give a
brief justification.

Return ONLY a valid JSON array:
[
  {{"domain": "selection", "rating": "low|unclear|high", "justification": "..."}},
  {{"domain": "performance", "rating": "low|unclear|high", "justification": "..."}},
  {{"domain": "detection", "rating": "low|unclear|high", "justification": "..."}},
  {{"domain": "attrition", "rating": "low|unclear|high", "justification": "..."}},
  {{"domain": "reporting", "rating": "low|unclear|high", "justification": "..."}}
]

### Study
Title: {title}
Abstract: {abstract}
Design: {design}
Sample size: {sample_size}
"""


def _parse_rob(raw: str, pmid: str) -> RiskOfBiasResult:
    """Parse the LLM risk-of-bias response."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        items_raw = json.loads(cleaned)
        if isinstance(items_raw, list):
            items = [RiskOfBiasItem(**it) for it in items_raw]
            return RiskOfBiasResult(pmid=pmid, domains=items)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("RoB parse error for %s: %s", pmid, exc)

    # Fallback: all unclear
    return RiskOfBiasResult(
        pmid=pmid,
        domains=[
            RiskOfBiasItem(domain=d, rating="unclear", justification="Could not parse LLM output")
            for d in ("selection", "performance", "detection", "attrition", "reporting")
        ],
    )


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def assess_risk_of_bias(
    studies: List[StudyRecord],
    extractions: List[ExtractionResult],
    cfg: Dict[str, Any],
) -> List[RiskOfBiasResult]:
    """Assess risk of bias for each included study.

    Returns a list of RiskOfBiasResult objects and persists them to JSON.
    """
    logger.info("Assessing risk of bias for %d studies", len(studies))

    # Build a lookup from PMID → extraction data
    ext_map: Dict[str, ExtractionResult] = {e.pmid: e for e in extractions}

    results: List[RiskOfBiasResult] = []

    for study in tqdm(studies, desc="Risk of bias"):
        ext = ext_map.get(study.pmid)
        prompt = _ROB_PROMPT.format(
            title=study.title,
            abstract=study.abstract,
            design=ext.study_design if ext else "unknown",
            sample_size=ext.sample_size if ext else "unknown",
        )

        raw = call_llm(prompt, cfg)
        rob = _parse_rob(raw, study.pmid)
        results.append(rob)

    save_json(
        [r.model_dump() for r in results],
        cfg["paths"]["risk_of_bias"],
    )

    logger.info("Risk-of-bias assessment complete")
    return results
