"""
screening.py — Two-pass automated study screening.

Pass 1: LLM-based inclusion/exclusion with justification.
Pass 2: Confidence scoring and threshold filtering.

Ambiguous studies (confidence between exclude and include thresholds)
are flagged for human review.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from tqdm import tqdm

from utils import (
    PICOModel,
    ScreeningDecision,
    StudyRecord,
    call_llm,
    get_db_connection,
    now_iso,
    save_json,
)

logger = logging.getLogger("systematic_review.screening")

# ------------------------------------------------------------------ #
#  Screening prompt                                                    #
# ------------------------------------------------------------------ #

_SCREEN_PROMPT = """\
You are a systematic-review screener.  Based on the PICO criteria
below, decide whether the following study should be INCLUDED or
EXCLUDED from the review.

### PICO Criteria
- Population: {population}
- Intervention: {intervention}
- Comparison: {comparison}
- Outcome: {outcome}

### Study
Title: {title}
Abstract: {abstract}

### Instructions
Return ONLY a valid JSON object:
{{
  "decision": "include" | "exclude",
  "confidence": <float 0-1>,
  "justification": "<brief reason>"
}}
"""


def _parse_screening(raw: str) -> Dict[str, Any]:
    """Parse the LLM screening response into a dict."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Regex fallback
    decision = "exclude"
    if re.search(r'"decision"\s*:\s*"include"', cleaned, re.IGNORECASE):
        decision = "include"

    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', cleaned)
    confidence = float(conf_match.group(1)) if conf_match else 0.5

    just_match = re.search(r'"justification"\s*:\s*"([^"]*)"', cleaned)
    justification = just_match.group(1) if just_match else cleaned[:200]

    return {
        "decision": decision,
        "confidence": confidence,
        "justification": justification,
    }


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def screen_studies(
    studies: List[StudyRecord],
    pico: PICOModel,
    cfg: Dict[str, Any],
) -> List[StudyRecord]:
    """Screen *studies* against *pico* criteria.

    Returns only the included studies.  All decisions (include, exclude,
    ambiguous) are stored in the database and in PRISMA JSON.
    """
    scr_cfg = cfg.get("screening", {})
    thresh_inc = scr_cfg.get("threshold_include", 0.75)
    thresh_exc = scr_cfg.get("threshold_exclude", 0.25)

    logger.info("Screening %d studies (inc≥%.2f, exc≤%.2f)", len(studies), thresh_inc, thresh_exc)

    included: List[StudyRecord] = []
    excluded: List[str] = []
    ambiguous: List[str] = []
    all_decisions: List[Dict[str, Any]] = []

    conn = get_db_connection(cfg)

    for study in tqdm(studies, desc="Screening"):
        prompt = _SCREEN_PROMPT.format(
            population=pico.population,
            intervention=pico.intervention,
            comparison=pico.comparison,
            outcome=pico.outcome,
            title=study.title,
            abstract=study.abstract,
        )

        raw = call_llm(prompt, cfg)
        parsed = _parse_screening(raw)

        decision_str = parsed.get("decision", "exclude").lower()
        confidence = float(parsed.get("confidence", 0.5))
        justification = parsed.get("justification", "")

        # Apply threshold logic
        if confidence >= thresh_inc and decision_str == "include":
            final = "include"
            included.append(study)
        elif confidence <= thresh_exc or decision_str == "exclude":
            final = "exclude"
            excluded.append(study.pmid)
        else:
            final = "ambiguous"
            ambiguous.append(study.pmid)
            # Include ambiguous for now — flagged for human review
            included.append(study)

        dec = ScreeningDecision(
            pmid=study.pmid,
            decision=final,
            confidence=confidence,
            justification=justification,
        )

        all_decisions.append(dec.model_dump())

        conn.execute(
            """
            INSERT OR REPLACE INTO screening_decisions
                (pmid, decision, confidence, justification, decided_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (study.pmid, final, confidence, justification, now_iso()),
        )

    conn.commit()
    conn.close()

    # PRISMA flow data
    prisma = {
        "total_screened": len(studies),
        "included": len([d for d in all_decisions if d["decision"] == "include"]),
        "excluded": len(excluded),
        "ambiguous": len(ambiguous),
        "ambiguous_pmids": ambiguous,
        "decisions": all_decisions,
    }
    save_json(prisma, cfg["paths"]["prisma"])

    logger.info(
        "Screening complete: %d included, %d excluded, %d ambiguous",
        len(included), len(excluded), len(ambiguous),
    )
    return included


# ------------------------------------------------------------------ #
#  Taxonomy-based screening (no LLM required)                          #
# ------------------------------------------------------------------ #

def screen_studies_by_taxonomy(
    studies: List[StudyRecord],
    taxonomy: Dict[str, Any],
    cfg: Dict[str, Any],
) -> List[StudyRecord]:
    """Screen *studies* using keyword rules from a taxonomy file.

    This does NOT require LLM.  It checks each study's title+abstract
    against the ``keywords``, ``inclusion_criteria``, and
    ``exclusion_criteria`` defined in the taxonomy JSON.

    Returns only the included studies.
    """
    keywords = [kw.lower() for kw in taxonomy.get("keywords", [])]
    inclusion = [c.lower() for c in taxonomy.get("inclusion_criteria", [])]
    exclusion = [c.lower() for c in taxonomy.get("exclusion_criteria", [])]

    # Classification rules (free-text patterns)
    include_rules = [r.lower() for r in taxonomy.get("classification_rules", {}).get("include_if_any", [])]
    exclude_rules = [r.lower() for r in taxonomy.get("classification_rules", {}).get("exclude_if_any", [])]

    logger.info("Taxonomy screening: %d studies, %d keywords, %d inc/%d exc criteria",
                len(studies), len(keywords), len(inclusion), len(exclusion))

    included: List[StudyRecord] = []
    excluded_ids: List[str] = []
    ambiguous_ids: List[str] = []
    all_decisions: List[Dict[str, Any]] = []

    conn = get_db_connection(cfg)

    for study in tqdm(studies, desc="Taxonomy screening"):
        text = f"{study.title} {study.abstract}".lower()

        # Score: how many keywords match
        kw_hits = sum(1 for kw in keywords if kw in text)
        kw_score = kw_hits / max(len(keywords), 1)

        # Check exclusion rules first
        excluded_by_rule = any(rule in text for rule in exclude_rules)

        # Check inclusion rules
        included_by_rule = any(rule in text for rule in include_rules)

        # Decision logic
        if excluded_by_rule and not included_by_rule:
            decision = "exclude"
            confidence = 0.8
            justification = "Matched exclusion rule"
            excluded_ids.append(study.pmid)
        elif kw_score >= 0.3 or included_by_rule:
            decision = "include"
            confidence = min(0.5 + kw_score, 1.0)
            justification = f"Keyword match score: {kw_score:.2f} ({kw_hits}/{len(keywords)} keywords)"
            included.append(study)
        elif kw_score >= 0.15:
            decision = "ambiguous"
            confidence = kw_score
            justification = f"Low keyword match: {kw_score:.2f} — flagged for review"
            ambiguous_ids.append(study.pmid)
            included.append(study)  # include ambiguous for review
        else:
            decision = "exclude"
            confidence = 1.0 - kw_score
            justification = f"Insufficient keyword match: {kw_score:.2f}"
            excluded_ids.append(study.pmid)

        dec = ScreeningDecision(
            pmid=study.pmid,
            decision=decision,
            confidence=confidence,
            justification=justification,
        )
        all_decisions.append(dec.model_dump())

        conn.execute(
            """
            INSERT OR REPLACE INTO screening_decisions
                (pmid, decision, confidence, justification, decided_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (study.pmid, decision, confidence, justification, now_iso()),
        )

    conn.commit()
    conn.close()

    # PRISMA flow data
    prisma = {
        "mode": "taxonomy",
        "total_screened": len(studies),
        "included": len([d for d in all_decisions if d["decision"] == "include"]),
        "excluded": len(excluded_ids),
        "ambiguous": len(ambiguous_ids),
        "ambiguous_pmids": ambiguous_ids,
        "decisions": all_decisions,
    }
    save_json(prisma, cfg["paths"]["prisma"])

    logger.info(
        "Taxonomy screening: %d included, %d excluded, %d ambiguous",
        len(included), len(excluded_ids), len(ambiguous_ids),
    )
    return included

