"""paper_profiler.py — Structured evidence profiling for studies/chunks.

Builds canonical `PaperProfile` objects and deterministic validation flags,
then persists:
- data/processed/paper_profiles.json
- data/processed/validation_report.json
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from utils import AnchoredClaim, Chunk, PaperProfile, StudyRecord, ValidationFlag, save_json
from validators import (
    detect_domains,
    detect_phase_regimes,
    detect_system_topologies,
    detect_working_fluids,
    infer_pressure_range_bar,
    infer_temperature_range_k,
    validate_phase_co2,
    validate_text,
)

logger = logging.getLogger("systematic_review.paper_profiler")


_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "were", "was",
    "uma", "para", "com", "dos", "das", "nos", "nas", "como", "sobre",
}

_QUANT_RE = re.compile(
    r"\b\d+(?:[\.,]\d+)?\s*(?:K|°\s?C|º\s?C|bar|MPa|kPa|%)\b",
    re.IGNORECASE,
)
_QUAL_HINT_RE = re.compile(
    r"\b(increase|decrease|improv|redu|enhanc|higher|lower|melhor|pior|aument|reduz)\w*\b",
    re.IGNORECASE,
)


def _paper_id(study: StudyRecord) -> str:
    # Prefer pipeline-stable ID used across modules (PMID/filename key).
    if study.pmid:
        return study.pmid.strip()
    if study.doi:
        return study.doi.strip()
    return "unknown_paper"


def _extract_keywords(text: str, limit: int = 15) -> List[str]:
    words = re.findall(r"[A-Za-zÀ-ÿ0-9_\-]{4,}", text.lower())
    words = [w for w in words if w not in _STOPWORDS]
    if not words:
        return []
    cnt = Counter(words)
    return [w for w, _ in cnt.most_common(limit)]


def _chunk_sentence_claims(chunk: Chunk, max_per_chunk: int = 2) -> List[AnchoredClaim]:
    claims: List[AnchoredClaim] = []
    seen: set[str] = set()

    sentences = re.split(r"(?<=[\.!?])\s+", chunk.text or "")
    for sentence in sentences:
        s = sentence.strip()
        if len(s) < 30:
            continue

        claim_type = None
        if _QUANT_RE.search(s):
            claim_type = "quant"
        elif _QUAL_HINT_RE.search(s):
            claim_type = "qual"

        if claim_type is None:
            continue

        key = s.lower()
        if key in seen:
            continue
        seen.add(key)

        claims.append(AnchoredClaim(
            claim=s[:240],
            claim_type=claim_type,
            evidence_chunk_ids=[chunk.chunk_id],
        ))

        if len(claims) >= max_per_chunk:
            break

    return claims


def _build_profile(study: StudyRecord, study_chunks: List[Chunk]) -> PaperProfile:
    paper_id = _paper_id(study)
    combined_text = "\n".join([
        study.title or "",
        study.abstract or "",
        study.raw_text or "",
    ])

    domains = detect_domains(combined_text)
    fluids = detect_working_fluids(combined_text)
    regimes = detect_phase_regimes(combined_text)
    topologies = detect_system_topologies(combined_text)

    temp_range = infer_temperature_range_k(combined_text)
    press_range = infer_pressure_range_bar(combined_text)

    anchored_claims: List[AnchoredClaim] = []
    for ch in study_chunks:
        anchored_claims.extend(_chunk_sentence_claims(ch))

    # Keep a bounded list to avoid blowing up prompt contexts downstream.
    anchored_claims = anchored_claims[:50]

    keywords = _extract_keywords(combined_text)

    return PaperProfile(
        paper_id=paper_id,
        domains=domains,
        working_fluids=fluids,
        phase_regimes=regimes,
        temperature_range_K=temp_range,
        pressure_range_bar=press_range,
        system_topologies=topologies,
        keywords=keywords,
        anchored_claims=anchored_claims,
    )


def _build_validation_report(flags: List[ValidationFlag]) -> Dict[str, Any]:
    by_sev = {"low": 0, "medium": 0, "high": 0}
    by_type: Dict[str, int] = defaultdict(int)
    for f in flags:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
        by_type[f.flag_type] += 1

    return {
        "summary": {
            "total_flags": len(flags),
            "low": by_sev.get("low", 0),
            "medium": by_sev.get("medium", 0),
            "high": by_sev.get("high", 0),
            "by_type": dict(sorted(by_type.items())),
        },
        "flags": [f.model_dump() for f in flags],
    }


def profile_and_validate(
    studies: List[StudyRecord],
    chunks: List[Chunk],
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, PaperProfile], Dict[str, Any]]:
    """Create paper profiles and deterministic validation reports.

    Returns
    -------
    profiles_by_paper : dict[paper_id, PaperProfile]
    validation_report : dict (serializable)
    """
    profiles: Dict[str, PaperProfile] = {}
    flags: List[ValidationFlag] = []

    chunks_by_study: Dict[str, List[Chunk]] = defaultdict(list)
    for ch in chunks:
        chunks_by_study[ch.study_pmid].append(ch)

    for study in studies:
        study_chunks = chunks_by_study.get(study.pmid, [])
        profile = _build_profile(study, study_chunks)
        profiles[profile.paper_id] = profile

        # Physical consistency.
        flags.extend(validate_phase_co2(profile))

        # Corruption checks on raw study text and chunks.
        text_blob = "\n".join([study.title, study.abstract, study.raw_text])
        flags.extend(validate_text(text_blob, profile.paper_id))
        for ch in study_chunks:
            chunk_flags = validate_text(ch.text, ch.chunk_id)
            for cf in chunk_flags:
                if profile.paper_id not in cf.related_ids:
                    cf.related_ids.append(profile.paper_id)
            flags.extend(chunk_flags)

    # Persist profiles.
    save_json(
        {paper_id: profile.model_dump() for paper_id, profile in profiles.items()},
        "data/processed/paper_profiles.json",
    )

    validation_report = _build_validation_report(flags)
    save_json(validation_report, "data/processed/validation_report.json")

    logger.info(
        "Paper profiling complete: %d profiles, %d validation flags",
        len(profiles), validation_report["summary"]["total_flags"],
    )

    return profiles, validation_report
