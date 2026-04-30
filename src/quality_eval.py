"""quality_eval.py — Deterministic quality metrics for generated sections."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from utils import save_json
from validators import find_corruption_tokens


_NUMERIC_CLAIM_RE = re.compile(
    r"\b\d+(?:[\.,]\d+)?\s*(?:%|K|°\s?C|º\s?C|bar|MPa|kPa)\b"
    r"|\b\d+(?:[\.,]\d+)?\s*[-–]\s*\d+(?:[\.,]\d+)?\b",
    re.IGNORECASE,
)
_CHUNK_REF_RE = re.compile(r"\[chunk:([a-f0-9]{8,32})\]", re.IGNORECASE)
_PAPER_REF_RE = re.compile(r"\[paper:([^\]]+)\]", re.IGNORECASE)


def _forbidden_hits(text: str, forbidden_terms: List[str]) -> List[str]:
    text_low = text.lower()
    return [t for t in forbidden_terms if t and t.lower() in text_low]


def _grounding_score(text: str) -> float:
    numeric = _NUMERIC_CLAIM_RE.findall(text)
    chunk_refs = _CHUNK_REF_RE.findall(text)

    n_numeric = len(numeric)
    if n_numeric == 0:
        return 1.0

    # We cap coverage to avoid over-rewarding repetitive refs.
    covered = min(n_numeric, len(set(chunk_refs)))
    return round(covered / max(n_numeric, 1), 3)


def _physics_high_hits(text: str, validation_report: Dict[str, Any]) -> int:
    chunk_ids = set(_CHUNK_REF_RE.findall(text))
    paper_ids = set(_PAPER_REF_RE.findall(text))
    related_ids = chunk_ids.union(paper_ids)
    if not related_ids:
        return 0

    high_flags = [
        f for f in validation_report.get("flags", [])
        if f.get("severity") == "high"
    ]

    hits = 0
    for flag in high_flags:
        related = set(flag.get("related_ids", []))
        if related_ids.intersection(related):
            hits += 1
    return hits


def evaluate_section(
    section_id: str,
    section_text: str,
    forbidden_terms: List[str],
    validation_report: Dict[str, Any],
    section_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    section_meta = section_meta or {}
    filter_stats = section_meta.get("filter_stats", {})
    if not isinstance(filter_stats, dict):
        filter_stats = {}

    forbidden = _forbidden_hits(section_text, forbidden_terms)
    grounding = _grounding_score(section_text)
    corruption_tokens = find_corruption_tokens(section_text)
    physics_hits = _physics_high_hits(section_text, validation_report)
    evidence_count_after_filter = int(
        section_meta.get(
            "n_evidence_after_filter",
            section_meta.get("n_evidence", filter_stats.get("evidence_count_after_filter", 0)),
        ) or 0
    )
    avg_similarity = float(filter_stats.get("avg_similarity", 0.0) or 0.0)
    domain_purity = float(filter_stats.get("domain_purity", 1.0) or 0.0)
    foreign_evidence_share = float(filter_stats.get("foreign_evidence_share", 0.0) or 0.0)
    rejected_section_foreign = int(
        filter_stats.get("rejected_section_foreign_evidence_share", 0) or 0
    )

    contamination_score = max(0.0, 1.0 - min(1.0, len(forbidden) / 3.0))
    physics_consistency_score = 1.0 if physics_hits == 0 else max(0.0, 1.0 - 0.25 * physics_hits)

    return {
        "section_id": section_id,
        "forbidden_term_hits": forbidden,
        "cross_topic_contamination_score": round(contamination_score, 3),
        "physics_high_flag_hits": physics_hits,
        "high_flags_count": physics_hits,
        "physics_consistency_score": round(physics_consistency_score, 3),
        "grounding_score": grounding,
        "domain_purity": round(max(0.0, min(1.0, domain_purity)), 3),
        "avg_similarity": round(max(0.0, avg_similarity), 3),
        "foreign_evidence_share": round(max(0.0, min(1.0, foreign_evidence_share)), 3),
        "rejected_section_foreign_evidence_share": rejected_section_foreign,
        "evidence_count_after_filter": evidence_count_after_filter,
        "table_corruption_tokens": corruption_tokens,
        "table_integrity_score": round(1.0 if not corruption_tokens else max(0.0, 1.0 - len(corruption_tokens) / 5.0), 3),
    }


def build_quality_report(
    sections: List[Dict[str, Any]],
    taxonomy_entries: List[Dict[str, Any]],
    validation_report: Dict[str, Any],
    out_path: str = "data/processed/quality_report.json",
) -> Dict[str, Any]:
    entry_map = {
        f"{e.get('parent', '')} / {e.get('folder', '')}": e
        for e in taxonomy_entries
    }

    per_section: List[Dict[str, Any]] = []
    for sec in sections:
        sid = f"{sec.get('parent', '')} / {sec.get('folder', '')}"
        entry = entry_map.get(sid, {})
        metrics = evaluate_section(
            sid,
            sec.get("content", ""),
            entry.get("forbidden_terms", []),
            validation_report,
            section_meta=sec,
        )
        per_section.append(metrics)

    if per_section:
        avg_contam = sum(s["cross_topic_contamination_score"] for s in per_section) / len(per_section)
        avg_phys = sum(s["physics_consistency_score"] for s in per_section) / len(per_section)
        avg_ground = sum(s["grounding_score"] for s in per_section) / len(per_section)
        avg_domain = sum(s["domain_purity"] for s in per_section) / len(per_section)
        avg_sim = sum(s["avg_similarity"] for s in per_section) / len(per_section)
        avg_foreign_share = sum(s["foreign_evidence_share"] for s in per_section) / len(per_section)
    else:
        avg_contam = avg_phys = avg_ground = avg_domain = avg_sim = 1.0
        avg_foreign_share = 0.0

    report = {
        "summary": {
            "sections": len(per_section),
            "avg_cross_topic_contamination_score": round(avg_contam, 3),
            "avg_physics_consistency_score": round(avg_phys, 3),
            "avg_grounding_score": round(avg_ground, 3),
            "avg_domain_purity": round(avg_domain, 3),
            "avg_similarity": round(avg_sim, 3),
            "avg_foreign_evidence_share": round(avg_foreign_share, 3),
            "sections_with_corruption": sum(1 for s in per_section if s["table_corruption_tokens"]),
        },
        "sections": per_section,
    }

    save_json(report, out_path)
    return report
