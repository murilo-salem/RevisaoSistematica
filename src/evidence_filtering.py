"""evidence_filtering.py — Shared hard-gating helpers for evidence routing."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from utils import Chunk, ChunkTag


def build_high_flag_related_ids(validation_report: Dict[str, Any] | None) -> Set[str]:
    """Collect related IDs from HIGH severity validation flags."""
    related_ids: Set[str] = set()
    if not isinstance(validation_report, dict):
        return related_ids

    for flag in validation_report.get("flags", []):
        if not isinstance(flag, dict):
            continue
        if str(flag.get("severity", "")).lower() != "high":
            continue
        for rid in flag.get("related_ids", []):
            if rid:
                related_ids.add(str(rid))
    return related_ids


def _normalise_terms(values: List[str] | None) -> List[str]:
    if not values:
        return []
    cleaned = [str(v).strip().lower() for v in values if str(v).strip()]
    return sorted(set(cleaned))


def _profile_for_chunk(
    chunk: Chunk,
    profiles_by_paper: Dict[str, Dict[str, Any]] | None,
) -> Dict[str, Any]:
    if not profiles_by_paper:
        return {}
    pid = chunk.paper_id or chunk.study_pmid
    prof = profiles_by_paper.get(pid, {})
    return prof if isinstance(prof, dict) else {}


def _contains_any(text: str, terms: List[str]) -> bool:
    if not text or not terms:
        return False
    text_low = text.lower()
    norm_text = _normalise_text(text_low)
    padded_norm = f" {norm_text} "

    for term in terms:
        t = str(term).strip().lower()
        if not t:
            continue
        if t in text_low:
            return True
        norm_t = _normalise_text(t)
        if norm_t and f" {norm_t} " in padded_norm:
            return True
    return False


def _normalise_text(text: str) -> str:
    cleaned = re.sub(r"[\W_]+", " ", (text or "").lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _overlap_ratio(values: List[str], allowed: List[str]) -> float:
    if not allowed:
        return 1.0
    if not values:
        return 0.0
    value_set = set(v.lower() for v in values if v)
    allowed_set = set(allowed)
    return 1.0 if value_set.intersection(allowed_set) else 0.0


def _foreign_domain_ratio(values: List[str], allowed: List[str]) -> float:
    """Share of profile domains that are outside the section's allowed set."""
    if not values:
        return 1.0
    if not allowed:
        return 0.0
    value_set = {str(v).lower() for v in values if str(v).strip()}
    if not value_set:
        return 1.0
    allowed_set = set(allowed)
    foreign = [d for d in value_set if d not in allowed_set]
    return len(foreign) / max(len(value_set), 1)


def filter_and_rank_tags(
    tags: List[ChunkTag],
    chunks_by_id: Dict[str, Chunk],
    folder: str,
    parent: str,
    top_k: int | None = None,
    min_eligibility: float = 0.2,
    taxonomy_entry: Dict[str, Any] | None = None,
    profiles_by_paper: Dict[str, Dict[str, Any]] | None = None,
    high_flag_related_ids: Set[str] | None = None,
) -> Tuple[List[ChunkTag], Dict[str, Any]]:
    """Hard-filter section evidence first, then rank by similarity."""
    taxonomy_entry = taxonomy_entry or {}
    allowed_domains = _normalise_terms(taxonomy_entry.get("allowed_domains", []))
    required_phase_regimes = _normalise_terms(taxonomy_entry.get("required_phase_regimes", []))
    forbidden_terms = _normalise_terms(taxonomy_entry.get("forbidden_terms", []))
    required_any_terms = _normalise_terms(taxonomy_entry.get("required_any_terms", []))
    min_similarity = float(taxonomy_entry.get("min_similarity", 0.0) or 0.0)
    max_foreign_domain_ratio = float(
        taxonomy_entry.get("max_foreign_domain_ratio", 1.0) or 1.0
    )
    max_foreign_domain_ratio = max(0.0, min(1.0, max_foreign_domain_ratio))
    max_foreign_evidence_share = float(
        taxonomy_entry.get("max_foreign_evidence_share", 1.0) or 1.0
    )
    max_foreign_evidence_share = max(0.0, min(1.0, max_foreign_evidence_share))

    stats: Dict[str, Any] = {
        "candidates_total": len(tags),
        "after_section_filter": 0,
        "accepted": 0,
        "rejected_low_eligibility": 0,
        "rejected_min_similarity": 0,
        "rejected_missing_chunk": 0,
        "rejected_high_flags": 0,
        "rejected_allowed_domains": 0,
        "rejected_foreign_domain_ratio": 0,
        "rejected_required_phase_regimes": 0,
        "rejected_forbidden_terms": 0,
        "rejected_required_any_terms": 0,
        "rejected_section_foreign_evidence_share": 0,
        "avg_similarity": 0.0,
        "domain_purity": 1.0,
        "avg_foreign_domain_ratio": 0.0,
        "foreign_evidence_share": 0.0,
        "evidence_count_after_filter": 0,
    }

    section_tags = [t for t in tags if t.folder == folder and t.parent == parent]
    stats["after_section_filter"] = len(section_tags)

    # Domain purity at the candidate level (before hard filtering).
    if allowed_domains:
        domain_hits = 0
        domain_total = 0
        for tag in section_tags:
            if tag.eligibility_score < min_eligibility:
                continue
            chunk = chunks_by_id.get(tag.chunk_id)
            if not chunk:
                continue
            prof = _profile_for_chunk(chunk, profiles_by_paper)
            domain_total += 1
            ratio = _overlap_ratio(prof.get("domains", []), allowed_domains)
            if ratio > 0:
                domain_hits += 1
        stats["domain_purity"] = round(domain_hits / max(domain_total, 1), 3) if domain_total else 0.0

    filtered: List[ChunkTag] = []
    accepted_foreign_ratios: List[float] = []
    high_flag_related_ids = high_flag_related_ids or set()

    for tag in section_tags:
        if tag.eligibility_score < min_eligibility:
            stats["rejected_low_eligibility"] += 1
            continue
        if tag.similarity < min_similarity:
            stats["rejected_min_similarity"] += 1
            continue

        chunk = chunks_by_id.get(tag.chunk_id)
        if not chunk:
            stats["rejected_missing_chunk"] += 1
            continue

        paper_id = chunk.paper_id or chunk.study_pmid
        if high_flag_related_ids and (
            chunk.chunk_id in high_flag_related_ids or paper_id in high_flag_related_ids
        ):
            stats["rejected_high_flags"] += 1
            continue

        chunk_low = chunk.text.lower()
        prof = _profile_for_chunk(chunk, profiles_by_paper)
        prof_domains = [str(d).lower() for d in prof.get("domains", [])]
        prof_phases = [str(p).lower() for p in prof.get("phase_regimes", [])]

        if allowed_domains and not set(prof_domains).intersection(allowed_domains):
            stats["rejected_allowed_domains"] += 1
            continue

        if allowed_domains and prof_domains:
            foreign_ratio = _foreign_domain_ratio(prof_domains, allowed_domains)
            if foreign_ratio > max_foreign_domain_ratio:
                stats["rejected_foreign_domain_ratio"] += 1
                continue

        if required_phase_regimes and not set(prof_phases).intersection(required_phase_regimes):
            stats["rejected_required_phase_regimes"] += 1
            continue

        if forbidden_terms and _contains_any(chunk_low, forbidden_terms):
            stats["rejected_forbidden_terms"] += 1
            continue

        if required_any_terms:
            profile_blob = " ".join([
                " ".join(str(v) for v in prof.get("keywords", [])),
                " ".join(str(v) for v in prof.get("working_fluids", [])),
                " ".join(str(v) for v in prof.get("system_topologies", [])),
                " ".join(str(v) for v in prof.get("domains", [])),
            ]).lower()
            searchable = f"{chunk_low}\n{profile_blob}"
            if not _contains_any(searchable, required_any_terms):
                stats["rejected_required_any_terms"] += 1
                continue

        filtered.append(tag)
        if allowed_domains:
            if prof_domains:
                foreign_ratio = _foreign_domain_ratio(prof_domains, allowed_domains)
                accepted_foreign_ratios.append(foreign_ratio)

    filtered.sort(key=lambda t: t.similarity, reverse=True)
    if top_k is not None:
        filtered = filtered[:top_k]

    # Final section-level guard: if most selected evidence has foreign domains,
    # fail closed instead of letting the writer mix cross-topic content.
    if allowed_domains and filtered:
        selected_foreign: List[bool] = []
        for tag in filtered:
            chunk = chunks_by_id.get(tag.chunk_id)
            if not chunk:
                selected_foreign.append(True)
                continue
            prof = _profile_for_chunk(chunk, profiles_by_paper)
            prof_domains = [str(d).lower() for d in prof.get("domains", [])]
            if not prof_domains:
                selected_foreign.append(True)
                continue
            selected_foreign.append(_foreign_domain_ratio(prof_domains, allowed_domains) > 0.0)

        foreign_share = sum(1 for has_foreign in selected_foreign if has_foreign) / max(
            len(selected_foreign), 1
        )
        stats["foreign_evidence_share"] = round(foreign_share, 3)
        if foreign_share > max_foreign_evidence_share:
            stats["rejected_section_foreign_evidence_share"] = len(filtered)
            stats["accepted"] = 0
            stats["evidence_count_after_filter"] = 0
            stats["avg_similarity"] = 0.0
            return [], stats

    stats["accepted"] = len(filtered)
    stats["evidence_count_after_filter"] = len(filtered)
    if filtered:
        stats["avg_similarity"] = round(
            sum(t.similarity for t in filtered) / len(filtered), 3
        )
    if accepted_foreign_ratios:
        stats["avg_foreign_domain_ratio"] = round(
            sum(accepted_foreign_ratios) / len(accepted_foreign_ratios), 3
        )

    return filtered, stats
