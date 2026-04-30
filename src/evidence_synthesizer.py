"""
evidence_synthesizer.py — Consensus, contradiction, and gap analysis.

Analyses tagged chunks grouped by taxonomy theme and produces a
structured SynthesisMap per section.  The map feeds downstream writing
(thesis generation, critical analysis) and conclusions (research agenda).

Part of Épico 2: Consensus/Contradiction/Gap Analyzer.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from utils import Chunk, ChunkTag, StudyRecord, call_llm, load_json, save_json
from evidence_filtering import build_high_flag_related_ids, filter_and_rank_tags

logger = logging.getLogger("systematic_review.evidence_synthesizer")


# ------------------------------------------------------------------ #
#  Data models                                                         #
# ------------------------------------------------------------------ #

class ConsensusPoint(BaseModel):
    """A point on which multiple studies converge."""
    statement: str
    supporting_studies: List[str] = Field(default_factory=list)  # PMIDs
    strength: str = "moderate"  # strong | moderate | weak
    evidence_chunk_ids: List[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    """A disagreement between two or more studies."""
    point: str
    study_a: str = ""      # PMID or citation
    finding_a: str = ""
    study_b: str = ""      # PMID or citation
    finding_b: str = ""
    possible_reason: str = ""
    evidence_chunk_ids: List[str] = Field(default_factory=list)


class KnowledgeGap(BaseModel):
    """An area where evidence is insufficient or absent."""
    description: str
    priority: str = "medium"  # high | medium | low
    suggested_approach: str = ""


class SynthesisMap(BaseModel):
    """Structured synthesis of a single theme/section."""
    theme: str
    consensus_points: List[ConsensusPoint] = Field(default_factory=list)
    contradictions: List[Contradiction] = Field(default_factory=list)
    knowledge_gaps: List[KnowledgeGap] = Field(default_factory=list)


# ------------------------------------------------------------------ #
#  LLM prompt                                                          #
# ------------------------------------------------------------------ #

_SYNTHESIS_PROMPT = """\
You are an evidence synthesis specialist for systematic reviews.

Analyse the evidence excerpts below, which come from different published
studies on the same theme.  Your task is to identify:
  1. Points of CONSENSUS (claims supported by multiple studies)
  2. CONTRADICTIONS (studies that disagree on a finding)
  3. KNOWLEDGE GAPS (questions the evidence does not answer)

### Theme: {theme}

### Evidence
{evidence}

### Output format
Return ONLY valid JSON with this structure (no extra text):
{{
  "consensus_points": [
    {{
      "statement": "...",
      "supporting_studies": ["PMID or (Author, Year)", ...],
      "strength": "strong | moderate | weak",
      "evidence_chunk_ids": ["chunk_id1", "chunk_id2"]
    }}
  ],
  "contradictions": [
    {{
      "point": "...",
      "study_a": "PMID or citation",
      "finding_a": "...",
      "study_b": "PMID or citation",
      "finding_b": "...",
      "possible_reason": "hypothesis for the discrepancy",
      "evidence_chunk_ids": ["chunk_id1", "chunk_id2"]
    }}
  ],
  "knowledge_gaps": [
    {{
      "description": "...",
      "priority": "high | medium | low",
      "suggested_approach": "how this gap could be addressed"
    }}
  ]
}}

### Rules
- Each consensus point must cite ≥2 studies.
- Strength is "strong" if ≥3 studies agree with consistent methodology.
- Contradictions should include a plausible reason for the divergence.
- Knowledge gaps should be actionable (suggest a study design or approach).
- consensus_points and contradictions MUST include evidence_chunk_ids.
- Use ONLY chunk IDs present in the provided evidence blocks.
- Return at least 1 item in each category. If none exists, state why.
"""


# ------------------------------------------------------------------ #
#  Core logic                                                          #
# ------------------------------------------------------------------ #

def _build_evidence_for_theme(
    folder: str,
    parent: str,
    tags: List[ChunkTag],
    chunks_by_id: Dict[str, Chunk],
    studies_by_pmid: Dict[str, StudyRecord],
    top_k: int = 15,
    min_eligibility: float = 0.2,
    taxonomy_entry: Dict[str, Any] | None = None,
    profiles_by_paper: Dict[str, Dict[str, Any]] | None = None,
    high_flag_related_ids: set[str] | None = None,
) -> tuple[str, List[str]]:
    """Gather and format evidence for a single theme."""
    relevant_tags, _ = filter_and_rank_tags(
        tags=tags,
        chunks_by_id=chunks_by_id,
        folder=folder,
        parent=parent,
        top_k=top_k,
        min_eligibility=min_eligibility,
        taxonomy_entry=taxonomy_entry,
        profiles_by_paper=profiles_by_paper,
        high_flag_related_ids=high_flag_related_ids,
    )

    if not relevant_tags:
        return "", []

    parts: List[str] = []
    used_chunk_ids: List[str] = []
    for tag in relevant_tags:
        chunk = chunks_by_id.get(tag.chunk_id)
        if not chunk:
            continue
        study = studies_by_pmid.get(chunk.study_pmid)
        if study:
            author = study.authors or chunk.study_pmid
            year = study.year or "n.d."
            label = f"({author}, {year})"
        else:
            label = f"({chunk.study_pmid})"

        # Include metadata if available
        meta_parts = []
        if chunk.study_metadata:
            scale = chunk.study_metadata.get("study_scale", "")
            if scale:
                meta_parts.append(f"Scale: {scale}")
            rob = chunk.study_metadata.get("rob_overall", "")
            if rob:
                meta_parts.append(f"RoB: {rob}")
        meta_str = f" [{', '.join(meta_parts)}]" if meta_parts else ""

        parts.append(f"--- {label}{meta_str} [chunk_id:{chunk.chunk_id}] ---\n{chunk.text}")
        used_chunk_ids.append(chunk.chunk_id)

    return "\n\n".join(parts), used_chunk_ids


def _parse_synthesis(
    raw: str,
    theme: str,
    valid_chunk_ids: List[str] | None = None,
) -> SynthesisMap:
    """Parse LLM output into a SynthesisMap."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    valid_ids = set(valid_chunk_ids or [])

    try:
        data = json.loads(cleaned)
        consensus_points = []
        for cp in data.get("consensus_points", []):
            if not isinstance(cp, dict):
                continue
            ids = cp.get("evidence_chunk_ids", [])
            if valid_ids:
                ids = [cid for cid in ids if cid in valid_ids]
            cp["evidence_chunk_ids"] = ids
            consensus_points.append(ConsensusPoint(**cp))

        contradictions = []
        for c in data.get("contradictions", []):
            if not isinstance(c, dict):
                continue
            ids = c.get("evidence_chunk_ids", [])
            if valid_ids:
                ids = [cid for cid in ids if cid in valid_ids]
            c["evidence_chunk_ids"] = ids
            contradictions.append(Contradiction(**c))

        return SynthesisMap(
            theme=theme,
            consensus_points=consensus_points,
            contradictions=contradictions,
            knowledge_gaps=[KnowledgeGap(**g) for g in data.get("knowledge_gaps", [])],
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Synthesis parse error for '%s': %s", theme, exc)
        return SynthesisMap(
            theme=theme,
            knowledge_gaps=[KnowledgeGap(
                description="Synthesis could not be parsed from LLM output",
                priority="low",
                suggested_approach="Manual review of evidence",
            )],
        )


def _grounding_ok(smap: SynthesisMap) -> bool:
    if not smap.consensus_points and not smap.contradictions:
        return True
    for cp in smap.consensus_points:
        if not cp.evidence_chunk_ids:
            return False
    for c in smap.contradictions:
        if not c.evidence_chunk_ids:
            return False
    return True


def synthesize_theme(
    folder: str,
    parent: str,
    tags: List[ChunkTag],
    chunks_by_id: Dict[str, Chunk],
    studies_by_pmid: Dict[str, StudyRecord],
    cfg: Dict[str, Any],
    top_k: int = 15,
    taxonomy_entry: Dict[str, Any] | None = None,
    profiles_by_paper: Dict[str, Dict[str, Any]] | None = None,
    high_flag_related_ids: set[str] | None = None,
) -> SynthesisMap:
    """Produce a SynthesisMap for a single taxonomy theme."""
    theme = f"{parent} / {folder}"

    min_eligibility = cfg.get("routing", {}).get("min_eligibility_score", 0.2)
    evidence, evidence_chunk_ids = _build_evidence_for_theme(
        folder,
        parent,
        tags,
        chunks_by_id,
        studies_by_pmid,
        top_k,
        min_eligibility,
        taxonomy_entry=taxonomy_entry,
        profiles_by_paper=profiles_by_paper,
        high_flag_related_ids=high_flag_related_ids,
    )

    if not evidence:
        logger.info("  No evidence for %s — returning empty map", theme)
        return SynthesisMap(
            theme=theme,
            knowledge_gaps=[KnowledgeGap(
                description=f"No evidence found for: {theme}",
                priority="high",
                suggested_approach="Targeted literature search",
            )],
        )

    prompt = _SYNTHESIS_PROMPT.format(theme=theme, evidence=evidence)
    grounding_retry_prompt = (
        "Your previous JSON lacked required evidence_chunk_ids. "
        "Return JSON again, and for each consensus_points/contradictions item "
        "you MUST attach evidence_chunk_ids using ONLY IDs present in evidence."
    )

    max_retries = cfg.get("evidence_synthesizer", {}).get("max_retries", 1)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            raw = call_llm(prompt, cfg)
            parsed = _parse_synthesis(raw, theme, evidence_chunk_ids)

            if _grounding_ok(parsed):
                return parsed

            # One mandatory correction pass for evidence anchoring.
            corrected = call_llm(
                f"{prompt}\n\n{grounding_retry_prompt}",
                cfg,
            )
            parsed2 = _parse_synthesis(corrected, theme, evidence_chunk_ids)
            if _grounding_ok(parsed2):
                return parsed2

            logger.warning("Synthesis for '%s' missing evidence_chunk_ids after retry", theme)
            return SynthesisMap(
                theme=theme,
                consensus_points=[],
                contradictions=[],
                knowledge_gaps=[KnowledgeGap(
                    description="Insufficient evidence anchoring in provided corpus",
                    priority="high",
                    suggested_approach="Revise evidence mapping and regenerate synthesis",
                )],
            )
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                logger.warning(
                    "Synthesis attempt %d/%d failed for '%s': %s — retrying",
                    attempt + 1, max_retries + 1, theme, exc,
                )

    logger.error("Synthesis failed for '%s' after %d attempts: %s", theme, max_retries + 1, last_err)
    return SynthesisMap(theme=theme)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def synthesize_all_themes(
    taxonomy_entries: List[Dict[str, str]],
    chunks: List[Chunk],
    tags: List[ChunkTag],
    studies: List[StudyRecord],
    cfg: Dict[str, Any],
) -> Dict[str, SynthesisMap]:
    """Run evidence synthesis for all taxonomy themes.

    Returns a dict mapping "parent / folder" → SynthesisMap.
    Also persists the results to ``data/processed/synthesis_map.json``.
    """
    from tqdm import tqdm

    chunks_by_id = {c.chunk_id: c for c in chunks}
    studies_by_pmid = {s.pmid: s for s in studies}
    top_k = cfg.get("evidence_synthesizer", {}).get("top_k_evidence", 15)
    profiles_by_paper: Dict[str, Dict[str, Any]] = {}
    try:
        loaded_profiles = load_json("data/processed/paper_profiles.json")
        if isinstance(loaded_profiles, dict):
            profiles_by_paper = loaded_profiles
    except Exception:
        profiles_by_paper = {}

    high_flag_related_ids: set[str] = set()
    if cfg.get("validators", {}).get("exclude_high_flagged_evidence", True):
        try:
            validation_report = load_json("data/processed/validation_report.json")
            high_flag_related_ids = build_high_flag_related_ids(
                validation_report if isinstance(validation_report, dict) else {}
            )
        except Exception:
            high_flag_related_ids = set()

    logger.info("Synthesizing evidence for %d themes", len(taxonomy_entries))

    results: Dict[str, SynthesisMap] = {}
    for entry in tqdm(taxonomy_entries, desc="Evidence synthesis"):
        theme_key = f"{entry['parent']} / {entry['folder']}"
        synthesis = synthesize_theme(
            entry["folder"],
            entry["parent"],
            tags,
            chunks_by_id,
            studies_by_pmid,
            cfg,
            top_k,
            taxonomy_entry=entry,
            profiles_by_paper=profiles_by_paper,
            high_flag_related_ids=high_flag_related_ids,
        )
        results[theme_key] = synthesis
        logger.debug(
            "  %s: %d consensus, %d contradictions, %d gaps",
            theme_key,
            len(synthesis.consensus_points),
            len(synthesis.contradictions),
            len(synthesis.knowledge_gaps),
        )

    # Persist
    serialized = {k: v.model_dump() for k, v in results.items()}
    save_json(serialized, "data/processed/synthesis_map.json")

    # Summary stats
    total_consensus = sum(len(s.consensus_points) for s in results.values())
    total_contradictions = sum(len(s.contradictions) for s in results.values())
    total_gaps = sum(len(s.knowledge_gaps) for s in results.values())
    logger.info(
        "Synthesis complete: %d consensus, %d contradictions, %d gaps across %d themes",
        total_consensus, total_contradictions, total_gaps, len(results),
    )

    return results
