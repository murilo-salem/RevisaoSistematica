"""
content_analyzer.py — Content analysis, chunking, and tag mapping.

Stages 3–6 of the local pipeline:
  3. Content Analysis   — embed articles + taxonomy prompts
  4. Chunking           — split articles into semantic chunks
  5. Tag Mapping        — assign chunks to taxonomy sections via cosine similarity
  6. Correlation Check  — report which sections have evidence
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from utils import (
    Chunk,
    ChunkTag,
    StudyRecord,
    ValidationFlag,
    get_db_connection,
    now_iso,
    save_json,
)

logger = logging.getLogger("systematic_review.content_analyzer")

# ------------------------------------------------------------------ #
#  Text chunking                                                       #
# ------------------------------------------------------------------ #

def _chunk_text(
    text: str,
    max_tokens: int = 400,
    overlap: int = 50,
) -> List[Tuple[str, int, int]]:
    """Split *text* into overlapping chunks.

    Returns list of (chunk_text, start_char, end_char).
    Uses sentence boundaries when possible.
    """
    if not text or not text.strip():
        return []

    # Approximate tokens as words (good enough for chunking)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[Tuple[str, int, int]] = []

    current_words: List[str] = []
    current_start = 0
    char_pos = 0

    for sentence in sentences:
        words = sentence.split()
        if not words:
            char_pos += len(sentence) + 1
            continue

        # If adding this sentence exceeds max, save current chunk
        if len(current_words) + len(words) > max_tokens and current_words:
            chunk_text = " ".join(current_words)
            chunks.append((chunk_text, current_start, current_start + len(chunk_text)))

            # Keep overlap words from end of current chunk
            overlap_words = current_words[-overlap:] if len(current_words) > overlap else []
            current_words = overlap_words
            current_start = char_pos - len(" ".join(overlap_words)) if overlap_words else char_pos

        current_words.extend(words)
        char_pos += len(sentence) + 1  # +1 for the space/newline after

    # Last chunk
    if current_words:
        chunk_text = " ".join(current_words)
        chunks.append((chunk_text, current_start, current_start + len(chunk_text)))

    return chunks


def _make_chunk_id(pmid: str, index: int) -> str:
    """Deterministic chunk ID from study PMID and chunk index."""
    raw = f"{pmid}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ #
#  Embedding helpers                                                   #
# ------------------------------------------------------------------ #

def _load_model(model_name: str, device: str = "cuda"):
    """Load sentence-transformer model.  Falls back to CPU."""
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception:
        logger.warning("Could not load model on %s — falling back to CPU", device)
        model = SentenceTransformer(model_name, device="cpu")
    return model


def _embed_texts(
    texts: List[str],
    model,
    batch_size: int = 32,
    desc: str = "Embedding",
) -> np.ndarray:
    """Batch-encode texts and return numpy matrix."""
    return model.encode(
        texts,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_numpy=True,
    )


def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of A and each row of B.

    Returns shape (len(A), len(B)).
    """
    # Normalise rows
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T


def _merge_validation_flags(
    validation_report: Dict[str, Any],
    extra_flags: List[ValidationFlag],
) -> Dict[str, Any]:
    """Append new flags to an existing validation report and recompute summary."""
    report = dict(validation_report or {})
    existing_flags = list(report.get("flags", []))
    existing_flags.extend([f.model_dump() for f in extra_flags])

    by_type: Dict[str, int] = {}
    low = medium = high = 0
    for fl in existing_flags:
        sev = str(fl.get("severity", "low")).lower()
        if sev == "high":
            high += 1
        elif sev == "medium":
            medium += 1
        else:
            low += 1
        ft = fl.get("flag_type", "unknown")
        by_type[ft] = by_type.get(ft, 0) + 1

    report["summary"] = {
        "total_flags": len(existing_flags),
        "low": low,
        "medium": medium,
        "high": high,
        "by_type": dict(sorted(by_type.items())),
    }
    report["flags"] = existing_flags
    return report


# ------------------------------------------------------------------ #
#  Scope filtering (paper-level, before chunking)                      #
# ------------------------------------------------------------------ #

_SCOPE_CLUSTER_PATTERNS: Dict[str, List[re.Pattern]] = {
    "refrigeration_hp": [
        re.compile(r"\br744\b", re.IGNORECASE),
        re.compile(r"\btranscritical\b|\bsubcritical\b", re.IGNORECASE),
        re.compile(r"\bgas cooler\b|\bevaporator\b|\bcondenser\b", re.IGNORECASE),
        re.compile(r"\bejector\b|\bparallel compression\b|\beconomizer\b", re.IGNORECASE),
    ],
    "cold_storage_ctes": [
        re.compile(r"\bctes\b|cold thermal energy storage", re.IGNORECASE),
        re.compile(r"\bcold storage\b|\bcold chain\b", re.IGNORECASE),
        re.compile(r"\bdry ice\b|\bsolid co2\b|\bco2 snow\b", re.IGNORECASE),
        re.compile(r"\bsublimation\b|\bboil[- ]off\b|\bsub[- ]zero\b|\bultra[- ]low temperature\b", re.IGNORECASE),
    ],
    "cryogenic_systems": [
        re.compile(r"\blaes\b|liquid air energy storage", re.IGNORECASE),
        re.compile(r"\bliquid air\b|\bcryogenic (?:energy storage|battery|tank|systems?)\b", re.IGNORECASE),
        re.compile(r"\bcold recovery\b|\bliquefaction\b|\bregasification\b", re.IGNORECASE),
    ],
    "co2_phase_thermo": [
        re.compile(r"\btriple point\b|\bponto triplo\b", re.IGNORECASE),
        re.compile(r"\bphase diagram\b|\bphase equilibrium\b", re.IGNORECASE),
        re.compile(r"\bnucleate boiling\b|\bpool boiling\b|\bcritical heat flux\b|\bchf\b", re.IGNORECASE),
    ],
}

_SCOPE_ANCHOR_PATTERNS: List[re.Pattern] = [
    re.compile(r"\br744\b", re.IGNORECASE),
    re.compile(r"\blaes\b|liquid air energy storage", re.IGNORECASE),
    re.compile(r"\bdry ice\b|\bsolid co2\b|\bco2 snow\b", re.IGNORECASE),
    re.compile(r"\btriple point\b|\bponto triplo\b", re.IGNORECASE),
    re.compile(r"\bnucleate boiling\b|\bpool boiling\b|\bcritical heat flux\b|\bchf\b", re.IGNORECASE),
    re.compile(r"\bcold thermal energy storage\b|\bctes\b", re.IGNORECASE),
]

_SCOPE_NEGATIVE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bpyrolysis\b|\bbio[- ]oil\b|\bbiochar\b", re.IGNORECASE),
    re.compile(r"\bbiodiesel\b|\bmicroalgae\b|\banaerobic digestion\b|\bbiogas\b", re.IGNORECASE),
    re.compile(r"\bfermentation\b|\bsyngas\b|\bsteam reforming\b|\bmethanol\b|\bglycerol\b|\btransesterification\b", re.IGNORECASE),
    re.compile(r"\bwgs\b|\bpsa\b|\basu\b|\brplug\b|\brgibbs\b|\baspen\b", re.IGNORECASE),
    re.compile(r"\breactor\b|\bcatalyst\b|\bhydrotreat(?:ing|ment)\b|\bfischer[- ]tropsch\b", re.IGNORECASE),
    re.compile(r"\bdac\b|\bccu\b|\bdirect air capture\b|\bcarbon capture\b|\bengine emissions?\b|\bice emissions?\b", re.IGNORECASE),
    re.compile(r"\blow[- ]carbon fuels?\b|\bdiesel engine\b|\bspark ignition\b|\bcompression ignition\b", re.IGNORECASE),
]


def _study_scope_features(study: StudyRecord) -> Dict[str, Any]:
    text = "\n".join([
        study.title or "",
        study.abstract or "",
        (study.raw_text or "")[:6000],
    ])
    text_low = text.lower()

    cluster_hits: Dict[str, int] = {}
    for cluster, patterns in _SCOPE_CLUSTER_PATTERNS.items():
        n = sum(1 for p in patterns if p.search(text))
        if n > 0:
            cluster_hits[cluster] = n

    anchor_hits = sum(1 for p in _SCOPE_ANCHOR_PATTERNS if p.search(text))
    negative_hits = sum(1 for p in _SCOPE_NEGATIVE_PATTERNS if p.search(text_low))
    positive_score = len(cluster_hits) + min(anchor_hits, 3)

    return {
        "cluster_hits": cluster_hits,
        "anchor_hits": anchor_hits,
        "negative_hits": negative_hits,
        "positive_score": positive_score,
    }


def _filter_studies_by_scope(
    studies: List[StudyRecord],
    routing_cfg: Dict[str, Any],
) -> tuple[List[StudyRecord], Dict[str, Any]]:
    min_positive = int(routing_cfg.get("scope_min_positive_score", 2) or 2)
    max_neg_wo_anchor = int(routing_cfg.get("scope_max_negative_without_anchor", 2) or 2)
    max_neg_with_anchor = int(routing_cfg.get("scope_max_negative_with_anchor", 6) or 6)

    kept: List[StudyRecord] = []
    dropped: List[Dict[str, Any]] = []

    for study in studies:
        feats = _study_scope_features(study)
        has_anchor = feats["anchor_hits"] > 0

        in_scope = feats["positive_score"] >= min_positive
        if in_scope and not has_anchor and feats["negative_hits"] >= max_neg_wo_anchor:
            in_scope = False
        if in_scope and has_anchor and feats["negative_hits"] > max_neg_with_anchor:
            in_scope = False

        if in_scope:
            kept.append(study)
        else:
            dropped.append({
                "paper_id": study.pmid,
                "title": study.title,
                "positive_score": feats["positive_score"],
                "anchor_hits": feats["anchor_hits"],
                "negative_hits": feats["negative_hits"],
                "cluster_hits": feats["cluster_hits"],
            })

    report = {
        "enabled": True,
        "input_studies": len(studies),
        "kept_studies": len(kept),
        "dropped_studies": len(dropped),
        "settings": {
            "scope_min_positive_score": min_positive,
            "scope_max_negative_without_anchor": max_neg_wo_anchor,
            "scope_max_negative_with_anchor": max_neg_with_anchor,
        },
        "dropped": dropped[:500],
    }
    return kept, report


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def analyze_and_chunk(
    studies: List[StudyRecord],
    taxonomy_entries: List[Dict[str, str]],
    cfg: Dict[str, Any],
    study_metadata: Dict[str, Dict[str, Any]] | None = None,
) -> Tuple[List[Chunk], List[ChunkTag], Dict[str, Any]]:
    """Run stages 3–6: chunk articles, embed, tag, and report.

    Parameters
    ----------
    studies : list[StudyRecord]
        Deduplicated articles.
    taxonomy_entries : list[dict]
        Each dict has keys ``prompt``, ``folder``, ``parent``.
    cfg : dict
        Global config.

    Returns
    -------
    chunks : list[Chunk]
    tags   : list[ChunkTag]
    coverage : dict   — per-section coverage statistics
    """
    if study_metadata is None:
        study_metadata = {}
    analyzer_cfg = cfg.get("content_analyzer", {})
    routing_cfg = cfg.get("routing", {})
    validators_cfg = cfg.get("validators", {})
    model_name = analyzer_cfg.get("model", "all-MiniLM-L6-v2")
    max_tokens = analyzer_cfg.get("chunk_max_tokens", 400)
    overlap = analyzer_cfg.get("chunk_overlap", 50)
    sim_threshold = analyzer_cfg.get("similarity_threshold", 0.25)
    top_k = analyzer_cfg.get("top_k_tags", 3)
    batch_size = analyzer_cfg.get("embedding_batch_size", 32)
    guardrails_enabled = routing_cfg.get("guardrails_enabled", True)
    scope_filter_enabled = routing_cfg.get("scope_filter_enabled", True)
    strictness_default = routing_cfg.get("strictness_default", "soft")
    validators_enabled = validators_cfg.get("enabled", True)

    if scope_filter_enabled:
        studies, scope_report = _filter_studies_by_scope(studies, routing_cfg)
        save_json(scope_report, "data/processed/scope_report.json")
        logger.info(
            "Scope filter: %d → %d studies kept (%d dropped as out-of-scope)",
            scope_report["input_studies"],
            scope_report["kept_studies"],
            scope_report["dropped_studies"],
        )
        if not studies:
            logger.warning("All studies were marked out-of-scope before chunking.")
            empty_coverage: Dict[str, Any] = {"sections": [], "summary": {
                "total_sections": len(taxonomy_entries),
                "sections_with_evidence": 0,
                "sections_without_evidence": len(taxonomy_entries),
                "total_chunks": 0,
                "total_tags": 0,
            }}
            save_json([], "data/processed/chunks.json")
            save_json([], "data/processed/chunk_tags.json")
            save_json(empty_coverage, "data/processed/coverage_report.json")
            return [], [], empty_coverage

    # Determine device
    sys_caps = cfg.get("system", {})
    device = "cuda" if sys_caps.get("cuda", False) else "cpu"

    # ---- Stage 3: Content analysis (load model) -------------------- #
    logger.info("Loading embedding model '%s' on %s (batch_size=%d)", model_name, device, batch_size)
    model = _load_model(model_name, device)

    # ---- Stage 4: Chunking ----------------------------------------- #
    logger.info("Chunking %d articles (max_tokens=%d, overlap=%d)", len(studies), max_tokens, overlap)
    all_chunks: List[Chunk] = []

    for study in tqdm(studies, desc="Chunking articles"):
        # Combine title + abstract as the article text
        full_text = f"{study.title}. {study.abstract}" if study.abstract else study.title
        raw_chunks = _chunk_text(full_text, max_tokens, overlap)

        if not raw_chunks:
            # If text is very short, treat the whole thing as one chunk
            chunk_id = _make_chunk_id(study.pmid, 0)
            all_chunks.append(Chunk(
                chunk_id=chunk_id,
                study_pmid=study.pmid,
                text=full_text,
                start_char=0,
                end_char=len(full_text),
                paper_id=study.pmid,
                study_metadata=study_metadata.get(study.pmid, {}),
            ))
        else:
            for i, (text, sc, ec) in enumerate(raw_chunks):
                chunk_id = _make_chunk_id(study.pmid, i)
                all_chunks.append(Chunk(
                    chunk_id=chunk_id,
                    study_pmid=study.pmid,
                    text=text,
                    start_char=sc,
                    end_char=ec,
                    paper_id=study.pmid,
                    study_metadata=study_metadata.get(study.pmid, {}),
                ))

    logger.info("Created %d chunks from %d articles", len(all_chunks), len(studies))

    # ---- Structured profiling + deterministic validation ---------- #
    profiles_by_paper: Dict[str, Any] = {}
    validation_report: Dict[str, Any] = {"summary": {}, "flags": []}
    if guardrails_enabled or validators_enabled:
        try:
            from paper_profiler import profile_and_validate
            profiles_by_paper, validation_report = profile_and_validate(studies, all_chunks, cfg)
        except Exception as exc:
            logger.warning("Paper profiling failed: %s", exc)
            profiles_by_paper = {}
            validation_report = {"summary": {"total_flags": 0}, "flags": []}

    # ---- Stage 5: Embedding + Tag mapping -------------------------- #
    logger.info("Embedding %d chunks + %d taxonomy prompts", len(all_chunks), len(taxonomy_entries))

    chunk_texts = [c.text for c in all_chunks]
    prompt_texts = [e["prompt"] for e in taxonomy_entries]

    chunk_embeds = _embed_texts(chunk_texts, model, batch_size=batch_size, desc="Embedding chunks")
    prompt_embeds = _embed_texts(prompt_texts, model, batch_size=batch_size, desc="Embedding prompts")

    # Compute similarity matrix: (n_chunks, n_prompts)
    sim_matrix = _cosine_matrix(chunk_embeds, prompt_embeds)

    all_tags: List[ChunkTag] = []
    routing_flags: List[ValidationFlag] = []
    if guardrails_enabled:
        from validators import evaluate_routing_guardrails

    for i, chunk in enumerate(tqdm(all_chunks, desc="Tag mapping")):
        scores = sim_matrix[i]
        # Get top-k indices above threshold
        sorted_indices = np.argsort(scores)[::-1]

        assigned = 0
        for j in sorted_indices:
            if assigned >= top_k:
                break
            base_score = float(scores[j])

            entry = taxonomy_entries[j]
            entry_min_similarity = float(entry.get("min_similarity", 0.0) or 0.0)
            effective_threshold = max(sim_threshold, entry_min_similarity)
            if base_score < effective_threshold:
                break

            eligibility_score = 1.0
            routing_notes: List[str] = []
            is_eligible = True
            final_score = base_score

            if guardrails_enabled:
                profile = profiles_by_paper.get(chunk.paper_id or chunk.study_pmid)
                strictness = entry.get("routing_strictness", strictness_default)
                eligibility_score, routing_notes, extra_flags, is_eligible = evaluate_routing_guardrails(
                    profile=profile,
                    chunk_text=chunk.text,
                    chunk_id=chunk.chunk_id,
                    allowed_domains=entry.get("allowed_domains", []),
                    required_phase_regimes=entry.get("required_phase_regimes", []),
                    required_any_terms=entry.get("required_any_terms", []),
                    forbidden_terms=entry.get("forbidden_terms", []),
                    max_foreign_domain_ratio=entry.get("max_foreign_domain_ratio", 1.0),
                    routing_strictness=strictness,
                )
                routing_flags.extend(extra_flags)
                final_score = base_score * max(0.0, min(1.0, eligibility_score))

                # In soft mode we allow mild penalties, but still drop very weak links.
                routed_threshold = sim_threshold * (0.6 if strictness == "soft" else 1.0)
                if (not is_eligible) or final_score < routed_threshold:
                    continue

            tag = ChunkTag(
                chunk_id=chunk.chunk_id,
                folder=entry["folder"],
                parent=entry["parent"],
                similarity=final_score,
                eligibility_score=eligibility_score,
                routing_notes=routing_notes,
            )
            all_tags.append(tag)
            assigned += 1

    logger.info("Created %d chunk-tag mappings", len(all_tags))

    if routing_flags:
        validation_report = _merge_validation_flags(validation_report, routing_flags)
        save_json(validation_report, "data/processed/validation_report.json")

    # ---- Stage 6: Correlation / coverage report -------------------- #
    coverage: Dict[str, Any] = {"sections": [], "summary": {}}
    total_with_evidence = 0

    for entry in taxonomy_entries:
        section_tags = [
            t for t in all_tags
            if t.folder == entry["folder"] and t.parent == entry["parent"]
        ]
        has_evidence = len(section_tags) > 0
        if has_evidence:
            total_with_evidence += 1

        avg_sim = (
            sum(t.similarity for t in section_tags) / len(section_tags)
            if section_tags else 0.0
        )
        avg_elig = (
            sum(t.eligibility_score for t in section_tags) / len(section_tags)
            if section_tags else 0.0
        )

        coverage["sections"].append({
            "parent": entry["parent"],
            "folder": entry["folder"],
            "n_chunks": len(section_tags),
            "avg_similarity": round(avg_sim, 4),
            "avg_eligibility": round(avg_elig, 4),
            "status": "✅ has evidence" if has_evidence else "⚠️ no evidence",
        })

    coverage["summary"] = {
        "total_sections": len(taxonomy_entries),
        "sections_with_evidence": total_with_evidence,
        "sections_without_evidence": len(taxonomy_entries) - total_with_evidence,
        "total_chunks": len(all_chunks),
        "total_tags": len(all_tags),
    }

    # Log coverage
    for sec in coverage["sections"]:
        if sec["n_chunks"] == 0:
            logger.warning("  ⚠️  No evidence for: %s / %s", sec["parent"], sec["folder"])
    logger.info(
        "Coverage: %d/%d sections have evidence (%d chunks, %d tags)",
        total_with_evidence, len(taxonomy_entries), len(all_chunks), len(all_tags),
    )

    # ---- Persist to DB and JSON ------------------------------------ #
    conn = get_db_connection(cfg)
    ts = now_iso()

    for chunk in all_chunks:
        conn.execute(
            "INSERT OR REPLACE INTO chunks (chunk_id, study_pmid, text, start_char, end_char, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (chunk.chunk_id, chunk.study_pmid, chunk.text, chunk.start_char, chunk.end_char, ts),
        )

    for tag in all_tags:
        conn.execute(
            "INSERT INTO chunk_tags (chunk_id, folder, parent, similarity, created_at) VALUES (?, ?, ?, ?, ?)",
            (tag.chunk_id, tag.folder, tag.parent, tag.similarity, ts),
        )

    conn.commit()
    conn.close()

    save_json([c.model_dump() for c in all_chunks], "data/processed/chunks.json")
    save_json([t.model_dump() for t in all_tags], "data/processed/chunk_tags.json")
    save_json(coverage, "data/processed/coverage_report.json")

    return all_chunks, all_tags, coverage
