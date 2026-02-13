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


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def analyze_and_chunk(
    studies: List[StudyRecord],
    taxonomy_entries: List[Dict[str, str]],
    cfg: Dict[str, Any],
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
    analyzer_cfg = cfg.get("content_analyzer", {})
    model_name = analyzer_cfg.get("model", "all-MiniLM-L6-v2")
    max_tokens = analyzer_cfg.get("chunk_max_tokens", 400)
    overlap = analyzer_cfg.get("chunk_overlap", 50)
    sim_threshold = analyzer_cfg.get("similarity_threshold", 0.25)
    top_k = analyzer_cfg.get("top_k_tags", 3)

    # Determine device
    sys_caps = cfg.get("system", {})
    device = "cuda" if sys_caps.get("cuda", False) else "cpu"

    # ---- Stage 3: Content analysis (load model) -------------------- #
    logger.info("Loading embedding model '%s' on %s", model_name, device)
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
                ))

    logger.info("Created %d chunks from %d articles", len(all_chunks), len(studies))

    # ---- Stage 5: Embedding + Tag mapping -------------------------- #
    logger.info("Embedding %d chunks + %d taxonomy prompts", len(all_chunks), len(taxonomy_entries))

    chunk_texts = [c.text for c in all_chunks]
    prompt_texts = [e["prompt"] for e in taxonomy_entries]

    chunk_embeds = _embed_texts(chunk_texts, model, desc="Embedding chunks")
    prompt_embeds = _embed_texts(prompt_texts, model, desc="Embedding prompts")

    # Compute similarity matrix: (n_chunks, n_prompts)
    sim_matrix = _cosine_matrix(chunk_embeds, prompt_embeds)

    all_tags: List[ChunkTag] = []

    for i, chunk in enumerate(tqdm(all_chunks, desc="Tag mapping")):
        scores = sim_matrix[i]
        # Get top-k indices above threshold
        sorted_indices = np.argsort(scores)[::-1]

        assigned = 0
        for j in sorted_indices:
            if assigned >= top_k:
                break
            if scores[j] < sim_threshold:
                break

            entry = taxonomy_entries[j]
            tag = ChunkTag(
                chunk_id=chunk.chunk_id,
                folder=entry["folder"],
                parent=entry["parent"],
                similarity=float(scores[j]),
            )
            all_tags.append(tag)
            assigned += 1

    logger.info("Created %d chunk-tag mappings", len(all_tags))

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

        coverage["sections"].append({
            "parent": entry["parent"],
            "folder": entry["folder"],
            "n_chunks": len(section_tags),
            "avg_similarity": round(avg_sim, 4),
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
