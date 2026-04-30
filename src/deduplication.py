"""
deduplication.py — Remove duplicate studies using semantic similarity.

Uses sentence-transformers embeddings and cosine similarity to detect
near-duplicate titles+abstracts, keeping only unique records.

Falls back to normalised exact-title matching when sentence-transformers
or PyTorch are not available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from utils import StudyRecord, get_db_connection, now_iso

logger = logging.getLogger("systematic_review.deduplication")

# Lazy import — sentence-transformers requires PyTorch >= 2.4
# Lazy import — sentence-transformers requires PyTorch >= 2.4
_EMBEDDINGS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _EMBEDDINGS_AVAILABLE = True
except Exception as exc:
    logger.warning(
        "sentence-transformers not available or failed to load (%s). "
        "Falling back to exact-title deduplication.", exc
    )
except SystemExit:
    raise
except BaseException as exc:
    # Catch-all for weird import crashes (e.g. DLL load failed, KeyboardInterrupt during import)
    logger.warning(
        "sentence-transformers crashed during import (%s). "
        "Falling back to exact-title deduplication.", exc
    )


# ------------------------------------------------------------------ #
#  Fallback: exact-title deduplication                                 #
# ------------------------------------------------------------------ #

def _deduplicate_exact(
    studies: List[StudyRecord],
    cfg: Dict[str, Any],
) -> List[StudyRecord]:
    """Remove studies with identical normalised titles."""
    conn = get_db_connection(cfg)
    keep: List[StudyRecord] = []
    seen_titles: Dict[str, str] = {}  # norm_title → first pmid

    for s in studies:
        norm = s.title.strip().lower()
        if norm in seen_titles:
            conn.execute(
                "INSERT INTO dedup_log (kept_pmid, removed_pmid, similarity, decided_at) VALUES (?, ?, ?, ?)",
                (seen_titles[norm], s.pmid, 1.0, now_iso()),
            )
        else:
            seen_titles[norm] = s.pmid
            keep.append(s)

    conn.commit()
    conn.close()
    logger.info("Exact-title dedup: %d → %d studies", len(studies), len(keep))
    return keep


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def deduplicate(
    studies: List[StudyRecord],
    cfg: Dict[str, Any],
) -> List[StudyRecord]:
    """Return a deduplicated list of studies.

    Uses semantic (embedding-based) deduplication when available;
    otherwise falls back to exact-title matching.
    """
    if not studies:
        return studies

    dedup_cfg = cfg.get("deduplication", {})
    model_name = dedup_cfg.get("model", "all-MiniLM-L6-v2")
    threshold = dedup_cfg.get("similarity_threshold", 0.95)

    # Check global system capabilities if available, else fall back to module-level check
    sys_caps = cfg.get("system", {})
    has_embeddings = sys_caps.get("sentence_transformers", _EMBEDDINGS_AVAILABLE)

    if not has_embeddings:
        return _deduplicate_exact(studies, cfg)

    logger.info(
        "Deduplicating %d studies (model=%s, threshold=%.2f)",
        len(studies), model_name, threshold,
    )

    # Load embedding model
    model = SentenceTransformer(model_name)

    # Build texts for embedding
    texts = [f"{s.title} {s.abstract}" for s in studies]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    # Pairwise duplicate detection
    keep: List[StudyRecord] = []
    removed_indices: set[int] = set()
    conn = get_db_connection(cfg)

    for i in tqdm(range(len(embeddings)), desc="Deduplicating"):
        if i in removed_indices:
            continue

        keep.append(studies[i])

        for j in range(i + 1, len(embeddings)):
            if j in removed_indices:
                continue

            sim = cosine_similarity(
                [embeddings[i]], [embeddings[j]]
            )[0][0]

            if sim > threshold:
                removed_indices.add(j)
                logger.debug(
                    "Duplicate: %s ↔ %s (sim=%.4f)",
                    studies[i].pmid, studies[j].pmid, sim,
                )
                conn.execute(
                    "INSERT INTO dedup_log (kept_pmid, removed_pmid, similarity, decided_at) VALUES (?, ?, ?, ?)",
                    (studies[i].pmid, studies[j].pmid, float(sim), now_iso()),
                )

    conn.commit()
    conn.close()

    logger.info(
        "Deduplication complete: %d → %d studies (%d removed)",
        len(studies), len(keep), len(removed_indices),
    )
    return keep
