"""
organizer.py — Organize articles into folders based on taxonomy sections.

Stage 7 of the local pipeline:
  For each taxonomy entry, create a subfolder under data/organized/
  and generate a section_articles.json with metadata, similarity scores,
  and representative excerpts.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from utils import Chunk, ChunkTag, StudyRecord, _resolve

logger = logging.getLogger("systematic_review.organizer")


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _sanitize_dirname(name: str) -> str:
    """Sanitize a string for use as a directory name (Windows-safe)."""
    # Replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Collapse multiple underscores / spaces
    sanitized = re.sub(r'[_\s]+', '_', sanitized).strip('_')
    # Truncate to reasonable length
    return sanitized[:120] if sanitized else "unnamed"


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def organize_by_taxonomy(
    chunks: List[Chunk],
    tags: List[ChunkTag],
    studies: List[StudyRecord],
    taxonomy_entries: List[Dict[str, str]],
    cfg: Dict[str, Any],
) -> str:
    """Organize articles into folders by taxonomy section.

    Creates hierarchy::

        data/organized/{parent}/{folder}/section_articles.json

    Each JSON contains article metadata, similarity scores, chunk counts,
    and a representative excerpt.

    Returns the path to the organized output directory.
    """
    base_dir = _resolve("data/organized")
    base_dir.mkdir(parents=True, exist_ok=True)

    # Build lookups
    studies_by_pmid: Dict[str, StudyRecord] = {s.pmid: s for s in studies}
    chunks_by_id: Dict[str, Chunk] = {c.chunk_id: c for c in chunks}

    # Group tags by (parent, folder)
    section_tags: Dict[tuple, List[ChunkTag]] = defaultdict(list)
    for tag in tags:
        section_tags[(tag.parent, tag.folder)].append(tag)

    total_articles = 0
    total_sections = 0

    for entry in taxonomy_entries:
        parent = entry["parent"]
        folder = entry["folder"]

        parent_dir = base_dir / _sanitize_dirname(parent)
        folder_dir = parent_dir / _sanitize_dirname(folder)
        folder_dir.mkdir(parents=True, exist_ok=True)

        # Collect relevant tags for this section
        relevant_tags = section_tags.get((parent, folder), [])

        # Group by study and compute per-study stats
        study_chunks: Dict[str, List[ChunkTag]] = defaultdict(list)
        for tag in relevant_tags:
            chunk = chunks_by_id.get(tag.chunk_id)
            if chunk:
                study_chunks[chunk.study_pmid].append(tag)

        articles: List[Dict[str, Any]] = []
        for pmid, pmid_tags in study_chunks.items():
            study = studies_by_pmid.get(pmid)

            # Best similarity for this study–section pair
            best_tag = max(pmid_tags, key=lambda t: t.similarity)
            best_chunk = chunks_by_id.get(best_tag.chunk_id)

            article_info: Dict[str, Any] = {
                "pmid": pmid,
                "author": study.authors if study else pmid,
                "year": study.year if study else None,
                "citation": f"({study.authors}, {study.year})" if study and study.authors and study.year
                            else f"({pmid})",
                "similarity": round(best_tag.similarity, 4),
                "n_chunks": len(pmid_tags),
                "best_excerpt": best_chunk.text[:300] if best_chunk else "",
                # Enriched metadata (Épico 1) — from chunk.study_metadata
                "study_scale": best_chunk.study_metadata.get("study_scale", "") if best_chunk else "",
                "sample_size": best_chunk.study_metadata.get("sample_size") if best_chunk else None,
                "rob_overall": best_chunk.study_metadata.get("rob_overall", "") if best_chunk else "",
                "limitations": best_chunk.study_metadata.get("limitations", "") if best_chunk else "",
            }
            articles.append(article_info)

        # Sort by similarity descending
        articles.sort(key=lambda a: a["similarity"], reverse=True)

        # Write section_articles.json
        out_path = folder_dir / "section_articles.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(articles, fh, indent=2, ensure_ascii=False)

        total_articles += len(articles)
        total_sections += 1

        if articles:
            logger.debug(
                "  %s / %s: %d articles (best sim=%.3f)",
                parent, folder, len(articles), articles[0]["similarity"],
            )

    logger.info(
        "Organized %d articles across %d sections → %s",
        total_articles, total_sections, base_dir,
    )
    return str(base_dir)
