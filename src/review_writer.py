"""
review_writer.py — Write the systematic review using tagged chunks.

Stages 7–8 of the local pipeline:
  7. Section Writing    — for each taxonomy entry, gather top chunks and
                          call the LLM to write that section
  8. Assemble Document  — combine sections in hierarchical order
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List

from tqdm import tqdm

from utils import (
    Chunk,
    ChunkTag,
    call_llm,
    save_json,
    _resolve,
)

logger = logging.getLogger("systematic_review.review_writer")

# ------------------------------------------------------------------ #
#  Section writing                                                     #
# ------------------------------------------------------------------ #

_SECTION_PROMPT = """\
You are a scientific review writer.  Write a section for a systematic
review using ONLY the evidence provided below.

### Section
Chapter: {parent}
Section: {folder}
Content guidance: {prompt}

### Evidence (excerpts from published articles)
{evidence}

### Instructions
- Write in formal academic English (or match the language of the evidence).
- Use the evidence to support your statements.  Cite by [Study ID] when
  referencing a specific excerpt.
- Do NOT invent information beyond what the evidence provides.
- If the evidence is thin, acknowledge gaps.
- Produce 2–4 well-structured paragraphs.
- Do NOT repeat the section title.
"""

_SECTION_NO_EVIDENCE_PROMPT = """\
You are a scientific review writer.  Write a SHORT section for a
systematic review.

### Section
Chapter: {parent}
Section: {folder}
Content guidance: {prompt}

### Instructions
- There was no direct evidence found in the analysed articles for this
  specific section.
- Write 1–2 paragraphs providing general context based on the content
  guidance above.
- Clearly state that the reviewed literature did not provide direct
  evidence for this topic and that further research is recommended.
- Write in formal academic English.
"""


def _gather_evidence(
    folder: str,
    parent: str,
    all_tags: List[ChunkTag],
    chunks_by_id: Dict[str, Chunk],
    top_k: int = 10,
) -> str:
    """Collect the top-k most relevant chunks for a given section."""
    relevant = [
        t for t in all_tags
        if t.folder == folder and t.parent == parent
    ]
    # Sort by similarity descending
    relevant.sort(key=lambda t: t.similarity, reverse=True)
    relevant = relevant[:top_k]

    if not relevant:
        return ""

    lines: List[str] = []
    seen_chunks = set()
    for tag in relevant:
        if tag.chunk_id in seen_chunks:
            continue
        seen_chunks.add(tag.chunk_id)
        chunk = chunks_by_id.get(tag.chunk_id)
        if chunk:
            lines.append(
                f"[Study {chunk.study_pmid}] (relevance: {tag.similarity:.2f})\n"
                f"{chunk.text}\n"
            )

    return "\n---\n".join(lines)


def _write_section(
    prompt: str,
    folder: str,
    parent: str,
    evidence: str,
    cfg: Dict[str, Any],
) -> str:
    """Call the LLM to write one section."""
    if evidence:
        llm_prompt = _SECTION_PROMPT.format(
            parent=parent,
            folder=folder,
            prompt=prompt,
            evidence=evidence,
        )
    else:
        llm_prompt = _SECTION_NO_EVIDENCE_PROMPT.format(
            parent=parent,
            folder=folder,
            prompt=prompt,
        )

    try:
        return call_llm(llm_prompt, cfg)
    except Exception as exc:
        logger.error("LLM failed for %s / %s: %s", parent, folder, exc)
        return f"*[Section could not be generated: {exc}]*"


# ------------------------------------------------------------------ #
#  Document assembly                                                   #
# ------------------------------------------------------------------ #

def _assemble_markdown(
    sections: List[Dict[str, Any]],
    topic: str,
) -> str:
    """Combine sections into a single Markdown document."""
    lines: List[str] = [
        f"# {topic}\n",
        "---\n",
    ]

    current_parent = None
    for sec in sections:
        # New chapter heading
        if sec["parent"] != current_parent:
            current_parent = sec["parent"]
            lines.append(f"\n## {current_parent}\n")

        lines.append(f"\n### {sec['folder']}\n")
        lines.append(sec["content"])
        lines.append("\n")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def write_review(
    taxonomy_entries: List[Dict[str, str]],
    all_chunks: List[Chunk],
    all_tags: List[ChunkTag],
    topic: str,
    cfg: Dict[str, Any],
) -> str:
    """Write the full review document.

    Returns the path to the generated Markdown file.
    """
    writer_cfg = cfg.get("review_writer", {})
    top_k_evidence = writer_cfg.get("top_k_evidence", 10)

    # Build lookup
    chunks_by_id = {c.chunk_id: c for c in all_chunks}

    # Group entries by parent to maintain chapter order
    ordered_parents: List[str] = []
    seen_parents: set = set()
    for entry in taxonomy_entries:
        if entry["parent"] not in seen_parents:
            ordered_parents.append(entry["parent"])
            seen_parents.add(entry["parent"])

    # Sort entries by parent order, then by original position
    entry_order = {entry["folder"]: i for i, entry in enumerate(taxonomy_entries)}
    sorted_entries = sorted(
        taxonomy_entries,
        key=lambda e: (ordered_parents.index(e["parent"]), entry_order.get(e["folder"], 0)),
    )

    # ---- Stage 7: Write each section ------------------------------- #
    logger.info("Writing %d sections via LLM", len(sorted_entries))
    sections: List[Dict[str, Any]] = []

    for entry in tqdm(sorted_entries, desc="Writing sections"):
        evidence = _gather_evidence(
            entry["folder"], entry["parent"],
            all_tags, chunks_by_id, top_k_evidence,
        )

        n_evidence = len(evidence.split("---")) if evidence else 0
        logger.info(
            "  Writing: %s / %s (%d evidence chunks)",
            entry["parent"], entry["folder"], n_evidence,
        )

        content = _write_section(
            entry["prompt"], entry["folder"], entry["parent"],
            evidence, cfg,
        )

        sections.append({
            "parent": entry["parent"],
            "folder": entry["folder"],
            "prompt": entry["prompt"],
            "n_evidence": n_evidence,
            "content": content,
        })

    # Save individual sections for later editing
    save_json(sections, "data/results/review_sections.json")

    # ---- Stage 8: Assemble document -------------------------------- #
    logger.info("Assembling final document")
    document = _assemble_markdown(sections, topic)

    out_path = _resolve("data/results/systematic_review.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")

    logger.info("Review saved → %s", out_path)
    return str(out_path)
