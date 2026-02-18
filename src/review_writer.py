"""
review_writer.py — Write the systematic review using tagged chunks.

Stages 7–8 of the local pipeline:
  7. Section Writing    — for each taxonomy entry, gather top chunks and
                          call the LLM to write that section
  8. Assemble Document  — combine sections in hierarchical order

Supports **parallel writing** when ``review_writer.parallel_workers > 1``
in config (requires Ollama ``OLLAMA_NUM_PARALLEL`` to match).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
) -> tuple[str, int]:
    """Collect the top-k most relevant chunks for a given section.

    Returns ``(evidence_text, n_chunks)``.
    """
    relevant = [
        t for t in all_tags
        if t.folder == folder and t.parent == parent
    ]
    # Sort by similarity descending
    relevant.sort(key=lambda t: t.similarity, reverse=True)
    relevant = relevant[:top_k]

    if not relevant:
        return "", 0

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

    return "\n---\n".join(lines), len(lines)


def _write_section(
    prompt: str,
    folder: str,
    parent: str,
    evidence: str,
    cfg: Dict[str, Any],
    max_retries: int = 1,
) -> str:
    """Call the LLM to write one section, with optional retries."""
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

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return call_llm(llm_prompt, cfg)
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                logger.warning(
                    "LLM attempt %d/%d failed for %s / %s: %s — retrying",
                    attempt + 1, max_retries + 1, parent, folder, exc,
                )

    logger.error("LLM failed for %s / %s after %d attempts: %s", parent, folder, max_retries + 1, last_err)
    return f"*[Section could not be generated: {last_err}]*"


def _write_single_entry(
    entry: Dict[str, str],
    all_tags: List[ChunkTag],
    chunks_by_id: Dict[str, Chunk],
    cfg: Dict[str, Any],
    top_k_evidence: int,
    max_retries: int,
) -> Dict[str, Any]:
    """Write a single section. Used as the unit of work for parallel execution."""
    evidence, n_evidence = _gather_evidence(
        entry["folder"], entry["parent"],
        all_tags, chunks_by_id, top_k_evidence,
    )

    logger.info(
        "  Writing: %s / %s (%d evidence chunks, ~%d chars)",
        entry["parent"], entry["folder"], n_evidence, len(evidence),
    )

    content = _write_section(
        entry["prompt"], entry["folder"], entry["parent"],
        evidence, cfg, max_retries,
    )

    return {
        "parent": entry["parent"],
        "folder": entry["folder"],
        "prompt": entry["prompt"],
        "n_evidence": n_evidence,
        "content": content,
    }


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
    parallel_workers = writer_cfg.get("parallel_workers", 1)
    max_retries = writer_cfg.get("max_retries", 1)

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
    n = len(sorted_entries)

    if parallel_workers > 1:
        # ---- Parallel writing ---- #
        logger.info(
            "Writing %d sections via LLM (%d parallel workers)",
            n, parallel_workers,
        )
        # We need to maintain order, so we map futures back to indices
        sections: List[Dict[str, Any] | None] = [None] * n

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_idx = {}
            for idx, entry in enumerate(sorted_entries):
                future = executor.submit(
                    _write_single_entry,
                    entry, all_tags, chunks_by_id, cfg,
                    top_k_evidence, max_retries,
                )
                future_to_idx[future] = idx

            with tqdm(total=n, desc="Writing sections") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        sections[idx] = future.result()
                    except Exception as exc:
                        entry = sorted_entries[idx]
                        logger.error("Worker failed for %s / %s: %s", entry["parent"], entry["folder"], exc)
                        sections[idx] = {
                            "parent": entry["parent"],
                            "folder": entry["folder"],
                            "prompt": entry["prompt"],
                            "n_evidence": 0,
                            "content": f"*[Section failed: {exc}]*",
                        }
                    pbar.update(1)

        # Filter out any None (shouldn't happen, but safety)
        sections = [s for s in sections if s is not None]
    else:
        # ---- Sequential writing ---- #
        logger.info("Writing %d sections via LLM (sequential)", n)
        sections = []
        for entry in tqdm(sorted_entries, desc="Writing sections"):
            result = _write_single_entry(
                entry, all_tags, chunks_by_id, cfg,
                top_k_evidence, max_retries,
            )
            sections.append(result)

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
