"""
post_processor.py — Post-process the systematic review for narrative refinement.

Transforms the v1 review (author-centric paragraphs) into a consensus-oriented
academic narrative.  Each section is sent to the LLM for reorganisation by theme
rather than by author.

Configuration via ``cfg["post_processing"]``::

    post_processing:
      enabled: true
      parallel_workers: 1
      max_retries: 1
      preserve_v1: true
"""

from __future__ import annotations

import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger("systematic_review.post_processor")


# ------------------------------------------------------------------ #
#  Refinement prompt                                                   #
# ------------------------------------------------------------------ #

_REFINE_PROMPT = """\
You are an academic editor refining a systematic review section.

### Original section
{section_text}

### Refinement rules
1. Analyse the section to identify convergence and divergence across
   the cited authors.  Make the consensus and disagreement EXPLICIT in
   the revised text.
2. Reorganise content by THEME, not by author.  The subject of each
   sentence must be the phenomenon, finding, or concept — NOT the
   author name.  Author citations appear only parenthetically.
3. At most 2 consecutive sentences may begin with an author reference.
4. Strengthen transitions between paragraphs — each paragraph should
   logically follow from the previous one.
5. Where studies present contradictory findings, discuss possible
   reasons for the divergence (e.g. different methodologies, sample
   sizes, geographic contexts).
6. Distinguish between what authors OBSERVE (empirical data) and what
   they CONCLUDE (interpretation or recommendation).
7. Preserve ALL citations exactly as given — do NOT add, remove, or
   modify any (Author, Year) reference.
8. Keep formal academic language.
9. Maintain the same approximate length — do NOT shorten significantly.
10. Do NOT add a heading or title — return only the refined body text.
11. CRITICAL ANALYSIS: If any claim is stated without qualification
   (e.g. "yield of 98%"), add context about experimental conditions,
   scalability limitations, or contradictions from other cited studies.
12. Use hedging language for conclusions not fully verified at scale:
   "embora", "sob condições controladas", "a escalabilidade permanece
   um desafio", "resultados preliminares sugerem".
13. Flag results obtained only under idealised laboratory conditions
   and note whether pilot-scale or field validation exists in the
   cited evidence.
14. ELIMINATE REDUNDANCIES: If the same concept, finding, or argument
   appears more than once, merge all occurrences into a single,
   comprehensive statement.  Remove duplicate citations within the
   same parenthetical group (e.g. "(X, 2023; X, 2023)" → "(X, 2023)").
15. REPLACE GENERIC CONNECTORS: Do NOT start any paragraph with
   "Além disso", "No entanto", "Adicionalmente", "Por outro lado", or
   "Em contrapartida".  Replace them with substantive topic sentences
   that preview the paragraph's argument and narratively link it to
   the previous paragraph.
16. Ensure ALL text is in the same language as the original section.
   Remove or translate any stray sentences, phrases, or headings in
   a different language.
17. Normalise terminology: use consistent spelling throughout (e.g.
   "catálise" not "catalise").
18. Ensure the same citation appears in the same format throughout
   (e.g. always "Verma, 2023" or always "J. Verma, 2023", not both).

Return the refined section text only, nothing else.
"""


_DEDUP_PROMPT = """\
You are a redundancy editor for an academic systematic review.

The chapter below contains REDUNDANT content: the same concept, finding,
or argument is explained more than once across different paragraphs.

### Chapter: {chapter_title}

{chapter_text}

### Your task
1. Identify paragraphs or sentences that repeat the same information.
2. MERGE duplicate explanations into a single, comprehensive statement
   at the FIRST occurrence.
3. At subsequent occurrences, replace with a brief cross-reference
   (e.g. "conforme discutido anteriormente" or simply remove the
   redundant passage).
4. Preserve ALL citations — do NOT remove any (Author, Year) reference.
   Merge citation groups if needed (e.g. two identical claims in
   different paragraphs become one claim citing all relevant authors).
5. Keep the same headings, section structure, and paragraph order.
6. Do NOT add new content or change the meaning.
7. Return the chapter text with redundancies eliminated.
"""


_COHERENCE_PROMPT = """\
You are an academic editor reviewing a CHAPTER of a systematic review
for internal coherence.  The chapter contains multiple sections.

### Chapter: {chapter_title}

{chapter_text}

### Coherence rules
1. Remove information that is redundantly repeated across sections
   within this chapter.  Each concept or finding should appear in ONLY
   ONE section — the most appropriate one for that topic.
2. Ensure consistent terminology — the same concept should use the
   same term throughout all sections.
3. Add brief NARRATIVE transitional sentences at the END of each
   section to connect it to the next section's topic.  Do NOT use
   generic connectors like "Além disso" or "No entanto".
4. If two sections present contradictory claims, add a note
   acknowledging the discrepancy and suggesting possible reasons.
5. Preserve ALL citations exactly as given.
6. Preserve ALL section headings (### lines) exactly as given.
7. Do NOT change the number of sections or their order.
8. Remove duplicate citations within the same parenthetical group
   (e.g. "(X, 2023; X, 2023)" → "(X, 2023)").
9. Replace any generic paragraph-opening connectors ("Além disso",
   "Adicionalmente", etc.) with substantive topic sentences.
10. Remove or translate any stray English phrases or sentences — the
    entire text must be in the same language.
11. Ensure citation format is consistent throughout the chapter (e.g.
    always "Verma, 2023" not sometimes "J. Verma, 2023").
8. Return the entire chapter text (with ### headings) with your edits.
"""


_ARGUMENTATION_PROMPT = """\
You are an argumentation validator for a systematic literature review.

### Chapter: {chapter_title}
### Chapter thesis (the narrative should converge toward this):
{chapter_thesis}

### Current chapter text:
{chapter_text}

### Task
1. Check whether each section contributes evidence or analysis that
   supports, qualifies, or contextualises the thesis stated above.
2. Identify paragraphs that are DISCONNECTED from the thesis —
   they present information but do not link back to the central argument.
3. For each disconnected paragraph, add 1–2 sentences that explicitly
   connect its content to the thesis (e.g. "Estes resultados reforçam
   a tese de que...").
4. At the end of the LAST section, ensure there is a brief synthesis
   paragraph that ties back to the thesis.
5. Preserve ALL citations, section headings, and structure.
6. Return the entire revised chapter text.
"""


def _validate_argumentation(
    sections: List[_Section],
    chapter_theses: Dict[str, str],
    cfg: Dict[str, Any],
    max_retries: int = 1,
) -> List[_Section]:
    """Validate and strengthen argumentation across chapter sections.

    For each chapter that has a thesis, sends the combined text to the
    LLM to check alignment and add connecting sentences.

    Returns the updated list of sections.
    """
    from utils import call_llm
    from collections import defaultdict

    if not chapter_theses:
        return sections

    # Group level-3 sections by parent
    chapter_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, sec in enumerate(sections):
        if sec.level == 3:
            chapter_groups[sec.parent].append(idx)

    result = list(sections)

    for parent, indices in chapter_groups.items():
        thesis = chapter_theses.get(parent)
        if not thesis or thesis.startswith("(no specific thesis"):
            continue

        # Build combined chapter text
        parts = []
        for idx in indices:
            s = sections[idx]
            parts.append(f"### {s.heading}\n\n{s.body}")
        chapter_text = "\n\n".join(parts)

        prompt = _ARGUMENTATION_PROMPT.format(
            chapter_title=parent,
            chapter_thesis=thesis,
            chapter_text=chapter_text,
        )

        try:
            refined_text = call_llm(prompt, cfg)
        except Exception as exc:
            logger.warning(
                "Argumentation validation failed for '%s': %s — skipping",
                parent, exc,
            )
            continue

        # Parse refined text back into sections
        _, refined_sections = _split_sections(refined_text)
        refined_level3 = [s for s in refined_sections if s.level == 3]

        if len(refined_level3) == len(indices):
            for idx, new_sec in zip(indices, refined_level3):
                result[idx] = _Section(
                    heading=sections[idx].heading,
                    body=new_sec.body,
                    level=3,
                    parent=parent,
                )
            logger.info(
                "  ✓ Argumentation validated for '%s' (%d sections)",
                parent, len(indices),
            )
        else:
            logger.warning(
                "Argumentation validation changed section count for '%s' "
                "(%d → %d) — keeping original",
                parent, len(indices), len(refined_level3),
            )

    return result


# ------------------------------------------------------------------ #
#  Data structures                                                     #
# ------------------------------------------------------------------ #

@dataclass
class _Section:
    """Represents a parsed section from the Markdown review."""
    heading: str
    body: str
    level: int  # heading level: 2 for ##, 3 for ###
    parent: str = ""


# ------------------------------------------------------------------ #
#  Section parsing                                                     #
# ------------------------------------------------------------------ #

def _split_sections(md_text: str) -> tuple[str, List[_Section]]:
    """Parse a Markdown document into its preamble and sections.

    Sections are identified by ``##`` and ``###`` headings.

    Returns
    -------
    preamble : str
        Content before the first ``##`` heading (title, horizontal rule, etc.)
    sections : list[_Section]
        Parsed sections with heading, body, and level.
    """
    lines = md_text.splitlines()
    preamble_lines: List[str] = []
    sections: List[_Section] = []
    current_heading: Optional[str] = None
    current_level: int = 0
    current_body: List[str] = []
    current_parent: str = ""

    for line in lines:
        # Detect ## or ### headings
        m = re.match(r'^(#{2,3})\s+(.+)$', line)
        if m:
            # Save previous section
            if current_heading is not None:
                sections.append(_Section(
                    heading=current_heading,
                    body='\n'.join(current_body).strip(),
                    level=current_level,
                    parent=current_parent,
                ))

            level = len(m.group(1))
            heading = m.group(2).strip()

            if level == 2:
                current_parent = heading

            current_heading = heading
            current_level = level
            current_body = []
        elif current_heading is not None:
            current_body.append(line)
        else:
            preamble_lines.append(line)

    # Save last section
    if current_heading is not None:
        sections.append(_Section(
            heading=current_heading,
            body='\n'.join(current_body).strip(),
            level=current_level,
            parent=current_parent,
        ))

    preamble = '\n'.join(preamble_lines).strip()
    return preamble, sections


# ------------------------------------------------------------------ #
#  LLM refinement                                                      #
# ------------------------------------------------------------------ #

def _refine_section(
    section: _Section,
    cfg: Dict[str, Any],
    max_retries: int = 1,
) -> _Section:
    """Send a single section to the LLM for refinement.

    Returns a new _Section with the refined body.
    """
    from utils import call_llm

    # Skip sections with very little content
    if len(section.body.split()) < 20:
        logger.debug("Skipping refinement for short section: %s", section.heading)
        return section

    prompt = _REFINE_PROMPT.format(section_text=section.body)

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            refined_text = call_llm(prompt, cfg)
            return _Section(
                heading=section.heading,
                body=refined_text,
                level=section.level,
                parent=section.parent,
            )
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                logger.warning(
                    "Refinement attempt %d/%d failed for '%s': %s — retrying",
                    attempt + 1, max_retries + 1, section.heading, exc,
                )

    logger.error(
        "Refinement failed for '%s' after %d attempts: %s",
        section.heading, max_retries + 1, last_err,
    )
    return section  # Return original on failure


# ------------------------------------------------------------------ #
#  Document reassembly                                                 #
# ------------------------------------------------------------------ #

def _reassemble_markdown(sections: List[_Section], preamble: str) -> str:
    """Reconstruct a Markdown document from sections."""
    parts: List[str] = []

    if preamble:
        parts.append(preamble)
        parts.append('')

    for sec in sections:
        prefix = '#' * sec.level
        parts.append(f'{prefix} {sec.heading}')
        parts.append('')
        if sec.body:
            parts.append(sec.body)
            parts.append('')

    return '\n'.join(parts)


# ------------------------------------------------------------------ #
#  Programmatic textual cleanup (no LLM)                               #
# ------------------------------------------------------------------ #

# Common Portuguese spelling normalisations
_SPELLING_FIXES = {
    'catalise': 'catálise',
    'catalises': 'catálises',
    'Catalise': 'Catálise',
    'transesterificaçao': 'transesterificação',
    'Transesterificaçao': 'Transesterificação',
    'saponificaçao': 'saponificação',
    'esterificaçao': 'esterificação',
}

# English fragments often left by the LLM
_STRAY_ENGLISH_PATTERNS = [
    re.compile(r'^\s*Future research directions[^.]*\.\s*$', re.MULTILINE),
    re.compile(r'^\s*In summary[^.]*\.\s*$', re.MULTILINE),
    re.compile(r'^\s*In conclusion[^.]*\.\s*$', re.MULTILINE),
    re.compile(r'^\s*Overall[,;][^.]*\.\s*$', re.MULTILINE),
    re.compile(r'^\s*This section (discusses|presents|reviews|examines)[^.]*\.\s*$',
               re.MULTILINE),
]


def _dedup_citations_in_group(match: re.Match) -> str:
    """Remove duplicate (Author, Year) entries in a single parenthetical group."""
    inner = match.group(1)
    # Split by ';' and deduplicate while keeping order
    parts = [p.strip() for p in inner.split(';')]
    seen: set = set()
    unique: list = []
    for part in parts:
        key = part.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(part)
    return '(' + '; '.join(unique) + ')'


def _textual_cleanup(text: str) -> str:
    """Apply deterministic, regex-based cleanups to the final Markdown.

    This runs WITHOUT LLM calls and catches issues the prompts may miss:
    - Duplicate citations in the same parenthetical group
    - Stray English sentences in Portuguese text
    - Common spelling normalisations
    """
    # 1. Deduplicate citations within parenthetical groups
    #    Matches: (Author, 2023; Author, 2023; Other, 2024)
    text = re.sub(
        r'\(([^)]*\d{4}[^)]*;[^)]*)\)',
        _dedup_citations_in_group,
        text,
    )

    # 2. Remove stray English sentences
    for pattern in _STRAY_ENGLISH_PATTERNS:
        text = pattern.sub('', text)

    # 3. Apply spelling normalisations
    for wrong, correct in _SPELLING_FIXES.items():
        text = text.replace(wrong, correct)

    # 4. Clean up excessive blank lines (more than 2 → 2)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text


def _check_chapter_coherence(
    sections: List[_Section],
    cfg: Dict[str, Any],
    max_retries: int = 1,
) -> List[_Section]:
    """Check and improve coherence across sections within a chapter.

    Groups sections by their parent (## heading) and sends each
    chapter's combined text to the LLM for cross-section review.

    Returns the updated list of sections.
    """
    from utils import call_llm
    from collections import defaultdict

    # Group level-3 sections by parent
    chapter_groups: Dict[str, List[int]] = defaultdict(list)
    for i, sec in enumerate(sections):
        if sec.level == 3 and sec.body.strip():
            chapter_groups[sec.parent].append(i)

    if not chapter_groups:
        return sections

    result = list(sections)  # copy

    for chapter_title, indices in chapter_groups.items():
        if len(indices) < 2:
            # No cross-section coherence needed for single-section chapters
            continue

        # Build chapter text with ### headings
        chapter_parts: List[str] = []
        for idx in indices:
            sec = sections[idx]
            chapter_parts.append(f"### {sec.heading}")
            chapter_parts.append("")
            chapter_parts.append(sec.body)
            chapter_parts.append("")

        chapter_text = "\n".join(chapter_parts)

        logger.info(
            "  Coherence check: %s (%d sections, ~%d chars)",
            chapter_title, len(indices), len(chapter_text),
        )

        prompt = _COHERENCE_PROMPT.format(
            chapter_title=chapter_title,
            chapter_text=chapter_text,
        )

        last_err = None
        refined_text = None
        for attempt in range(max_retries + 1):
            try:
                refined_text = call_llm(prompt, cfg)
                break
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    logger.warning(
                        "Coherence attempt %d/%d failed for '%s': %s — retrying",
                        attempt + 1, max_retries + 1, chapter_title, exc,
                    )

        if refined_text is None:
            logger.warning(
                "Coherence check failed for '%s' after %d attempts: %s — keeping original",
                chapter_title, max_retries + 1, last_err,
            )
            continue

        # Parse the refined chapter text back into sections
        _, refined_sections = _split_sections(refined_text)
        refined_level3 = [s for s in refined_sections if s.level == 3]

        if len(refined_level3) == len(indices):
            for idx, new_sec in zip(indices, refined_level3):
                result[idx] = _Section(
                    heading=sections[idx].heading,  # keep original heading
                    body=new_sec.body,
                    level=3,
                    parent=chapter_title,
                )
        else:
            logger.warning(
                "Coherence check returned %d sections for '%s' (expected %d) — keeping original",
                len(refined_level3), chapter_title, len(indices),
            )

    return result


def _dedup_chapters(
    sections: List[_Section],
    cfg: Dict[str, Any],
    max_retries: int = 1,
) -> List[_Section]:
    """Remove redundant content within each chapter.

    Groups level-3 sections by parent (## heading) and sends each
    chapter's combined text to the LLM for deduplication.

    Returns the updated list of sections.
    """
    from utils import call_llm
    from collections import defaultdict

    # Group level-3 sections by parent
    chapter_groups: Dict[str, List[int]] = defaultdict(list)
    for i, sec in enumerate(sections):
        if sec.level == 3 and sec.body.strip():
            chapter_groups[sec.parent].append(i)

    if not chapter_groups:
        return sections

    result = list(sections)  # copy

    for chapter_title, indices in chapter_groups.items():
        if len(indices) < 2:
            continue

        # Build chapter text with ### headings
        chapter_parts: List[str] = []
        for idx in indices:
            sec = sections[idx]
            chapter_parts.append(f"### {sec.heading}")
            chapter_parts.append("")
            chapter_parts.append(sec.body)
            chapter_parts.append("")

        chapter_text = "\n".join(chapter_parts)

        logger.info(
            "  Deduplication: %s (%d sections, ~%d chars)",
            chapter_title, len(indices), len(chapter_text),
        )

        prompt = _DEDUP_PROMPT.format(
            chapter_title=chapter_title,
            chapter_text=chapter_text,
        )

        last_err = None
        refined_text = None
        for attempt in range(max_retries + 1):
            try:
                refined_text = call_llm(prompt, cfg)
                break
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    logger.warning(
                        "Dedup attempt %d/%d failed for '%s': %s — retrying",
                        attempt + 1, max_retries + 1, chapter_title, exc,
                    )

        if refined_text is None:
            logger.warning(
                "Dedup failed for '%s' after %d attempts: %s — keeping original",
                chapter_title, max_retries + 1, last_err,
            )
            continue

        # Parse the deduped chapter text back into sections
        _, deduped_sections = _split_sections(refined_text)
        deduped_level3 = [s for s in deduped_sections if s.level == 3]

        if len(deduped_level3) == len(indices):
            for idx, new_sec in zip(indices, deduped_level3):
                result[idx] = _Section(
                    heading=sections[idx].heading,  # keep original heading
                    body=new_sec.body,
                    level=3,
                    parent=chapter_title,
                )
        else:
            logger.warning(
                "Dedup returned %d sections for '%s' (expected %d) — keeping original",
                len(deduped_level3), chapter_title, len(indices),
            )

    return result


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def post_process_review(
    input_path: str | Path,
    output_path: str | Path | None = None,
    cfg: Dict[str, Any] | None = None,
) -> str:
    """Refine a systematic review document via LLM post-processing.

    Parameters
    ----------
    input_path : str or Path
        Path to the v1 Markdown review.
    output_path : str or Path, optional
        Path for the refined output. Defaults to same file (overwrite).
    cfg : dict, optional
        Global configuration. Reads ``post_processing`` section.

    Returns
    -------
    str
        Path to the refined document.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    output_path = Path(output_path)

    if cfg is None:
        cfg = {}

    pp_cfg = cfg.get("post_processing", {})
    enabled = pp_cfg.get("enabled", True)
    parallel_workers = pp_cfg.get("parallel_workers", 1)
    max_retries = pp_cfg.get("max_retries", 1)
    preserve_v1 = pp_cfg.get("preserve_v1", True)

    if not enabled:
        logger.info("Post-processing disabled in config — skipping")
        return str(input_path)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return str(input_path)

    # Read v1
    md_text = input_path.read_text(encoding='utf-8')
    preamble, sections = _split_sections(md_text)

    # Filter to only ### sections (subsections with actual content)
    refinable = [s for s in sections if s.level == 3 and s.body.strip()]
    logger.info(
        "Post-processing: %d refinable sections of %d total",
        len(refinable), len(sections),
    )

    if not refinable:
        logger.info("No refinable sections found — skipping")
        return str(input_path)

    # Preserve v1 backup
    if preserve_v1 and output_path == input_path:
        backup_path = input_path.with_suffix('.v1.md')
        shutil.copy2(input_path, backup_path)
        logger.info("v1 backup saved → %s", backup_path)

    # Build a mapping for refined sections
    refined_map: Dict[str, _Section] = {}

    if parallel_workers > 1:
        # Parallel refinement
        logger.info("Refining %d sections (%d parallel workers)", len(refinable), parallel_workers)
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_heading = {}
            for sec in refinable:
                future = executor.submit(_refine_section, sec, cfg, max_retries)
                future_to_heading[future] = sec.heading

            with tqdm(total=len(refinable), desc="Refining sections") as pbar:
                for future in as_completed(future_to_heading):
                    heading = future_to_heading[future]
                    try:
                        refined_sec = future.result()
                        refined_map[heading] = refined_sec
                    except Exception as exc:
                        logger.error("Worker failed for '%s': %s", heading, exc)
                    pbar.update(1)
    else:
        # Sequential refinement
        logger.info("Refining %d sections (sequential)", len(refinable))
        for sec in tqdm(refinable, desc="Refining sections"):
            refined = _refine_section(sec, cfg, max_retries)
            refined_map[sec.heading] = refined

    # Merge refined sections back
    merged: List[_Section] = []
    for sec in sections:
        if sec.heading in refined_map:
            merged.append(refined_map[sec.heading])
        else:
            merged.append(sec)

    # ---- Cross-chapter deduplication (Épico 4) ---- #
    logger.info("Running per-chapter deduplication")
    merged = _dedup_chapters(merged, cfg, max_retries)

    # ---- Cross-section coherence check ---- #
    logger.info("Running cross-section coherence check")
    merged = _check_chapter_coherence(merged, cfg, max_retries)

    # Reassemble
    result = _reassemble_markdown(merged, preamble)

    # ---- Programmatic textual cleanup ---- #
    logger.info("Running programmatic textual cleanup")
    result = _textual_cleanup(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding='utf-8')

    logger.info("Refined review saved → %s", output_path)
    return str(output_path)
