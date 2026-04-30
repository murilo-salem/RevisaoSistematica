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

import ast
import copy
import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set

from tqdm import tqdm

from utils import (
    Chunk,
    ChunkTag,
    StudyRecord,
    call_llm,
    load_json,
    save_json,
    _resolve,
)
from concept_registry import ConceptRegistry
from table_generator import TableRegistry, detect_table_opportunity
from evidence_filtering import build_high_flag_related_ids, filter_and_rank_tags

logger = logging.getLogger("systematic_review.review_writer")

# ------------------------------------------------------------------ #
#  Section writing                                                     #
# ------------------------------------------------------------------ #

_LANG_NAMES = {
    "pt": "português acadêmico formal",
    "en": "formal academic English",
    "es": "español académico formal",
}

_SECTION_PROMPT = """\
You are a specialist scientific review writer.  Write a section for a
systematic literature review using ONLY the evidence provided below.
Follow the multi-stage reasoning process before producing the final text.

### Section
Chapter: {parent}
Section: {folder}
Content guidance: {prompt}

### Chapter thesis
{chapter_thesis}
The narrative must converge toward the thesis above. Each paragraph
should contribute evidence or analysis that supports it.

### Concepts already covered in previous sections
{covered_concepts}
Do NOT repeat full explanations of these concepts. If you need to
reference them, do so briefly (e.g. "conforme discutido anteriormente").

### Evidence (excerpts from published articles)
{evidence}

### Stage 1 — Author Analysis (internal reasoning, do NOT include in output)
For each cited work in the evidence above, identify:
  (a) the author’s main claim or thesis,
  (b) the methodology employed (scale: lab, pilot, simulation, field),
  (c) the key quantitative or qualitative finding,
  (d) any limitation explicitly acknowledged by the author,
  (e) conditions under which the result was obtained (e.g. optimised
      lab conditions, specific catalyst, limited sample).

### Stage 2 — Thematic Synthesis (internal reasoning, do NOT include in output)
Group the findings from Stage 1 by THEME (not by author).  Identify:
  • Points of consensus across multiple studies
  • Divergences or contradictions between authors
  • Methodological gaps or under-explored areas

### Stage 2.5 — Critical Analysis (internal reasoning, do NOT include in output)
For each major finding from Stage 2, evaluate critically:
  • Were results obtained under idealised/laboratory conditions?
    If so, what are the scalability challenges?
  • Does another study in the evidence contradict or qualify this
    finding?  What might explain the discrepancy (different
    methodology, feedstock, geographic context, sample size)?
  • Are there cost, environmental, or practical constraints that the
    original authors did not fully address?
  • Is the sample size or number of replicates sufficient to
    generalise the conclusion?

### Stage 3 — Write Section (THIS is the output)
Using your analysis from Stages 1, 2, and 2.5, write the section
following these rules:
- CRITICAL ANALYSIS IS MANDATORY.  Do NOT simply report results
  uncritically.  Every major claim must be qualified with context:
  conditions, limitations, or contradictions from other studies.
- Use hedging and qualifying language: "embora", "no entanto",
  "apesar de", "sob condições controladas", "a escalabilidade
  permanece incerta", "resultados preliminares sugerem".
- When a study reports a positive result, actively look for and
  mention counterpoints or caveats from other studies in the evidence.
- The subject of every sentence should be the phenomenon, finding, or
  concept — NOT the author name.  Authors appear only in parenthetical
  citations.
- At most 2 consecutive sentences may start with an author reference.
- Cite every claim using the exact (Author, Year) format from the
  evidence — e.g. (A. Saravanan, 2019).  Do NOT invent, remove, or
  alter any citation.
- For numeric/comparative claims, append the source chunk ID in-line as
  [chunk:xxxxxxxxxxxxxxxx] using IDs provided in the evidence package.
- Distinguish between what authors OBSERVE (data) and what they
  CONCLUDE (interpretation).
- When studies disagree, present both sides and analyse possible
  reasons for the divergence.
- Do NOT invent information beyond what the evidence provides.
- If the evidence is thin, explicitly acknowledge gaps and recommend
  future research directions.
- Write in {language_name}.  ALL text must be in {language_name} —
  do NOT leave any sentence, phrase, or heading in English.
- Produce 3–6 well-structured paragraphs.
- Do NOT include the section title, stage labels, or preamble — return
  only the body text.
- DO NOT repeat the same citation more than twice in the same
  paragraph.  Never place the same (Author, Year) reference twice
  inside a single parenthetical group.
- DO NOT start paragraphs with generic connectors such as "Além
  disso", "No entanto", "Adicionalmente", "Por outro lado", or "Em
  contrapartida".  Instead, open each paragraph with a substantive
  topic sentence that previews the paragraph's main point and connects
  it to the previous paragraph's conclusion.
- Use NARRATIVE transitions that link ideas, not mechanical fillers.
  Example: instead of "Além disso, o uso de catalisadores..." write
  "Enquanto as propriedades do biodiesel apresentam vantagens claras, a
  sua produção em escala enfrenta barreiras econômicas significativas."
- Use consistent terminology: once you choose a term for a concept
  (e.g. "catálise" not "catalise"), use that exact term throughout.

### Example of critical writing style (for reference only)
BAD (uncritical): "O rendimento de biodiesel alcançou 98% (X, 2023)."
GOOD (critical): "Embora um rendimento de 98% tenha sido alcançado sob
condições laboratoriais otimizadas (X, 2023), a escalabilidade desse
processo permanece um desafio, uma vez que o catalisador utilizado
apresenta custo elevado para aplicação industrial (Y, 2024). Resultados
em escala piloto demonstraram rendimentos significativamente inferiores,
da ordem de 82–85% (Z, 2022)."
"""


def _make_citation(study: StudyRecord | None, paper_id: str | None = None) -> str:
    """Build an ``(Author, Year)`` citation from study metadata.

    Falls back to the PMID (filename) formatted as a readable name.
    """
    if study:
        author = study.authors if study.authors else study.pmid.replace('_', ' ')
        if study.year:
            return f"({author}, {study.year})"
        return f"({author})"
    if paper_id:
        return f"(Study {paper_id})"
    return "(Study unknown)"


def _call_llm_with_optional_model(
    prompt: str,
    cfg: Dict[str, Any],
    model_override: str | None = None,
) -> str:
    """Call LLM, optionally overriding model for this single request."""
    model_name = (model_override or "").strip()
    if not model_name:
        return call_llm(prompt, cfg)
    cfg_local = copy.deepcopy(cfg)
    cfg_local.setdefault("llm", {})["model"] = model_name
    return call_llm(prompt, cfg_local)

_SECTION_NO_EVIDENCE_PROMPT = """\
You are a specialist scientific review writer.  Write a SHORT section
for a systematic literature review.

### Section
Chapter: {parent}
Section: {folder}
Content guidance: {prompt}

### Instructions
- There was no direct evidence found in the analysed articles for this
  specific section after strict scope/routing filters.
- Write 1–2 paragraphs providing general context based on the content
  guidance above.
- Clearly state that the reviewed literature did not provide direct
  evidence for this topic (or was largely out-of-scope) and that
  further targeted research is recommended.
- Write in {language_name}.
"""


_THESIS_PROMPT = """\
You are an academic thesis formulator for a systematic literature review.

### Chapter: {chapter}
### Research questions for this chapter:
{research_questions}

### Task
Based on the research questions above, formulate a concise THESIS
STATEMENT (1–2 sentences) that captures the central argument this
chapter should develop.  The thesis must:
- Be falsifiable or debatable (not a trivial fact)
- Guide the narrative toward a specific conclusion
- Acknowledge complexity (e.g. "although X, evidence suggests Y")

Write the thesis in {language_name}.  Return ONLY the thesis statement.
"""


def _generate_chapter_thesis(
    chapter: str,
    research_questions: List[str],
    cfg: Dict[str, Any],
) -> str:
    """Generate a thesis statement for a chapter using the LLM."""
    if not research_questions:
        return "(no specific thesis — write based on evidence)"

    lang = cfg.get("review_writer", {}).get("language", "pt")
    language_name = _LANG_NAMES.get(lang, _LANG_NAMES["pt"])

    rq_text = "\n".join(f"- {q}" for q in research_questions)
    prompt = _THESIS_PROMPT.format(
        chapter=chapter,
        research_questions=rq_text,
        language_name=language_name,
    )

    try:
        thesis = call_llm(prompt, cfg).strip()
        logger.info("  Thesis for '%s': %s", chapter, thesis[:120])
        return thesis
    except Exception as exc:
        logger.warning("Thesis generation failed for '%s': %s", chapter, exc)
        return "(no specific thesis — write based on evidence)"


_EVIDENCE_SUMMARY_PROMPT = """\
You are an academic research assistant and relevance judge.
Evaluate whether this excerpt is relevant to the target section scope.

### Target section
Chapter: {parent}
Section: {folder}
Section guidance: {section_prompt}
Required anchors (at least one expected): {required_any_terms}
Forbidden signals (if dominant, mark irrelevant): {forbidden_terms}

### Excerpt
{excerpt}

### Citation
{citation}

### Output format
Return ONLY valid JSON:
{{
  "is_relevant": true,
  "confidence": 0.0,
  "why": "short reason",
  "claim": "main claim in one sentence",
  "method": "methodology or Not specified",
  "findings": "key finding in one sentence"
}}

### Rules
- confidence must be a number between 0 and 1.
- Set is_relevant=false when excerpt is mostly outside the target scope.
- If forbidden signals dominate, set is_relevant=false.
"""


_POLISH_PROMPT = """\
You are an academic editor.  Below is a draft section of a systematic
literature review.  Polish it according to the rules.

### Draft
{draft_text}

### Polishing rules
1. Improve coherence: ensure each paragraph flows logically from the
   previous one.
2. Eliminate redundant statements that repeat the same idea.
3. Strengthen topic sentences so each paragraph has a clear focus.
4. Ensure the subject of every sentence is the phenomenon or finding,
   NOT the author name.  Authors appear only in parenthetical
   citations.
5. Vary sentence structure to avoid monotony.
6. CRITICAL ANALYSIS: Review every claim in the text.  If a result is
   stated without qualification (e.g. "achieved 98% yield"), add
   qualifying context such as experimental conditions, scale
   limitations, or contradictions from other cited studies.
   Use hedging language: "embora", "sob condições controladas",
   "resultados preliminares sugerem", "a generalização requer cautela".
7. Where two citations present conflicting results, ensure the text
   explicitly discusses possible reasons for the discrepancy.
8. ELIMINATE REDUNDANCIES: If the same concept or finding is stated
   more than once, merge the repetitions into a single, comprehensive
   statement.  Remove duplicate citations within the same parenthetical
   group (e.g. "(X, 2023; X, 2023)" → "(X, 2023)").
9. REPLACE GENERIC CONNECTORS: Do NOT start any paragraph with "Além
   disso", "No entanto", "Adicionalmente", or similar filler phrases.
   Replace them with substantive topic sentences that preview the
   paragraph's argument and link it to the previous paragraph.
10. Ensure ALL text is in {language_name}.  Remove or translate any
    stray English phrases or sentences.
11. Normalise terminology: use consistent spelling throughout (e.g.
    "catálise" not "catalise", "transesterificação" consistently).
12. Preserve ALL citations exactly as given — do NOT add, remove, or
    modify any (Author, Year) reference.
13. Do NOT add a heading or title.
14. Maintain the same approximate length.

Return the polished section text only, nothing else.
"""


def _format_evidence_from_tags(
    relevant: List[ChunkTag],
    chunks_by_id: Dict[str, Chunk],
    studies_by_pmid: Dict[str, StudyRecord],
    profiles_by_paper: Dict[str, Dict[str, Any]] | None = None,
) -> tuple[str, int]:
    """Render selected evidence tags into the writer evidence package."""
    if not relevant:
        return "", 0

    lines: List[str] = []
    seen_chunks = set()
    for tag in relevant:
        if tag.chunk_id in seen_chunks:
            continue
        seen_chunks.add(tag.chunk_id)
        chunk = chunks_by_id.get(tag.chunk_id)
        if not chunk:
            continue
        study = studies_by_pmid.get(chunk.study_pmid)
        citation = _make_citation(study, chunk.paper_id or chunk.study_pmid)
        profile_note = ""
        if profiles_by_paper:
            prof = profiles_by_paper.get(chunk.paper_id or chunk.study_pmid, {})
            domains = prof.get("domains", [])[:3] if isinstance(prof, dict) else []
            regimes = prof.get("phase_regimes", [])[:2] if isinstance(prof, dict) else []
            bits = []
            if domains:
                bits.append("domains=" + ",".join(domains))
            if regimes:
                bits.append("phase=" + ",".join(regimes))
            if bits:
                profile_note = " [" + " | ".join(bits) + "]"
        lines.append(
            f"[{citation}] [chunk:{chunk.chunk_id}] [paper:{chunk.paper_id or chunk.study_pmid}] "
            f"(relevance: {tag.similarity:.2f}, eligibility: {tag.eligibility_score:.2f}){profile_note}\n"
            f"{chunk.text}\n"
        )

    return "\n---\n".join(lines), len(lines)


def _gather_evidence(
    folder: str,
    parent: str,
    all_tags: List[ChunkTag],
    chunks_by_id: Dict[str, Chunk],
    studies_by_pmid: Dict[str, StudyRecord],
    top_k: int = 10,
    min_eligibility: float = 0.2,
    taxonomy_entry: Dict[str, Any] | None = None,
    profiles_by_paper: Dict[str, Dict[str, Any]] | None = None,
    high_flag_related_ids: Set[str] | None = None,
) -> tuple[str, int]:
    """Collect the top-k most relevant chunks for a given section.

    Returns ``(evidence_text, n_chunks)``.
    """
    relevant, _ = filter_and_rank_tags(
        tags=all_tags,
        chunks_by_id=chunks_by_id,
        folder=folder,
        parent=parent,
        top_k=top_k,
        min_eligibility=min_eligibility,
        taxonomy_entry=taxonomy_entry,
        profiles_by_paper=profiles_by_paper,
        high_flag_related_ids=high_flag_related_ids,
    )
    return _format_evidence_from_tags(relevant, chunks_by_id, studies_by_pmid, profiles_by_paper)


def _pre_summarize_evidence(
    evidence_text: str,
    chunks_by_id: Dict[str, Chunk],
    studies_by_pmid: Dict[str, StudyRecord],
    all_tags: List[ChunkTag],
    folder: str,
    parent: str,
    top_k: int,
    cfg: Dict[str, Any],
    min_eligibility: float = 0.2,
    taxonomy_entry: Dict[str, Any] | None = None,
    profiles_by_paper: Dict[str, Dict[str, Any]] | None = None,
    high_flag_related_ids: Set[str] | None = None,
    selected_tags: List[ChunkTag] | None = None,
) -> tuple[str, int, Dict[str, int]]:
    """Summarize each evidence chunk into a structured brief.

    Returns:
      - formatted evidence text with structured summaries
      - number of kept chunks after relevance check
      - stats dict (kept/dropped/parse_errors)
    """
    taxonomy_entry = taxonomy_entry or {}
    writer_cfg = cfg.get("review_writer", {})
    required_any_terms = [str(t) for t in taxonomy_entry.get("required_any_terms", []) if str(t).strip()]
    forbidden_terms = [str(t) for t in taxonomy_entry.get("forbidden_terms", []) if str(t).strip()]
    drop_parse_failures = bool(writer_cfg.get("drop_presummary_parse_failures", True))
    min_confidence = float(writer_cfg.get("presummary_min_confidence", 0.6) or 0.0)
    relevance_judge_model = str(writer_cfg.get("relevance_judge_model", "") or "").strip()

    def _parse_summary_json(raw_text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
        candidates = [cleaned]
        obj_match = re.search(r"\{[\s\S]*\}", cleaned)
        if obj_match:
            candidates.append(obj_match.group(0))

        for cand in candidates:
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                sanitized = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cand)
                try:
                    parsed = json.loads(sanitized)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue
            except Exception:
                continue
        return {}

    if selected_tags is None:
        relevant, _ = filter_and_rank_tags(
            tags=all_tags,
            chunks_by_id=chunks_by_id,
            folder=folder,
            parent=parent,
            top_k=top_k,
            min_eligibility=min_eligibility,
            taxonomy_entry=taxonomy_entry,
            profiles_by_paper=profiles_by_paper,
            high_flag_related_ids=high_flag_related_ids,
        )
    else:
        relevant = selected_tags[:top_k]

    if not relevant:
        return evidence_text, 0, {
            "kept": 0,
            "dropped_irrelevant": 0,
            "dropped_low_confidence": 0,
            "parse_errors": 0,
        }

    summaries: List[str] = []
    seen = set()
    kept = 0
    dropped_irrelevant = 0
    dropped_low_confidence = 0
    parse_errors = 0
    for tag in relevant:
        if tag.chunk_id in seen:
            continue
        seen.add(tag.chunk_id)
        chunk = chunks_by_id.get(tag.chunk_id)
        if not chunk:
            continue
        study = studies_by_pmid.get(chunk.study_pmid)
        citation = _make_citation(study, chunk.paper_id or chunk.study_pmid)

        try:
            raw = _call_llm_with_optional_model(
                _EVIDENCE_SUMMARY_PROMPT.format(
                    parent=parent,
                    folder=folder,
                    section_prompt=(taxonomy_entry or {}).get("prompt", ""),
                    required_any_terms=", ".join(required_any_terms[:12]) or "none",
                    forbidden_terms=", ".join(forbidden_terms[:20]) or "none",
                    excerpt=chunk.text,
                    citation=f"{citation} [chunk:{chunk.chunk_id}]",
                ),
                cfg,
                relevance_judge_model,
            )
            parsed = _parse_summary_json(raw)
            if not parsed:
                parse_errors += 1
                if not drop_parse_failures:
                    summaries.append(
                        f"[{citation}] [chunk:{chunk.chunk_id}] "
                        f"(relevance: {tag.similarity:.2f}, eligibility: {tag.eligibility_score:.2f})\n"
                        f"{chunk.text}"
                    )
                    kept += 1
                else:
                    dropped_irrelevant += 1
                continue

            if not bool(parsed.get("is_relevant", False)):
                dropped_irrelevant += 1
                continue
            try:
                confidence = float(parsed.get("confidence", 1.0))
            except Exception:
                confidence = 1.0
            confidence = max(0.0, min(1.0, confidence))
            if confidence < min_confidence:
                dropped_low_confidence += 1
                continue

            claim = str(parsed.get("claim", "") or "Not specified").strip()
            method = str(parsed.get("method", "") or "Not specified").strip()
            findings = str(parsed.get("findings", "") or "Not specified").strip()
            why = str(parsed.get("why", "") or "").strip()
            brief = (
                f"- Claim: {claim}\n"
                f"- Method: {method}\n"
                f"- Findings: {findings}\n"
                f"- Relevance: {why or 'Relevant to section scope'}\n"
                f"- Judge confidence: {confidence:.2f}"
            )
            summaries.append(
                f"[{citation}] [chunk:{chunk.chunk_id}] "
                f"(relevance: {tag.similarity:.2f}, eligibility: {tag.eligibility_score:.2f})\n{brief}"
            )
            kept += 1
        except Exception as exc:
            parse_errors += 1
            logger.warning("Pre-summarization failed for %s: %s", citation, exc)
            if not drop_parse_failures:
                summaries.append(
                    f"[{citation}] [chunk:{chunk.chunk_id}] "
                    f"(relevance: {tag.similarity:.2f}, eligibility: {tag.eligibility_score:.2f})\n{chunk.text}"
                )
                kept += 1
            else:
                dropped_irrelevant += 1

    return "\n---\n".join(summaries), kept, {
        "kept": kept,
        "dropped_irrelevant": dropped_irrelevant,
        "dropped_low_confidence": dropped_low_confidence,
        "parse_errors": parse_errors,
    }


def _write_section(
    prompt: str,
    folder: str,
    parent: str,
    evidence: str,
    cfg: Dict[str, Any],
    max_retries: int = 1,
    covered_concepts: str = "(none yet — this is the first section)",
    chapter_thesis: str = "(no specific thesis — write based on evidence)",
) -> str:
    """Call the LLM to write one section, with optional retries."""
    lang = cfg.get("review_writer", {}).get("language", "pt")
    language_name = _LANG_NAMES.get(lang, _LANG_NAMES["pt"])

    if evidence:
        llm_prompt = _SECTION_PROMPT.format(
            parent=parent,
            folder=folder,
            prompt=prompt,
            evidence=evidence,
            language_name=language_name,
            covered_concepts=covered_concepts,
            chapter_thesis=chapter_thesis,
        )
    else:
        llm_prompt = _SECTION_NO_EVIDENCE_PROMPT.format(
            parent=parent,
            folder=folder,
            prompt=prompt,
            language_name=language_name,
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


def _polish_section(
    draft: str,
    cfg: Dict[str, Any],
    max_retries: int = 1,
) -> str:
    """Send a draft section back to the LLM for polishing."""
    lang = cfg.get("review_writer", {}).get("language", "pt")
    language_name = _LANG_NAMES.get(lang, _LANG_NAMES["pt"])

    prompt = _POLISH_PROMPT.format(
        draft_text=draft,
        language_name=language_name,
    )

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return call_llm(prompt, cfg)
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                logger.warning("Polish attempt %d/%d failed: %s — retrying", attempt + 1, max_retries + 1, exc)

    logger.warning("Polish failed after %d attempts: %s — returning draft", max_retries + 1, last_err)
    return draft  # Graceful fallback to draft


def _apply_thermo_sanity_fixes(text: str) -> tuple[str, int]:
    """Fix a short list of common thermodynamic sign mistakes."""
    if not text:
        return text, 0

    parts = re.split(r"(?<=[.!?])\s+", text)
    fixed_parts: List[str] = []
    fixes = 0

    for part in parts:
        low = part.lower()
        updated = part

        # Sublimation is endothermic.
        if ("sublimação" in low or "sublimation" in low) and (
            "exotérm" in low or "exotherm" in low
        ):
            updated = re.sub(r"exot[eé]rmic\w*", "endotérmico", updated, flags=re.IGNORECASE)
            updated = re.sub(r"exothermic", "endothermic", updated, flags=re.IGNORECASE)

        # Deposition is exothermic.
        low_updated = updated.lower()
        if ("deposição" in low_updated or "deposition" in low_updated) and (
            "endotérm" in low_updated or "endotherm" in low_updated
        ):
            updated = re.sub(r"endot[eé]rmic\w*", "exotérmico", updated, flags=re.IGNORECASE)
            updated = re.sub(r"endothermic", "exothermic", updated, flags=re.IGNORECASE)

        if updated != part:
            fixes += 1
        fixed_parts.append(updated)

    return " ".join(fixed_parts), fixes


# ------------------------------------------------------------------ #
#  Concept extraction (for redundancy prevention)                      #
# ------------------------------------------------------------------ #

_CONCEPT_EXTRACT_PROMPT = """\
You are a concept extraction assistant.  From the text below, identify
the 5–10 most important TECHNICAL CONCEPTS that are explained or
defined (not merely mentioned).  Return ONLY a JSON array of strings.

Example output: ["transesterificação", "catálise heterogênea", "biodiesel de segunda geração"]

### Text
{text}
"""


def _extract_concepts(text: str, cfg: dict) -> list[str]:
    """Use LLM to extract key concepts from a section's text."""
    if not text or len(text) < 100:
        return []
    writer_cfg = cfg.get("review_writer", {})
    concept_model = str(writer_cfg.get("concept_extractor_model", "") or "").strip()
    # Use only the first 3000 chars to save context window
    snippet = text[:3000]
    prompt = _CONCEPT_EXTRACT_PROMPT.format(text=snippet)

    def _normalise_list(values: List[Any]) -> List[str]:
        cleaned = []
        seen = set()
        for v in values:
            c = str(v).strip()
            if not c:
                continue
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(c)
        return cleaned[:20]

    def _try_parse_json_array(raw_text: str) -> List[str]:
        cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
        candidates = [cleaned]

        # If model added extra text, try extracting the first JSON array block.
        m = re.search(r"\[[\s\S]*\]", cleaned)
        if m:
            candidates.append(m.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return _normalise_list(parsed)
            except json.JSONDecodeError:
                # Escape stray backslashes to tolerate invalid \escape.
                sanitized = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)
                try:
                    parsed = json.loads(sanitized)
                    if isinstance(parsed, list):
                        return _normalise_list(parsed)
                except Exception:
                    pass
                try:
                    parsed = ast.literal_eval(sanitized)
                    if isinstance(parsed, list):
                        return _normalise_list(parsed)
                except Exception:
                    continue
            except Exception:
                continue
        return []

    try:
        raw = _call_llm_with_optional_model(prompt, cfg, concept_model)
        parsed = _try_parse_json_array(raw)
        if parsed:
            return parsed
        # Last-resort fallback 1: collect quoted strings.
        quoted = re.findall(r'"([^"\n]{2,120})"', raw)
        if quoted:
            return _normalise_list(quoted)
        # Last-resort fallback 2: bullet points or comma-separated list.
        bullets = re.findall(r"(?:^|\n)\s*(?:[-*•]|\d+\.)\s+([^\n]{2,120})", raw)
        if bullets:
            return _normalise_list(bullets)
        if "," in raw:
            parts = [p.strip(" -\t\r\n") for p in raw.split(",")]
            parts = [p for p in parts if 2 <= len(p) <= 120]
            if parts:
                return _normalise_list(parts)
    except Exception as exc:
        logger.debug("Concept extraction failed: %s", exc)
    return []


def _write_single_entry(
    entry: Dict[str, str],
    all_tags: List[ChunkTag],
    chunks_by_id: Dict[str, Chunk],
    studies_by_pmid: Dict[str, StudyRecord],
    cfg: Dict[str, Any],
    top_k_evidence: int,
    max_retries: int,
    min_eligibility: float = 0.2,
    profiles_by_paper: Dict[str, Dict[str, Any]] | None = None,
    high_flag_related_ids: Set[str] | None = None,
    section_synthesis: Dict[str, Any] | None = None,
    concept_registry: ConceptRegistry | None = None,
    table_registry: TableRegistry | None = None,
    chapter_thesis: str = "(no specific thesis — write based on evidence)",
) -> Dict[str, Any]:
    """Write a single section. Used as the unit of work for parallel execution."""
    writer_cfg = cfg.get("review_writer", {})
    use_pre_summarize = writer_cfg.get("evidence_pre_summarize", False)
    use_two_pass = writer_cfg.get("two_pass_writing", False)
    min_required_evidence = int(writer_cfg.get("min_required_evidence", 1) or 1)

    # Build covered concepts string for prompt injection
    covered_concepts_str = "(none yet — this is the first section)"
    if concept_registry and concept_registry.size() > 0:
        concepts = concept_registry.get_covered_concepts()
        covered_concepts_str = "; ".join(concepts[:50])  # cap to avoid prompt overflow

    selected_tags, filter_stats = filter_and_rank_tags(
        tags=all_tags,
        chunks_by_id=chunks_by_id,
        folder=entry["folder"],
        parent=entry["parent"],
        top_k=top_k_evidence,
        min_eligibility=min_eligibility,
        taxonomy_entry=entry,
        profiles_by_paper=profiles_by_paper,
        high_flag_related_ids=high_flag_related_ids,
    )
    evidence, n_evidence = _format_evidence_from_tags(
        selected_tags, chunks_by_id, studies_by_pmid, profiles_by_paper
    )

    # Append synthesis-level anchors so the writer can preserve grounding.
    synthesis_payload: Dict[str, Any] = {}
    if section_synthesis:
        if hasattr(section_synthesis, "model_dump"):
            try:
                synthesis_payload = section_synthesis.model_dump()
            except Exception:
                synthesis_payload = {}
        elif isinstance(section_synthesis, dict):
            synthesis_payload = section_synthesis

    if synthesis_payload and evidence:
        anchor_lines: List[str] = []
        for cp in synthesis_payload.get("consensus_points", [])[:6]:
            if not isinstance(cp, dict):
                continue
            ids = cp.get("evidence_chunk_ids", [])
            stmt = cp.get("statement", "")
            if stmt and ids:
                anchor_lines.append(f"- {stmt} [chunks:{', '.join(ids)}]")
        for c in synthesis_payload.get("contradictions", [])[:4]:
            if not isinstance(c, dict):
                continue
            ids = c.get("evidence_chunk_ids", [])
            point = c.get("point", "")
            if point and ids:
                anchor_lines.append(f"- Contradiction: {point} [chunks:{', '.join(ids)}]")
        if anchor_lines:
            evidence = (
                "### Anchored synthesis hints (must remain grounded)\n"
                + "\n".join(anchor_lines)
                + "\n\n"
                + evidence
            )

    # --- Optional: pre-summarize evidence into structured briefs ---
    if use_pre_summarize and evidence:
        logger.info(
            "  Pre-summarizing evidence for: %s / %s (%d chunks)",
            entry["parent"], entry["folder"], n_evidence,
        )
        evidence, n_after_presummary, presummary_stats = _pre_summarize_evidence(
            evidence,
            chunks_by_id,
            studies_by_pmid,
            all_tags,
            entry["folder"],
            entry["parent"],
            top_k_evidence,
            cfg,
            min_eligibility=min_eligibility,
            taxonomy_entry=entry,
            profiles_by_paper=profiles_by_paper,
            high_flag_related_ids=high_flag_related_ids,
            selected_tags=selected_tags,
        )
        n_evidence = n_after_presummary
        filter_stats["presummary_kept"] = presummary_stats.get("kept", n_after_presummary)
        filter_stats["presummary_dropped_irrelevant"] = presummary_stats.get("dropped_irrelevant", 0)
        filter_stats["presummary_dropped_low_confidence"] = presummary_stats.get("dropped_low_confidence", 0)
        filter_stats["presummary_parse_errors"] = presummary_stats.get("parse_errors", 0)

    if n_evidence < min_required_evidence:
        logger.warning(
            "  Evidence below minimum for %s / %s: %d < %d (forcing insufficient-evidence mode)",
            entry["parent"], entry["folder"], n_evidence, min_required_evidence,
        )
        evidence = ""

    logger.info(
        "  Writing: %s / %s (%d evidence chunks after hard gate, ~%d chars, purity=%.2f)",
        entry["parent"], entry["folder"], n_evidence, len(evidence), filter_stats.get("domain_purity", 1.0),
    )

    # --- Pass 1: Draft ---
    content = _write_section(
        entry["prompt"], entry["folder"], entry["parent"],
        evidence, cfg, max_retries,
        covered_concepts=covered_concepts_str,
        chapter_thesis=chapter_thesis,
    )

    # --- Optional Pass 2: Polish ---
    if use_two_pass and content and not content.startswith("*["):
        logger.info(
            "  Polishing: %s / %s (~%d chars)",
            entry["parent"], entry["folder"], len(content),
        )
        content = _polish_section(content, cfg, max_retries)

    # Deterministic thermo sanity patch for known sign mistakes.
    if content and not content.startswith("*["):
        content, thermo_fixes = _apply_thermo_sanity_fixes(content)
        if thermo_fixes:
            filter_stats["thermo_fixes_applied"] = thermo_fixes

    # --- Register new concepts for subsequent sections ---
    if concept_registry and content and not content.startswith("*[") and n_evidence > 0:
        new_concepts = _extract_concepts(content, cfg)
        section_label = f"{entry['parent']} / {entry['folder']}"
        concept_registry.register_many(new_concepts, section_label)
        logger.debug(
            "  Registered %d concepts from %s / %s",
            len(new_concepts), entry["parent"], entry["folder"],
        )

    # --- Table detection (Épico 6) ---
    if table_registry is not None and content and not content.startswith("*[") and n_evidence > 0:
        table_spec = detect_table_opportunity(
            entry["folder"],
            entry["parent"],
            all_tags,
            chunks_by_id,
            studies_by_pmid,
            cfg,
            taxonomy_entry=entry,
            profiles_by_paper=profiles_by_paper,
            high_flag_related_ids=high_flag_related_ids,
        )
        if table_spec:
            table_md = table_registry.register(table_spec)
            try:
                from validators import sanitize_table_text
                table_md = sanitize_table_text(table_md)
            except Exception:
                pass
            content = content + "\n\n" + table_md
            logger.info(
                "  Table generated for %s / %s: %s",
                entry["parent"], entry["folder"], table_spec.table_id,
            )

    return {
        "parent": entry["parent"],
        "folder": entry["folder"],
        "prompt": entry["prompt"],
        "n_evidence": n_evidence,
        "n_evidence_after_filter": filter_stats.get("evidence_count_after_filter", n_evidence),
        "filter_stats": filter_stats,
        "allowed_domains": entry.get("allowed_domains", []),
        "required_phase_regimes": entry.get("required_phase_regimes", []),
        "required_any_terms": entry.get("required_any_terms", []),
        "forbidden_terms": entry.get("forbidden_terms", []),
        "min_similarity": entry.get("min_similarity", 0.0),
        "content": content,
    }


# ------------------------------------------------------------------ #
#  Document assembly                                                   #
# ------------------------------------------------------------------ #

def _assemble_markdown(
    sections: List[Dict[str, Any]],
    topic: str,
    cfg: Dict[str, Any] | None = None,
    synthesis_maps: Dict | None = None,
) -> str:
    """Combine sections into a single Markdown document.

    If cfg and synthesis_maps are provided, generates:
    - Executive summary (LLM-based)
    - Limitations of this review
    - Prioritized research agenda from knowledge gaps
    """
    lines: List[str] = [
        f"# {topic}\n",
        "---\n",
    ]

    # --- Executive Summary (Épico 5) ---
    if cfg:
        body_preview = "\n".join(
            f"## {sec['parent']}\n### {sec['folder']}\n{sec['content'][:300]}..."
            for sec in sections[:8]
        )
        exec_summary = _generate_executive_summary(topic, body_preview, cfg)
        if exec_summary:
            lines.append("\n## Resumo Executivo\n")
            lines.append(exec_summary)
            lines.append("\n")

    current_parent = None
    for sec in sections:
        # New chapter heading
        if sec["parent"] != current_parent:
            current_parent = sec["parent"]
            lines.append(f"\n## {current_parent}\n")

        lines.append(f"\n### {sec['folder']}\n")
        lines.append(sec["content"])
        lines.append("\n")

    # --- Limitations section (Épico 5) ---
    limitations_text = _generate_limitations(topic, cfg or {})
    if limitations_text:
        lines.append("\n## Limita\u00e7\u00f5es desta Revis\u00e3o\n")
        lines.append(limitations_text)
        lines.append("\n")

    # --- Research Agenda from knowledge gaps (Épico 5) ---
    if synthesis_maps:
        agenda = _generate_research_agenda(synthesis_maps)
        if agenda:
            lines.append("\n## Agenda de Pesquisa Priorit\u00e1ria\n")
            lines.append(agenda)
            lines.append("\n")

    return "\n".join(lines)


_EXECUTIVE_SUMMARY_PROMPT = """\
You are an academic editor.  Write a concise EXECUTIVE SUMMARY
(3–4 paragraphs) for a systematic literature review titled:

\"{topic}\"

### Preview of chapter contents
{body_preview}

### Instructions
- Summarize the scope, key findings, and practical implications.
- Highlight areas of consensus and key contradictions found.
- Write in {language_name}.
- Return ONLY the summary text.
"""


def _generate_executive_summary(
    topic: str,
    body_preview: str,
    cfg: Dict[str, Any],
) -> str:
    """Generate an executive summary using the LLM."""
    lang = cfg.get("review_writer", {}).get("language", "pt")
    language_name = _LANG_NAMES.get(lang, _LANG_NAMES["pt"])
    prompt = _EXECUTIVE_SUMMARY_PROMPT.format(
        topic=topic, body_preview=body_preview, language_name=language_name,
    )
    try:
        return call_llm(prompt, cfg).strip()
    except Exception as exc:
        logger.warning("Executive summary generation failed: %s", exc)
        return ""


def _generate_limitations(topic: str, cfg: Dict[str, Any]) -> str:
    """Generate a standard limitations section."""
    lang = cfg.get("review_writer", {}).get("language", "pt")
    if lang == "pt":
        return (
            "Esta revis\u00e3o sistem\u00e1tica apresenta limita\u00e7\u00f5es inerentes \u00e0 "
            "metodologia adotada. Primeiramente, a sele\u00e7\u00e3o de bases de dados "
            "e termos de busca pode ter exclu\u00eddo estudos relevantes publicados "
            "em peri\u00f3dicos n\u00e3o indexados ou em idiomas n\u00e3o cobertos. "
            "Em segundo lugar, a heterogeneidade das metodologias empregadas "
            "pelos estudos prim\u00e1rios dificulta a compara\u00e7\u00e3o direta de "
            "resultados quantitativos. Adicionalmente, o vi\u00e9s de publica\u00e7\u00e3o "
            "pode superestimar efeitos positivos, uma vez que estudos com "
            "resultados negativos ou inconclusivos tendem a ser menos publicados. "
            "Por fim, a an\u00e1lise cr\u00edtica apresentada reflete o estado da "
            "literatura at\u00e9 a data de realiza\u00e7\u00e3o desta revis\u00e3o, n\u00e3o incorporando "
            "publica\u00e7\u00f5es posteriores."
        )
    return (
        "This systematic review has inherent limitations. First, the choice "
        "of databases and search terms may have excluded relevant studies. "
        "Second, heterogeneity in primary study methodologies limits direct "
        "quantitative comparison. Additionally, publication bias may "
        "overestimate positive effects. Finally, the analysis reflects the "
        "literature up to the date of this review."
    )


def _generate_research_agenda(synthesis_maps: Dict) -> str:
    """Generate a prioritized research agenda from knowledge gaps."""
    all_gaps = []
    for theme, smap in synthesis_maps.items():
        for gap in getattr(smap, 'knowledge_gaps', []):
            all_gaps.append({
                "theme": theme,
                "description": getattr(gap, 'description', str(gap)),
                "priority": getattr(gap, 'priority', 'medium'),
                "suggested_approach": getattr(gap, 'suggested_approach', ''),
            })

    if not all_gaps:
        return ""

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    all_gaps.sort(key=lambda g: priority_order.get(g["priority"], 1))

    lines = [
        "Com base nas lacunas de conhecimento identificadas nesta revis\u00e3o, "
        "prop\u00f5e-se a seguinte agenda de pesquisa, organizada por prioridade:\n"
    ]

    for i, gap in enumerate(all_gaps, 1):
        priority_label = {
            "high": "\ud83d\udfe5 Alta",
            "medium": "\ud83d\udfe8 M\u00e9dia",
            "low": "\ud83d\udfe9 Baixa",
        }.get(gap["priority"], "\ud83d\udfe8 M\u00e9dia")

        lines.append(f"**{i}. {gap['description']}** ({priority_label})")
        if gap["suggested_approach"]:
            lines.append(f"   - *Abordagem sugerida*: {gap['suggested_approach']}")
        lines.append(f"   - *Tema*: {gap['theme']}")
        lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def write_review(
    taxonomy_entries: List[Dict[str, str]],
    all_chunks: List[Chunk],
    all_tags: List[ChunkTag],
    studies: List[StudyRecord],
    topic: str,
    cfg: Dict[str, Any],
    synthesis_maps: Dict | None = None,
) -> str:
    """Write the full review document.

    Returns the path to the generated Markdown file.
    """
    writer_cfg = cfg.get("review_writer", {})
    routing_cfg = cfg.get("routing", {})
    top_k_evidence = writer_cfg.get("top_k_evidence", 10)
    parallel_workers = writer_cfg.get("parallel_workers", 1)
    max_retries = writer_cfg.get("max_retries", 1)
    min_eligibility = routing_cfg.get("min_eligibility_score", 0.2)

    profiles_by_paper: Dict[str, Dict[str, Any]] = {}
    try:
        loaded_profiles = load_json("data/processed/paper_profiles.json")
        if isinstance(loaded_profiles, dict):
            profiles_by_paper = loaded_profiles
    except Exception:
        profiles_by_paper = {}

    high_flag_related_ids: Set[str] = set()
    if cfg.get("validators", {}).get("exclude_high_flagged_evidence", True):
        try:
            validation_report = load_json("data/processed/validation_report.json")
            high_flag_related_ids = build_high_flag_related_ids(
                validation_report if isinstance(validation_report, dict) else {}
            )
        except Exception:
            high_flag_related_ids = set()

    # Concept registry for cross-section deduplication (Épico 4)
    concept_registry = ConceptRegistry()
    # Table registry for comparative tables (Épico 6)
    table_registry = TableRegistry()

    # Build lookups
    chunks_by_id = {c.chunk_id: c for c in all_chunks}
    studies_by_pmid = {s.pmid: s for s in studies}

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

    # ---- Generate chapter theses (Épico 3) ----------------------- #
    chapter_theses: Dict[str, str] = {}
    chapter_rqs: Dict[str, List[str]] = defaultdict(list)
    for entry in taxonomy_entries:
        rqs = entry.get("research_questions", [])
        if rqs:
            chapter_rqs[entry["parent"]].extend(rqs)

    for parent, rqs in chapter_rqs.items():
        logger.info("  Generating thesis for chapter: %s", parent)
        chapter_theses[parent] = _generate_chapter_thesis(parent, rqs, cfg)

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
                theme_key = f"{entry['parent']} / {entry['folder']}"
                future = executor.submit(
                    _write_single_entry,
                    entry=entry,
                    all_tags=all_tags,
                    chunks_by_id=chunks_by_id,
                    studies_by_pmid=studies_by_pmid,
                    cfg=cfg,
                    top_k_evidence=top_k_evidence,
                    max_retries=max_retries,
                    min_eligibility=min_eligibility,
                    profiles_by_paper=profiles_by_paper,
                    high_flag_related_ids=high_flag_related_ids,
                    section_synthesis=(synthesis_maps or {}).get(theme_key, {}),
                    concept_registry=concept_registry,
                    table_registry=table_registry,
                    chapter_thesis=chapter_theses.get(
                        entry["parent"], "(no specific thesis — write based on evidence)"
                    ),
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
            theme_key = f"{entry['parent']} / {entry['folder']}"
            result = _write_single_entry(
                entry=entry,
                all_tags=all_tags,
                chunks_by_id=chunks_by_id,
                studies_by_pmid=studies_by_pmid,
                cfg=cfg,
                top_k_evidence=top_k_evidence,
                max_retries=max_retries,
                min_eligibility=min_eligibility,
                profiles_by_paper=profiles_by_paper,
                high_flag_related_ids=high_flag_related_ids,
                section_synthesis=(synthesis_maps or {}).get(theme_key, {}),
                concept_registry=concept_registry,
                table_registry=table_registry,
                chapter_thesis=chapter_theses.get(
                    entry["parent"], "(no specific thesis — write based on evidence)"
                ),
            )
            sections.append(result)

    # Save individual sections for later editing
    save_json(sections, "data/results/review_sections.json")

    # Deterministic quality report (grounding/contamination/physics/table integrity)
    try:
        from quality_eval import build_quality_report
        validation_report = load_json("data/processed/validation_report.json")
        build_quality_report(
            sections=sections,
            taxonomy_entries=taxonomy_entries,
            validation_report=validation_report if isinstance(validation_report, dict) else {},
            out_path="data/processed/quality_report.json",
        )
    except Exception as exc:
        logger.warning("Could not generate quality report: %s", exc)

    # Persist concept registry
    concept_registry.to_json(str(_resolve("data/results/concept_registry.json")))
    logger.info("Concept registry: %d concepts tracked", concept_registry.size())
    logger.info("Table registry: %d tables generated", table_registry.size())

    # ---- Stage 8: Assemble document -------------------------------- #
    logger.info("Assembling final document")
    document = _assemble_markdown(sections, topic, cfg, synthesis_maps)

    out_path = _resolve("data/results/systematic_review.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sanitize document to thoroughly remove any surrogate escapes or invalid unicode
    sanitized_document = document.encode("utf-8", "replace").decode("utf-8")
    out_path.write_text(sanitized_document, encoding="utf-8")

    logger.info("Review saved → %s", out_path)
    return str(out_path)
