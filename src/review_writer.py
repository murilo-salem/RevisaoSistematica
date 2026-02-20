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

import json
import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from tqdm import tqdm

from utils import (
    Chunk,
    ChunkTag,
    StudyRecord,
    call_llm,
    save_json,
    _resolve,
)
from concept_registry import ConceptRegistry
from table_generator import TableRegistry, detect_table_opportunity

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


def _make_citation(study: StudyRecord) -> str:
    """Build an ``(Author, Year)`` citation from study metadata.

    Falls back to the PMID (filename) formatted as a readable name.
    """
    author = study.authors if study.authors else study.pmid.replace('_', ' ')
    if study.year:
        return f"({author}, {study.year})"
    return f"({author})"

_SECTION_NO_EVIDENCE_PROMPT = """\
You are a specialist scientific review writer.  Write a SHORT section
for a systematic literature review.

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
You are an academic research assistant.  Summarise the following
excerpt from a published article into a structured evidence brief.

### Excerpt
{excerpt}

### Citation
{citation}

### Output format (return ONLY this, no preamble)
- **Claim**: [main claim or thesis in 1 sentence]
- **Method**: [methodology in 1 sentence, or "Not specified"]
- **Finding**: [key quantitative or qualitative finding in 1 sentence]
- **Limitation**: [limitation acknowledged by the author, or "Not stated"]
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


def _gather_evidence(
    folder: str,
    parent: str,
    all_tags: List[ChunkTag],
    chunks_by_id: Dict[str, Chunk],
    studies_by_pmid: Dict[str, StudyRecord],
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
            study = studies_by_pmid.get(chunk.study_pmid)
            citation = _make_citation(study) if study else f"(Study {chunk.study_pmid})"
            lines.append(
                f"[{citation}] (relevance: {tag.similarity:.2f})\n"
                f"{chunk.text}\n"
            )

    return "\n---\n".join(lines), len(lines)


def _pre_summarize_evidence(
    evidence_text: str,
    chunks_by_id: Dict[str, Chunk],
    studies_by_pmid: Dict[str, StudyRecord],
    all_tags: List[ChunkTag],
    folder: str,
    parent: str,
    top_k: int,
    cfg: Dict[str, Any],
) -> str:
    """Summarize each evidence chunk into a structured brief.

    Returns formatted evidence text with structured summaries
    replacing raw chunks.  Falls back to the original evidence
    on any error.
    """
    relevant = [
        t for t in all_tags
        if t.folder == folder and t.parent == parent
    ]
    relevant.sort(key=lambda t: t.similarity, reverse=True)
    relevant = relevant[:top_k]

    if not relevant:
        return evidence_text

    summaries: List[str] = []
    seen = set()
    for tag in relevant:
        if tag.chunk_id in seen:
            continue
        seen.add(tag.chunk_id)
        chunk = chunks_by_id.get(tag.chunk_id)
        if not chunk:
            continue
        study = studies_by_pmid.get(chunk.study_pmid)
        citation = _make_citation(study) if study else f"(Study {chunk.study_pmid})"

        try:
            brief = call_llm(
                _EVIDENCE_SUMMARY_PROMPT.format(
                    excerpt=chunk.text,
                    citation=citation,
                ),
                cfg,
            )
            summaries.append(f"[{citation}] (relevance: {tag.similarity:.2f})\n{brief}")
        except Exception as exc:
            logger.warning("Pre-summarization failed for %s: %s — using raw chunk", citation, exc)
            summaries.append(
                f"[{citation}] (relevance: {tag.similarity:.2f})\n{chunk.text}"
            )

    return "\n---\n".join(summaries)


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
    # Use only the first 3000 chars to save context window
    snippet = text[:3000]
    prompt = _CONCEPT_EXTRACT_PROMPT.format(text=snippet)
    try:
        raw = call_llm(prompt, cfg)
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        concepts = json.loads(cleaned)
        if isinstance(concepts, list):
            return [str(c).strip() for c in concepts if c]
    except (json.JSONDecodeError, Exception) as exc:
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
    concept_registry: ConceptRegistry | None = None,
    table_registry: TableRegistry | None = None,
    chapter_thesis: str = "(no specific thesis — write based on evidence)",
) -> Dict[str, Any]:
    """Write a single section. Used as the unit of work for parallel execution."""
    writer_cfg = cfg.get("review_writer", {})
    use_pre_summarize = writer_cfg.get("evidence_pre_summarize", False)
    use_two_pass = writer_cfg.get("two_pass_writing", False)

    # Build covered concepts string for prompt injection
    covered_concepts_str = "(none yet — this is the first section)"
    if concept_registry and concept_registry.size() > 0:
        concepts = concept_registry.get_covered_concepts()
        covered_concepts_str = "; ".join(concepts[:50])  # cap to avoid prompt overflow

    evidence, n_evidence = _gather_evidence(
        entry["folder"], entry["parent"],
        all_tags, chunks_by_id, studies_by_pmid, top_k_evidence,
    )

    # --- Optional: pre-summarize evidence into structured briefs ---
    if use_pre_summarize and evidence:
        logger.info(
            "  Pre-summarizing evidence for: %s / %s (%d chunks)",
            entry["parent"], entry["folder"], n_evidence,
        )
        evidence = _pre_summarize_evidence(
            evidence, chunks_by_id, studies_by_pmid,
            all_tags, entry["folder"], entry["parent"],
            top_k_evidence, cfg,
        )

    logger.info(
        "  Writing: %s / %s (%d evidence chunks, ~%d chars)",
        entry["parent"], entry["folder"], n_evidence, len(evidence),
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

    # --- Register new concepts for subsequent sections ---
    if concept_registry and content and not content.startswith("*["):
        new_concepts = _extract_concepts(content, cfg)
        section_label = f"{entry['parent']} / {entry['folder']}"
        concept_registry.register_many(new_concepts, section_label)
        logger.debug(
            "  Registered %d concepts from %s / %s",
            len(new_concepts), entry["parent"], entry["folder"],
        )

    # --- Table detection (Épico 6) ---
    if table_registry is not None and content and not content.startswith("*["):
        table_spec = detect_table_opportunity(
            entry["folder"], entry["parent"],
            all_tags, chunks_by_id, studies_by_pmid, cfg,
        )
        if table_spec:
            table_md = table_registry.register(table_spec)
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
    top_k_evidence = writer_cfg.get("top_k_evidence", 10)
    parallel_workers = writer_cfg.get("parallel_workers", 1)
    max_retries = writer_cfg.get("max_retries", 1)

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
                future = executor.submit(
                    _write_single_entry,
                    entry, all_tags, chunks_by_id, studies_by_pmid, cfg,
                    top_k_evidence, max_retries, concept_registry, table_registry,
                    chapter_theses.get(entry["parent"], "(no specific thesis — write based on evidence)"),
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
                entry, all_tags, chunks_by_id, studies_by_pmid, cfg,
                top_k_evidence, max_retries, concept_registry, table_registry,
                chapter_theses.get(entry["parent"], "(no specific thesis — write based on evidence)"),
            )
            sections.append(result)

    # Save individual sections for later editing
    save_json(sections, "data/results/review_sections.json")

    # Persist concept registry
    concept_registry.to_json(str(_resolve("data/results/concept_registry.json")))
    logger.info("Concept registry: %d concepts tracked", concept_registry.size())
    logger.info("Table registry: %d tables generated", table_registry.size())

    # ---- Stage 8: Assemble document -------------------------------- #
    logger.info("Assembling final document")
    document = _assemble_markdown(sections, topic, cfg, synthesis_maps)

    out_path = _resolve("data/results/systematic_review.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")

    logger.info("Review saved → %s", out_path)
    return str(out_path)
