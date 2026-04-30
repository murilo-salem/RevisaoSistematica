"""
manuscript.py — Generate the systematic-review manuscript in LaTeX.

Each section (Introduction, Methods, Results, Discussion) is produced
by a controlled LLM prompt that receives concrete data so the model
does not need to hallucinate facts.  The output is rendered through a
Jinja2 LaTeX template.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader

from utils import (
    ExtractionResult,
    PICOModel,
    RiskOfBiasResult,
    call_llm,
    save_json,
    _resolve,
)

logger = logging.getLogger("systematic_review.manuscript")


# ------------------------------------------------------------------ #
#  Section generators                                                  #
# ------------------------------------------------------------------ #

def _gen_introduction(pico: PICOModel, cfg: Dict[str, Any]) -> str:
    prompt = f"""\
Write the Introduction section of a systematic review manuscript.

Topic: {pico.topic}
PICO:
  Population: {pico.population}
  Intervention: {pico.intervention}
  Comparison: {pico.comparison}
  Outcome: {pico.outcome}

Requirements:
- State the clinical/scientific rationale.
- Summarise existing knowledge gaps.
- State the objective of this systematic review clearly.
- Write in formal academic English.
- Do NOT invent references.  Use placeholders like [REF] where a citation would go.
- 3–5 paragraphs.
"""
    return call_llm(prompt, cfg)


def _gen_methods(
    pico: PICOModel,
    n_retrieved: int,
    n_after_dedup: int,
    n_included: int,
    cfg: Dict[str, Any],
) -> str:
    prompt = f"""\
Write the Methods section of a systematic review manuscript.

Search strategy:
  Boolean query: {pico.query}
  Database: PubMed
  Records retrieved: {n_retrieved}
  After deduplication: {n_after_dedup}
  After screening: {n_included}

Screening approach:
  Automated screening using a large-language model
  Inclusion threshold: {cfg['screening']['threshold_include']}
  Exclusion threshold: {cfg['screening']['threshold_exclude']}
  Ambiguous cases flagged for human review

Data extraction:
  Structured extraction of study design, sample size, intervention,
  comparators, outcomes, and effect sizes.

Risk-of-bias assessment:
  Five domains evaluated: selection, performance, detection, attrition,
  reporting.

Requirements:
- Follow PRISMA 2020 reporting guidelines.
- Be precise and reproducible.
- Write in formal academic English.
- Do NOT invent references.
"""
    return call_llm(prompt, cfg)


def _gen_results(
    extractions: List[ExtractionResult],
    synthesis: Dict[str, Any],
    rob: List[RiskOfBiasResult],
    cfg: Dict[str, Any],
) -> str:
    # Build a compact table for the LLM
    study_rows = "\n".join(
        f"- PMID {e.pmid}: {e.study_design}, n={e.sample_size}, "
        f"effect={e.effect_size}, CI=[{e.ci_lower}, {e.ci_upper}]"
        for e in extractions
    )

    synthesis_summary = ""
    if synthesis.get("type") == "meta-analysis":
        synthesis_summary = (
            f"Pooled effect = {synthesis['pooled_effect']:.3f} "
            f"(95% CI {synthesis['pooled_ci_lower']:.3f}–{synthesis['pooled_ci_upper']:.3f}), "
            f"k = {synthesis['k']}, I² = {synthesis['I2']:.1f}%, τ² = {synthesis['tau2']:.4f}"
        )
    else:
        synthesis_summary = synthesis.get("summary", "Thematic analysis performed.")

    prompt = f"""\
Write the Results section of a systematic review manuscript.

Included studies (n={len(extractions)}):
{study_rows}

Synthesis:
{synthesis_summary}

Requirements:
- Present study characteristics in prose (a table will be included separately).
- Report the synthesis results clearly.
- Mention risk-of-bias summary.
- Write in formal academic English.
- Do NOT invent data — use only what is provided above.
"""
    return call_llm(prompt, cfg)


def _gen_discussion(
    pico: PICOModel,
    synthesis: Dict[str, Any],
    n_included: int,
    cfg: Dict[str, Any],
) -> str:
    prompt = f"""\
Write the Discussion section of a systematic review manuscript.

Topic: {pico.topic}
Key finding: {synthesis.get('pooled_effect', synthesis.get('summary', 'see results'))}
Number of included studies: {n_included}

Requirements:
- Interpret the findings in the context of existing literature.
- Discuss strengths and limitations of the review.
- Discuss heterogeneity if applicable (I² = {synthesis.get('I2', 'N/A')}).
- Suggest implications for practice and future research.
- Write in formal academic English.
- Do NOT invent references.  Use [REF] placeholders.
- 4–6 paragraphs.
"""
    return call_llm(prompt, cfg)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def generate_manuscript(
    pico: PICOModel,
    extractions: List[ExtractionResult],
    synthesis: Dict[str, Any],
    rob: List[RiskOfBiasResult],
    n_retrieved: int,
    n_after_dedup: int,
    cfg: Dict[str, Any],
) -> str:
    """Generate the full LaTeX manuscript and write it to disk.

    Returns the path to the generated ``.tex`` file.
    """
    logger.info("Generating manuscript sections via LLM…")

    introduction = _gen_introduction(pico, cfg)
    methods = _gen_methods(pico, n_retrieved, n_after_dedup, len(extractions), cfg)
    results = _gen_results(extractions, synthesis, rob, cfg)
    discussion = _gen_discussion(pico, synthesis, len(extractions), cfg)

    # Save individual sections for iterative editing
    sections = {
        "introduction": introduction,
        "methods": methods,
        "results": results,
        "discussion": discussion,
    }
    save_json(sections, "data/results/manuscript_sections.json")

    # Render LaTeX via Jinja2
    template_dir = str(_resolve("templates"))
    env = Environment(
        loader=FileSystemLoader(template_dir),
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="<#",
        comment_end_string="#>",
    )
    template = env.get_template("manuscript.tex.jinja")

    # Prepare template variables
    has_forest = synthesis.get("type") == "meta-analysis"
    content = template.render(
        title=pico.topic.title(),
        introduction=introduction,
        methods=methods,
        results=results,
        discussion=discussion,
        has_forest_plot=has_forest,
        n_studies=len(extractions),
    )

    out_path = _resolve(cfg["paths"]["manuscript"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    logger.info("Manuscript saved → %s", out_path)
    return str(out_path)
