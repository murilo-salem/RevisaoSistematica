"""
models.py — Pydantic data models for the multi-agent system.

New models that extend the existing ones in utils.py for agent-specific
data structures: ThemeEvidence, CriticalAnalysis, ReviewReport, etc.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
#  Mapping agent models                                                #
# ------------------------------------------------------------------ #

class ContradictionDetail(BaseModel):
    """A detailed contradiction between studies."""
    point: str = ""
    study_a: str = ""
    finding_a: str = ""
    study_b: str = ""
    finding_b: str = ""
    possible_cause: str = ""


class ThemeEvidence(BaseModel):
    """Aggregated evidence for a single theme/section.

    Produced by the mapping agent after grouping chunks and performing
    preliminary consensus/contradiction analysis.
    """
    theme_id: str
    parent: str = ""
    folder: str = ""
    chunk_ids: List[str] = Field(default_factory=list)
    preliminary_consensus: str = ""
    preliminary_contradictions: List[ContradictionDetail] = Field(default_factory=list)
    coverage_stats: Dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ #
#  Critical analysis models                                            #
# ------------------------------------------------------------------ #

class ComparativeItem(BaseModel):
    """A structured comparison between two studies."""
    study_a: str = ""
    study_b: str = ""
    methodology_diff: str = ""     # e.g. "lab vs pilot scale"
    result_diff: str = ""          # e.g. "98% vs 82% yield"
    possible_cause: str = ""       # e.g. "different catalyst concentration"
    robustness_note: str = ""      # per-claim evidence strength


class CriticalAnalysis(BaseModel):
    """Deep methodological analysis for a theme.

    Produced by the critical agent after evaluating study quality,
    identifying contradictions, and assessing evidence robustness.
    """
    theme: str
    methodological_quality_summary: str = ""
    contradictions_detailed: List[ContradictionDetail] = Field(default_factory=list)
    comparative_analysis: List[ComparativeItem] = Field(default_factory=list)
    robustness_rating: str = "média"  # "alta" | "média" | "baixa"
    contextual_factors: str = ""


# ------------------------------------------------------------------ #
#  Review agent models                                                 #
# ------------------------------------------------------------------ #

class CriterionScore(BaseModel):
    """Score and comment for a single quality criterion."""
    score: float = 0.0
    comment: str = ""


class ReviewReport(BaseModel):
    """Quality assessment of a written section.

    Produced by the review agent, used by the coordinator to decide
    whether to iterate.
    """
    section_id: str = ""
    score: float = 0.0  # overall 0–10
    thesis_clarity: CriterionScore = Field(default_factory=CriterionScore)
    redundancy: CriterionScore = Field(default_factory=CriterionScore)
    citation_usage: CriterionScore = Field(default_factory=CriterionScore)
    hedging: CriterionScore = Field(default_factory=CriterionScore)
    critical_depth: CriterionScore = Field(default_factory=CriterionScore)
    overall_feedback: str = ""
