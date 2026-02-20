"""
review_agent.py — Agent for quality review of written sections.

Evaluates section text on multiple quality criteria and produces
actionable feedback for the writing agent.  The coordinator uses the
overall score to decide whether to iterate.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from agents.base_agent import BaseAgent, Message, AgentResult
from agents.models import CriterionScore, ReviewReport

logger = logging.getLogger("systematic_review.agents.review")


# ------------------------------------------------------------------ #
#  Review prompt                                                       #
# ------------------------------------------------------------------ #

_REVIEW_PROMPT = """\
You are a senior academic editor reviewing a section from a systematic
literature review.

### Section text
{section_text}

### Section thesis
{thesis}

### Task
Evaluate this section on the following criteria, scoring each from 0 to
10 (10 = excellent).  Provide a brief comment for each criterion.

1. **Thesis clarity**: Does the section clearly support or extend the
   thesis?  Is the argumentation coherent and progressive?

2. **Redundancy**: Is there unnecessary repetition of concepts,
   findings, or phrasing?  (10 = no redundancy)

3. **Cross-section redundancy**: Given the summaries of OTHER already-
   approved sections below, does this section repeat explanations or
   concepts that were already covered elsewhere?  If so, list the
   specific overlaps.  (10 = no cross-section overlap)

   ### Previously approved sections
   {approved_sections_summary}

4. **Citation usage**: Are citations used correctly and consistently?
   Does every claim have a supporting citation?

5. **Hedging**: Does the text appropriately use hedging language for
   uncertain findings without being overly cautious or overly
   assertive?

6. **Critical depth**: Does the section go beyond mere description?
   Does it analyse, compare, and contextualise the evidence?

Finally, compute an OVERALL SCORE (weighted average with critical depth
and thesis clarity having 2× weight) and provide 2–3 specific,
actionable suggestions for improvement.

Return ONLY a valid JSON object:
{{
  "thesis_clarity": {{"score": float, "comment": "..."}},
  "redundancy": {{"score": float, "comment": "..."}},
  "cross_section_redundancy": {{"score": float, "comment": "..."}},
  "citation_usage": {{"score": float, "comment": "..."}},
  "hedging": {{"score": float, "comment": "..."}},
  "critical_depth": {{"score": float, "comment": "..."}},
  "overall_score": float,
  "overall_feedback": "2–3 specific suggestions"
}}
"""


class ReviewAgent(BaseAgent):
    """Evaluate section quality and produce actionable feedback."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__("review", cfg)

    def process(self, message: Message) -> AgentResult:
        """Process a review request.

        Expected payload keys:
          - section_text: the draft section text
          - thesis: the chapter thesis
          - synthesis_map: optional context
          - section_id: theme key for tracking
          - approved_sections_summary: summaries of other approved sections
        """
        section_text = message.payload.get("section_text", "")
        thesis = message.payload.get("thesis", "")
        synthesis_map = message.payload.get("synthesis_map", {})
        section_id = message.payload.get("section_id", "")
        approved_summary = message.payload.get(
            "approved_sections_summary", "(no previous sections)",
        )

        if not section_text.strip():
            return AgentResult(
                success=True,
                data={
                    "score": 10.0,
                    "overall_feedback": "Empty section — nothing to review.",
                    "section_id": section_id,
                },
            )

        self.logger.info("Reviewing section: %s", section_id)

        prompt = _REVIEW_PROMPT.format(
            section_text=section_text[:4000],  # cap to avoid context overflow
            thesis=thesis or "(no thesis provided)",
            approved_sections_summary=approved_summary,
        )

        try:
            raw = self.call_llm(prompt)
            report = self._parse_response(raw, section_id)
        except Exception as exc:
            self.logger.error("Review failed: %s", exc)
            # On failure, return a passing score to avoid blocking
            return AgentResult(
                success=False,
                data={"score": 8.0, "overall_feedback": f"Review error: {exc}"},
                errors=[str(exc)],
            )

        self.logger.info(
            "Review complete for %s: score=%.1f", section_id, report.score,
        )

        return AgentResult(
            success=True,
            data={
                "score": report.score,
                "overall_feedback": report.overall_feedback,
                "thesis_clarity": report.thesis_clarity.model_dump(),
                "redundancy": report.redundancy.model_dump(),
                "citation_usage": report.citation_usage.model_dump(),
                "hedging": report.hedging.model_dump(),
                "critical_depth": report.critical_depth.model_dump(),
                "section_id": section_id,
            },
        )

    def _parse_response(self, raw: str, section_id: str) -> ReviewReport:
        """Parse the LLM JSON into a ReviewReport."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            self.logger.warning("Could not parse review JSON: %s...", raw[:200])
            return ReviewReport(
                section_id=section_id,
                score=7.0,
                overall_feedback=raw[:500],
            )

        def _parse_criterion(d: dict) -> CriterionScore:
            return CriterionScore(
                score=float(d.get("score", 0)),
                comment=d.get("comment", ""),
            )

        return ReviewReport(
            section_id=section_id,
            score=float(data.get("overall_score", 0)),
            thesis_clarity=_parse_criterion(data.get("thesis_clarity", {})),
            redundancy=_parse_criterion(data.get("redundancy", {})),
            citation_usage=_parse_criterion(data.get("citation_usage", {})),
            hedging=_parse_criterion(data.get("hedging", {})),
            critical_depth=_parse_criterion(data.get("critical_depth", {})),
            overall_feedback=data.get("overall_feedback", ""),
        )
