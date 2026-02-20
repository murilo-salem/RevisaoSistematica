"""
critical_agent.py — Agent for deep methodological critical analysis.

Evaluates study quality across themes, identifies contradictions with
causal reasoning, and assesses evidence robustness.  Uses a capable
model for deep reasoning.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from agents.base_agent import BaseAgent, Message, AgentResult
from agents.models import CriticalAnalysis, ContradictionDetail, ComparativeItem

logger = logging.getLogger("systematic_review.agents.critical")


# ------------------------------------------------------------------ #
#  Critical analysis prompt                                            #
# ------------------------------------------------------------------ #

_CRITICAL_PROMPT = """\
You are an experienced scientific reviewer analysing the literature on
the theme "{theme}".

Below are excerpts from several studies, each annotated with metadata
(author, year, scale, risk-of-bias rating, funding).

### Your task — produce a CRITICAL ANALYSIS containing:

1. **Methodological quality summary**: Are the studies generally well-
   conducted?  Are there common problems (small samples, lack of
   controls, unclear methods)?  Specifically evaluate:
   - Study design quality (randomisation, blinding, controls)
   - Sample sizes and statistical power
   - Reproducibility potential

2. **Contradictions**: Identify any contradictory results between
   studies.  For EACH contradiction, explain the possible
   methodological or contextual cause (e.g. different catalyst,
   temperature, scale, feedstock).

3. **Comparative analysis**: For the 2–4 most important findings in
   this theme, provide a STRUCTURED COMPARISON between studies:
   - Compare study designs (scale: lab/pilot/industrial, sample size,
     methodology)
   - Explain HOW methodological differences may explain result
     differences
   - Rate the robustness of each specific claim (strong/moderate/weak)
     based on: number of corroborating studies, risk of bias, sample
     size, and consistency across scales.

4. **Evidence robustness**: Classify the OVERALL body of evidence as
   "alta" (high), "média" (moderate), or "baixa" (low), with
   justification citing specific studies.

5. **Contextual factors**: Is there evidence of influence from
   geographic context, funding source, or study scale on the results?

### Studies

{formatted_studies}

### Output format

Return ONLY a valid JSON object with these keys:
{{
  "methodological_quality_summary": "...",
  "contradictions": [
    {{"point": "...", "study_a": "...", "finding_a": "...",
      "study_b": "...", "finding_b": "...", "possible_cause": "..."}}
  ],
  "comparative_analysis": [
    {{"study_a": "...", "study_b": "...",
      "methodology_diff": "e.g. lab-scale with KOH vs pilot-scale with NaOH",
      "result_diff": "e.g. 98% yield vs 82% yield",
      "possible_cause": "e.g. catalyst concentration and reaction time differed",
      "robustness_note": "strong|moderate|weak — with brief justification"}}
  ],
  "robustness_rating": "alta|média|baixa",
  "contextual_factors": "..."
}}
"""


class CriticalAgent(BaseAgent):
    """Deep methodological analysis for a theme."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__("critical", cfg)

    def process(self, message: Message) -> AgentResult:
        """Process a critical analysis request.

        Expected payload keys:
          - parent, folder: theme identifiers
          - studies, chunks, tags: data from mapping
          - extractions, risk_of_bias: from extraction agent
        """
        parent = message.payload.get("parent", "")
        folder = message.payload.get("folder", "")
        studies = message.payload.get("studies", [])
        chunks = message.payload.get("chunks", [])
        tags = message.payload.get("tags", [])
        extractions = message.payload.get("extractions", [])
        rob_results = message.payload.get("risk_of_bias", [])

        theme = f"{parent} / {folder}"
        self.logger.info("Critical analysis for: %s", theme)

        # Build lookups
        from utils import StudyRecord
        chunks_by_id = {c.chunk_id: c for c in chunks}
        studies_by_pmid = {s.pmid: s for s in studies}
        ext_by_pmid = {e["pmid"]: e for e in extractions} if extractions else {}
        rob_by_pmid = {r["pmid"]: r for r in rob_results} if rob_results else {}

        # Gather relevant chunks for this theme
        relevant_tags = [
            t for t in tags
            if t.parent == parent and t.folder == folder
        ]
        relevant_tags.sort(key=lambda t: t.similarity, reverse=True)
        top_tags = relevant_tags[:20]

        if not top_tags:
            self.logger.warning("No evidence for theme: %s", theme)
            return AgentResult(
                success=True,
                data={"critical_analysis": CriticalAnalysis(theme=theme).model_dump()},
            )

        # Format studies with metadata
        formatted = self._format_studies(
            top_tags, chunks_by_id, studies_by_pmid, ext_by_pmid, rob_by_pmid,
        )

        # Call LLM
        prompt = _CRITICAL_PROMPT.format(
            theme=theme,
            formatted_studies=formatted,
        )

        try:
            raw = self.call_llm(prompt)
            analysis = self._parse_response(raw, theme)
        except Exception as exc:
            self.logger.error("Critical analysis failed: %s", exc)
            analysis = CriticalAnalysis(theme=theme)

        return AgentResult(
            success=True,
            data={"critical_analysis": analysis.model_dump()},
        )

    # ---- Helpers -------------------------------------------------- #

    def _format_studies(
        self,
        tags: list,
        chunks_by_id: dict,
        studies_by_pmid: dict,
        ext_by_pmid: dict,
        rob_by_pmid: dict,
    ) -> str:
        """Format study excerpts with metadata for the prompt."""
        parts: List[str] = []
        seen_pmids: set = set()

        for tag in tags:
            chunk = chunks_by_id.get(tag.chunk_id)
            if not chunk:
                continue
            pmid = chunk.study_pmid
            study = studies_by_pmid.get(pmid)
            ext = ext_by_pmid.get(pmid, {})
            rob = rob_by_pmid.get(pmid, {})

            author = study.authors if study and study.authors else pmid
            year = study.year if study else "n.d."
            scale = ext.get("study_scale", "unknown")
            rob_overall = rob.get("overall_rating", "unclear")
            funding = ext.get("funding_source", "unknown")

            label = (
                f"--- ({author}, {year}) | Scale: {scale} "
                f"| RoB: {rob_overall} | Funding: {funding} ---"
            )

            if pmid not in seen_pmids:
                parts.append(f"{label}\n{chunk.text[:600]}")
                seen_pmids.add(pmid)

        return "\n\n".join(parts)

    def _parse_response(self, raw: str, theme: str) -> CriticalAnalysis:
        """Parse the LLM JSON response into a CriticalAnalysis."""
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            self.logger.warning("Could not parse critical analysis JSON")
            return CriticalAnalysis(
                theme=theme,
                methodological_quality_summary=raw[:500],
            )

        contradictions = []
        for c in data.get("contradictions", []):
            contradictions.append(ContradictionDetail(
                point=c.get("point", ""),
                study_a=c.get("study_a", ""),
                finding_a=c.get("finding_a", ""),
                study_b=c.get("study_b", ""),
                finding_b=c.get("finding_b", ""),
                possible_cause=c.get("possible_cause", ""),
            ))

        comparisons = []
        for comp in data.get("comparative_analysis", []):
            comparisons.append(ComparativeItem(
                study_a=comp.get("study_a", ""),
                study_b=comp.get("study_b", ""),
                methodology_diff=comp.get("methodology_diff", ""),
                result_diff=comp.get("result_diff", ""),
                possible_cause=comp.get("possible_cause", ""),
                robustness_note=comp.get("robustness_note", ""),
            ))

        return CriticalAnalysis(
            theme=theme,
            methodological_quality_summary=data.get(
                "methodological_quality_summary", "",
            ),
            contradictions_detailed=contradictions,
            comparative_analysis=comparisons,
            robustness_rating=data.get("robustness_rating", "média"),
            contextual_factors=data.get("contextual_factors", ""),
        )
