"""
synthesis_agent.py — Agent for evidence synthesis and thesis generation.

Wraps ``evidence_synthesizer.py`` and enriches its output with a
preliminary thesis derived from critical analysis findings.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, Message, AgentResult

logger = logging.getLogger("systematic_review.agents.synthesis")


# ------------------------------------------------------------------ #
#  Thesis generation prompt                                            #
# ------------------------------------------------------------------ #

_SYNTHESIS_THESIS_PROMPT = """\
You are an academic thesis formulator for a systematic literature review.

### Theme: {theme}

### Critical analysis summary
{critical_summary}

### Consensus points
{consensus}

### Contradictions
{contradictions}

### Knowledge gaps
{gaps}

### Task
Based on the analysis above, produce a JSON object with:

1. A concise THESIS STATEMENT (1–2 sentences) that captures the
   central argument for this section.  The thesis must:
   - Be falsifiable or debatable
   - Acknowledge complexity (e.g. "although X, evidence suggests Y")
   - Guide the narrative toward a specific conclusion

2. GROUPED RESEARCH GAPS: Organize the knowledge gaps into thematic
   groups.  Each group should have:
   - A group label (e.g. "Scalability", "Long-term Stability")
   - 3–5 specific gaps within that group (no duplicates)
   - A priority level for each gap: "alta" or "média"
   - A brief justification for each gap
   DO NOT repeat the same gap in different wording.  If two gaps
   are essentially the same concept, merge them into one.

3. A RESEARCH PRIORITIES list (3–5 items) ordered by importance,
   derived from the most critical gaps.

Return ONLY JSON:
{{
  "thesis": "...",
  "grouped_gaps": [
    {{
      "group": "Group Label",
      "gaps": [
        {{"description": "...", "priority": "alta|m\u00e9dia", "justification": "..."}}
      ]
    }}
  ],
  "research_priorities": ["...", "..."]
}}
"""


class SynthesisAgent(BaseAgent):
    """Synthesise evidence into consensus/contradiction/gap maps + thesis."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__("synthesis", cfg)

    def process(self, message: Message) -> AgentResult:
        """Process a synthesis request.

        Expected payload keys:
          - parent, folder: theme identifiers
          - studies, chunks, tags: data
          - critical_analysis: from critical agent (dict)
        """
        parent = message.payload.get("parent", "")
        folder = message.payload.get("folder", "")
        studies = message.payload.get("studies", [])
        chunks = message.payload.get("chunks", [])
        tags = message.payload.get("tags", [])
        critical_analysis = message.payload.get("critical_analysis", {})

        theme = f"{parent} / {folder}"
        self.logger.info("Synthesising theme: %s", theme)

        # ---- Run existing evidence synthesis ----------------------- #
        from evidence_synthesizer import synthesize_theme
        from utils import Chunk, ChunkTag, StudyRecord

        chunks_by_id = {c.chunk_id: c for c in chunks}
        studies_by_pmid = {s.pmid: s for s in studies}

        synthesis_map = synthesize_theme(
            folder, parent, tags, chunks_by_id, studies_by_pmid, self.cfg,
        )

        # ---- Generate thesis from synthesis + critical analysis ---- #
        thesis = ""
        research_priorities: List[str] = []

        consensus_text = "\n".join(
            f"- {cp.statement}" for cp in synthesis_map.consensus_points
        ) or "(none identified)"

        contradiction_text = "\n".join(
            f"- {c.point}: {c.possible_reason}"
            for c in synthesis_map.contradictions
        ) or "(none identified)"

        gap_text = "\n".join(
            f"- [{g.priority}] {g.description}"
            for g in synthesis_map.knowledge_gaps
        ) or "(none identified)"

        critical_summary = critical_analysis.get(
            "methodological_quality_summary", "(no analysis available)",
        )

        try:
            prompt = _SYNTHESIS_THESIS_PROMPT.format(
                theme=theme,
                critical_summary=critical_summary,
                consensus=consensus_text,
                contradictions=contradiction_text,
                gaps=gap_text,
            )

            raw = self.call_llm(prompt)
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            data = json.loads(cleaned)
            thesis = data.get("thesis", "")
            research_priorities = data.get("research_priorities", [])
            grouped_gaps = data.get("grouped_gaps", [])
        except (json.JSONDecodeError, Exception) as exc:
            self.logger.warning("Thesis generation failed: %s", exc)
            grouped_gaps = []

        self.logger.info(
            "Synthesis complete for %s: %d consensus, %d contradictions, %d gaps",
            theme,
            len(synthesis_map.consensus_points),
            len(synthesis_map.contradictions),
            len(synthesis_map.knowledge_gaps),
        )

        return AgentResult(
            success=True,
            data={
                "synthesis_map": synthesis_map.model_dump(),
                "thesis": thesis,
                "research_priorities": research_priorities,
                "grouped_gaps": grouped_gaps,
            },
        )

    # ---- Agenda consolidation ------------------------------------- #

    _AGENDA_PROMPT = """\
You are a research agenda editor for a systematic literature review.

Below are research gaps collected from multiple themes.  Many are
duplicates or near-duplicates phrased differently.

### Raw gaps
{raw_gaps}

### Task
Consolidate these into a SINGLE deduplicated research agenda with at
most 30 items.  Rules:
- MERGE gaps that describe the same concept in different words
- ORDER by priority (alta first, then média)
- Each item must have: description, priority (alta/média), suggested_approach
- NO duplicates — if two themes list the same gap, keep only one

Return ONLY a JSON array:
[
  {{"description": "...", "priority": "alta|m\u00e9dia", "suggested_approach": "..."}}
]
"""

    def consolidate_agenda(
        self, all_gaps: List[Dict[str, str]],
    ) -> Optional[List[Dict[str, str]]]:
        """Consolidate gaps from all themes into a clean agenda."""
        if not all_gaps:
            return None

        raw_text = "\n".join(
            f"- [{g.get('priority', 'medium')}] ({g.get('theme', '?')}): "
            f"{g.get('description', '')}"
            for g in all_gaps
        )

        prompt = self._AGENDA_PROMPT.format(raw_gaps=raw_text)

        try:
            raw = self.call_llm(prompt)
            cleaned = re.sub(r"```(?:json)?", "", raw).strip()
            agenda = json.loads(cleaned)
            if isinstance(agenda, list):
                self.logger.info(
                    "Consolidated %d raw gaps into %d agenda items",
                    len(all_gaps), len(agenda),
                )
                return agenda[:30]
        except (json.JSONDecodeError, Exception) as exc:
            self.logger.warning("Agenda consolidation failed: %s", exc)

        return None
