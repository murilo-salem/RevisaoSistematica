"""
extraction_agent.py — Agent for structured data extraction and RoB assessment.

Wraps the existing ``extraction.py`` and ``risk_of_bias.py`` modules,
running them as a single agent step.  Can use a lightweight model for speed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agents.base_agent import BaseAgent, Message, AgentResult

logger = logging.getLogger("systematic_review.agents.extraction")


class ExtractionAgent(BaseAgent):
    """Extract structured metadata and assess risk of bias for each study."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__("extraction", cfg)

    def process(self, message: Message) -> AgentResult:
        """Process an extraction request.

        Expected payload keys:
          - studies: List[StudyRecord]
        """
        studies = message.payload.get("studies", [])
        if not studies:
            return AgentResult(
                success=False,
                errors=["No studies provided for extraction"],
            )

        self.logger.info("Extracting data from %d studies", len(studies))

        # ---- Extraction ------------------------------------------- #
        from extraction import extract_data
        extractions = extract_data(studies, self.cfg)

        # ---- Risk of bias ----------------------------------------- #
        from risk_of_bias import assess_risk_of_bias
        rob_results = assess_risk_of_bias(studies, extractions, self.cfg)

        # Derive methodology_quality from overall_rating
        for rob in rob_results:
            quality = "medium"
            if rob.overall_rating == "low":
                quality = "high"
            elif rob.overall_rating == "high":
                quality = "low"
            # store it for downstream use
            rob_dict = rob.model_dump()
            rob_dict["methodology_quality"] = quality

        self.logger.info(
            "Extraction complete: %d extractions, %d RoB assessments",
            len(extractions), len(rob_results),
        )

        return AgentResult(
            success=True,
            data={
                "extractions": [e.model_dump() for e in extractions],
                "risk_of_bias": [r.model_dump() for r in rob_results],
                "n_extractions": len(extractions),
                "n_rob": len(rob_results),
            },
        )
