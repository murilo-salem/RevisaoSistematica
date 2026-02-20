"""
writing_agent.py — Agent for section writing with feedback integration.

Wraps ``review_writer.py`` section-writing functions, adding the ability
to incorporate feedback from the review agent on subsequent iterations.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from agents.base_agent import BaseAgent, Message, AgentResult

logger = logging.getLogger("systematic_review.agents.writing")


class WritingAgent(BaseAgent):
    """Write section text using thesis, evidence, and review feedback."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__("writing", cfg)

    def process(self, message: Message) -> AgentResult:
        """Process a section writing request.

        Expected payload keys:
          - entry: Dict with 'parent', 'folder', 'prompt' keys
          - studies, chunks, tags: data
          - thesis: chapter thesis string
          - synthesis_map: dict
          - covered_concepts: list of already-covered concepts
        Message fields:
          - feedback: review feedback for revisions
          - iteration: current iteration number
        """
        entry = message.payload.get("entry", {})
        studies = message.payload.get("studies", [])
        chunks = message.payload.get("chunks", [])
        tags = message.payload.get("tags", [])
        thesis = message.payload.get("thesis", "")
        synthesis_map = message.payload.get("synthesis_map", {})
        covered_concepts = message.payload.get("covered_concepts", [])
        covered_with_sections = message.payload.get("covered_with_sections", [])
        feedback = message.feedback
        iteration = message.iteration

        parent = entry.get("parent", "")
        folder = entry.get("folder", "")
        prompt = entry.get("prompt", "")
        theme = f"{parent} / {folder}"

        self.logger.info(
            "Writing section: %s (iteration=%d, has_feedback=%s)",
            theme, iteration, bool(feedback),
        )

        # ---- Build inputs for _write_single_entry ------------------ #
        from utils import Chunk, ChunkTag, StudyRecord
        from concept_registry import ConceptRegistry

        chunks_by_id = {c.chunk_id: c for c in chunks}
        studies_by_pmid = {s.pmid: s for s in studies}

        # Build a concept registry from covered concepts
        concept_registry = ConceptRegistry()
        for concept in covered_concepts:
            concept_registry.register(concept, "previous_section")

        # ---- Gather evidence --------------------------------------- #
        from review_writer import _gather_evidence, _write_section
        from review_writer import _pre_summarize_evidence, _extract_concepts

        top_k = self.cfg.get("retrieval", {}).get("top_k_evidence", 10)
        evidence, n_chunks = _gather_evidence(
            folder, parent, tags, chunks_by_id, studies_by_pmid, top_k,
        )

        if not evidence.strip():
            self.logger.info("No evidence for %s, skipping", theme)
            return AgentResult(
                success=True,
                data={"text": "", "new_concepts": []},
            )

        # Pre-summarise evidence
        evidence = _pre_summarize_evidence(
            evidence, chunks_by_id, studies_by_pmid, tags,
            folder, parent, top_k, self.cfg,
        )

        # Build covered concepts string with section references
        if covered_with_sections:
            # Structured format: concept (introduced in Section)
            items = [
                f"- {cs['concept']} (see: {cs['section']})"
                for cs in covered_with_sections[:40]
            ]
            covered_str = "\n".join(items) if items else "(none yet)"
        elif covered_concepts:
            covered_str = ", ".join(covered_concepts[:50])
        else:
            covered_str = "(none yet)"

        # ---- Modify prompt if we have feedback --------------------- #
        effective_prompt = prompt
        if feedback and iteration > 0:
            effective_prompt = (
                f"{prompt}\n\n"
                f"### REVISION FEEDBACK (iteration {iteration})\n"
                f"The following feedback was provided by a quality reviewer.\n"
                f"Address ALL points in this revision:\n\n"
                f"{feedback}"
            )

        # ---- Write section ----------------------------------------- #
        max_retries = self.cfg.get("multi_agent", {}).get("write_retries", 1)

        text = _write_section(
            prompt=effective_prompt,
            folder=folder,
            parent=parent,
            evidence=evidence,
            cfg=self.cfg,
            max_retries=max_retries,
            covered_concepts=covered_str,
            chapter_thesis=thesis or "(no specific thesis — write based on evidence)",
        )

        # ---- Extract new concepts ---------------------------------- #
        new_concepts: List[str] = []
        if text.strip():
            try:
                new_concepts = _extract_concepts(text, self.cfg)
            except Exception as exc:
                self.logger.debug("Concept extraction failed: %s", exc)

        self.logger.info(
            "Wrote %d chars for %s, %d new concepts",
            len(text), theme, len(new_concepts),
        )

        return AgentResult(
            success=True,
            data={
                "text": text,
                "new_concepts": new_concepts,
                "n_evidence_chunks": n_chunks,
            },
        )
