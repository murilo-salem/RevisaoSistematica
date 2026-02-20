"""
blackboard.py — Shared state for the multi-agent system.

The Blackboard is the central data store that all agents read from and
write to.  It holds articles, chunks, analysis results, section drafts,
review reports, and the full execution audit log.

Persisted to JSON after each major step for recovery and debugging.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from concept_registry import ConceptRegistry

from utils import now_iso

logger = logging.getLogger("systematic_review.agents.blackboard")


@dataclass
class Blackboard:
    """Global mutable state shared across all agents.

    All data produced by agents is stored here.  The coordinator is
    responsible for loading/saving the blackboard to disk.
    """

    # ---- Inputs --------------------------------------------------- #
    topic: str = ""
    taxonomy_entries: List[Dict[str, str]] = field(default_factory=list)

    # ---- Stage outputs (serialisable summaries) ------------------- #
    # Raw objects (StudyRecord, Chunk, etc.) are kept by reference in
    # the coordinator — we store only serialisable summaries here.
    n_articles: int = 0
    n_chunks: int = 0
    n_tags: int = 0

    # ---- Extraction ----------------------------------------------- #
    extraction_results: List[Dict[str, Any]] = field(default_factory=list)
    risk_of_bias_results: List[Dict[str, Any]] = field(default_factory=list)

    # ---- Mapping -------------------------------------------------- #
    theme_evidence: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ---- Critical analysis ---------------------------------------- #
    critical_analyses: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ---- Synthesis ------------------------------------------------ #
    synthesis_maps: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    chapter_theses: Dict[str, str] = field(default_factory=dict)

    # ---- Writing -------------------------------------------------- #
    section_drafts: Dict[str, str] = field(default_factory=dict)
    section_reviews: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    iteration_count: Dict[str, int] = field(default_factory=dict)
    approved_sections: Dict[str, str] = field(default_factory=dict)

    # ---- Final output --------------------------------------------- #
    final_document: Optional[str] = None

    # ---- Research agenda ------------------------------------------ #
    research_agenda: List[Dict[str, Any]] = field(default_factory=list)

    # ---- Audit ---------------------------------------------------- #
    execution_log: List[Dict[str, Any]] = field(default_factory=list)

    # ---- Concept tracking (real registry) ------------------------- #
    concept_registry: ConceptRegistry = field(default_factory=ConceptRegistry)
    covered_concepts: List[str] = field(default_factory=list)  # legacy compat

    # ---- Table tracking ------------------------------------------- #
    table_count: int = 0

    # ---------------------------------------------------------------- #
    #  Logging                                                          #
    # ---------------------------------------------------------------- #

    def log_event(
        self,
        agent: str,
        action: str,
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Append an event to the execution log."""
        entry = {
            "timestamp": now_iso(),
            "agent": agent,
            "action": action,
        }
        if details:
            entry["details"] = details
        self.execution_log.append(entry)

    # ---------------------------------------------------------------- #
    #  Persistence                                                      #
    # ---------------------------------------------------------------- #

    def save(self, path: str | Path) -> None:
        """Save the blackboard to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "topic": self.topic,
            "n_articles": self.n_articles,
            "n_chunks": self.n_chunks,
            "n_tags": self.n_tags,
            "chapter_theses": self.chapter_theses,
            "section_drafts": {k: v[:200] + "..." if len(v) > 200 else v
                               for k, v in self.section_drafts.items()},
            "iteration_count": self.iteration_count,
            "approved_sections": list(self.approved_sections.keys()),
            "final_document": self.final_document is not None,
            "execution_log": self.execution_log[-50:],  # last 50 events
            "covered_concepts": self.concept_registry.get_covered_concepts()[:100],
            "research_agenda": self.research_agenda[:30],
            "table_count": self.table_count,
        }

        # Also persist concept registry separately
        try:
            reg_path = path.parent / "concept_registry.json"
            self.concept_registry.to_json(reg_path)
        except Exception as exc:
            logger.debug("Could not save concept registry: %s", exc)

        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.debug("Blackboard saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "Blackboard":
        """Load blackboard from a JSON file (partial recovery)."""
        path = Path(path)
        bb = cls()
        if not path.exists():
            return bb
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            bb.topic = data.get("topic", "")
            bb.n_articles = data.get("n_articles", 0)
            bb.n_chunks = data.get("n_chunks", 0)
            bb.chapter_theses = data.get("chapter_theses", {})
            bb.iteration_count = data.get("iteration_count", {})
            bb.execution_log = data.get("execution_log", [])
            logger.info("Blackboard loaded from %s", path)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not load blackboard: %s", exc)
        return bb

    # ---------------------------------------------------------------- #
    #  Convenience                                                      #
    # ---------------------------------------------------------------- #

    def get_theme_key(self, parent: str, folder: str) -> str:
        """Canonical key for a theme (parent / folder)."""
        return f"{parent} / {folder}"

    def summary(self) -> Dict[str, Any]:
        """Return a compact status summary."""
        return {
            "topic": self.topic,
            "articles": self.n_articles,
            "chunks": self.n_chunks,
            "themes_mapped": len(self.theme_evidence),
            "themes_analysed": len(self.critical_analyses),
            "themes_synthesised": len(self.synthesis_maps),
            "sections_drafted": len(self.section_drafts),
            "sections_approved": len(self.approved_sections),
            "total_iterations": sum(self.iteration_count.values()),
            "has_final_doc": self.final_document is not None,
        }
