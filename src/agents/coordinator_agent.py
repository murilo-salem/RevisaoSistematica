"""
coordinator_agent.py — Central orchestrator for the multi-agent pipeline.

Manages the execution plan, dispatches tasks to specialised agents,
handles the write → review iteration loop, and assembles the final
document.

Flow
----
1. Extraction  (extraction_agent)
2. Mapping     (mapping_agent)
3. Per-theme loop:
   a. Critical analysis  (critical_agent)
   b. Synthesis          (synthesis_agent)
   c. Writing            (writing_agent)
   d. Review             (review_agent)
   e. If score < threshold → iterate (c) with feedback
4. Formatting + assembly  (formatting_agent)
5. Optional debate for controversial themes  (debate_agent)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, Message, AgentResult
from agents.blackboard import Blackboard

logger = logging.getLogger("systematic_review.agents.coordinator")


class CoordinatorAgent:
    """Orchestrates the multi-agent systematic review pipeline.

    Not a BaseAgent subclass — it *manages* agents rather than
    processing messages itself.
    """

    def __init__(self, cfg: Dict[str, Any], blackboard: Blackboard) -> None:
        self.cfg = cfg
        self.bb = blackboard
        self.agents: Dict[str, BaseAgent] = {}

        # Multi-agent config
        ma_cfg = cfg.get("multi_agent", {})
        self.max_iterations: int = ma_cfg.get("max_iterations", 3)
        self.quality_threshold: float = ma_cfg.get("quality_threshold", 7.0)
        self.parallel_themes: int = ma_cfg.get("parallel_themes", 2)
        self.debate_enabled: bool = ma_cfg.get("debate_enabled", True)
        self.debate_controversy_threshold: int = ma_cfg.get(
            "debate_controversy_threshold", 2,
        )

    # ---- Agent registration --------------------------------------- #

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for use in the pipeline."""
        self.agents[agent.name] = agent
        logger.info("Registered agent: %s", agent.name)

    def _dispatch(self, agent_name: str, message: Message) -> AgentResult:
        """Send a message to a named agent and return the result."""
        agent = self.agents.get(agent_name)
        if agent is None:
            logger.error("Agent %r not registered", agent_name)
            return AgentResult(
                success=False,
                errors=[f"Agent {agent_name!r} not registered"],
            )
        return agent.timed_process(message)

    # ---- Main pipeline -------------------------------------------- #

    def run(
        self,
        studies: list,
        taxonomy_entries: List[Dict[str, str]],
        topic: str,
        chunks: list | None = None,
        tags: list | None = None,
    ) -> str | None:
        """Execute the full multi-agent pipeline.

        Parameters
        ----------
        studies : list[StudyRecord]
        taxonomy_entries : list[dict]
        topic : str
        chunks : list[Chunk], optional
            Pre-computed chunks (from content_analyzer).
        tags : list[ChunkTag], optional
            Pre-computed tags.

        Returns
        -------
        str or None
            Path to the final document, or None on failure.
        """
        t0 = time.time()
        self.bb.topic = topic
        self.bb.taxonomy_entries = taxonomy_entries
        self.bb.log_event("coordinator", "pipeline_start", {"topic": topic})

        logger.info("=" * 60)
        logger.info("MULTI-AGENT PIPELINE START")
        logger.info("  Topic: %s", topic)
        logger.info("  Articles: %d | Taxonomy entries: %d", len(studies), len(taxonomy_entries))
        logger.info("  Max iterations: %d | Quality threshold: %.1f",
                     self.max_iterations, self.quality_threshold)
        logger.info("=" * 60)

        # ---- Phase 1: Extraction ---------------------------------- #
        logger.info("▶ Phase 1 — Extraction")
        result = self._dispatch("extraction", Message(
            task="extract",
            payload={"studies": studies},
            source="coordinator",
        ))
        if not result.success:
            logger.error("Extraction failed: %s", result.errors)
            return None
        self.bb.extraction_results = result.data.get("extractions", [])
        self.bb.risk_of_bias_results = result.data.get("risk_of_bias", [])
        self.bb.n_articles = len(studies)
        self._save_blackboard()

        # ---- Phase 2: Mapping ------------------------------------- #
        logger.info("▶ Phase 2 — Evidence Mapping")
        result = self._dispatch("mapping", Message(
            task="map",
            payload={
                "studies": studies,
                "taxonomy_entries": taxonomy_entries,
                "chunks": chunks,
                "tags": tags,
            },
            source="coordinator",
        ))
        if not result.success:
            logger.error("Mapping failed: %s", result.errors)
            return None

        # Store mapping results
        mapped_chunks = result.data.get("chunks", chunks or [])
        mapped_tags = result.data.get("tags", tags or [])
        theme_evidence = result.data.get("theme_evidence", {})
        self.bb.n_chunks = len(mapped_chunks)
        self.bb.n_tags = len(mapped_tags)
        self.bb.theme_evidence = {
            k: v.model_dump() if hasattr(v, "model_dump") else v
            for k, v in theme_evidence.items()
        }
        self._save_blackboard()

        # ---- Phase 3: Per-theme analysis + writing loop ----------- #
        logger.info("▶ Phase 3 — Theme Analysis & Writing (with review loop)")

        # Group entries by parent (chapter)
        chapter_entries: Dict[str, List[Dict[str, str]]] = {}
        for entry in taxonomy_entries:
            chapter_entries.setdefault(entry["parent"], []).append(entry)

        all_theme_keys = [
            self.bb.get_theme_key(e["parent"], e["folder"])
            for e in taxonomy_entries
        ]

        if self.parallel_themes > 1 and len(all_theme_keys) > 1:
            self._process_themes_parallel(
                taxonomy_entries, studies, mapped_chunks, mapped_tags,
            )
        else:
            self._process_themes_sequential(
                taxonomy_entries, studies, mapped_chunks, mapped_tags,
            )

        self._save_blackboard()

        # ---- Phase 3.5: Table detection per theme ------------------- #
        logger.info("▶ Phase 3.5 — Table Detection")
        self._detect_tables_for_themes(
            taxonomy_entries, studies, mapped_chunks, mapped_tags,
        )

        # ---- Phase 3.6: Agenda consolidation ---------------------- #
        logger.info("▶ Phase 3.6 — Research Agenda Consolidation")
        self._consolidate_research_agenda()

        # ---- Phase 4: Debate (optional) --------------------------- #
        if self.debate_enabled and "debate" in self.agents:
            self._run_debates(taxonomy_entries)

        # ---- Phase 5: Formatting & Assembly ----------------------- #
        logger.info("▶ Phase 5 — Formatting & Assembly")
        result = self._dispatch("formatting", Message(
            task="format",
            payload={
                "approved_sections": self.bb.approved_sections,
                "taxonomy_entries": taxonomy_entries,
                "topic": topic,
                "synthesis_maps": self.bb.synthesis_maps,
                "chapter_theses": self.bb.chapter_theses,
                "studies": studies,
                "chunks": mapped_chunks,
                "tags": mapped_tags,
                "research_agenda": self.bb.research_agenda,
                "table_count": self.bb.table_count,
            },
            source="coordinator",
        ))
        if result.success:
            self.bb.final_document = result.data.get("output_path")

        # ---- Done ------------------------------------------------- #
        elapsed = time.time() - t0
        self.bb.log_event("coordinator", "pipeline_complete", {
            "elapsed_s": round(elapsed, 2),
            "summary": self.bb.summary(),
        })
        self._save_blackboard()

        logger.info("=" * 60)
        logger.info("MULTI-AGENT PIPELINE COMPLETE")
        logger.info("  Elapsed: %.1fs", elapsed)
        logger.info("  Sections: %d drafted, %d approved",
                     len(self.bb.section_drafts), len(self.bb.approved_sections))
        logger.info("  Iterations: %d total", sum(self.bb.iteration_count.values()))
        if self.bb.final_document:
            logger.info("  Output → %s", self.bb.final_document)
        logger.info("=" * 60)

        return self.bb.final_document

    # ---- Theme processing ----------------------------------------- #

    def _process_single_theme(
        self,
        entry: Dict[str, str],
        studies: list,
        chunks: list,
        tags: list,
    ) -> None:
        """Run the analysis → synthesis → write → review loop for one theme."""
        parent = entry["parent"]
        folder = entry["folder"]
        theme_key = self.bb.get_theme_key(parent, folder)

        logger.info("  Processing theme: %s", theme_key)

        # ---- 3a. Critical Analysis -------------------------------- #
        crit_result = self._dispatch("critical", Message(
            task="analyse",
            payload={
                "parent": parent,
                "folder": folder,
                "studies": studies,
                "chunks": chunks,
                "tags": tags,
                "extractions": self.bb.extraction_results,
                "risk_of_bias": self.bb.risk_of_bias_results,
            },
            source="coordinator",
        ))
        if crit_result.success:
            self.bb.critical_analyses[theme_key] = crit_result.data.get(
                "critical_analysis", {},
            )

        # ---- 3b. Synthesis ---------------------------------------- #
        synth_result = self._dispatch("synthesis", Message(
            task="synthesise",
            payload={
                "parent": parent,
                "folder": folder,
                "studies": studies,
                "chunks": chunks,
                "tags": tags,
                "critical_analysis": self.bb.critical_analyses.get(theme_key, {}),
            },
            source="coordinator",
        ))
        if synth_result.success:
            self.bb.synthesis_maps[theme_key] = synth_result.data.get(
                "synthesis_map", {},
            )
            thesis = synth_result.data.get("thesis", "")
            if thesis:
                self.bb.chapter_theses.setdefault(parent, thesis)

        # ---- 3c–d. Write → Review loop ---------------------------- #
        self.bb.iteration_count[theme_key] = 0
        feedback = ""

        for iteration in range(self.max_iterations):
            self.bb.iteration_count[theme_key] = iteration + 1

            # Write
            write_result = self._dispatch("writing", Message(
                task="write_section",
                payload={
                    "entry": entry,
                    "studies": studies,
                    "chunks": chunks,
                    "tags": tags,
                    "thesis": self.bb.chapter_theses.get(parent, ""),
                    "synthesis_map": self.bb.synthesis_maps.get(theme_key, {}),
                    "covered_concepts": self.bb.concept_registry.get_covered_concepts(),
                    "covered_with_sections": self.bb.concept_registry.get_covered_with_sections(),
                },
                source="coordinator",
                iteration=iteration,
                feedback=feedback,
            ))

            if not write_result.success:
                logger.warning("Writing failed for %s (iter %d)", theme_key, iteration)
                break

            draft = write_result.data.get("text", "")
            self.bb.section_drafts[theme_key] = draft

            # Update covered concepts via registry
            new_concepts = write_result.data.get("new_concepts", [])
            theme_label = f"{parent} / {folder}"
            self.bb.concept_registry.register_many(new_concepts, theme_label)
            # legacy compat
            self.bb.covered_concepts.extend(new_concepts)

            # Build summary of previously approved sections for redundancy check
            approved_summary = self._build_approved_summary(theme_key)

            review_result = self._dispatch("review", Message(
                task="review",
                payload={
                    "section_text": draft,
                    "thesis": self.bb.chapter_theses.get(parent, ""),
                    "synthesis_map": self.bb.synthesis_maps.get(theme_key, {}),
                    "section_id": theme_key,
                    "approved_sections_summary": approved_summary,
                },
                source="coordinator",
                iteration=iteration,
            ))

            if review_result.success:
                score = review_result.data.get("score", 0.0)
                self.bb.section_reviews[theme_key] = review_result.data
                self.bb.log_event("coordinator", "review_score", {
                    "theme": theme_key,
                    "iteration": iteration,
                    "score": score,
                })

                if score >= self.quality_threshold:
                    logger.info(
                        "  ✓ %s approved (score=%.1f, iter=%d)",
                        theme_key, score, iteration + 1,
                    )
                    self.bb.approved_sections[theme_key] = draft
                    return
                else:
                    feedback = review_result.data.get("overall_feedback", "")
                    logger.info(
                        "  ↻ %s needs revision (score=%.1f, iter=%d): %s",
                        theme_key, score, iteration + 1, feedback[:100],
                    )
            else:
                # Review failed — accept draft anyway
                logger.warning("Review failed for %s, accepting draft", theme_key)
                self.bb.approved_sections[theme_key] = draft
                return

        # Max iterations reached — accept last draft
        logger.warning(
            "  ⚠ %s: max iterations reached (%d), accepting last draft",
            theme_key, self.max_iterations,
        )
        self.bb.approved_sections[theme_key] = self.bb.section_drafts.get(
            theme_key, "",
        )

    def _process_themes_sequential(
        self,
        entries: List[Dict[str, str]],
        studies: list,
        chunks: list,
        tags: list,
    ) -> None:
        """Process all themes one by one."""
        for i, entry in enumerate(entries):
            logger.info(
                "  Theme %d/%d: %s / %s",
                i + 1, len(entries), entry["parent"], entry["folder"],
            )
            self._process_single_theme(entry, studies, chunks, tags)

    def _process_themes_parallel(
        self,
        entries: List[Dict[str, str]],
        studies: list,
        chunks: list,
        tags: list,
    ) -> None:
        """Process themes in parallel using ThreadPoolExecutor."""
        logger.info(
            "  Processing %d themes with %d parallel workers",
            len(entries), self.parallel_themes,
        )
        with ThreadPoolExecutor(max_workers=self.parallel_themes) as pool:
            futures = {
                pool.submit(
                    self._process_single_theme, entry, studies, chunks, tags,
                ): entry
                for entry in entries
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    theme_key = self.bb.get_theme_key(
                        entry["parent"], entry["folder"],
                    )
                    logger.error("Theme %s failed: %s", theme_key, exc)

    # ---- Debate --------------------------------------------------- #

    def _run_debates(self, taxonomy_entries: List[Dict[str, str]]) -> None:
        """Trigger debate agent for highly controversial themes."""
        logger.info("▶ Phase 4 — Debate (controversial themes)")
        for entry in taxonomy_entries:
            theme_key = self.bb.get_theme_key(entry["parent"], entry["folder"])
            ca = self.bb.critical_analyses.get(theme_key, {})

            # Check if theme is controversial enough
            contradictions = ca.get("contradictions_detailed", [])
            robustness = ca.get("robustness_rating", "média")

            if (len(contradictions) >= self.debate_controversy_threshold
                    and robustness == "baixa"):
                logger.info("  Debate triggered for: %s", theme_key)
                result = self._dispatch("debate", Message(
                    task="debate",
                    payload={
                        "theme_key": theme_key,
                        "contradictions": contradictions,
                        "current_draft": self.bb.approved_sections.get(theme_key, ""),
                        "entry": entry,
                    },
                    source="coordinator",
                ))
                if result.success:
                    debate_text = result.data.get("debate_section", "")
                    if debate_text:
                        # Append debate section to the approved draft
                        existing = self.bb.approved_sections.get(theme_key, "")
                        self.bb.approved_sections[theme_key] = (
                            existing + "\n\n" + debate_text
                        )

    # ---- Approved summary for redundancy check -------------------- #

    def _build_approved_summary(self, current_theme: str) -> str:
        """Build a summary of previously approved sections for the review
        agent to detect cross-section redundancies."""
        summaries: List[str] = []
        for key, text in self.bb.approved_sections.items():
            if key == current_theme:
                continue
            # Take first 300 chars of each section as a digest
            preview = text[:300].replace("\n", " ").strip()
            if preview:
                summaries.append(f"[{key}]: {preview}...")
        if not summaries:
            return "(no previous sections approved yet)"
        return "\n".join(summaries[:15])  # cap at 15 sections

    # ---- Table detection ------------------------------------------ #

    def _detect_tables_for_themes(
        self,
        entries: List[Dict[str, str]],
        studies: list,
        chunks: list,
        tags: list,
    ) -> None:
        """Detect table opportunities and insert markers into approved sections."""
        try:
            from table_generator import detect_table_opportunity, TableRegistry
            from utils import Chunk, ChunkTag, StudyRecord
        except ImportError as exc:
            logger.warning("Table generation unavailable: %s", exc)
            return

        chunks_by_id = {c.chunk_id: c for c in chunks}
        studies_by_pmid = {s.pmid: s for s in studies}
        table_registry = TableRegistry()

        for entry in entries:
            parent = entry["parent"]
            folder = entry["folder"]
            theme_key = self.bb.get_theme_key(parent, folder)

            if theme_key not in self.bb.approved_sections:
                continue

            try:
                table_spec = detect_table_opportunity(
                    folder, parent, tags, chunks_by_id, studies_by_pmid,
                    self.cfg,
                )
                if table_spec:
                    md = table_registry.register(table_spec)
                    marker = table_registry.get_marker(table_spec.table_id)
                    # Append the actual table at the end of the section
                    self.bb.approved_sections[theme_key] += (
                        f"\n\n{md}"
                    )
                    logger.info(
                        "  Table generated for %s: %s", theme_key,
                        table_spec.caption[:60],
                    )
            except Exception as exc:
                logger.debug("Table detection failed for %s: %s", theme_key, exc)

        self.bb.table_count = table_registry.size()
        logger.info("  Total tables generated: %d", self.bb.table_count)

    # ---- Research agenda consolidation ----------------------------- #

    def _consolidate_research_agenda(self) -> None:
        """Consolidate research gaps from all themes into a deduplicated
        agenda of max 30 items."""
        if "synthesis" not in self.agents:
            logger.warning("Synthesis agent not registered, skipping agenda")
            return

        # Collect all gaps from synthesis maps
        all_gaps: List[Dict[str, str]] = []
        for theme_key, smap in self.bb.synthesis_maps.items():
            gaps = smap.get("knowledge_gaps", [])
            for gap in gaps:
                if isinstance(gap, dict):
                    all_gaps.append({
                        "theme": theme_key,
                        "description": gap.get("description", str(gap)),
                        "priority": gap.get("priority", "medium"),
                    })
                else:
                    all_gaps.append({
                        "theme": theme_key,
                        "description": str(gap),
                        "priority": "medium",
                    })

        if not all_gaps:
            logger.info("  No research gaps to consolidate")
            return

        logger.info("  Consolidating %d raw gaps into agenda", len(all_gaps))

        # Use synthesis agent to consolidate
        agent = self.agents["synthesis"]
        result = agent.consolidate_agenda(all_gaps)
        if result:
            self.bb.research_agenda = result[:30]
            logger.info(
                "  Research agenda: %d items", len(self.bb.research_agenda),
            )
        else:
            # Fallback: simple dedup by description
            seen = set()
            unique = []
            for g in all_gaps:
                desc_lower = g["description"].lower().strip()
                if desc_lower not in seen:
                    seen.add(desc_lower)
                    unique.append(g)
            self.bb.research_agenda = unique[:30]
            logger.info(
                "  Research agenda (fallback dedup): %d items",
                len(self.bb.research_agenda),
            )

    # ---- Persistence ---------------------------------------------- #

    def _save_blackboard(self) -> None:
        """Save blackboard to the configured path."""
        from utils import _resolve
        path = _resolve("data/processed/blackboard.json")
        self.bb.save(path)
