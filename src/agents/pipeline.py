"""
pipeline.py — Multi-agent pipeline entry point.

Replaces the outline mode of ``run_pipeline_local()`` when the
``--multi-agent`` flag is used.  Sets up all agents, runs existing
pre-processing stages (loading, dedup, chunking), then hands off
to the CoordinatorAgent for the analysis → writing → review loop.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from utils import load_config, now_iso, _resolve

logger = logging.getLogger("systematic_review.agents.pipeline")


def run_multi_agent_pipeline(
    taxonomy_path: str | None = None,
    cfg: Dict[str, Any] | None = None,
) -> None:
    """Execute the multi-agent pipeline for systematic review.

    Reuses the same loading/dedup/chunking stages from the monolithic
    pipeline, then delegates to the coordinator for analysis and writing.
    """
    if cfg is None:
        cfg = load_config()

    logger = logging.getLogger("systematic_review.agents.pipeline")
    logger.info("=" * 60)
    logger.info("MULTI-AGENT PIPELINE START")
    logger.info("=" * 60)

    t0 = time.time()

    # ---- Load taxonomy -------------------------------------------- #
    from local_loader import load_taxonomy
    taxonomy = load_taxonomy(taxonomy_path)
    is_outline = taxonomy.get("type") == "outline"

    if not is_outline:
        logger.error("Multi-agent mode only supports outline taxonomy.")
        return

    topic = taxonomy.get("topic", cfg.get("topic", "Systematic Review"))
    entries = taxonomy.get("entries", [])

    # ---- Stage 1: Load articles ----------------------------------- #
    logger.info("▶ Stage 1 — Loading articles")
    from local_loader import load_local_studies
    studies = load_local_studies(cfg)
    if not studies:
        logger.warning("No articles found.")
        return
    logger.info("  ✓ Loaded %d articles", len(studies))

    # ---- Stage 2: Deduplication ----------------------------------- #
    logger.info("▶ Stage 2 — Deduplication")
    from deduplication import deduplicate
    n_before = len(studies)
    studies = deduplicate(studies, cfg)
    logger.info("  ✓ %d → %d studies", n_before, len(studies))

    # ---- Stage 3: PDF conversion ---------------------------------- #
    pdf_input = str(_resolve("data/raw/pdf"))
    raw_dir = str(_resolve("data/raw"))
    try:
        from pdf_converter import convert_pdfs
        n_conv = convert_pdfs(cfg, input_dir=pdf_input, output_dir=raw_dir)
        if n_conv > 0:
            logger.info("  ✓ Converted %d PDFs", n_conv)
    except ImportError:
        pass

    # ---- Stage 4: Content analysis + chunking --------------------- #
    logger.info("▶ Stage 4 — Content Analysis")
    from content_analyzer import analyze_and_chunk
    all_chunks, all_tags, coverage = analyze_and_chunk(studies, entries, cfg)
    logger.info("  ✓ %d chunks, %d tags", len(all_chunks), len(all_tags))

    # ---- Set up agents -------------------------------------------- #
    logger.info("▶ Setting up multi-agent system")

    from agents.blackboard import Blackboard
    from agents.coordinator_agent import CoordinatorAgent
    from agents.extraction_agent import ExtractionAgent
    from agents.mapping_agent import MappingAgent
    from agents.critical_agent import CriticalAgent
    from agents.synthesis_agent import SynthesisAgent
    from agents.writing_agent import WritingAgent
    from agents.review_agent import ReviewAgent
    from agents.formatting_agent import FormattingAgent
    from agents.debate_agent import DebateAgent

    blackboard = Blackboard()
    coordinator = CoordinatorAgent(cfg, blackboard)

    # Register all agents
    coordinator.register_agent(ExtractionAgent(cfg))
    coordinator.register_agent(MappingAgent(cfg))
    coordinator.register_agent(CriticalAgent(cfg))
    coordinator.register_agent(SynthesisAgent(cfg))
    coordinator.register_agent(WritingAgent(cfg))
    coordinator.register_agent(ReviewAgent(cfg))
    coordinator.register_agent(FormattingAgent(cfg))

    if cfg.get("multi_agent", {}).get("debate_enabled", True):
        coordinator.register_agent(DebateAgent(cfg))

    # ---- Run coordinator ------------------------------------------ #
    logger.info("▶ Running coordinator with %d agents", len(coordinator.agents))
    output_path = coordinator.run(
        studies=studies,
        taxonomy_entries=entries,
        topic=topic,
        chunks=all_chunks,
        tags=all_tags,
    )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("MULTI-AGENT PIPELINE COMPLETE (%.1fs)", elapsed)
    if output_path:
        logger.info("Output → %s", output_path)
    logger.info("=" * 60)
