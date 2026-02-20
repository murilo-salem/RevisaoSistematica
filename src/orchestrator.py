"""
orchestrator.py — Pipeline orchestrator for the systematic-review system.

Two modes:
    run_pipeline(topic)       — Full online: LLM → PubMed → Screen → … → Manuscript
    run_pipeline_local(tax)   — Offline:     Local files → Taxonomy screen → … → Report

Each stage writes intermediate results, logs timing, and updates the
pipeline state file so runs can be audited or resumed.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from utils import (
    PICOModel,
    init_database,
    load_config,
    save_json,
    load_json,
    setup_logging,
    now_iso,
    _resolve,
)

logger = logging.getLogger("systematic_review.orchestrator")


# ------------------------------------------------------------------ #
#  Shared helpers                                                      #
# ------------------------------------------------------------------ #

def _init(cfg: Dict[str, Any] | None, label: str, topic: str) -> tuple:
    if cfg is None:
        cfg = load_config()
    setup_logging(cfg)
    init_database(cfg)

    logger.info("=" * 60)
    logger.info("SYSTEMATIC REVIEW PIPELINE — %s", label)
    logger.info("Topic: %s", topic)
    logger.info("Config hash: %s", cfg.get("version", {}).get("config_hash", "?"))
    logger.info("=" * 60)

    state: Dict[str, Any] = {
        "topic": topic,
        "mode": label,
        "started_at": now_iso(),
        "config_hash": cfg.get("version", {}).get("config_hash"),
        "stages": {},
    }
    return cfg, state


def _save_state(state: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    save_json(state, cfg["paths"]["pipeline_state"])


# ------------------------------------------------------------------ #
#  ONLINE pipeline (original — requires Ollama + PubMed)               #
# ------------------------------------------------------------------ #

def run_pipeline(topic: str, cfg: Dict[str, Any] | None = None) -> None:
    """Execute the full systematic-review pipeline for *topic*."""

    cfg, state = _init(cfg, "ONLINE", topic)

    # ---- Stage 1: Query Builder ---------------------------------- #
    logger.info("▶ Stage 1/7 — Query Builder")
    t0 = time.time()
    from query_builder import build_query
    pico, query = build_query(topic, cfg)
    elapsed = time.time() - t0
    state["stages"]["query_builder"] = {"elapsed_s": round(elapsed, 2), "query": query}
    _save_state(state, cfg)
    logger.info("  ✓ Query built in %.1fs", elapsed)

    # ---- Stage 2: Retrieval -------------------------------------- #
    logger.info("▶ Stage 2/7 — PubMed Retrieval")
    t0 = time.time()
    from retrieval import search_pubmed
    studies = search_pubmed(query, cfg)
    n_retrieved = len(studies)
    elapsed = time.time() - t0
    state["stages"]["retrieval"] = {"elapsed_s": round(elapsed, 2), "n_retrieved": n_retrieved}
    _save_state(state, cfg)
    logger.info("  ✓ Retrieved %d studies in %.1fs", n_retrieved, elapsed)

    if not studies:
        logger.warning("No studies retrieved — pipeline cannot continue.")
        state["finished_at"] = now_iso()
        state["status"] = "no_results"
        _save_state(state, cfg)
        return

    # ---- Stage 3: Deduplication ---------------------------------- #
    logger.info("▶ Stage 3/7 — Deduplication")
    t0 = time.time()
    from deduplication import deduplicate
    studies = deduplicate(studies, cfg)
    n_after_dedup = len(studies)
    elapsed = time.time() - t0
    state["stages"]["deduplication"] = {
        "elapsed_s": round(elapsed, 2),
        "n_before": n_retrieved,
        "n_after": n_after_dedup,
    }
    _save_state(state, cfg)
    logger.info("  ✓ %d → %d studies in %.1fs", n_retrieved, n_after_dedup, elapsed)

    # ---- Stage 4: Screening -------------------------------------- #
    logger.info("▶ Stage 4/7 — Screening")
    t0 = time.time()
    from screening import screen_studies
    included = screen_studies(studies, pico, cfg)
    elapsed = time.time() - t0
    state["stages"]["screening"] = {
        "elapsed_s": round(elapsed, 2),
        "n_screened": n_after_dedup,
        "n_included": len(included),
    }
    _save_state(state, cfg)
    logger.info("  ✓ %d included of %d in %.1fs", len(included), n_after_dedup, elapsed)

    if not included:
        logger.warning("No studies included after screening — pipeline cannot continue.")
        state["finished_at"] = now_iso()
        state["status"] = "no_included"
        _save_state(state, cfg)
        return

    # ---- Stage 5: Data Extraction -------------------------------- #
    logger.info("▶ Stage 5/7 — Data Extraction")
    t0 = time.time()
    from extraction import extract_data
    extracted = extract_data(included, cfg)
    elapsed = time.time() - t0
    state["stages"]["extraction"] = {
        "elapsed_s": round(elapsed, 2),
        "n_extracted": len(extracted),
    }
    _save_state(state, cfg)
    logger.info("  ✓ Extracted data from %d studies in %.1fs", len(extracted), elapsed)

    # ---- Stage 6: Risk-of-Bias Assessment ------------------------ #
    logger.info("▶ Stage 6/7 — Risk-of-Bias Assessment")
    t0 = time.time()
    from risk_of_bias import assess_risk_of_bias
    rob = assess_risk_of_bias(included, extracted, cfg)
    elapsed = time.time() - t0
    state["stages"]["risk_of_bias"] = {"elapsed_s": round(elapsed, 2), "n_assessed": len(rob)}
    _save_state(state, cfg)
    logger.info("  ✓ Assessed %d studies in %.1fs", len(rob), elapsed)

    # ---- Stage 7a: Synthesis ------------------------------------- #
    logger.info("▶ Stage 7/7 — Synthesis + Manuscript")
    t0 = time.time()
    from synthesis import run_synthesis
    synthesis = run_synthesis(extracted, cfg)
    elapsed_synth = time.time() - t0

    # ---- Stage 7b: Manuscript Generation ------------------------- #
    from manuscript import generate_manuscript
    t1 = time.time()
    out_path = generate_manuscript(
        pico, extracted, synthesis, rob,
        n_retrieved, n_after_dedup, cfg,
    )
    elapsed_ms = time.time() - t1

    state["stages"]["synthesis"] = {
        "elapsed_s": round(elapsed_synth, 2),
        "type": synthesis.get("type", "unknown"),
    }
    state["stages"]["manuscript"] = {
        "elapsed_s": round(elapsed_ms, 2),
        "output": out_path,
    }
    state["finished_at"] = now_iso()
    state["status"] = "completed"
    _save_state(state, cfg)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Manuscript → %s", out_path)
    logger.info("=" * 60)


# ------------------------------------------------------------------ #
#  LOCAL / OFFLINE pipeline (no Ollama required for core flow)         #
# ------------------------------------------------------------------ #

def run_pipeline_local(
    taxonomy_path: str | None = None,
    cfg: Dict[str, Any] | None = None,
) -> None:
    """Execute an offline pipeline using local articles + taxonomy.

    Two modes depending on the taxonomy file format:

    **Outline mode** (``type == "outline"``)
        1. Load articles
        2. Deduplicate
        3. Content analysis (embed articles + prompts)
        4. Chunking (split articles into semantic chunks)
        5. Tag mapping (assign chunks to taxonomy sections)
        6. Correlation check (coverage report)
        7. Section writing (LLM writes each section using chunks)
        8. Assemble review document

    **Keyword mode** (traditional JSON/MD taxonomy)
        1. Load → 2. Dedup → 3. Screen → 4. Extract → 5. Synthesise
    """
    from local_loader import load_local_studies, load_taxonomy

    taxonomy = load_taxonomy(taxonomy_path)
    topic = taxonomy.get("topic", "Local systematic review")
    is_outline = taxonomy.get("type") == "outline"
    n_stages = 10 if is_outline else 5

    cfg, state = _init(cfg, "LOCAL", topic)
    state["taxonomy_path"] = taxonomy_path or "config/taxonomia.json"
    state["taxonomy_type"] = "outline" if is_outline else "keyword"

    # Build a PICOModel from the taxonomy
    pico_data = taxonomy.get("pico", {})
    pico = PICOModel(
        topic=topic,
        population=pico_data.get("population", ""),
        intervention=pico_data.get("intervention", ""),
        comparison=pico_data.get("comparison", ""),
        outcome=pico_data.get("outcome", ""),
        query="LOCAL — see taxonomy",
    )
    save_json(pico.model_dump(), cfg["paths"]["pico"])

    # ---- Stage 0: PDF Conversion (optional) ----------------------- #
    pdf_input = _resolve(cfg.get("paths", {}).get("pdf_input", "data/pdfs"))
    raw_dir = _resolve(cfg.get("paths", {}).get("raw_dir", "data/raw"))
    
    if pdf_input.exists():
        from pdf_converter import convert_pdfs
        logger.info("▶ Stage 0/9 — Checking for PDF documents")
        n_conv = convert_pdfs(cfg, input_dir=pdf_input, output_dir=raw_dir)
        if n_conv > 0:
            logger.info("  ✓ Converted %d new PDFs to text", n_conv)
    
    # ---- Stage 1: Load local articles ----------------------------- #
    logger.info("▶ Stage 1/%d — Loading local articles", n_stages)
    t0 = time.time()
    studies = load_local_studies(cfg)
    n_loaded = len(studies)
    elapsed = time.time() - t0
    state["stages"]["local_load"] = {"elapsed_s": round(elapsed, 2), "n_loaded": n_loaded}
    _save_state(state, cfg)
    logger.info("  ✓ Loaded %d articles in %.1fs", n_loaded, elapsed)

    if not studies:
        logger.warning("No articles found in data/raw/ — pipeline cannot continue.")
        logger.info("Supported formats: .txt, .json, .csv, .bib")
        state["finished_at"] = now_iso()
        state["status"] = "no_articles"
        _save_state(state, cfg)
        return

    # ---- Stage 2: Deduplication ---------------------------------- #
    logger.info("▶ Stage 2/%d — Deduplication", n_stages)
    t0 = time.time()
    from deduplication import deduplicate
    studies = deduplicate(studies, cfg)
    n_after_dedup = len(studies)
    elapsed = time.time() - t0
    state["stages"]["deduplication"] = {
        "elapsed_s": round(elapsed, 2),
        "n_before": n_loaded,
        "n_after": n_after_dedup,
    }
    _save_state(state, cfg)
    logger.info("  ✓ %d → %d studies in %.1fs", n_loaded, n_after_dedup, elapsed)

    # ================================================================ #
    #  OUTLINE MODE — content analysis + chunking + review writing      #
    # ================================================================ #
    if is_outline:
        entries = taxonomy.get("entries", [])

        # ---- Stage 3: Content Analysis ----------------------------- #
        logger.info("▶ Stage 3/%d — Content Analysis", n_stages)
        t0 = time.time()
        from content_analyzer import analyze_and_chunk
        all_chunks, all_tags, coverage = analyze_and_chunk(studies, entries, cfg)
        elapsed = time.time() - t0

        state["stages"]["content_analysis"] = {
            "elapsed_s": round(elapsed, 2),
            "n_chunks": len(all_chunks),
            "n_tags": len(all_tags),
        }
        _save_state(state, cfg)
        logger.info(
            "  ✓ %d chunks, %d tags in %.1fs",
            len(all_chunks), len(all_tags), elapsed,
        )

        # ---- Stage 4: Chunking — (done inside analyze_and_chunk) --- #
        logger.info("▶ Stage 4/%d — Chunking (done)", n_stages)
        state["stages"]["chunking"] = {
            "elapsed_s": 0,
            "n_chunks": len(all_chunks),
            "note": "Included in content_analysis stage",
        }

        # ---- Stage 5: Tag Mapping — (done inside analyze_and_chunk) - #
        logger.info("▶ Stage 5/%d — Tag Mapping (done)", n_stages)
        state["stages"]["tag_mapping"] = {
            "elapsed_s": 0,
            "n_tags": len(all_tags),
            "note": "Included in content_analysis stage",
        }

        # ---- Stage 6: Correlation Check ----------------------------- #
        logger.info("▶ Stage 6/%d — Correlation Check", n_stages)
        summary = coverage.get("summary", {})
        logger.info(
            "  Coverage: %d/%d sections have evidence",
            summary.get("sections_with_evidence", 0),
            summary.get("total_sections", 0),
        )
        state["stages"]["correlation"] = {"coverage": summary}
        _save_state(state, cfg)

        # ---- Stage 7: Evidence Synthesis (Épico 2) ------------------ #
        logger.info("▶ Stage 7/%d — Evidence Synthesis", n_stages)
        t0 = time.time()
        from evidence_synthesizer import synthesize_all_themes
        synthesis_maps = synthesize_all_themes(
            entries, all_chunks, all_tags, studies, cfg,
        )
        elapsed = time.time() - t0
        state["stages"]["evidence_synthesis"] = {
            "elapsed_s": round(elapsed, 2),
            "n_themes": len(synthesis_maps),
            "n_consensus": sum(len(s.consensus_points) for s in synthesis_maps.values()),
            "n_contradictions": sum(len(s.contradictions) for s in synthesis_maps.values()),
            "n_gaps": sum(len(s.knowledge_gaps) for s in synthesis_maps.values()),
        }
        _save_state(state, cfg)
        logger.info("  ✓ Synthesized %d themes in %.1fs", len(synthesis_maps), elapsed)

        # ---- Stage 8: Organize into Folders ----------------------- #
        logger.info("▶ Stage 8/%d — Organize into Folders", n_stages)
        t0 = time.time()
        from organizer import organize_by_taxonomy
        organized_dir = organize_by_taxonomy(
            all_chunks, all_tags, studies, entries, cfg,
        )
        elapsed = time.time() - t0
        state["stages"]["organize"] = {
            "elapsed_s": round(elapsed, 2),
            "output_dir": organized_dir,
        }
        _save_state(state, cfg)
        logger.info("  ✓ Organized into folders in %.1fs → %s", elapsed, organized_dir)

        # ---- Stage 9: Section Writing ------------------------------ #
        logger.info("▶ Stage 9/%d — Section Writing", n_stages)
        t0 = time.time()
        from review_writer import write_review
        out_path = write_review(entries, all_chunks, all_tags, studies, topic, cfg, synthesis_maps)
        elapsed = time.time() - t0

        state["stages"]["section_writing"] = {
            "elapsed_s": round(elapsed, 2),
            "n_sections": len(entries),
        }
        _save_state(state, cfg)
        logger.info("  ✓ Wrote %d sections in %.1fs", len(entries), elapsed)

        # ---- Stage 10: Assemble Review ----------------------------- #
        logger.info("▶ Stage 10/%d — Assemble Review (done)", n_stages)
        state["stages"]["assembly"] = {"output": out_path}

        state["finished_at"] = now_iso()
        state["status"] = "completed"
        _save_state(state, cfg)

        logger.info("=" * 60)
        logger.info("LOCAL PIPELINE COMPLETE (OUTLINE MODE)")
        logger.info("Articles: %d → %d (after dedup)", n_loaded, n_after_dedup)
        logger.info("Chunks: %d | Tags: %d", len(all_chunks), len(all_tags))
        logger.info("Sections: %d", len(entries))
        logger.info("Review → %s", out_path)
        logger.info("=" * 60)
        return

    # ================================================================ #
    #  KEYWORD MODE — traditional taxonomy screening                    #
    # ================================================================ #

    # ---- Stage 3: Taxonomy-based screening ----------------------- #
    logger.info("▶ Stage 3/%d — Taxonomy Screening", n_stages)
    t0 = time.time()
    from screening import screen_studies_by_taxonomy
    included = screen_studies_by_taxonomy(studies, taxonomy, cfg)
    elapsed = time.time() - t0
    state["stages"]["screening"] = {
        "elapsed_s": round(elapsed, 2),
        "mode": "taxonomy",
        "n_screened": n_after_dedup,
        "n_included": len(included),
    }
    _save_state(state, cfg)
    logger.info("  ✓ %d included of %d in %.1fs", len(included), n_after_dedup, elapsed)

    if not included:
        logger.warning("No studies included after taxonomy screening.")
        state["finished_at"] = now_iso()
        state["status"] = "no_included"
        _save_state(state, cfg)
        return

    # ---- Stage 4: Data Extraction (LLM if available) ------------- #
    logger.info("▶ Stage 4/%d — Data Extraction", n_stages)
    t0 = time.time()
    extracted = []
    try:
        from extraction import extract_data
        extracted = extract_data(included, cfg)
    except Exception as exc:
        logger.warning("LLM extraction failed (%s) — saving raw records only", exc)
        from utils import ExtractionResult
        for s in included:
            extracted.append(ExtractionResult(
                pmid=s.pmid,
                study_design="unknown",
                population=s.abstract[:200] if s.abstract else "",
                intervention="",
                outcome="",
                notes="Auto-extracted without LLM — manual review required",
            ))
        save_json([e.model_dump() for e in extracted], cfg["paths"]["extracted"])

    elapsed = time.time() - t0
    state["stages"]["extraction"] = {
        "elapsed_s": round(elapsed, 2),
        "n_extracted": len(extracted),
    }
    _save_state(state, cfg)
    logger.info("  ✓ Extracted data from %d studies in %.1fs", len(extracted), elapsed)

    # ---- Stage 5: Synthesis -------------------------------------- #
    logger.info("▶ Stage 5/%d — Synthesis", n_stages)
    t0 = time.time()
    synthesis = {"type": "local", "k": len(extracted)}
    try:
        from synthesis import run_synthesis
        synthesis = run_synthesis(extracted, cfg)
    except Exception as exc:
        logger.warning("Synthesis failed (%s) — using basic summary", exc)
        synthesis["summary"] = f"Local analysis of {len(extracted)} studies. Manual synthesis required."
        save_json(synthesis, "data/results/synthesis_local.json")

    elapsed = time.time() - t0
    state["stages"]["synthesis"] = {
        "elapsed_s": round(elapsed, 2),
        "type": synthesis.get("type", "local"),
    }

    state["finished_at"] = now_iso()
    state["status"] = "completed"
    _save_state(state, cfg)

    logger.info("=" * 60)
    logger.info("LOCAL PIPELINE COMPLETE")
    logger.info("Included: %d studies", len(included))
    logger.info("Extracted: %d records", len(extracted))
    logger.info("Results → data/results/")
    logger.info("=" * 60)


