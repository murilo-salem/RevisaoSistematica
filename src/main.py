"""main.py — CLI entry point for the systematic-review pipeline.

Usage
-----
Online mode (requires Ollama + internet):
    python src/main.py --topic "effect of exercise on depression"

Local / Offline mode:
    python src/main.py --local
    python src/main.py --local --taxonomy config/biodiesel_prompts.json

High-end GPU profile (RTX 5090):
    python src/main.py --local --profile 5090 --taxonomy config/biodiesel_prompts.json

Multi-agent mode:
    python src/main.py --local --multi-agent --taxonomy config/biodiesel_prompts.json

Resume a previous run:
    python src/main.py --resume

Custom config:
    python src/main.py --config path.yaml --topic "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the src directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import load_config, load_json, setup_logging, _resolve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Systematic Review Automation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  Online:   python src/main.py --topic "effect of exercise on depression"
  Local:    python src/main.py --local
  Local+T:  python src/main.py --local --taxonomy config/biodiesel_prompts.json
  5090:     python src/main.py --local --profile 5090 --taxonomy config/biodiesel_prompts.json
  Agents:   python src/main.py --local --multi-agent --taxonomy config/biodiesel_prompts.json
  Resume:   python src/main.py --resume
""",
    )
    parser.add_argument(
        "--topic", "-t",
        type=str,
        default=None,
        help="Research topic for the systematic review (online mode)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to a custom config.yaml",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from the last saved pipeline state",
    )
    parser.add_argument(
        "--local", "-l",
        action="store_true",
        help="Run in local/offline mode: load articles from data/raw/ "
             "and screen via taxonomy (no Ollama required for core flow)",
    )
    parser.add_argument(
        "--taxonomy", "-x",
        type=str,
        default=None,
        help="Path to taxonomy JSON file (default: config/taxonomia.json)",
    )
    parser.add_argument(
        "--profile", "-p",
        type=str,
        default=None,
        help="Hardware profile: '5090' loads config/config_5090.yaml. "
             "Overrides --config.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only mode (bypasses GPU detection)",
    )
    parser.add_argument(
        "--multi-agent", "-m",
        action="store_true",
        help="Use the multi-agent pipeline with iterative review loops "
             "(requires --local mode)",
    )

    args = parser.parse_args()

    # Resolve config path: --profile takes precedence over --config
    config_path = args.config
    if args.profile:
        profile_path = f"config/config_{args.profile}.yaml"
        config_path = profile_path
        print(f"Using hardware profile: {args.profile} ({profile_path})")

    # Load configuration
    cfg = load_config(config_path)

    # Force CPU if requested
    if args.cpu:
        cfg.setdefault("system", {})["force_cpu"] = True

    logger = setup_logging(cfg)

    if args.profile:
        logger.info("Hardware profile: %s", args.profile)

    # ---- System Check -------------------------------------------- #
    from utils import check_system_capabilities
    caps = check_system_capabilities(cfg)
    cfg["system"] = caps

    logger.info("System Capabilities:")
    logger.info("  • PyTorch:             %s", "✅" if caps["torch"] else "❌")
    logger.info("  • CUDA (GPU):          %s", f"✅ ({caps['gpu_name']})" if caps["cuda"] else "❌")
    logger.info("  • SentenceTransformers: %s", "✅" if caps["sentence_transformers"] else "❌")

    if args.cpu:
        logger.info("  • Mode:                Manual CPU Override")
    elif not caps["cuda"]:
        logger.warning("No GPU detected or PyTorch not available. ML features will run on CPU or be disabled.")

    # ---- Local / Offline mode ------------------------------------ #
    if args.local:
        if getattr(args, 'multi_agent', False):
            logger.info("Running in LOCAL + MULTI-AGENT mode")
            from agents.pipeline import run_multi_agent_pipeline
            run_multi_agent_pipeline(taxonomy_path=args.taxonomy, cfg=cfg)
        else:
            logger.info("Running in LOCAL mode")
            from orchestrator import run_pipeline_local
            run_pipeline_local(taxonomy_path=args.taxonomy, cfg=cfg)
        return

    # ---- Resume mode --------------------------------------------- #
    if args.resume:
        state_path = _resolve(cfg["paths"]["pipeline_state"])
        if state_path.exists():
            state = load_json(cfg["paths"]["pipeline_state"])
            topic = state.get("topic", "")
            mode = state.get("mode", "ONLINE")
            logger.info("Resuming pipeline for topic: %s (mode: %s)", topic, mode)

            if mode == "LOCAL":
                from orchestrator import run_pipeline_local
                run_pipeline_local(
                    taxonomy_path=state.get("taxonomy_path"),
                    cfg=cfg,
                )
            else:
                from orchestrator import run_pipeline
                run_pipeline(topic, cfg)
        else:
            print("No saved state found. Please provide a topic or use --local.")
            sys.exit(1)
        return

    # ---- Online mode --------------------------------------------- #
    if args.topic:
        topic = args.topic
    else:
        topic = input("Tema da revisão: ").strip()
        if not topic:
            print("No topic provided. Exiting.")
            sys.exit(1)

    from orchestrator import run_pipeline
    run_pipeline(topic, cfg)


if __name__ == "__main__":
    main()
