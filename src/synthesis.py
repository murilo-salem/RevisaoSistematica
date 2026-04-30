"""
synthesis.py — Data synthesis: meta-analysis or thematic analysis.

When sufficient numeric effect-size data is available a fixed-effects
meta-analysis is computed (inverse-variance weighting) along with
heterogeneity statistics (I², Q, τ²).  A forest plot is generated via
matplotlib.

When numeric data is insufficient the module falls back to an LLM-driven
thematic analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from utils import ExtractionResult, call_llm, save_json, _resolve

logger = logging.getLogger("systematic_review.synthesis")


# ------------------------------------------------------------------ #
#  Meta-analysis helpers                                               #
# ------------------------------------------------------------------ #

def _inverse_variance_meta(
    effects: List[float],
    ci_lowers: List[float],
    ci_uppers: List[float],
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Fixed-effect meta-analysis using inverse-variance weighting.

    Parameters
    ----------
    effects : list[float]
        Point estimates (e.g. standardised mean difference, odds ratio).
    ci_lowers, ci_uppers : list[float]
        Lower and upper confidence-interval bounds.
    confidence_level : float
        Confidence level for the pooled CI (default 0.95).

    Returns
    -------
    dict with keys: pooled_effect, pooled_ci_lower, pooled_ci_upper,
    Q, I2, tau2, k.
    """
    z_crit = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    # Standard errors from CIs
    se = np.array([(u - l) / (2 * z_crit) for l, u in zip(ci_lowers, ci_uppers)])
    se = np.where(se <= 0, 1e-6, se)  # guard against zero SE

    weights = 1.0 / (se ** 2)
    es = np.array(effects)

    pooled = np.sum(weights * es) / np.sum(weights)
    pooled_se = 1.0 / np.sqrt(np.sum(weights))
    pooled_ci_l = pooled - z_crit * pooled_se
    pooled_ci_u = pooled + z_crit * pooled_se

    # Cochran's Q
    Q = np.sum(weights * (es - pooled) ** 2)
    k = len(effects)
    df = max(k - 1, 1)
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0.0
    tau2 = max(0, (Q - df) / (np.sum(weights) - np.sum(weights ** 2) / np.sum(weights))) if Q > df else 0.0

    return {
        "pooled_effect": float(pooled),
        "pooled_ci_lower": float(pooled_ci_l),
        "pooled_ci_upper": float(pooled_ci_u),
        "pooled_se": float(pooled_se),
        "Q": float(Q),
        "I2": float(I2),
        "tau2": float(tau2),
        "k": k,
    }


def _generate_forest_plot(
    labels: List[str],
    effects: List[float],
    ci_lowers: List[float],
    ci_uppers: List[float],
    pooled: Dict[str, Any],
    out_path: str,
) -> None:
    """Create and save a forest plot."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.6)))

    y_pos = list(range(len(labels)))

    # Individual studies
    for i, (label, e, lo, hi) in enumerate(zip(labels, effects, ci_lowers, ci_uppers)):
        ax.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt="o", color="#2563eb",
                     capsize=4, markersize=6, linewidth=1.5)

    # Pooled diamond
    pe = pooled["pooled_effect"]
    pl = pooled["pooled_ci_lower"]
    pu = pooled["pooled_ci_upper"]
    diamond_y = len(labels)
    ax.fill([pl, pe, pu, pe], [diamond_y, diamond_y + 0.3, diamond_y, diamond_y - 0.3],
            color="#dc2626", alpha=0.7)
    labels.append("Pooled")
    y_pos.append(diamond_y)

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Effect Size")
    ax.set_title(f"Forest Plot (k={pooled['k']}, I²={pooled['I2']:.1f}%)")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Forest plot saved → %s", out_path)


# ------------------------------------------------------------------ #
#  Thematic analysis fallback                                          #
# ------------------------------------------------------------------ #

def _thematic_analysis(
    extractions: List[ExtractionResult],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Use LLM to identify themes across studies when quantitative
    synthesis is not feasible."""

    summaries = "\n".join(
        f"- [{e.pmid}] {e.study_design}: {e.intervention} → {e.outcome}"
        for e in extractions
    )

    prompt = f"""\
You are a qualitative synthesis expert.

Given the following studies, identify the main themes and provide a
narrative synthesis.  Group findings by theme and note agreements and
disagreements.

Studies:
{summaries}

Return a structured summary with clearly labelled themes.
"""
    response = call_llm(prompt, cfg)
    return {"type": "thematic", "summary": response, "k": len(extractions)}


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def run_synthesis(
    extractions: List[ExtractionResult],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run meta-analysis (if possible) or thematic analysis.

    Returns a results dictionary used by the manuscript generator.
    """
    min_studies = cfg.get("synthesis", {}).get("min_studies_for_meta", 3)
    confidence = cfg.get("synthesis", {}).get("confidence_level", 0.95)

    # Collect studies with complete numeric data
    numeric = [
        e for e in extractions
        if e.effect_size is not None and e.ci_lower is not None and e.ci_upper is not None
    ]

    if len(numeric) >= min_studies:
        logger.info("Running meta-analysis with %d studies", len(numeric))

        effects = [e.effect_size for e in numeric]
        lowers = [e.ci_lower for e in numeric]
        uppers = [e.ci_upper for e in numeric]
        labels = [f"{e.pmid}: {e.intervention[:30]}" for e in numeric]

        meta = _inverse_variance_meta(effects, lowers, uppers, confidence)
        meta["type"] = "meta-analysis"

        # Forest plot
        plot_path = str(_resolve(cfg["paths"]["forest_plot"]))
        _generate_forest_plot(labels, effects, lowers, uppers, meta, plot_path)

        save_json(meta, "data/results/meta_analysis.json")
        return meta
    else:
        logger.info(
            "Insufficient numeric data (%d/%d) — falling back to thematic analysis",
            len(numeric), min_studies,
        )
        result = _thematic_analysis(extractions, cfg)
        save_json(result, "data/results/thematic_analysis.json")
        return result
