"""phase_co2_validator.py — Physical consistency checks for CO2 phase regimes."""

from __future__ import annotations

from typing import List

from utils import PaperProfile, ValidationFlag

CO2_TRIPLE_T_K = 216.58
CO2_TRIPLE_P_BAR = 5.18
CO2_CRITICAL_T_K = 304.13
CO2_CRITICAL_P_BAR = 73.8


def _has_any(items: list[str], needle: str) -> bool:
    needle = needle.lower()
    return any(needle in it.lower() for it in items)


def validate_profile(profile: PaperProfile) -> List[ValidationFlag]:
    """Validate coarse CO2 phase/range consistency for one paper profile."""
    flags: List[ValidationFlag] = []

    regimes = [r.lower() for r in profile.phase_regimes]
    fluids = [f.lower() for f in profile.working_fluids]

    t_min = profile.temperature_range_K.min
    t_max = profile.temperature_range_K.max
    p_min = profile.pressure_range_bar.min
    p_max = profile.pressure_range_bar.max

    uses_co2 = (
        _has_any(fluids, "co2")
        or _has_any(profile.keywords, "co2")
        or _has_any(regimes, "co2")
    )

    if not uses_co2:
        return flags

    if "solid" in regimes and t_max is not None and t_max > CO2_TRIPLE_T_K + 2.0:
        flags.append(ValidationFlag(
            flag_type="phase_co2",
            severity="high",
            message=(
                f"CO2 em regime sólido com Tmax={t_max:.2f} K acima do ponto "
                f"triplo ({CO2_TRIPLE_T_K:.2f} K)."
            ),
            related_ids=[profile.paper_id],
        ))

    if "near_triple_point" in regimes:
        if t_max is not None and t_max > 230.0:
            flags.append(ValidationFlag(
                flag_type="phase_co2",
                severity="high",
                message="Regime near_triple_point com temperatura incompatível (>230 K).",
                related_ids=[profile.paper_id],
            ))
        if p_min is not None and p_min < 3.0:
            flags.append(ValidationFlag(
                flag_type="phase_co2",
                severity="medium",
                message="Regime near_triple_point com pressão muito baixa (<3 bar).",
                related_ids=[profile.paper_id],
            ))

    if "supercritical" in regimes:
        if t_max is not None and t_max < CO2_CRITICAL_T_K:
            flags.append(ValidationFlag(
                flag_type="phase_co2",
                severity="high",
                message="Regime supercritical sem temperatura acima do ponto crítico.",
                related_ids=[profile.paper_id],
            ))
        if p_max is not None and p_max < CO2_CRITICAL_P_BAR:
            flags.append(ValidationFlag(
                flag_type="phase_co2",
                severity="high",
                message="Regime supercritical sem pressão acima do ponto crítico.",
                related_ids=[profile.paper_id],
            ))

    if "transcritical" in regimes:
        # espera-se cruzamento de regime: ao menos um indicativo acima do crítico
        has_high_t = t_max is not None and t_max >= CO2_CRITICAL_T_K
        has_high_p = p_max is not None and p_max >= CO2_CRITICAL_P_BAR
        if not (has_high_t or has_high_p):
            flags.append(ValidationFlag(
                flag_type="phase_co2",
                severity="high",
                message="Regime transcritical sem indício de cruzamento do ponto crítico.",
                related_ids=[profile.paper_id],
            ))

    return flags
