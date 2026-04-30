"""domain_validator.py — Domain detection and routing guardrail checks."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from utils import PaperProfile, ValidationFlag


_DOMAIN_PATTERNS: Dict[str, List[re.Pattern]] = {
    "refrigeration": [
        re.compile(r"\brefrigeration\b", re.IGNORECASE),
        re.compile(r"\bcold storage\b", re.IGNORECASE),
        re.compile(r"\bvcrc\b", re.IGNORECASE),
        re.compile(r"\bevaporator\b|\bcondenser\b", re.IGNORECASE),
    ],
    "heat_transfer": [
        re.compile(r"\bboiling\b|\bnucleate\b", re.IGNORECASE),
        re.compile(r"\bheat transfer\b|\bthermal conductivity\b", re.IGNORECASE),
        re.compile(r"\bcritical heat flux\b|\bCHF\b", re.IGNORECASE),
    ],
    "materials": [
        re.compile(r"\bphase change material\b|\bPCM\b", re.IGNORECASE),
        re.compile(r"\bencapsulation\b|\bcomposite\b", re.IGNORECASE),
        re.compile(r"\bporous\b|\bmicrostructure\b", re.IGNORECASE),
    ],
    "energy_storage": [
        re.compile(r"\bthermal energy storage\b|\bTES\b|\bCTES\b|cold thermal energy storage", re.IGNORECASE),
        re.compile(r"\bLAES\b|\bliquid air\b", re.IGNORECASE),
        re.compile(r"\bcharge\b|\bdischarge\b", re.IGNORECASE),
    ],
    "process_chemical": [
        re.compile(r"\bWGS\b|water[- ]gas shift", re.IGNORECASE),
        re.compile(r"\bPSA\b|pressure swing adsorption", re.IGNORECASE),
        re.compile(r"\bASU\b|air separation unit", re.IGNORECASE),
        re.compile(r"\bRPlug\b|\bRGibbs\b|\bAspen\b", re.IGNORECASE),
        re.compile(r"\breactor\b|\bcatalyst\b|\bsyngas\b", re.IGNORECASE),
        re.compile(r"\bpyrolysis\b|\bbio[- ]oil\b|\bbiochar\b|\bbiodiesel\b", re.IGNORECASE),
        re.compile(r"\bmicroalgae\b|\banaerobic digestion\b|\bbiogas\b|\bfermentation\b", re.IGNORECASE),
        re.compile(r"\bsteam reforming\b|\bmethanol\b|\bglycerol\b|\btransesterification\b", re.IGNORECASE),
        re.compile(r"\bdiesel engine\b|\bengine emissions?\b|\blow[- ]carbon fuels?\b", re.IGNORECASE),
        re.compile(r"\bdac\b|\bccu\b|\bdirect air capture\b|\bcarbon capture\b", re.IGNORECASE),
    ],
}

_FLUID_PATTERNS: Dict[str, List[re.Pattern]] = {
    "CO2_R744": [
        re.compile(r"\bR744\b", re.IGNORECASE),
        re.compile(r"\bCO2\b|\bCO₂\b", re.IGNORECASE),
    ],
    "CO2_solid": [
        re.compile(r"\bdry ice\b", re.IGNORECASE),
        re.compile(r"\bsolid CO2\b|\bCO2 snow\b|\bCO₂ snow\b", re.IGNORECASE),
    ],
    "air": [
        re.compile(r"\bair\b", re.IGNORECASE),
        re.compile(r"\bliquid air\b", re.IGNORECASE),
    ],
}

_PHASE_PATTERNS: Dict[str, List[re.Pattern]] = {
    "solid": [
        re.compile(r"\bsolid\b", re.IGNORECASE),
        re.compile(r"\bdry ice\b", re.IGNORECASE),
    ],
    "near_triple_point": [
        re.compile(r"\btriple point\b|\bponto triplo\b", re.IGNORECASE),
    ],
    "subcritical": [
        re.compile(r"\bsubcritical\b|\bsubcr[ií]tico\b", re.IGNORECASE),
    ],
    "transcritical": [
        re.compile(r"\btranscritical\b|\btranscr[ií]tico\b", re.IGNORECASE),
    ],
    "supercritical": [
        re.compile(r"\bsupercritical\b|\bsupercr[ií]tico\b", re.IGNORECASE),
    ],
}

_TOPOLOGY_PATTERNS: Dict[str, List[re.Pattern]] = {
    "vcrc": [re.compile(r"\bvcrc\b|vapou?r compression", re.IGNORECASE)],
    "brayton": [re.compile(r"\bbrayton\b", re.IGNORECASE)],
    "laes": [re.compile(r"\blaes\b|liquid air energy storage", re.IGNORECASE)],
    "tes_tank": [re.compile(r"\bTES\b|thermal energy storage tank", re.IGNORECASE)],
    "pcm_module": [re.compile(r"\bPCM\b|phase change material module", re.IGNORECASE)],
}


def _match_labels(text: str, patterns: Dict[str, List[re.Pattern]]) -> List[str]:
    labels: List[str] = []
    for label, regs in patterns.items():
        if any(r.search(text) for r in regs):
            labels.append(label)
    return sorted(labels)


def detect_domains(text: str) -> List[str]:
    return _match_labels(text, _DOMAIN_PATTERNS)


def detect_working_fluids(text: str) -> List[str]:
    return _match_labels(text, _FLUID_PATTERNS)


def detect_phase_regimes(text: str) -> List[str]:
    regimes = _match_labels(text, _PHASE_PATTERNS)
    if not regimes:
        return ["unknown"]
    return regimes


def detect_system_topologies(text: str) -> List[str]:
    return _match_labels(text, _TOPOLOGY_PATTERNS)


def _normalise_text(text: str) -> str:
    cleaned = re.sub(r"[\W_]+", " ", (text or "").lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _contains_term(text_low: str, norm_text: str, term: str) -> bool:
    t = str(term or "").strip().lower()
    if not t:
        return False
    if t in text_low:
        return True
    norm_t = _normalise_text(t)
    if not norm_t:
        return False
    return f" {norm_t} " in f" {norm_text} "


def evaluate_routing_guardrails(
    profile: PaperProfile | None,
    chunk_text: str,
    chunk_id: str,
    allowed_domains: List[str] | None,
    required_phase_regimes: List[str] | None,
    required_any_terms: List[str] | None,
    forbidden_terms: List[str] | None,
    max_foreign_domain_ratio: float = 1.0,
    routing_strictness: str = "soft",
) -> Tuple[float, List[str], List[ValidationFlag], bool]:
    """Evaluate domain/phase/term guardrails for one chunk-section pair.

    Returns: (eligibility_score, routing_notes, flags, is_eligible)
    """
    score = 1.0
    notes: List[str] = []
    flags: List[ValidationFlag] = []
    is_hard = (routing_strictness or "soft").lower() == "hard"

    allowed_domains = [d.lower() for d in (allowed_domains or [])]
    required_phase_regimes = [p.lower() for p in (required_phase_regimes or [])]
    required_any_terms = [t.lower() for t in (required_any_terms or [])]
    forbidden_terms = [t.lower() for t in (forbidden_terms or [])]
    max_foreign_domain_ratio = max(0.0, min(1.0, float(max_foreign_domain_ratio or 1.0)))

    profile_domains = [d.lower() for d in (profile.domains if profile else [])]
    profile_regimes = [r.lower() for r in (profile.phase_regimes if profile else [])]

    if allowed_domains:
        overlap = sorted(set(profile_domains).intersection(allowed_domains))
        if overlap:
            notes.append(f"domain_match={','.join(overlap)}")
        else:
            score *= 0.25
            notes.append("domain_mismatch")
            flags.append(ValidationFlag(
                flag_type="domain_mismatch",
                severity="high" if is_hard else "medium",
                message=(
                    "Perfil do paper não corresponde aos domínios permitidos da seção."
                ),
                related_ids=[chunk_id, profile.paper_id if profile else ""],
            ))
            if is_hard:
                return score, notes, flags, False

        if profile_domains:
            foreign = [d for d in set(profile_domains) if d not in set(allowed_domains)]
            foreign_ratio = len(foreign) / max(len(set(profile_domains)), 1)
            if foreign_ratio > max_foreign_domain_ratio:
                score *= 0.2
                notes.append(f"foreign_domain_ratio={foreign_ratio:.2f}")
                flags.append(ValidationFlag(
                    flag_type="foreign_domain_ratio",
                    severity="high" if is_hard else "medium",
                    message=(
                        "Perfil do paper contém excesso de domínios fora do escopo "
                        f"(ratio={foreign_ratio:.2f}, max={max_foreign_domain_ratio:.2f})."
                    ),
                    related_ids=[chunk_id, profile.paper_id if profile else ""],
                ))
                if is_hard:
                    return score, notes, flags, False

    if required_phase_regimes:
        overlap = sorted(set(profile_regimes).intersection(required_phase_regimes))
        if overlap:
            notes.append(f"phase_match={','.join(overlap)}")
        else:
            score *= 0.35
            notes.append("phase_mismatch")
            flags.append(ValidationFlag(
                flag_type="phase_mismatch",
                severity="high" if is_hard else "medium",
                message="Regime de fase do paper incompatível com a seção.",
                related_ids=[chunk_id, profile.paper_id if profile else ""],
            ))
            if is_hard:
                return score, notes, flags, False

    chunk_lower = chunk_text.lower()
    if required_any_terms:
        profile_blob = ""
        if profile:
            profile_blob = " ".join([
                " ".join(profile.keywords),
                " ".join(profile.working_fluids),
                " ".join(profile.system_topologies),
                " ".join(profile.domains),
            ]).lower()
        searchable = f"{chunk_lower}\n{profile_blob}"
        searchable_norm = _normalise_text(searchable)
        matched_required = [
            t for t in required_any_terms
            if _contains_term(searchable, searchable_norm, t)
        ]
        if matched_required:
            notes.append(f"required_any_terms={','.join(matched_required[:4])}")
        else:
            score *= 0.2
            notes.append("required_any_terms_missing")
            flags.append(ValidationFlag(
                flag_type="required_any_terms_missing",
                severity="high" if is_hard else "medium",
                message="Chunk não contém âncoras positivas mínimas exigidas pela seção.",
                related_ids=[chunk_id, profile.paper_id if profile else ""],
            ))
            if is_hard:
                return score, notes, flags, False

    chunk_norm = _normalise_text(chunk_lower)
    matched_forbidden = [
        t for t in forbidden_terms
        if _contains_term(chunk_lower, chunk_norm, t)
    ]
    if matched_forbidden:
        score *= 0.1
        notes.append(f"forbidden_terms={','.join(matched_forbidden[:4])}")
        flags.append(ValidationFlag(
            flag_type="forbidden_terms",
            severity="high",
            message="Chunk contém termos proibidos para esta seção.",
            related_ids=[chunk_id, profile.paper_id if profile else ""],
        ))
        if is_hard:
            return score, notes, flags, False

    return score, notes, flags, score >= 0.2
