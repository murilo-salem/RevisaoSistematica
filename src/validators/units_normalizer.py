"""units_normalizer.py — Unit extraction and normalization helpers.

Deterministic helpers used by the profiler and validators to:
- extract temperature/pressure values from raw text
- normalize to canonical SI-ish units used by the pipeline
  (temperature in K, pressure in bar)
"""

from __future__ import annotations

import re
from typing import List, Tuple

from utils import NumericRange


_TEMP_C_RE = re.compile(
    r"(?P<value>-?\d+(?:[\.,]\d+)?)\s*(?:°\s?C|º\s?C|deg\s?C|celsius)",
    re.IGNORECASE,
)
_TEMP_K_RE = re.compile(
    r"(?P<value>-?\d+(?:[\.,]\d+)?)\s*K\b",
    re.IGNORECASE,
)

_PRESS_BAR_RE = re.compile(
    r"(?P<value>\d+(?:[\.,]\d+)?)\s*bar\b",
    re.IGNORECASE,
)
_PRESS_MPA_RE = re.compile(
    r"(?P<value>\d+(?:[\.,]\d+)?)\s*MPa\b",
    re.IGNORECASE,
)
_PRESS_KPA_RE = re.compile(
    r"(?P<value>\d+(?:[\.,]\d+)?)\s*kPa\b",
    re.IGNORECASE,
)


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None


def normalize_temperature_to_k(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "c":
        return value + 273.15
    return value


def normalize_pressure_to_bar(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "mpa":
        return value * 10.0
    if unit == "kpa":
        return value / 100.0
    return value


def extract_temperatures_k(text: str) -> List[float]:
    values_k: List[float] = []

    for m in _TEMP_C_RE.finditer(text):
        v = _to_float(m.group("value"))
        if v is None:
            continue
        values_k.append(normalize_temperature_to_k(v, "c"))

    for m in _TEMP_K_RE.finditer(text):
        v = _to_float(m.group("value"))
        if v is None:
            continue
        values_k.append(normalize_temperature_to_k(v, "k"))

    return values_k


def extract_pressures_bar(text: str) -> List[float]:
    values_bar: List[float] = []

    for m in _PRESS_BAR_RE.finditer(text):
        v = _to_float(m.group("value"))
        if v is None:
            continue
        values_bar.append(normalize_pressure_to_bar(v, "bar"))

    for m in _PRESS_MPA_RE.finditer(text):
        v = _to_float(m.group("value"))
        if v is None:
            continue
        values_bar.append(normalize_pressure_to_bar(v, "mpa"))

    for m in _PRESS_KPA_RE.finditer(text):
        v = _to_float(m.group("value"))
        if v is None:
            continue
        values_bar.append(normalize_pressure_to_bar(v, "kpa"))

    return values_bar


def infer_temperature_range_k(text: str) -> NumericRange:
    values = extract_temperatures_k(text)
    if not values:
        return NumericRange(min=None, max=None, confidence=0.0)
    return NumericRange(
        min=min(values),
        max=max(values),
        confidence=min(1.0, len(values) / 5.0),
    )


def infer_pressure_range_bar(text: str) -> NumericRange:
    values = extract_pressures_bar(text)
    if not values:
        return NumericRange(min=None, max=None, confidence=0.0)
    return NumericRange(
        min=min(values),
        max=max(values),
        confidence=min(1.0, len(values) / 5.0),
    )
