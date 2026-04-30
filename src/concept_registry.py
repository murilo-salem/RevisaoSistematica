"""
concept_registry.py — Thread-safe registry of concepts already explained.

Prevents the same concept from being re-explained in multiple sections
by tracking which concepts have been covered and injecting the list
into writing prompts.

Part of Épico 4: Structural Redundancy Elimination.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger("systematic_review.concept_registry")


# ------------------------------------------------------------------ #
#  Normalisation helpers                                               #
# ------------------------------------------------------------------ #

def _normalise(text: str) -> str:
    """Normalise a concept string for deduplication.

    - Lowercase
    - Strip accents (NFD + strip combining marks)
    - Collapse whitespace
    - Strip leading/trailing whitespace
    """
    text = text.lower().strip()
    # Strip accents
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


# ------------------------------------------------------------------ #
#  Registry                                                            #
# ------------------------------------------------------------------ #

class ConceptRegistry:
    """Thread-safe registry of concepts already covered in the review.

    Usage::

        registry = ConceptRegistry()
        registry.register("catálise heterogênea", "Alkaline Catalysis")
        assert registry.already_covered("catalise heterogenea")  # normalised match
        covered = registry.get_covered_concepts()  # for prompt injection
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # _concepts: normalised_key → {"original": str, "section": str}
        self._concepts: Dict[str, Dict[str, str]] = {}

    # ---- Core API ------------------------------------------------- #

    def register(self, concept: str, section: str) -> None:
        """Mark *concept* as already explained in *section*."""
        key = _normalise(concept)
        if not key:
            return
        with self._lock:
            if key not in self._concepts:
                self._concepts[key] = {
                    "original": concept.strip(),
                    "section": section,
                }

    def register_many(self, concepts: List[str], section: str) -> None:
        """Register multiple concepts at once."""
        for c in concepts:
            self.register(c, section)

    def already_covered(self, concept: str) -> bool:
        """Check whether *concept* (or a normalised variant) is registered."""
        key = _normalise(concept)
        with self._lock:
            return key in self._concepts

    def get_covered_concepts(self) -> List[str]:
        """Return a list of original concept strings (sorted)."""
        with self._lock:
            return sorted(v["original"] for v in self._concepts.values())

    def get_covered_with_sections(self) -> List[Dict[str, str]]:
        """Return concepts with the section where they were first introduced."""
        with self._lock:
            return sorted(
                [{"concept": v["original"], "section": v["section"]}
                 for v in self._concepts.values()],
                key=lambda x: x["concept"],
            )

    def size(self) -> int:
        """Number of registered concepts."""
        with self._lock:
            return len(self._concepts)

    # ---- Persistence ---------------------------------------------- #

    def to_json(self, path: str | Path) -> None:
        """Save registry to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "concepts": list(self._concepts.values()),
                "count": len(self._concepts),
            }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Concept registry saved → %s (%d concepts)", path, data["count"])

    @classmethod
    def from_json(cls, path: str | Path) -> "ConceptRegistry":
        """Load registry from a previously saved JSON file."""
        path = Path(path)
        registry = cls()
        if not path.exists():
            return registry
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for entry in data.get("concepts", []):
                registry.register(entry["original"], entry.get("section", "unknown"))
            logger.info("Loaded %d concepts from %s", registry.size(), path)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Could not load concept registry from %s: %s", path, exc)
        return registry

    def __repr__(self) -> str:
        return f"ConceptRegistry(n={self.size()})"
