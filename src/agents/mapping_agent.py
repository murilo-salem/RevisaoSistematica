"""
mapping_agent.py — Agent for evidence mapping and preliminary analysis.

Wraps ``content_analyzer.py`` for chunking/embedding/tag mapping and
adds a preliminary consensus/contradiction detection step.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from agents.base_agent import BaseAgent, Message, AgentResult
from agents.models import ContradictionDetail, ThemeEvidence

logger = logging.getLogger("systematic_review.agents.mapping")


# ------------------------------------------------------------------ #
#  Contradiction detection prompt                                      #
# ------------------------------------------------------------------ #

_CONTRADICTION_PROMPT = """\
Analyse the two excerpts below from scientific articles on biodiesel.
Do they present contradictory results or conclusions?

Excerpt A ({author_a}):
{chunk_a}

Excerpt B ({author_b}):
{chunk_b}

If they contradict, respond with a JSON object:
{{"contradiction": true, "point": "...", "possible_cause": "..."}}

If they do NOT contradict, respond:
{{"contradiction": false}}

Return ONLY JSON.
"""


class MappingAgent(BaseAgent):
    """Map evidence to taxonomy themes with preliminary contradiction analysis."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__("mapping", cfg)

    def process(self, message: Message) -> AgentResult:
        """Process a mapping request.

        Expected payload keys:
          - studies: List[StudyRecord]
          - taxonomy_entries: List[Dict]
          - chunks: optional pre-computed chunks
          - tags: optional pre-computed tags
        """
        studies = message.payload.get("studies", [])
        entries = message.payload.get("taxonomy_entries", [])
        prior_chunks = message.payload.get("chunks")
        prior_tags = message.payload.get("tags")

        # ---- Chunking / embedding / tagging ----------------------- #
        if prior_chunks is not None and prior_tags is not None:
            self.logger.info(
                "Using pre-computed chunks (%d) and tags (%d)",
                len(prior_chunks), len(prior_tags),
            )
            all_chunks = prior_chunks
            all_tags = prior_tags
            coverage = {}
        else:
            self.logger.info("Running content analysis for %d studies", len(studies))
            from content_analyzer import analyze_and_chunk
            all_chunks, all_tags, coverage = analyze_and_chunk(
                studies, entries, self.cfg,
            )

        # Build lookups
        from utils import Chunk, ChunkTag, StudyRecord
        chunks_by_id: Dict[str, Any] = {c.chunk_id: c for c in all_chunks}
        studies_by_pmid: Dict[str, Any] = {s.pmid: s for s in studies}

        # ---- Build ThemeEvidence per theme ------------------------- #
        from collections import defaultdict

        section_tags: Dict[str, List] = defaultdict(list)
        for tag in all_tags:
            key = f"{tag.parent} / {tag.folder}"
            section_tags[key].append(tag)

        theme_evidence: Dict[str, ThemeEvidence] = {}

        for entry in entries:
            parent = entry["parent"]
            folder = entry["folder"]
            key = f"{parent} / {folder}"

            tags_for_theme = section_tags.get(key, [])
            tags_for_theme.sort(key=lambda t: t.similarity, reverse=True)
            top_tags = tags_for_theme[:15]

            # Collect unique studies
            study_pmids = set()
            chunk_ids = []
            for tag in top_tags:
                chunk = chunks_by_id.get(tag.chunk_id)
                if chunk:
                    study_pmids.add(chunk.study_pmid)
                    chunk_ids.append(tag.chunk_id)

            theme_evidence[key] = ThemeEvidence(
                theme_id=key,
                parent=parent,
                folder=folder,
                chunk_ids=chunk_ids,
                coverage_stats={
                    "n_chunks": len(top_tags),
                    "n_studies": len(study_pmids),
                    "avg_similarity": round(
                        sum(t.similarity for t in top_tags) / max(len(top_tags), 1), 3,
                    ),
                },
            )

        # ---- Preliminary contradiction detection (LLM) ------------ #
        self._detect_contradictions(
            theme_evidence, chunks_by_id, studies_by_pmid,
        )

        self.logger.info(
            "Mapping complete: %d chunks, %d tags, %d themes",
            len(all_chunks), len(all_tags), len(theme_evidence),
        )

        return AgentResult(
            success=True,
            data={
                "chunks": all_chunks,
                "tags": all_tags,
                "coverage": coverage,
                "theme_evidence": theme_evidence,
            },
        )

    # ---- Contradiction detection ---------------------------------- #

    def _detect_contradictions(
        self,
        theme_evidence: Dict[str, ThemeEvidence],
        chunks_by_id: Dict[str, Any],
        studies_by_pmid: Dict[str, Any],
    ) -> None:
        """Detect contradictions between top chunk pairs per theme."""
        for key, te in theme_evidence.items():
            if len(te.chunk_ids) < 2:
                continue

            # Compare first 3 pairs for efficiency
            pairs_checked = 0
            for i in range(min(len(te.chunk_ids), 4)):
                for j in range(i + 1, min(len(te.chunk_ids), 4)):
                    if pairs_checked >= 3:
                        break
                    chunk_a = chunks_by_id.get(te.chunk_ids[i])
                    chunk_b = chunks_by_id.get(te.chunk_ids[j])
                    if not chunk_a or not chunk_b:
                        continue

                    study_a = studies_by_pmid.get(chunk_a.study_pmid)
                    study_b = studies_by_pmid.get(chunk_b.study_pmid)
                    if chunk_a.study_pmid == chunk_b.study_pmid:
                        continue  # same study

                    author_a = (study_a.authors if study_a else chunk_a.study_pmid)
                    author_b = (study_b.authors if study_b else chunk_b.study_pmid)

                    try:
                        prompt = _CONTRADICTION_PROMPT.format(
                            author_a=author_a,
                            chunk_a=chunk_a.text[:500],
                            author_b=author_b,
                            chunk_b=chunk_b.text[:500],
                        )
                        raw = self.call_llm(prompt)
                        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
                        data = json.loads(cleaned)

                        if data.get("contradiction", False):
                            te.preliminary_contradictions.append(
                                ContradictionDetail(
                                    point=data.get("point", ""),
                                    study_a=author_a,
                                    study_b=author_b,
                                    possible_cause=data.get("possible_cause", ""),
                                )
                            )
                    except (json.JSONDecodeError, Exception) as exc:
                        self.logger.debug(
                            "Contradiction check failed for %s: %s", key, exc,
                        )

                    pairs_checked += 1
