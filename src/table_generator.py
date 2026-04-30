"""
table_generator.py — Automatic comparative table detection and generation.

Analyses evidence chunks for a theme and detects opportunities for
comparative tables (e.g. catalyst comparison, yield across feedstocks).
Generates well-formatted Markdown tables with captions and source citations.

Part of Épico 6: Table Curation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

from utils import Chunk, ChunkTag, StudyRecord, call_llm
from evidence_filtering import filter_and_rank_tags

logger = logging.getLogger("systematic_review.table_generator")


# ------------------------------------------------------------------ #
#  Data models                                                         #
# ------------------------------------------------------------------ #

class TableSpec(BaseModel):
    """Specification for a generated table."""
    table_id: str = ""
    caption: str = ""
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    source_studies: List[str] = Field(default_factory=list)  # PMIDs

    @field_validator("columns", "source_studies", mode="before")
    @classmethod
    def _coerce_str_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if not isinstance(value, list):
            return []
        out: List[str] = []
        for item in value:
            txt = str(item).strip()
            if txt:
                out.append(txt)
        return out

    @field_validator("rows", mode="before")
    @classmethod
    def _coerce_rows(cls, value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        rows: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                rows.append(dict(item))
        return rows


def _parse_table_json(raw_text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    candidates = [cleaned]
    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_match:
        candidates.append(obj_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            sanitized = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)
            try:
                parsed = json.loads(sanitized)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        except Exception:
            continue
    return {}


def _cell_to_text(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    text = str(value).strip()
    return text if text else "—"


# ------------------------------------------------------------------ #
#  Prompts                                                             #
# ------------------------------------------------------------------ #

_DETECT_TABLE_PROMPT = """\
You are an academic editor reviewing evidence for a systematic review section.

### Theme: {theme}

### Evidence excerpts
{evidence}

### Task
Determine whether this evidence contains COMPARATIVE QUANTITATIVE DATA
that would be better presented as a table than as prose.  Look for:
- Multiple studies reporting the same metric (yield, selectivity, cost, etc.)
- Comparisons across catalysts, feedstocks, temperatures, or methods
- Numerical data that readers would want to compare side by side

If a table is appropriate, return a JSON object:
{{
  "should_create_table": true,
  "caption": "Brief descriptive caption for the table",
  "columns": ["Column1", "Column2", ...],
  "rows": [
    {{"Column1": "value", "Column2": "value", ...}},
    ...
  ],
  "source_studies": ["(Author, Year)", ...]
}}

If NO table is appropriate, return:
{{
  "should_create_table": false
}}

Return ONLY the JSON, nothing else.
"""


# ------------------------------------------------------------------ #
#  Core logic                                                          #
# ------------------------------------------------------------------ #

def _gather_table_evidence(
    folder: str,
    parent: str,
    tags: List[ChunkTag],
    chunks_by_id: Dict[str, Chunk],
    studies_by_pmid: Dict[str, StudyRecord],
    top_k: int = 15,
    min_eligibility: float = 0.2,
    taxonomy_entry: Dict[str, Any] | None = None,
    profiles_by_paper: Dict[str, Dict[str, Any]] | None = None,
    high_flag_related_ids: Set[str] | None = None,
) -> str:
    """Gather evidence text for table detection."""
    relevant, _ = filter_and_rank_tags(
        tags=tags,
        chunks_by_id=chunks_by_id,
        folder=folder,
        parent=parent,
        top_k=top_k,
        min_eligibility=min_eligibility,
        taxonomy_entry=taxonomy_entry,
        profiles_by_paper=profiles_by_paper,
        high_flag_related_ids=high_flag_related_ids,
    )

    if not relevant:
        return ""

    parts: List[str] = []
    for tag in relevant:
        chunk = chunks_by_id.get(tag.chunk_id)
        if not chunk:
            continue
        study = studies_by_pmid.get(chunk.study_pmid)
        if study:
            label = f"({study.authors or chunk.study_pmid}, {study.year or 'n.d.'})"
        else:
            label = f"({chunk.study_pmid})"
        parts.append(f"--- {label} ---\n{chunk.text}")

    return "\n\n".join(parts)


def detect_table_opportunity(
    folder: str,
    parent: str,
    tags: List[ChunkTag],
    chunks_by_id: Dict[str, Chunk],
    studies_by_pmid: Dict[str, StudyRecord],
    cfg: Dict[str, Any],
    top_k: int = 15,
    taxonomy_entry: Dict[str, Any] | None = None,
    profiles_by_paper: Dict[str, Dict[str, Any]] | None = None,
    high_flag_related_ids: Set[str] | None = None,
) -> Optional[TableSpec]:
    """Detect whether a section benefits from a comparative table.

    Returns a TableSpec if a table should be created, None otherwise.
    """
    evidence = _gather_table_evidence(
        folder,
        parent,
        tags,
        chunks_by_id,
        studies_by_pmid,
        top_k,
        min_eligibility=cfg.get("routing", {}).get("min_eligibility_score", 0.2),
        taxonomy_entry=taxonomy_entry,
        profiles_by_paper=profiles_by_paper,
        high_flag_related_ids=high_flag_related_ids,
    )

    if not evidence or len(evidence) < 200:
        return None

    theme = f"{parent} / {folder}"
    prompt = _DETECT_TABLE_PROMPT.format(theme=theme, evidence=evidence)

    try:
        raw = call_llm(prompt, cfg)
        data = _parse_table_json(raw)
        if not data:
            return None

        if not data.get("should_create_table", False):
            return None

        table_id = f"table_{folder.lower().replace(' ', '_')}"
        return TableSpec(
            table_id=table_id,
            caption=data.get("caption", ""),
            columns=data.get("columns", []),
            rows=data.get("rows", []),
            source_studies=data.get("source_studies", []),
        )
    except (json.JSONDecodeError, Exception) as exc:
        logger.debug("Table detection failed for %s / %s: %s", parent, folder, exc)
        return None


def generate_markdown_table(table: TableSpec) -> str:
    """Convert a TableSpec into a formatted Markdown table with caption."""
    if not table.columns or not table.rows:
        return ""

    lines: List[str] = []

    # Caption
    if table.caption:
        lines.append(f"**Tabela: {table.caption}**")
        lines.append("")

    # Header row
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    lines.append(header)
    lines.append(separator)

    # Data rows
    for row in table.rows:
        cells = [_cell_to_text(row.get(col, "—")) for col in table.columns]
        lines.append("| " + " | ".join(cells) + " |")

    # Source attribution
    if table.source_studies:
        lines.append("")
        sources = "; ".join(table.source_studies)
        lines.append(f"*Fontes: {sources}*")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Table registry (in-memory for pipeline run)                         #
# ------------------------------------------------------------------ #

class TableRegistry:
    """Stores generated tables keyed by table_id for later insertion."""

    def __init__(self) -> None:
        self._tables: Dict[str, str] = {}  # table_id → markdown

    def register(self, table: TableSpec) -> str:
        """Register a table and return its markdown. Also returns the
        marker string that should be placed in the text."""
        md = generate_markdown_table(table)
        self._tables[table.table_id] = md
        return md

    def get_marker(self, table_id: str) -> str:
        """Return the marker string for a table."""
        return f"[TABLE: {table_id}]"

    def replace_markers(self, text: str) -> str:
        """Replace all [TABLE: id] markers with their markdown tables."""
        for table_id, md in self._tables.items():
            marker = self.get_marker(table_id)
            text = text.replace(marker, md)
        return text

    def size(self) -> int:
        return len(self._tables)
