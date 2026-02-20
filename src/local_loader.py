"""
local_loader.py — Load pre-existing articles from data/raw/ files.

Supports .txt, .csv, .json, and .bib files.  Each record is converted
to a StudyRecord so all downstream modules work unchanged.

Expected file formats:
  • .txt   — one article per file (filename used as ID)
  • .json  — list of objects with keys: id/pmid, title, abstract, …
  • .csv   — columns: id, title, abstract (+ optional: authors, year, …)
  • .bib   — BibTeX entries parsed into StudyRecords
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from utils import StudyRecord, get_db_connection, init_database, now_iso, _resolve

logger = logging.getLogger("systematic_review.local_loader")


# ------------------------------------------------------------------ #
#  Filename metadata extraction                                        #
# ------------------------------------------------------------------ #

def _parse_filename_metadata(stem: str) -> Dict[str, Any]:
    """Extract author and year from a filename stem.

    Examples
    --------
    >>> _parse_filename_metadata('A._Saravanan_2019')
    {'author': 'A. Saravanan', 'year': 2019}
    >>> _parse_filename_metadata('Brandon_Han_Hoe_Goh_2019')
    {'author': 'Brandon Han Hoe Goh', 'year': 2019}
    >>> _parse_filename_metadata('unknown_file')
    {'author': 'unknown file', 'year': None}
    """
    # Try to find a trailing 4-digit year
    m = re.match(r'^(.+?)_?(\d{4})$', stem)
    if m:
        raw_author = m.group(1)
        year = int(m.group(2))
    else:
        raw_author = stem
        year = None

    # Replace underscores with spaces, fix "A." patterns
    author = raw_author.replace('_', ' ').strip()
    # Collapse double spaces
    author = re.sub(r'\s+', ' ', author)

    return {'author': author, 'year': year}


def _format_citation(author: str, year: int | None) -> str:
    """Format an in-text citation.

    >>> _format_citation('A. Saravanan', 2019)
    '(A. Saravanan, 2019)'
    >>> _format_citation('A. Saravanan', None)
    '(A. Saravanan)'
    """
    if year:
        return f"({author}, {year})"
    return f"({author})"


# ------------------------------------------------------------------ #
#  Format-specific loaders                                             #
# ------------------------------------------------------------------ #

def _load_txt_files(raw_dir: Path) -> List[StudyRecord]:
    """Load .txt files — each file becomes one study."""
    records: List[StudyRecord] = []
    for txt in sorted(raw_dir.rglob("*.txt")):
        content = txt.read_text(encoding="utf-8", errors="replace")
        # Try to extract title from first non-empty line
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        title = lines[0] if lines else txt.stem
        abstract = "\n".join(lines[1:]) if len(lines) > 1 else content

        meta = _parse_filename_metadata(txt.stem)
        records.append(StudyRecord(
            pmid=txt.stem,
            title=title,
            abstract=abstract,
            authors=meta["author"],
            year=meta["year"],
            raw_text=content,
        ))
    return records


def _load_json_file(path: Path) -> List[StudyRecord]:
    """Load a JSON file — expects a list of study objects."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict):
        data = [data]

    records: List[StudyRecord] = []
    for item in data:
        rec = StudyRecord(
            pmid=str(item.get("id", item.get("pmid", ""))),
            title=item.get("title", ""),
            authors=item.get("authors", ""),
            journal=item.get("journal", ""),
            year=item.get("year"),
            abstract=item.get("abstract", ""),
            doi=item.get("doi", ""),
            raw_text=item.get("raw_text", f"{item.get('title', '')} {item.get('abstract', '')}"),
        )
        records.append(rec)
    return records


def _load_csv_file(path: Path) -> List[StudyRecord]:
    """Load a CSV file with headers: id,title,abstract,…"""
    records: List[StudyRecord] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rec = StudyRecord(
                pmid=row.get("id", row.get("pmid", "")),
                title=row.get("title", ""),
                authors=row.get("authors", ""),
                journal=row.get("journal", ""),
                year=int(row["year"]) if row.get("year", "").isdigit() else None,
                abstract=row.get("abstract", ""),
                doi=row.get("doi", ""),
                raw_text=f"{row.get('title', '')} {row.get('abstract', '')}",
            )
            records.append(rec)
    return records


def _load_bib_file(path: Path) -> List[StudyRecord]:
    """Minimal BibTeX parser — extracts title, author, year, abstract."""
    content = path.read_text(encoding="utf-8", errors="replace")
    entries = re.split(r"@\w+\{", content)

    records: List[StudyRecord] = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Extract the cite key
        cite_key = entry.split(",")[0].strip() if "," in entry else "unknown"

        def _field(name: str) -> str:
            m = re.search(rf"{name}\s*=\s*\{{([^}}]*)\}}", entry, re.IGNORECASE)
            return m.group(1).strip() if m else ""

        records.append(StudyRecord(
            pmid=cite_key,
            title=_field("title"),
            authors=_field("author"),
            journal=_field("journal"),
            year=int(_field("year")) if _field("year").isdigit() else None,
            abstract=_field("abstract"),
            doi=_field("doi"),
            raw_text=f"{_field('title')} {_field('abstract')}",
        ))

    return records


# ------------------------------------------------------------------ #
#  Taxonomy loader                                                     #
# ------------------------------------------------------------------ #

def _parse_taxonomy_md(content: str) -> Dict[str, Any]:
    """Parse a Markdown taxonomy file into a dictionary.

    Format:
    # Topic: ...
    ## PICO
    - Population: ...
    ## Keywords
    - keyword1
    ...
    """
    lines = content.splitlines()
    taxonomy = {"pico": {}, "keywords": [], "inclusion_criteria": [], "exclusion_criteria": [], "classification_rules": {"include_if_any": [], "exclude_if_any": []}}
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Topic (H1)
        if line.startswith("# "):
            val = line[2:].strip()
            if val.lower().startswith("topic:"):
                val = val[6:].strip()
            taxonomy["topic"] = val
            continue
            
        # Sections (H2)
        if line.startswith("## "):
            sec = line[3:].lower().strip()
            if "pico" in sec: current_section = "pico"
            elif "keyword" in sec: current_section = "keywords"
            elif "criteria" in sec or "rule" in sec:
                if "inclusion" in sec: current_section = "inclusion_criteria"
                elif "exclusion" in sec: current_section = "exclusion_criteria"
                else: current_section = "classification_rules"
            else: current_section = None
            continue
            
        # Lists
        if line.startswith("- ") or line.startswith("* ") or (line[0].isdigit() and line[1:3] == ". "):
            # strip bullet
            item = re.sub(r'^(\s*[-*+]|\s*\d+\.)\s+', '', line).strip()
            
            if current_section == "pico":
                if ":" in item:
                    k, v = item.split(":", 1)
                    taxonomy["pico"][k.strip().lower()] = v.strip()
                    
            elif current_section in ["keywords", "inclusion_criteria", "exclusion_criteria"]:
                taxonomy[current_section].append(item)
                
            elif current_section == "classification_rules":
                if "include" in item.lower() and "exclude" not in item.lower():
                    taxonomy["classification_rules"]["include_if_any"].append(item)
                else:
                    taxonomy["classification_rules"]["exclude_if_any"].append(item)
                    
    return taxonomy


def load_taxonomy(path: str | None = None) -> Dict[str, Any]:
    """Load the taxonomy from a JSON or Markdown file.

    Default path: ``config/taxonomia.json``
    """
    if path:
        tax_path = Path(path)
    else:
        # Try finding json or md in config
        base = _resolve("config/taxonomia")
        if base.with_suffix(".json").exists():
            tax_path = base.with_suffix(".json")
        elif base.with_suffix(".md").exists():
            tax_path = base.with_suffix(".md")
        else:
            tax_path = base.with_suffix(".json")

    if not tax_path.exists():
        logger.warning("Taxonomy not found at %s — returning empty", tax_path)
        return {}
        
    try:
        if tax_path.suffix.lower() == ".md":
            content = tax_path.read_text(encoding="utf-8", errors="replace")
            return _parse_taxonomy_md(content)
        else:
            with open(tax_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                # Support list-of-prompts format: [{prompt, folder, parent}, ...]
                if isinstance(data, list):
                    # Check if it looks like an outline (has prompt/folder/parent keys)
                    if data and isinstance(data[0], dict) and "prompt" in data[0]:
                        # Build hierarchical topic from parent categories
                        parents = sorted(set(item.get("parent", "") for item in data))
                        topic = " / ".join(parents) if parents else "Review Outline"
                        logger.info("Loaded outline taxonomy with %d entries (%d chapters)", len(data), len(parents))
                        return {
                            "type": "outline",
                            "topic": topic,
                            "entries": data,
                            "pico": {},
                            "keywords": [],
                        }
                    else:
                        logger.warning("Taxonomy file %s contains a list but not in outline format.", tax_path.name)
                        return {
                            "topic": "Unknown Topic",
                            "pico": {},
                            "keywords": [],
                            "_raw_list": data,
                        }
                return data
    except Exception as exc:
        logger.error("Failed to load taxonomy %s: %s", tax_path, exc)
        return {}


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

def load_local_studies(cfg: Dict[str, Any]) -> List[StudyRecord]:
    """Scan ``data/raw/`` for articles in supported formats and return
    a list of StudyRecords.

    Supported: .txt, .json, .csv, .bib

    All loaded records are also persisted to the SQLite database.
    """
    raw_dir = _resolve("data/raw")
    if not raw_dir.exists():
        logger.error("Raw data directory not found: %s", raw_dir)
        return []

    all_records: List[StudyRecord] = []

    # .txt files
    txt_records = _load_txt_files(raw_dir)
    if txt_records:
        logger.info("Loaded %d studies from .txt files", len(txt_records))
        all_records.extend(txt_records)

    # .json files
    for jf in sorted(raw_dir.rglob("*.json")):
        try:
            recs = _load_json_file(jf)
            logger.info("Loaded %d studies from %s", len(recs), jf.name)
            all_records.extend(recs)
        except Exception as exc:
            logger.error("Error loading %s: %s", jf.name, exc)

    # .csv files
    for cf in sorted(raw_dir.rglob("*.csv")):
        try:
            recs = _load_csv_file(cf)
            logger.info("Loaded %d studies from %s", len(recs), cf.name)
            all_records.extend(recs)
        except Exception as exc:
            logger.error("Error loading %s: %s", cf.name, exc)

    # .bib files
    for bf in sorted(raw_dir.rglob("*.bib")):
        try:
            recs = _load_bib_file(bf)
            logger.info("Loaded %d studies from %s", len(recs), bf.name)
            all_records.extend(recs)
        except Exception as exc:
            logger.error("Error loading %s: %s", bf.name, exc)

    logger.info("Total local studies loaded: %d", len(all_records))

    # Persist to database
    if all_records:
        init_database(cfg)
        conn = get_db_connection(cfg)
        for rec in all_records:
            conn.execute(
                """
                INSERT OR IGNORE INTO raw_studies
                    (pmid, title, authors, journal, year, abstract, doi, raw_text, retrieved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (rec.pmid, rec.title, rec.authors, rec.journal,
                 rec.year, rec.abstract, rec.doi, rec.raw_text, now_iso()),
            )
        conn.commit()
        conn.close()
        logger.info("All records persisted to database")

    return all_records
