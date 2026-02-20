"""
utils.py — Shared utilities for the systematic-review pipeline.

Provides:
    • YAML config loading
    • Ollama LLM helper (streaming)
    • SQLite connection factory
    • JSON I/O helpers
    • Pydantic data models
    • Centralised audit logging
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from pydantic import BaseModel, Field

# ------------------------------------------------------------------ #
#  Path helpers                                                        #
# ------------------------------------------------------------------ #

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(rel: str) -> Path:
    """Resolve a path relative to the project root."""
    return PROJECT_ROOT / rel


# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load and return the YAML configuration dictionary.

    Also injects ``config_hash`` for reproducibility tracking.
    """
    cfg_path = Path(path) if path else _resolve("config/config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)

    # Compute a deterministic hash of the config for versioning
    raw = yaml.dump(cfg, sort_keys=True)
    cfg.setdefault("version", {})["config_hash"] = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return cfg


# ------------------------------------------------------------------ #
#  Logging / Audit                                                     #
# ------------------------------------------------------------------ #

def setup_logging(cfg: Dict[str, Any]) -> logging.Logger:
    """Configure file + console logging and return the root logger."""
    log_path = _resolve(cfg["paths"]["audit_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("systematic_review")
    logger.setLevel(logging.DEBUG)

    # File handler — full audit trail
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    ))

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ------------------------------------------------------------------ #
#  LLM helper                                                         #
# ------------------------------------------------------------------ #

def call_llm(prompt: str, cfg: Dict[str, Any]) -> str:
    """Send *prompt* to the configured Ollama model and return the full
    generated text.

    Uses the ``/api/generate`` endpoint with **streaming enabled** so that
    the HTTP read-timeout applies to the gap *between* chunks rather than
    to the total generation time.  This prevents timeouts when large models
    (e.g. 70B) take several minutes to produce a full response.

    Timeout behaviour (configurable via ``llm.timeout`` in config YAML):
      • connect timeout  — fixed at 30 s (time to reach Ollama)
      • read timeout      — ``llm.timeout`` seconds (max silence between
        any two streamed chunks; default 120 s)
    """
    llm_cfg = cfg["llm"]
    url = f"{llm_cfg['base_url']}/api/generate"

    # Build options dict — always include temperature + seed, then
    # conditionally add advanced sampling knobs when present in config.
    opts: Dict[str, Any] = {
        "temperature": llm_cfg["temperature"],
        "seed": llm_cfg["seed"],
    }
    for key in ("num_ctx", "top_p", "repeat_penalty"):
        if key in llm_cfg:
            opts[key] = llm_cfg[key]

    payload = {
        "model": llm_cfg["model"],
        "prompt": prompt,
        "stream": True,
        "options": opts,
    }

    logger = logging.getLogger("systematic_review.llm")
    logger.debug("LLM prompt (%d chars): %.200s…", len(prompt), prompt)

    read_timeout = llm_cfg.get("timeout", 120)
    connect_timeout = 30

    logger.info(
        "LLM request: model=%s, prompt_chars=%d, read_timeout=%ds",
        llm_cfg["model"], len(prompt), read_timeout,
    )

    try:
        resp = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(connect_timeout, read_timeout),
        )
        resp.raise_for_status()

        chunks: list[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = data.get("response", "")
            if token:
                chunks.append(token)
            # Ollama signals completion with "done": true
            if data.get("done", False):
                break

        text = "".join(chunks)
        logger.debug("LLM response (%d chars): %.200s…", len(text), text)
        return text
    except requests.RequestException as exc:
        logger.error("LLM call failed: %s", exc)
        raise


# ------------------------------------------------------------------ #
#  Database                                                            #
# ------------------------------------------------------------------ #

def get_db_connection(cfg: Dict[str, Any]) -> sqlite3.Connection:
    """Return a SQLite connection, creating the DB file if needed."""
    db_path = _resolve(cfg["paths"]["database"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_database(cfg: Dict[str, Any]) -> None:
    """Create the schema tables if they do not exist."""
    conn = get_db_connection(cfg)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_studies (
            pmid        TEXT PRIMARY KEY,
            title       TEXT,
            authors     TEXT,
            journal     TEXT,
            year        INTEGER,
            abstract    TEXT,
            doi         TEXT,
            raw_text    TEXT,
            retrieved_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS dedup_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            kept_pmid       TEXT,
            removed_pmid    TEXT,
            similarity      REAL,
            decided_at      TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS screening_decisions (
            pmid            TEXT PRIMARY KEY,
            decision        TEXT,       -- include | exclude | ambiguous
            confidence      REAL,
            justification   TEXT,
            decided_at      TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id        TEXT PRIMARY KEY,
            study_pmid      TEXT,
            text            TEXT,
            start_char      INTEGER,
            end_char        INTEGER,
            created_at      TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunk_tags (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id        TEXT,
            folder          TEXT,
            parent          TEXT,
            similarity      REAL,
            created_at      TEXT
        )
    """)

    conn.commit()
    conn.close()


# ------------------------------------------------------------------ #
#  JSON I/O                                                            #
# ------------------------------------------------------------------ #

def save_json(data: Any, rel_path: str) -> Path:
    """Serialise *data* to JSON at *rel_path* (relative to project root)."""
    full = _resolve(rel_path)
    full.parent.mkdir(parents=True, exist_ok=True)
    with open(full, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
    return full


def load_json(rel_path: str) -> Any:
    """Load JSON from *rel_path* (relative to project root)."""
    full = _resolve(rel_path)
    with open(full, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ #
#  Pydantic Data Models                                                #
# ------------------------------------------------------------------ #

class PICOModel(BaseModel):
    """Structured PICO question."""
    population: str = ""
    intervention: str = ""
    comparison: str = ""
    outcome: str = ""
    query: str = ""
    topic: str = ""


class StudyRecord(BaseModel):
    """A single study record from PubMed."""
    pmid: str
    title: str = ""
    authors: str = ""
    journal: str = ""
    year: Optional[int] = None
    abstract: str = ""
    doi: str = ""
    raw_text: str = ""


class ScreeningDecision(BaseModel):
    """Screening outcome for one study."""
    pmid: str
    decision: str  # include | exclude | ambiguous
    confidence: float = 0.0
    justification: str = ""


class ExtractionResult(BaseModel):
    """Data extracted from an included study."""
    pmid: str = ""
    study_design: str = ""
    sample_size: Optional[int] = None
    population: str = ""
    intervention: str = ""
    comparison: str = ""
    outcome: str = ""
    effect_size: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    p_value: Optional[float] = None
    notes: str = ""
    # ---- Critical metadata (Épico 1) ----
    study_scale: str = ""          # laboratory | pilot | industrial | field | simulation
    geographic_scope: str = ""     # country / region described in the study
    funding_source: str = ""       # funding body, if disclosed
    conflict_of_interest: str = "" # COI statement, if disclosed
    limitations: str = ""          # key limitations noted by the authors


class RiskOfBiasItem(BaseModel):
    """Risk-of-bias assessment for one domain."""
    domain: str
    rating: str = "unclear"  # low | unclear | high
    justification: str = ""


class RiskOfBiasResult(BaseModel):
    """Full risk-of-bias assessment for one study."""
    pmid: str = ""
    domains: List[RiskOfBiasItem] = Field(default_factory=list)
    overall_rating: str = ""  # low | moderate | high (computed heuristically)


class TaxonomyEntry(BaseModel):
    """One entry from the hierarchical review outline."""
    prompt: str
    folder: str
    parent: str


class Chunk(BaseModel):
    """A semantic chunk extracted from an article."""
    chunk_id: str
    study_pmid: str
    text: str
    start_char: int = 0
    end_char: int = 0
    # Enriched metadata from extraction / risk-of-bias (Épico 1)
    study_metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkTag(BaseModel):
    """Mapping of a chunk to a taxonomy section."""
    chunk_id: str
    folder: str
    parent: str
    similarity: float = 0.0


# ------------------------------------------------------------------ #
#  Misc helpers                                                        #
# ------------------------------------------------------------------ #

def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ #
#  System Capabilities                                                 #
# ------------------------------------------------------------------ #

def check_system_capabilities(cfg: Dict[str, Any] | None = None) -> Dict[str, bool]:
    """Check for optional ML dependencies (torch, cuda, sentence-transformers).
    
    Returns a dict with:
      - 'torch': bool
      - 'cuda': bool 
      - 'sentence_transformers': bool
      - 'gpu_name': strOrNone
    """
    caps = {
        "torch": False,
        "cuda": False,
        "sentence_transformers": False,
        "gpu_name": None
    }
    
    force_cpu = False
    if cfg and "system" in cfg:
        force_cpu = cfg["system"].get("force_cpu", False)

    # Check PyTorch
    try:
        import torch
        caps["torch"] = True
        if not force_cpu and torch.cuda.is_available():
            # Try a small allocation to ensure drivers are actually working
            try:
                _ = torch.zeros(1).cuda()
                caps["cuda"] = True
                caps["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception as exc:
                logging.getLogger("systematic_review.utils").warning(
                    "CUDA reported available but failed to allocate memory (%s). Disabling GPU.", exc
                )
                caps["cuda"] = False
    except (ImportError, NameError, AttributeError, OSError):
        pass
        
    # Check Sentence Transformers (requires PyTorch >= 2.4 usually)
    if caps["torch"]:
        try:
            import sentence_transformers
            caps["sentence_transformers"] = True
        except (ImportError, NameError, AttributeError, OSError):
            pass
            
    return caps
