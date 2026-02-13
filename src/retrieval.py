"""
retrieval.py — Search PubMed via NCBI Entrez and store results in SQLite.

Fetches study metadata (title, authors, journal, year, abstract, DOI)
in batches using efetch in XML mode for structured parsing.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

from Bio import Entrez
from tqdm import tqdm

from utils import StudyRecord, get_db_connection, init_database, now_iso, save_json

logger = logging.getLogger("systematic_review.retrieval")


# ------------------------------------------------------------------ #
#  XML parsing helpers                                                 #
# ------------------------------------------------------------------ #

def _text(element: ET.Element | None, default: str = "") -> str:
    """Safely extract text from an XML element."""
    if element is None:
        return default
    return (element.text or default).strip()


def _parse_article(article_el: ET.Element) -> StudyRecord:
    """Parse a single ``<PubmedArticle>`` element into a StudyRecord."""
    medline = article_el.find(".//MedlineCitation")
    article = medline.find("Article") if medline is not None else None

    pmid = _text(medline.find("PMID")) if medline is not None else ""

    title = ""
    if article is not None:
        title = _text(article.find("ArticleTitle"))

    # Authors
    authors_list: List[str] = []
    if article is not None:
        for author in article.findall(".//Author"):
            last = _text(author.find("LastName"))
            fore = _text(author.find("ForeName"))
            if last:
                authors_list.append(f"{last} {fore}".strip())
    authors = "; ".join(authors_list)

    # Journal
    journal = ""
    if article is not None:
        journal_el = article.find("Journal")
        if journal_el is not None:
            journal = _text(journal_el.find("Title"))

    # Year
    year = None
    if article is not None:
        year_el = article.find(".//PubDate/Year")
        if year_el is not None and year_el.text and year_el.text.isdigit():
            year = int(year_el.text)

    # Abstract
    abstract = ""
    if article is not None:
        abs_el = article.find(".//Abstract")
        if abs_el is not None:
            parts = [_text(t) for t in abs_el.findall("AbstractText")]
            abstract = " ".join(parts)

    # DOI
    doi = ""
    if article is not None:
        for eid in article.findall(".//ELocationID"):
            if eid.get("EIdType") == "doi":
                doi = _text(eid)
                break

    return StudyRecord(
        pmid=pmid,
        title=title,
        authors=authors,
        journal=journal,
        year=year,
        abstract=abstract,
        doi=doi,
        raw_text=f"{title}\n{abstract}",
    )


# ------------------------------------------------------------------ #
#  PubMed search + fetch                                               #
# ------------------------------------------------------------------ #

def search_pubmed(query: str, cfg: Dict[str, Any]) -> List[StudyRecord]:
    """Search PubMed with *query* and return a list of StudyRecords.

    Results are also persisted in the SQLite database for audit.
    """
    ret_cfg = cfg["retrieval"]
    Entrez.email = ret_cfg["email"]
    if ret_cfg.get("api_key"):
        Entrez.api_key = ret_cfg["api_key"]

    max_results = ret_cfg.get("max_results", 500)
    batch_size = ret_cfg.get("batch_size", 50)

    logger.info("Searching PubMed: %s (max %d)", query, max_results)

    # Step 1 — esearch to get IDs
    handle = Entrez.esearch(db=ret_cfg.get("db", "pubmed"), term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    ids: List[str] = record.get("IdList", [])
    logger.info("Found %d IDs", len(ids))

    if not ids:
        return []

    # Step 2 — efetch in batches
    init_database(cfg)
    conn = get_db_connection(cfg)
    studies: List[StudyRecord] = []

    for start in tqdm(range(0, len(ids), batch_size), desc="Fetching studies"):
        batch_ids = ids[start : start + batch_size]
        try:
            fetch_handle = Entrez.efetch(
                db=ret_cfg.get("db", "pubmed"),
                id=",".join(batch_ids),
                rettype="xml",
                retmode="xml",
            )
            xml_data = fetch_handle.read()
            fetch_handle.close()

            root = ET.fromstring(xml_data)
            for art_el in root.findall("PubmedArticle"):
                rec = _parse_article(art_el)
                studies.append(rec)

                # Persist to SQLite
                conn.execute(
                    """
                    INSERT OR IGNORE INTO raw_studies
                        (pmid, title, authors, journal, year, abstract, doi, raw_text, retrieved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rec.pmid, rec.title, rec.authors, rec.journal,
                        rec.year, rec.abstract, rec.doi, rec.raw_text, now_iso(),
                    ),
                )
        except Exception as exc:
            logger.error("Error fetching batch starting at %d: %s", start, exc)

    conn.commit()
    conn.close()

    # Save retrieval log
    save_json(
        {
            "query": query,
            "total_ids": len(ids),
            "fetched": len(studies),
            "timestamp": now_iso(),
        },
        "data/results/retrieval_log.json",
    )

    logger.info("Retrieved %d studies total", len(studies))
    return studies
