"""
pdf_converter.py — Convert PDF files to plain text.

Supports two extraction backends:
  1. PyMuPDF (fitz) — primary extractor
  2. pdfminer.six   — fallback

Usage as CLI::

    python src/pdf_converter.py --input data/pdfs --output data/raw
    python src/pdf_converter.py --input data/pdfs --output data/raw --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

logger = logging.getLogger("systematic_review.pdf_converter")


# ------------------------------------------------------------------ #
#  Extraction backends                                                 #
# ------------------------------------------------------------------ #

def _extract_text_pymupdf(pdf_path: Path) -> Optional[str]:
    """Extract text from a PDF using PyMuPDF (fitz).

    Returns None on failure.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.debug("PyMuPDF not available")
        return None

    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text:
                pages.append(text)
        doc.close()
        if pages:
            return "\n\n".join(pages)
        return None
    except Exception as exc:
        logger.debug("PyMuPDF failed for %s: %s", pdf_path.name, exc)
        return None


def _extract_text_pdfminer(pdf_path: Path) -> Optional[str]:
    """Extract text from a PDF using pdfminer.six.

    Returns None on failure.
    """
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        logger.debug("pdfminer.six not available")
        return None

    try:
        text = extract_text(str(pdf_path))
        if text and text.strip():
            return text
        return None
    except Exception as exc:
        logger.debug("pdfminer failed for %s: %s", pdf_path.name, exc)
        return None


def extract_text(pdf_path: Path) -> Optional[str]:
    """Extract text from a PDF, trying PyMuPDF first then pdfminer.

    Returns the extracted text or None if both backends fail.
    """
    # Try PyMuPDF first (faster, better quality)
    text = _extract_text_pymupdf(pdf_path)
    if text:
        return text

    # Fallback to pdfminer
    text = _extract_text_pdfminer(pdf_path)
    if text:
        return text

    logger.warning("Could not extract text from %s", pdf_path.name)
    return None


# ------------------------------------------------------------------ #
#  Batch conversion                                                    #
# ------------------------------------------------------------------ #

def convert_pdfs(
    cfg: Dict[str, Any] | None = None,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> int:
    """Convert all PDFs in *input_dir* to .txt files in *output_dir*.

    Parameters
    ----------
    cfg : dict, optional
        Global config; reads ``paths.pdf_input`` and ``paths.raw_dir``.
    input_dir : str or Path, optional
        Override input directory.
    output_dir : str or Path, optional
        Override output directory.
    force : bool
        If True, re-convert files that already have a .txt output.

    Returns
    -------
    int
        Number of files successfully converted.
    """
    if input_dir is None:
        if cfg and "paths" in cfg:
            input_dir = Path(cfg["paths"].get("pdf_input", "data/pdfs"))
        else:
            input_dir = Path("data/pdfs")

    if output_dir is None:
        if cfg and "paths" in cfg:
            output_dir = Path(cfg["paths"].get("raw_dir", "data/raw"))
        else:
            output_dir = Path("data/raw")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDF files found in %s", input_dir)
        return 0

    logger.info("Converting %d PDFs from %s → %s", len(pdfs), input_dir, output_dir)

    converted = 0
    skipped = 0

    for pdf_path in tqdm(pdfs, desc="Converting PDFs"):
        out_path = output_dir / f"{pdf_path.stem}.txt"

        # Skip if already converted (unless --force)
        if out_path.exists() and not force:
            skipped += 1
            continue

        text = extract_text(pdf_path)
        if text:
            out_path.write_text(text, encoding="utf-8")
            converted += 1
            logger.debug("  ✓ %s → %s", pdf_path.name, out_path.name)
        else:
            logger.warning("  ✗ %s — no text extracted", pdf_path.name)

    logger.info(
        "Conversion complete: %d converted, %d skipped, %d failed",
        converted, skipped, len(pdfs) - converted - skipped,
    )
    return converted


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDF files to plain text (.txt)",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/pdfs",
        help="Directory containing PDF files (default: data/pdfs)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/raw",
        help="Output directory for .txt files (default: data/raw)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-conversion of already converted files",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = convert_pdfs(input_dir=args.input, output_dir=args.output, force=args.force)
    print(f"\nDone. {n} files converted.")


if __name__ == "__main__":
    main()
