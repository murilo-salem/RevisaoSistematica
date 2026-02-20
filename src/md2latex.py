"""
md2latex.py — Convert a Markdown systematic review to LaTeX.

Handles:
  - Section hierarchy (# → \\section, ## → \\subsection, ### → \\subsubsection)
  - Inline formatting (**bold**, *italic*)
  - Bullet lists → \\begin{itemize}
  - Special-character escaping for LaTeX
  - Language support (English / Portuguese via babel)

Usage::

    python src/md2latex.py --arquivo_md data/results/systematic_review.md
    python src/md2latex.py --arquivo_md review.md --output review.tex --lang pt
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("systematic_review.md2latex")

# ------------------------------------------------------------------ #
#  LaTeX template                                                      #
# ------------------------------------------------------------------ #

_LATEX_TEMPLATE = r"""\documentclass[12pt,a4paper]{{article}}

% --- Encoding & Fonts ---
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}

% --- Language ---
\usepackage[{babel_lang}]{{babel}}

% --- Layout ---
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{setspace}}
\onehalfspacing

% --- Figures, Tables, Maths ---
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}

% --- Lists ---
\usepackage{{enumitem}}

% --- Hyperlinks ---
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{{hyperref}}

% ==========================================================
\title{{{title}}}
\author{{Systematic Review Pipeline}}
\date{{\today}}

\begin{{document}}
\maketitle
\tableofcontents
\newpage

{body}

\end{{document}}
"""


# ------------------------------------------------------------------ #
#  LaTeX character escaping                                            #
# ------------------------------------------------------------------ #

_LATEX_SPECIAL = {
    '\\': r'\textbackslash{}',
    '&': r'\&',
    '%': r'\%',
    '$': r'\$',
    '#': r'\#',
    '_': r'\_',
    '{': r'\{',
    '}': r'\}',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
}

# Build a regex that matches any of the special characters
_SPECIAL_RE = re.compile(
    '(' + '|'.join(re.escape(k) for k in _LATEX_SPECIAL) + ')'
)


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in *text*."""
    return _SPECIAL_RE.sub(lambda m: _LATEX_SPECIAL[m.group()], text)


# ------------------------------------------------------------------ #
#  Inline formatting                                                   #
# ------------------------------------------------------------------ #

def _convert_inline(text: str) -> str:
    """Convert Markdown inline formatting to LaTeX.

    Handles **bold** → \\textbf{} and *italic* → \\textit{}.
    """
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'__(.+?)__', r'\\textbf{\1}', text)
    # Italic: *text* or _text_  (but not inside words with underscores)
    text = re.sub(r'(?<!\w)\*(.+?)\*(?!\w)', r'\\textit{\1}', text)
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\\textit{\1}', text)
    return text


# ------------------------------------------------------------------ #
#  Block detection                                                     #
# ------------------------------------------------------------------ #

def _is_heading(line: str) -> Optional[Tuple[int, str]]:
    """Detect a Markdown heading and return (level, text) or None."""
    m = re.match(r'^(#{1,6})\s+(.+)$', line)
    if m:
        return len(m.group(1)), m.group(2).strip()
    return None


def _is_bullet(line: str) -> Optional[str]:
    """Detect a Markdown bullet line and return the content, or None."""
    m = re.match(r'^\s*[-*+]\s+(.+)$', line)
    if m:
        return m.group(1).strip()
    return None


_REFERENCE_PATTERNS = re.compile(
    r'^#{1,3}\s*(references|referências|bibliography|bibliografia)',
    re.IGNORECASE,
)


def _is_reference_header(line: str) -> bool:
    """Detect a references section header."""
    return bool(_REFERENCE_PATTERNS.match(line))


# ------------------------------------------------------------------ #
#  Markdown → LaTeX converter                                          #
# ------------------------------------------------------------------ #

_HEADING_CMDS = {
    1: 'section',
    2: 'subsection',
    3: 'subsubsection',
    4: 'paragraph',
    5: 'subparagraph',
    6: 'subparagraph',
}


def _convert_md_to_latex(md_text: str) -> Tuple[str, str]:
    """Convert Markdown text to LaTeX body content.

    Returns (title, body) where title is extracted from the first ``#``
    heading if present.
    """
    lines = md_text.splitlines()
    output: List[str] = []
    title = "Systematic Review"
    in_list = False
    in_references = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # --- Skip horizontal rules ---
        if re.match(r'^-{3,}$|^\*{3,}$|^_{3,}$', line.strip()):
            i += 1
            continue

        # --- Headings ---
        heading = _is_heading(line)
        if heading:
            level, text = heading

            # Close any open list
            if in_list:
                output.append(r'\end{itemize}')
                in_list = False

            # Check for references section
            if _is_reference_header(line):
                in_references = True

            # Extract title from first H1
            if level == 1 and title == "Systematic Review":
                title = text
                i += 1
                continue

            cmd = _HEADING_CMDS.get(level, 'subparagraph')
            escaped_text = _escape_latex(text)
            output.append(f'\\{cmd}{{{escaped_text}}}')
            output.append('')
            i += 1
            continue

        # --- Bullet lists ---
        bullet = _is_bullet(line)
        if bullet:
            if not in_list:
                output.append(r'\begin{itemize}')
                in_list = True
            escaped = _escape_latex(bullet)
            escaped = _convert_inline(escaped)
            output.append(f'  \\item {escaped}')
            i += 1
            continue

        # Close list if we hit a non-bullet, non-empty line
        if in_list and line.strip():
            output.append(r'\end{itemize}')
            output.append('')
            in_list = False

        # --- Empty lines ---
        if not line.strip():
            output.append('')
            i += 1
            continue

        # --- Regular paragraph text ---
        # Collect contiguous text lines into a paragraph
        para_lines: List[str] = []
        while i < len(lines):
            cur = lines[i]
            # Stop at headings, bullets, empty lines, rules
            if not cur.strip():
                break
            if _is_heading(cur):
                break
            if _is_bullet(cur):
                break
            if re.match(r'^-{3,}$|^\*{3,}$|^_{3,}$', cur.strip()):
                break

            escaped = _escape_latex(cur)
            escaped = _convert_inline(escaped)
            para_lines.append(escaped)
            i += 1

        if para_lines:
            output.append(' '.join(para_lines))
            output.append('')
        continue

    # Close any open list at end
    if in_list:
        output.append(r'\end{itemize}')

    return title, '\n'.join(output)


# ------------------------------------------------------------------ #
#  Public API                                                          #
# ------------------------------------------------------------------ #

_BABEL_LANGS = {
    'en': 'english',
    'pt': 'brazilian',
}


def convert(
    md_path: str | Path,
    output_path: str | Path | None = None,
    lang: str = "pt",
) -> str:
    """Convert a Markdown file to LaTeX.

    Parameters
    ----------
    md_path : str or Path
        Path to the input Markdown file.
    output_path : str or Path, optional
        Path for the output .tex file.  Defaults to same name with .tex extension.
    lang : str
        Language code: ``"en"`` for English, ``"pt"`` for Portuguese.

    Returns
    -------
    str
        Path to the generated .tex file.
    """
    md_path = Path(md_path)
    if output_path is None:
        output_path = md_path.with_suffix('.tex')
    output_path = Path(output_path)

    md_text = md_path.read_text(encoding='utf-8')

    title, body = _convert_md_to_latex(md_text)

    babel_lang = _BABEL_LANGS.get(lang, 'english')

    # Escape title for LaTeX
    escaped_title = _escape_latex(title)

    latex = _LATEX_TEMPLATE.format(
        babel_lang=babel_lang,
        title=escaped_title,
        body=body,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding='utf-8')

    logger.info("LaTeX document saved → %s", output_path)
    return str(output_path)


# ------------------------------------------------------------------ #
#  CLI entry point                                                     #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Markdown systematic review to LaTeX",
    )
    parser.add_argument(
        "--arquivo_md",
        type=str,
        required=True,
        help="Path to the Markdown file to convert",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output .tex file path (default: same name with .tex extension)",
    )
    parser.add_argument(
        "--lang", "-l",
        type=str,
        default="pt",
        choices=["en", "pt"],
        help="Document language: 'en' for English, 'pt' for Portuguese (default: pt)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = convert(args.arquivo_md, args.output, args.lang)
    print(f"Done. LaTeX file saved to: {result}")


if __name__ == "__main__":
    main()
