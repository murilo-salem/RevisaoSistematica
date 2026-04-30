"""
debate_agent.py — Optional agent for structured debate on controversial themes.

When a theme has low robustness and multiple contradictions, this agent
creates a structured debate between pro/con positions, moderated by a
synthesiser, producing a balanced academic section.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agents.base_agent import BaseAgent, Message, AgentResult

logger = logging.getLogger("systematic_review.agents.debate")


# ------------------------------------------------------------------ #
#  Prompts                                                             #
# ------------------------------------------------------------------ #

_ADVOCATE_PROMPT = """\
You are a scientific advocate presenting the {position} position on
the following scientific question.

### Theme: {theme}

### Your position: {position_description}

### Contradictions to address
{contradictions}

### Task
Write a concise but rigorous argument (2–3 paragraphs) defending your
position using only the evidence described above.  Be specific about
methodology and results.  Cite studies where possible.

Write in {language_name}.
"""

_MODERATOR_PROMPT = """\
You are a scientific moderator synthesising a debate on a controversial
topic in a systematic literature review.

### Theme: {theme}

### Position A (supportive)
{position_a}

### Position B (critical/opposing)
{position_b}

### Task
Write a balanced SYNTHESIS section (3–4 paragraphs) that:
1. Presents both perspectives fairly
2. Explains WHY the evidence is contradictory (methodology, scale,
   conditions, etc.)
3. Identifies what ADDITIONAL evidence would be needed to resolve the
   controversy
4. Uses hedging language appropriately (e.g. "embora...", "sob
   certas condições...", "os resultados sugerem...")

Write in {language_name}.  Format as academic prose suitable for
insertion into a systematic review.

### Output
Start with a subsection heading:
#### Debate: {theme_short}
"""


class DebateAgent(BaseAgent):
    """Structured debate for controversial topics."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__("debate", cfg)

    def process(self, message: Message) -> AgentResult:
        """Process a debate request.

        Expected payload keys:
          - theme_key: str
          - contradictions: list of contradiction dicts
          - current_draft: the current approved section text
          - entry: taxonomy entry dict
        """
        theme_key = message.payload.get("theme_key", "")
        contradictions = message.payload.get("contradictions", [])
        current_draft = message.payload.get("current_draft", "")
        entry = message.payload.get("entry", {})

        if not contradictions:
            return AgentResult(
                success=True,
                data={"debate_section": ""},
            )

        self.logger.info("Debating theme: %s (%d contradictions)", theme_key, len(contradictions))

        lang = self.cfg.get("review", {}).get("language", "pt")
        lang_names = {
            "pt": "português acadêmico formal",
            "en": "formal academic English",
            "es": "español académico formal",
        }
        language_name = lang_names.get(lang, lang_names["pt"])

        # Format contradictions
        contra_text = "\n".join(
            f"- {c.get('point', '')}: "
            f"{c.get('study_a', '')} vs {c.get('study_b', '')} — "
            f"{c.get('possible_cause', '')}"
            for c in contradictions
        )

        theme_short = entry.get("folder", theme_key.split(" / ")[-1])

        # ---- Position A: Supportive ------------------------------- #
        try:
            position_a_text = self.call_llm(
                _ADVOCATE_PROMPT.format(
                    position="supportive",
                    position_description=(
                        "The results are generally positive and the methodology "
                        "is adequate. Contradictions can be explained by "
                        "differences in experimental conditions."
                    ),
                    theme=theme_key,
                    contradictions=contra_text,
                    language_name=language_name,
                )
            )
        except Exception as exc:
            self.logger.warning("Position A generation failed: %s", exc)
            position_a_text = "(Position A could not be generated)"

        # ---- Position B: Critical --------------------------------- #
        try:
            position_b_text = self.call_llm(
                _ADVOCATE_PROMPT.format(
                    position="critical",
                    position_description=(
                        "The results are unreliable, the methodology has "
                        "significant limitations, and the contradictions "
                        "undermine the conclusions drawn."
                    ),
                    theme=theme_key,
                    contradictions=contra_text,
                    language_name=language_name,
                )
            )
        except Exception as exc:
            self.logger.warning("Position B generation failed: %s", exc)
            position_b_text = "(Position B could not be generated)"

        # ---- Moderator synthesis ---------------------------------- #
        try:
            debate_section = self.call_llm(
                _MODERATOR_PROMPT.format(
                    theme=theme_key,
                    position_a=position_a_text,
                    position_b=position_b_text,
                    language_name=language_name,
                    theme_short=theme_short,
                )
            )
        except Exception as exc:
            self.logger.warning("Moderator synthesis failed: %s", exc)
            debate_section = ""

        self.logger.info(
            "Debate complete for %s: %d chars", theme_key, len(debate_section),
        )

        return AgentResult(
            success=True,
            data={"debate_section": debate_section},
        )
