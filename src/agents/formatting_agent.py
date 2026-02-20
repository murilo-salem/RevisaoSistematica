"""
formatting_agent.py — Agent for final document assembly and formatting.

Wraps ``post_processor.py`` for refinement and ``review_writer.py``
assembly functions for document construction.  Handles table marker
replacement, executive summary generation, and textual cleanup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from agents.base_agent import BaseAgent, Message, AgentResult

logger = logging.getLogger("systematic_review.agents.formatting")


class FormattingAgent(BaseAgent):
    """Assemble and format the final review document."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__("formatting", cfg)

    def process(self, message: Message) -> AgentResult:
        """Process a formatting/assembly request.

        Expected payload keys:
          - approved_sections: Dict[theme_key, text]
          - taxonomy_entries: List[Dict]
          - topic: str
          - synthesis_maps: Dict
          - chapter_theses: Dict
          - studies, chunks, tags: data (for post-processing)
        """
        approved = message.payload.get("approved_sections", {})
        entries = message.payload.get("taxonomy_entries", [])
        topic = message.payload.get("topic", "")
        synthesis_maps = message.payload.get("synthesis_maps", {})
        chapter_theses = message.payload.get("chapter_theses", {})
        research_agenda = message.payload.get("research_agenda", [])
        table_count = message.payload.get("table_count", 0)

        if not approved:
            return AgentResult(
                success=False,
                errors=["No approved sections to format"],
            )

        self.logger.info(
            "Formatting %d approved sections into final document", len(approved),
        )

        # ---- Build sections list for assembly ---------------------- #
        sections: List[Dict[str, Any]] = []
        for entry in entries:
            parent = entry.get("parent", "")
            folder = entry.get("folder", "")
            theme_key = f"{parent} / {folder}"

            text = approved.get(theme_key, "")
            sections.append({
                "folder": folder,
                "parent": parent,
                "prompt": entry.get("prompt", ""),
                "content": text,
            })

        # ---- Assemble the markdown document ----------------------- #
        from review_writer import _assemble_markdown

        md_text = _assemble_markdown(
            sections, topic,
            cfg=self.cfg,
            synthesis_maps=self._convert_synthesis_maps(synthesis_maps),
        )

        # ---- Post-process: textual cleanup (no LLM) --------------- #
        from post_processor import _textual_cleanup
        md_text = _textual_cleanup(md_text)

        # ---- Post-process: section refinement (LLM) --------------- #
        try:
            from post_processor import _split_sections, _refine_section
            from post_processor import _check_chapter_coherence
            from post_processor import _dedup_chapters
            from post_processor import _reassemble_markdown
            from post_processor import _validate_argumentation

            preamble, pp_sections = _split_sections(md_text)

            # Refine each section
            refined = []
            for sec in pp_sections:
                try:
                    refined.append(_refine_section(sec, self.cfg))
                except Exception:
                    refined.append(sec)  # keep original on error

            # Chapter coherence
            try:
                refined = _check_chapter_coherence(refined, self.cfg)
            except Exception as exc:
                self.logger.warning("Coherence check failed: %s", exc)

            # Deduplication
            try:
                refined = _dedup_chapters(refined, self.cfg)
            except Exception as exc:
                self.logger.warning("Dedup failed: %s", exc)

            # Argumentation validation
            if chapter_theses:
                try:
                    refined = _validate_argumentation(
                        refined, chapter_theses, self.cfg,
                    )
                except Exception as exc:
                    self.logger.warning("Argumentation validation failed: %s", exc)

            md_text = _reassemble_markdown(refined, preamble)

            # Final textual cleanup
            md_text = _textual_cleanup(md_text)

        except Exception as exc:
            self.logger.warning(
                "Post-processing partially failed: %s — using pre-processed text",
                exc,
            )

        # ---- Append research agenda ------------------------------ #
        if research_agenda:
            agenda_md = self._format_research_agenda(research_agenda)
            md_text += "\n\n" + agenda_md
            self.logger.info(
                "Research agenda appended (%d items)", len(research_agenda),
            )

        if table_count:
            self.logger.info("Tables included in document: %d", table_count)

        # ---- Write to file ---------------------------------------- #
        from utils import _resolve
        out_path = _resolve(self.cfg.get("paths", {}).get(
            "manuscript", "data/results/manuscript_v2.md",
        ))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_text, encoding="utf-8")

        self.logger.info("Final document written → %s (%d chars)", out_path, len(md_text))

        return AgentResult(
            success=True,
            data={
                "output_path": str(out_path),
                "n_sections": len(sections),
                "n_chars": len(md_text),
            },
        )

    def _convert_synthesis_maps(self, raw: Dict) -> Dict:
        """Try to convert back from dicts to SynthesisMap objects."""
        try:
            from evidence_synthesizer import SynthesisMap
            result = {}
            for key, value in raw.items():
                if isinstance(value, dict):
                    result[key] = SynthesisMap(**value)
                else:
                    result[key] = value
            return result
        except Exception:
            return raw

    @staticmethod
    def _format_research_agenda(
        agenda: List[Dict[str, Any]],
    ) -> str:
        """Format the consolidated research agenda as a markdown section."""
        lines = [
            "## Agenda de Pesquisas Futuras",
            "",
            "A seguir, apresenta-se a agenda consolidada de lacunas de pesquisa "
            "identificadas nesta revisão sistemática, ordenadas por prioridade.",
            "",
        ]

        # Group by priority
        alta = [g for g in agenda if g.get("priority", "").lower() == "alta"]
        media = [g for g in agenda if g.get("priority", "").lower() != "alta"]

        if alta:
            lines.append("### Prioridade Alta")
            lines.append("")
            for i, item in enumerate(alta, 1):
                desc = item.get("description", "")
                approach = item.get("suggested_approach", "")
                lines.append(f"{i}. **{desc}**")
                if approach:
                    lines.append(f"   - Abordagem sugerida: {approach}")
            lines.append("")

        if media:
            lines.append("### Prioridade Média")
            lines.append("")
            for i, item in enumerate(media, 1):
                desc = item.get("description", "")
                approach = item.get("suggested_approach", "")
                lines.append(f"{i}. **{desc}**")
                if approach:
                    lines.append(f"   - Abordagem sugerida: {approach}")

        return "\n".join(lines)
