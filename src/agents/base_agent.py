"""
base_agent.py — Base classes for the multi-agent system.

Defines the foundational abstractions:
  • Message       — inter-agent communication envelope
  • AgentResult   — standardised agent output
  • BaseAgent     — abstract base that all agents extend
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils import call_llm, now_iso

logger = logging.getLogger("systematic_review.agents.base")


# ------------------------------------------------------------------ #
#  Message protocol                                                    #
# ------------------------------------------------------------------ #

@dataclass
class Message:
    """Inter-agent communication envelope.

    Attributes
    ----------
    task : str
        The task identifier (e.g. "extract", "write_section", "review").
    payload : dict
        Task-specific data — contents vary per agent.
    source : str
        Name of the originating agent (or "coordinator").
    timestamp : str
        ISO-8601 timestamp of message creation.
    iteration : int
        Current iteration number (for retry loops).
    feedback : str
        Feedback from a previous iteration (e.g. from review_agent).
    """

    task: str
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: str = ""
    iteration: int = 0
    feedback: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = now_iso()


# ------------------------------------------------------------------ #
#  Agent result                                                        #
# ------------------------------------------------------------------ #

@dataclass
class AgentResult:
    """Standardised return type from every agent.

    Attributes
    ----------
    success : bool
        Whether the agent completed without fatal errors.
    data : dict
        Agent-specific output data.
    errors : list[str]
        Any error messages or warnings.
    metrics : dict
        Performance metrics (elapsed_s, tokens, etc.).
    """

    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------ #
#  Base agent                                                          #
# ------------------------------------------------------------------ #

class BaseAgent(ABC):
    """Abstract base class for all agents.

    Subclasses must implement ``process(message) -> AgentResult``.

    Parameters
    ----------
    name : str
        Human-readable agent identifier.
    cfg : dict
        Full pipeline configuration dictionary.
    """

    def __init__(self, name: str, cfg: Dict[str, Any]) -> None:
        self.name = name
        self.cfg = cfg
        self.logger = logging.getLogger(f"systematic_review.agents.{name}")

    # ---- Abstract interface --------------------------------------- #

    @abstractmethod
    def process(self, message: Message) -> AgentResult:
        """Process a message and return a result.

        This is the single entry point the coordinator calls.
        """
        ...

    # ---- LLM helper ---------------------------------------------- #

    def call_llm(
        self,
        prompt: str,
        model_override: Optional[str] = None,
    ) -> str:
        """Call the LLM, optionally overriding the model.

        Uses the per-agent model from ``cfg.multi_agent.agent_models.<name>``
        if available, otherwise falls back to the global model.
        """
        cfg = self.cfg

        # Per-agent model override
        if model_override is None:
            agent_models = cfg.get("multi_agent", {}).get("agent_models", {})
            model_override = agent_models.get(self.name)

        if model_override:
            # Temporarily swap the model in cfg for call_llm
            original_model = cfg.get("llm", {}).get("model")
            cfg.setdefault("llm", {})["model"] = model_override
            try:
                result = call_llm(prompt, cfg)
            finally:
                if original_model is not None:
                    cfg["llm"]["model"] = original_model
                elif "model" in cfg.get("llm", {}):
                    del cfg["llm"]["model"]
            return result

        return call_llm(prompt, cfg)

    # ---- Timing helper ------------------------------------------- #

    def timed_process(self, message: Message) -> AgentResult:
        """Call ``process()`` and record elapsed time in metrics."""
        t0 = time.time()
        self.logger.info(
            "Processing task=%s (iteration=%d)", message.task, message.iteration,
        )
        try:
            result = self.process(message)
        except Exception as exc:
            self.logger.error("Agent %s failed: %s", self.name, exc, exc_info=True)
            result = AgentResult(
                success=False,
                errors=[f"{type(exc).__name__}: {exc}"],
            )
        result.metrics["elapsed_s"] = round(time.time() - t0, 2)
        result.metrics.setdefault("agent", 0)
        self.logger.info(
            "Finished task=%s  success=%s  elapsed=%.1fs",
            message.task, result.success, result.metrics["elapsed_s"],
        )
        return result

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
