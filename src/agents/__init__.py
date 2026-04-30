"""
agents — Multi-agent system for systematic review automation.

Provides specialised agents that collaborate to produce a high-quality
systematic literature review through iterative refinement.
"""

from agents.base_agent import BaseAgent, Message, AgentResult
from agents.blackboard import Blackboard

__all__ = [
    "BaseAgent",
    "Message",
    "AgentResult",
    "Blackboard",
]
