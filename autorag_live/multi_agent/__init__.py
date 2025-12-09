"""
Multi-agent package initialization.
"""

from .collaboration import (
    AgentMessage,
    AgentProposal,
    AgentRole,
    MultiAgentOrchestrator,
    SpecializedAgent,
    create_default_agents,
)

__all__ = [
    "AgentRole",
    "AgentMessage",
    "AgentProposal",
    "SpecializedAgent",
    "MultiAgentOrchestrator",
    "create_default_agents",
]
