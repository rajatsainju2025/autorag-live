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
from .streaming import (
    AsyncStreamingMultiAgentOrchestrator,
    StreamEvent,
    StreamEventType,
    StreamingMultiAgentOrchestrator,
    create_sse_stream,
)

__all__ = [
    "AgentRole",
    "AgentMessage",
    "AgentProposal",
    "SpecializedAgent",
    "MultiAgentOrchestrator",
    "create_default_agents",
    # Streaming
    "StreamEventType",
    "StreamEvent",
    "StreamingMultiAgentOrchestrator",
    "AsyncStreamingMultiAgentOrchestrator",
    "create_sse_stream",
]
