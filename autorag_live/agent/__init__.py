"""
Agent package initialization and exports.

Provides main entry points for agentic RAG system.
"""

from .base import Action, Agent, AgentMemory, AgentState, Message, Observation
from .memory import ConversationBuffer, ConversationMemory, ConversationMessage
from .monitoring import AgentMetricsCollector, DistributedTracer, PerformanceMonitor
from .observations import ObservationBuffer, ObservationParser, ToolResponseHandler
from .rag_pipeline import AgenticRAGPipeline, RAGResponse
from .reasoning import Reasoner, ReasoningTrace
from .resilience import AdaptiveExecutor, ErrorRecoveryEngine, ResilientAgentWrapper
from .streaming import StreamEvent, StreamingResponseHandler
from .tools import ToolRegistry, get_tool_registry, register_builtin_tools

__all__ = [
    # Core agent
    "Agent",
    "AgentMemory",
    "AgentState",
    "Action",
    "Message",
    "Observation",
    # Memory
    "ConversationMemory",
    "ConversationBuffer",
    "ConversationMessage",
    # Reasoning
    "Reasoner",
    "ReasoningTrace",
    # RAG pipeline
    "AgenticRAGPipeline",
    "RAGResponse",
    # Streaming
    "StreamEvent",
    "StreamingResponseHandler",
    # Observations
    "ObservationBuffer",
    "ObservationParser",
    "ToolResponseHandler",
    # Tools
    "ToolRegistry",
    "get_tool_registry",
    "register_builtin_tools",
    # Monitoring
    "PerformanceMonitor",
    "DistributedTracer",
    "AgentMetricsCollector",
    # Resilience
    "ErrorRecoveryEngine",
    "AdaptiveExecutor",
    "ResilientAgentWrapper",
]
