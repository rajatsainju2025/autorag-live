"""
Core module for AutoRAG-Live.

Provides protocol-based interfaces and foundational abstractions
for building modular, state-of-the-art agentic RAG systems.
"""

from .protocols import (  # Base protocols; Data types; Enums; Registry
    AgentAction,
    AgentState,
    AgentStatus,
    BaseAgent,
    BaseEmbedder,
    BaseLLM,
    BaseMemory,
    BaseReranker,
    BaseRetriever,
    BaseTool,
    BaseVectorStore,
    ComponentRegistry,
    Document,
    EmbeddingResult,
    GenerationResult,
    Message,
    MessageRole,
    RetrievalResult,
    ToolCall,
    ToolResult,
    get_registry,
)

__all__ = [
    # Protocols
    "BaseLLM",
    "BaseRetriever",
    "BaseEmbedder",
    "BaseVectorStore",
    "BaseReranker",
    "BaseAgent",
    "BaseTool",
    "BaseMemory",
    # Data types
    "Document",
    "Message",
    "ToolCall",
    "ToolResult",
    "RetrievalResult",
    "GenerationResult",
    "EmbeddingResult",
    "AgentAction",
    "AgentState",
    # Enums
    "MessageRole",
    "AgentStatus",
    # Registry
    "ComponentRegistry",
    "get_registry",
]
