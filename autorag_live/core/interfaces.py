"""
Canonical interface consolidation for AutoRAG-Live.

This module re-exports the **single canonical version** of every shared type
and protocol that was previously duplicated across ``core.protocols``,
``types.types``, ``types.protocols``, and ``agent.base``.

Import Rule (going forward):
    >>> from autorag_live.core.interfaces import Document, Message, BaseLLM, BaseRetriever

Why this exists:
    The codebase evolved organically and accumulated **three** ``Document``
    classes, **two** ``Message`` classes, **two** ``AgentState`` enums, and
    **two** ``Observation`` dataclasses with incompatible field names.

    This module chooses a single canonical definition for each concept and
    provides thin compatibility aliases so existing code keeps working.

Canonical choices:
    - ``Document``       → ``core.protocols.Document``  (mutable, has embedding + score)
    - ``Message``        → ``core.protocols.Message``    (has MessageRole enum, tool_calls)
    - ``AgentState``     → ``agent.base.AgentState``     (IDLE/THINKING/ACTING/… enum)
    - ``AgentStatus``    → ``core.protocols.AgentStatus`` (alias for transition compat)
    - ``Observation``    → ``agent.base.Observation``    (action_id + result)
    - ``Action``         → ``agent.base.Action``         (tool_name + tool_input)
    - ``BaseLLM``        → ``core.protocols.BaseLLM``
    - ``BaseRetriever``  → ``core.protocols.BaseRetriever``
    - ``BaseReranker``   → ``core.protocols.BaseReranker``
    - ``BaseAgent``      → ``core.protocols.BaseAgent``
    - ``BaseTool``       → ``core.protocols.BaseTool``
    - Exceptions         → ``types.types.*``             (rich context, JSON-serialisable)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical agent types from agent.base
# ---------------------------------------------------------------------------
from autorag_live.agent.base import Action, AgentMemory, AgentState
from autorag_live.agent.base import Message as AgentMessage  # alias — prefer core.protocols.Message
from autorag_live.agent.base import Observation

# ---------------------------------------------------------------------------
# Canonical context types (new unified state)
# ---------------------------------------------------------------------------
from autorag_live.core.context import (
    ContextStage,
    EvalScore,
    RAGContext,
    ReasoningTrace,
    RetrievedDocument,
    StageLatency,
)

# ---------------------------------------------------------------------------
# Canonical protocols & data classes from core.protocols
# ---------------------------------------------------------------------------
from autorag_live.core.protocols import (
    AgentAction,
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
)
from autorag_live.core.protocols import RetrievalResult as ProtocolRetrievalResult
from autorag_live.core.protocols import ToolCall, ToolResult, get_registry

# ---------------------------------------------------------------------------
# Canonical graph engine
# ---------------------------------------------------------------------------
from autorag_live.core.state_graph import END, CompiledGraph, StateGraph

# ---------------------------------------------------------------------------
# Type aliases that were defined in types.types / types.protocols
# (kept for backward compatibility)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Canonical exceptions from types.types
# ---------------------------------------------------------------------------
from autorag_live.types.types import (
    AutoRAGError,
    ConfigurationError,
    DataError,
    DocumentId,
    DocumentText,
    EvaluationError,
    ModelError,
    OptimizerError,
    PipelineError,
    QueryText,
    RetrieverError,
    Score,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Compatibility map (old import path → new canonical)
# ---------------------------------------------------------------------------
# Consumers that imported from types.protocols:
#   ``from autorag_live.types.protocols import Retriever``
# should migrate to:
#   ``from autorag_live.core.interfaces import BaseRetriever``
#
# The old Retriever protocol (sync, .retrieve()/.add_documents()) is kept in
# types.protocols for backward compat but new code should use BaseRetriever.

__all__ = [
    # --- Unified context & graph ---
    "RAGContext",
    "ContextStage",
    "RetrievedDocument",
    "ReasoningTrace",
    "EvalScore",
    "StageLatency",
    "StateGraph",
    "CompiledGraph",
    "END",
    # --- Protocols ---
    "BaseLLM",
    "BaseRetriever",
    "BaseEmbedder",
    "BaseVectorStore",
    "BaseReranker",
    "BaseAgent",
    "BaseTool",
    "BaseMemory",
    "ComponentRegistry",
    "get_registry",
    # --- Data classes ---
    "Document",
    "Message",
    "MessageRole",
    "ToolCall",
    "ToolResult",
    "GenerationResult",
    "EmbeddingResult",
    "ProtocolRetrievalResult",
    "AgentAction",
    "AgentStatus",
    # --- Agent types ---
    "AgentState",
    "Action",
    "Observation",
    "AgentMemory",
    "AgentMessage",
    # --- Exceptions ---
    "AutoRAGError",
    "RetrieverError",
    "ConfigurationError",
    "EvaluationError",
    "OptimizerError",
    "PipelineError",
    "ModelError",
    "DataError",
    "ValidationError",
    # --- Type aliases ---
    "DocumentId",
    "DocumentText",
    "QueryText",
    "Score",
]
