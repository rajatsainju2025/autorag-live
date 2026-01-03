"""
Protocol-based interfaces for modular agentic RAG components.

This module defines abstract protocols (PEP 544) that enable loose coupling
and easy component swapping. Any implementation conforming to these protocols
can be used interchangeably.

Key Design Principles:
1. Protocol classes define the interface contract
2. Dataclasses provide structured data exchange
3. Async-first design for scalability
4. Type hints for IDE support and validation

Example:
    >>> class MyCustomLLM:
    ...     async def generate(self, messages, **kwargs) -> GenerationResult:
    ...         # Custom implementation
    ...         pass
    ...
    >>> # MyCustomLLM automatically implements BaseLLM protocol
    >>> llm: BaseLLM = MyCustomLLM()
"""

from __future__ import annotations

import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

# =============================================================================
# Enumerations
# =============================================================================


class MessageRole(str, Enum):
    """Standardized message roles for chat-based interactions."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"  # Legacy OpenAI format


class AgentStatus(str, Enum):
    """Agent execution lifecycle states."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    COMPLETE = "complete"
    ERROR = "error"


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GRAPH = "graph"
    RERANK = "rerank"


# =============================================================================
# Core Data Classes
# =============================================================================


@dataclass
class Document:
    """
    Universal document representation.

    Attributes:
        id: Unique document identifier
        content: Document text content
        metadata: Arbitrary metadata (source, timestamp, etc.)
        embedding: Optional pre-computed embedding vector
        score: Relevance score from retrieval
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: float = 0.0

    def __post_init__(self):
        """Validate document after initialization."""
        if not self.id:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create document from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            score=data.get("score", 0.0),
        )


@dataclass
class Message:
    """
    Chat message for LLM interactions.

    Attributes:
        role: Message role (system, user, assistant, tool)
        content: Message text content
        name: Optional name for function/tool messages
        tool_calls: Optional list of tool calls (assistant messages)
        tool_call_id: ID linking tool result to call (tool messages)
        metadata: Additional message metadata
    """

    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List["ToolCall"]] = None
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary."""
        result: Dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[List["ToolCall"]] = None) -> "Message":
        """Create assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str) -> "Message":
        """Create tool result message."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        )


@dataclass
class ToolCall:
    """
    Represents a tool/function call from the LLM.

    Attributes:
        id: Unique call identifier
        name: Tool/function name
        arguments: Arguments as dictionary or JSON string
        type: Call type (usually "function")
    """

    id: str
    name: str
    arguments: Union[Dict[str, Any], str]
    type: str = "function"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible format."""
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": (
                    self.arguments if isinstance(self.arguments, str) else str(self.arguments)
                ),
            },
        }

    def get_arguments(self) -> Dict[str, Any]:
        """Get arguments as dictionary."""
        if isinstance(self.arguments, dict):
            return self.arguments
        import json

        try:
            return json.loads(self.arguments)
        except json.JSONDecodeError:
            return {"raw": self.arguments}


@dataclass
class ToolResult:
    """
    Result from tool execution.

    Attributes:
        tool_call_id: ID of the originating tool call
        name: Tool name
        result: Execution result (any serializable value)
        success: Whether execution succeeded
        error: Error message if failed
        latency_ms: Execution time in milliseconds
    """

    tool_call_id: str
    name: str
    result: Any
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_message(self) -> Message:
        """Convert to tool message for LLM."""
        import json

        content = (
            json.dumps(self.result) if isinstance(self.result, (dict, list)) else str(self.result)
        )
        if not self.success and self.error:
            content = f"Error: {self.error}"
        return Message.tool(content, self.tool_call_id, self.name)


@dataclass
class RetrievalResult:
    """
    Result from retrieval operation.

    Attributes:
        documents: Retrieved documents
        query: Original query
        strategy: Retrieval strategy used
        latency_ms: Retrieval time
        metadata: Additional retrieval metadata
    """

    documents: List[Document]
    query: str
    strategy: RetrievalStrategy = RetrievalStrategy.DENSE
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_k(self) -> int:
        """Number of documents retrieved."""
        return len(self.documents)

    def get_context(self, max_docs: Optional[int] = None) -> str:
        """Get concatenated document content as context string."""
        docs = self.documents[:max_docs] if max_docs else self.documents
        return "\n\n".join(doc.content for doc in docs)


@dataclass
class GenerationResult:
    """
    Result from LLM generation.

    Attributes:
        content: Generated text content
        tool_calls: Tool calls if model requested them
        finish_reason: Why generation stopped
        usage: Token usage statistics
        model: Model identifier
        latency_ms: Generation time
    """

    content: str
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    latency_ms: float = 0.0

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return bool(self.tool_calls)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.usage.get("total_tokens", 0)


@dataclass
class EmbeddingResult:
    """
    Result from embedding operation.

    Attributes:
        embeddings: List of embedding vectors
        texts: Original texts
        model: Embedding model used
        dimensions: Embedding dimensions
        latency_ms: Embedding time
    """

    embeddings: List[List[float]]
    texts: List[str]
    model: str = ""
    dimensions: int = 0
    latency_ms: float = 0.0

    def __post_init__(self):
        """Set dimensions from embeddings if not provided."""
        if not self.dimensions and self.embeddings:
            self.dimensions = len(self.embeddings[0])


@dataclass
class AgentAction:
    """
    Represents an action taken by an agent.

    Attributes:
        id: Unique action identifier
        type: Action type (tool_call, final_answer, etc.)
        tool_name: Tool name if tool call
        tool_input: Tool arguments if tool call
        thought: Agent's reasoning for this action
        confidence: Confidence in this action (0-1)
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str = "tool_call"
    tool_name: Optional[str] = None
    tool_input: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    confidence: float = 1.0

    @classmethod
    def tool_call(
        cls,
        tool_name: str,
        tool_input: Dict[str, Any],
        thought: str = "",
    ) -> "AgentAction":
        """Create tool call action."""
        return cls(
            type="tool_call",
            tool_name=tool_name,
            tool_input=tool_input,
            thought=thought,
        )

    @classmethod
    def final_answer(cls, thought: str = "") -> "AgentAction":
        """Create final answer action."""
        return cls(type="final_answer", thought=thought)


@dataclass
class AgentState:
    """
    Complete agent state at a point in time.

    Attributes:
        status: Current agent status
        messages: Conversation history
        actions: Actions taken
        observations: Tool results received
        current_thought: Latest reasoning
        iteration: Current iteration number
        metadata: Additional state metadata
    """

    status: AgentStatus = AgentStatus.IDLE
    messages: List[Message] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)
    observations: List[ToolResult] = field(default_factory=list)
    current_thought: str = ""
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Add message to history."""
        self.messages.append(message)

    def add_action(self, action: AgentAction) -> None:
        """Add action to history."""
        self.actions.append(action)

    def add_observation(self, observation: ToolResult) -> None:
        """Add observation to history."""
        self.observations.append(observation)


# =============================================================================
# Protocol Definitions
# =============================================================================

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class BaseLLM(Protocol):
    """
    Protocol for Language Model implementations.

    Any class implementing these methods can be used as an LLM provider.
    Supports both sync and async generation, streaming, and tool calling.

    Example:
        >>> class OpenAILLM:
        ...     async def generate(self, messages, **kwargs):
        ...         # Call OpenAI API
        ...         return GenerationResult(content="Response")
        ...
        >>> llm: BaseLLM = OpenAILLM()
    """

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate a response from the model.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: Tool definitions for function calling
            tool_choice: Tool selection strategy
            **kwargs: Provider-specific options

        Returns:
            Generation result with content and optional tool calls
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from the model.

        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Provider-specific options

        Yields:
            Individual tokens as they're generated
        """
        ...

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        # Default implementation using word approximation
        return len(text.split()) * 4 // 3


@runtime_checkable
class BaseRetriever(Protocol):
    """
    Protocol for document retrieval implementations.

    Supports various retrieval strategies (dense, sparse, hybrid).
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of documents to retrieve
            filters: Metadata filters
            **kwargs: Retriever-specific options

        Returns:
            Retrieval result with documents
        """
        ...

    @abstractmethod
    async def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add documents to the retriever's index.

        Args:
            documents: Documents to add
            **kwargs: Indexing options

        Returns:
            List of document IDs
        """
        ...


@runtime_checkable
class BaseEmbedder(Protocol):
    """
    Protocol for text embedding implementations.
    """

    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        Generate embeddings for texts.

        Args:
            texts: Texts to embed
            **kwargs: Embedder-specific options

        Returns:
            Embedding result with vectors
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding vector dimensions."""
        ...


@runtime_checkable
class BaseVectorStore(Protocol):
    """
    Protocol for vector store implementations.

    Provides vector storage and similarity search.
    """

    @abstractmethod
    async def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[Document]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Add vectors to the store.

        Args:
            ids: Vector identifiers
            embeddings: Embedding vectors
            documents: Optional associated documents
            **kwargs: Store-specific options
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        *,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            k: Number of results
            filters: Metadata filters
            **kwargs: Search options

        Returns:
            Similar documents with scores
        """
        ...

    @abstractmethod
    async def delete(
        self,
        ids: List[str],
        **kwargs: Any,
    ) -> None:
        """
        Delete vectors from the store.

        Args:
            ids: Vector identifiers to delete
            **kwargs: Deletion options
        """
        ...


@runtime_checkable
class BaseReranker(Protocol):
    """
    Protocol for document reranking implementations.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[Document],
        *,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of documents to return
            **kwargs: Reranker-specific options

        Returns:
            Reranked documents
        """
        ...


@runtime_checkable
class BaseTool(Protocol):
    """
    Protocol for agent tool implementations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for invocation."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM."""
        ...

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for tool parameters.

        Returns:
            OpenAI-compatible function schema
        """
        ...

    @abstractmethod
    async def execute(
        self,
        **kwargs: Any,
    ) -> Any:
        """
        Execute the tool.

        Args:
            **kwargs: Tool arguments

        Returns:
            Tool execution result
        """
        ...


@runtime_checkable
class BaseMemory(Protocol):
    """
    Protocol for agent memory implementations.
    """

    @abstractmethod
    def add_message(self, message: Message) -> None:
        """Add message to memory."""
        ...

    @abstractmethod
    def get_messages(
        self,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Get messages from memory."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all messages."""
        ...

    @abstractmethod
    def get_context_window(
        self,
        max_tokens: int,
    ) -> List[Message]:
        """Get messages fitting within token budget."""
        ...


@runtime_checkable
class BaseAgent(Protocol):
    """
    Protocol for agent implementations.

    Defines the core agent loop with thinking, acting, and observing.
    """

    @abstractmethod
    async def step(
        self,
        state: AgentState,
    ) -> AgentState:
        """
        Execute one agent step.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        ...

    @abstractmethod
    async def run(
        self,
        query: str,
        *,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> str:
        """
        Run agent to completion.

        Args:
            query: User query
            max_iterations: Maximum iteration limit
            **kwargs: Additional options

        Returns:
            Final answer
        """
        ...


# =============================================================================
# Component Registry
# =============================================================================


class ComponentRegistry:
    """
    Registry for managing component implementations.

    Allows registering and retrieving implementations by type and name.

    Example:
        >>> registry = ComponentRegistry()
        >>> registry.register("llm", "openai", OpenAILLM())
        >>> llm = registry.get("llm", "openai")
    """

    def __init__(self):
        """Initialize empty registry."""
        self._components: Dict[str, Dict[str, Any]] = {}
        self._defaults: Dict[str, str] = {}

    def register(
        self,
        component_type: str,
        name: str,
        instance: Any,
        *,
        default: bool = False,
    ) -> None:
        """
        Register a component implementation.

        Args:
            component_type: Component category (llm, retriever, etc.)
            name: Implementation name
            instance: Component instance
            default: Set as default for this type
        """
        if component_type not in self._components:
            self._components[component_type] = {}
        self._components[component_type][name] = instance
        if default or component_type not in self._defaults:
            self._defaults[component_type] = name

    def get(
        self,
        component_type: str,
        name: Optional[str] = None,
    ) -> Any:
        """
        Get a component implementation.

        Args:
            component_type: Component category
            name: Implementation name (uses default if not specified)

        Returns:
            Component instance

        Raises:
            KeyError: If component not found
        """
        if component_type not in self._components:
            raise KeyError(f"No components registered for type: {component_type}")

        if name is None:
            name = self._defaults.get(component_type)
            if name is None:
                raise KeyError(f"No default set for type: {component_type}")

        if name not in self._components[component_type]:
            raise KeyError(f"Component not found: {component_type}/{name}")

        return self._components[component_type][name]

    def list_components(
        self,
        component_type: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        List registered components.

        Args:
            component_type: Optionally filter by type

        Returns:
            Dictionary of type -> list of names
        """
        if component_type:
            return {component_type: list(self._components.get(component_type, {}).keys())}
        return {k: list(v.keys()) for k, v in self._components.items()}

    def set_default(self, component_type: str, name: str) -> None:
        """Set default implementation for a component type."""
        if component_type not in self._components:
            raise KeyError(f"No components registered for type: {component_type}")
        if name not in self._components[component_type]:
            raise KeyError(f"Component not found: {component_type}/{name}")
        self._defaults[component_type] = name


# Global registry instance
_global_registry: Optional[ComponentRegistry] = None


def get_registry() -> ComponentRegistry:
    """
    Get the global component registry.

    Returns:
        Global ComponentRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ComponentRegistry()
    return _global_registry
