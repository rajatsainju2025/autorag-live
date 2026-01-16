"""
Agent orchestration module for AutoRAG-Live.

Provides coordination and management of multiple agents
for complex multi-step RAG workflows.

Features:
- Agent registry and lifecycle management
- Task routing and assignment
- Inter-agent communication
- Workflow orchestration
- Consensus mechanisms
- Agent monitoring
- Fault tolerance

Example usage:
    >>> orchestrator = AgentOrchestrator()
    >>> orchestrator.register_agent("retriever", RetrieverAgent())
    >>> orchestrator.register_agent("generator", GeneratorAgent())
    >>>
    >>> result = await orchestrator.execute_workflow(
    ...     query="What is RAG?",
    ...     workflow="rag_pipeline"
    ... )
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AgentState(Enum):
    """Agent lifecycle states."""

    IDLE = auto()
    BUSY = auto()
    PAUSED = auto()
    ERROR = auto()
    STOPPED = auto()


# =============================================================================
# OPTIMIZATION 8: Shared Agent Memory Workspace
# Based on: "Memory-Augmented Multi-Agent Systems" (DeepMind, 2024)
# and "Collaborative Intelligence" (Microsoft Research, 2024)
#
# Enables agents to share embeddings, retrieval results, and computation
# without redundant work. Implements copy-on-write semantics for efficiency.
# =============================================================================


@dataclass
class MemoryEntry:
    """Entry in the shared memory workspace."""

    key: str
    value: Any
    entry_type: str  # 'embedding', 'retrieval', 'computation', 'context'
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    creator_agent: str = ""
    ttl_seconds: float = 300.0  # Default 5 min TTL
    version: int = 1
    checksum: str = ""  # For change detection
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds


@dataclass
class MemoryView:
    """A read-only view of shared memory for an agent."""

    agent_id: str
    entries: Dict[str, MemoryEntry] = field(default_factory=dict)
    local_modifications: Dict[str, Any] = field(default_factory=dict)
    access_log: List[Tuple[float, str, str]] = field(default_factory=list)  # timestamp, key, action


class SharedMemoryWorkspace:
    """
    Shared memory workspace for multi-agent knowledge sharing.

    Implements:
    - Copy-on-write semantics for memory efficiency
    - TTL-based expiration for staleness control
    - Versioning for conflict detection
    - Namespace isolation per agent type
    - LRU eviction for memory bounds

    Example:
        >>> workspace = SharedMemoryWorkspace(max_entries=1000)
        >>>
        >>> # Agent writes retrieval results
        >>> await workspace.write(
        ...     key="query:what_is_rag",
        ...     value=retrieval_results,
        ...     entry_type="retrieval",
        ...     agent_id="retriever_1"
        ... )
        >>>
        >>> # Another agent reads shared results
        >>> results = await workspace.read("query:what_is_rag", "generator_1")
    """

    def __init__(
        self,
        max_entries: int = 10000,
        default_ttl: float = 300.0,
        cleanup_interval: float = 60.0,
    ):
        """
        Initialize shared memory workspace.

        Args:
            max_entries: Maximum entries before LRU eviction
            default_ttl: Default time-to-live in seconds
            cleanup_interval: Interval for cleanup task
        """
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval

        # Core storage
        self._entries: Dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()

        # Indexes for fast lookup
        self._type_index: Dict[str, Set[str]] = defaultdict(set)
        self._agent_index: Dict[str, Set[str]] = defaultdict(set)

        # Namespaced storage
        self._namespaces: Dict[str, Dict[str, MemoryEntry]] = defaultdict(dict)

        # Statistics
        self._stats = {
            "reads": 0,
            "writes": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def write(
        self,
        key: str,
        value: Any,
        entry_type: str,
        agent_id: str,
        ttl: Optional[float] = None,
        namespace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """
        Write entry to shared memory.

        Args:
            key: Unique key for entry
            value: Value to store
            entry_type: Type of entry
            agent_id: ID of writing agent
            ttl: Time-to-live (uses default if None)
            namespace: Optional namespace for isolation
            metadata: Additional metadata

        Returns:
            Created MemoryEntry
        """
        async with self._lock:
            self._stats["writes"] += 1

            # Check for existing entry (versioning)
            existing = self._entries.get(key)
            version = existing.version + 1 if existing else 1

            # Calculate checksum for change detection
            checksum = self._calculate_checksum(value)

            entry = MemoryEntry(
                key=key,
                value=value,
                entry_type=entry_type,
                creator_agent=agent_id,
                ttl_seconds=ttl or self.default_ttl,
                version=version,
                checksum=checksum,
                metadata=metadata or {},
            )

            # Store in main storage
            self._entries[key] = entry

            # Update indexes
            self._type_index[entry_type].add(key)
            self._agent_index[agent_id].add(key)

            # Store in namespace if specified
            if namespace:
                self._namespaces[namespace][key] = entry

            # Evict if over capacity
            if len(self._entries) > self.max_entries:
                await self._evict_lru()

            return entry

    async def read(
        self,
        key: str,
        agent_id: str,
        namespace: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Read entry from shared memory.

        Args:
            key: Entry key
            agent_id: ID of reading agent
            namespace: Optional namespace

        Returns:
            Entry value or None if not found
        """
        async with self._lock:
            self._stats["reads"] += 1

            # Check namespace first
            if namespace and key in self._namespaces.get(namespace, {}):
                entry = self._namespaces[namespace][key]
            else:
                entry = self._entries.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            # Check expiration
            if entry.is_expired:
                await self._remove_entry(key)
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return None

            # Update access tracking
            entry.accessed_at = time.time()
            entry.access_count += 1
            self._stats["hits"] += 1

            return entry.value

    async def read_many(
        self,
        keys: List[str],
        agent_id: str,
    ) -> Dict[str, Any]:
        """
        Batch read multiple entries.

        Args:
            keys: List of keys
            agent_id: Reading agent ID

        Returns:
            Dict of key -> value
        """
        results = {}
        for key in keys:
            value = await self.read(key, agent_id)
            if value is not None:
                results[key] = value
        return results

    async def query_by_type(
        self,
        entry_type: str,
        limit: int = 100,
    ) -> List[MemoryEntry]:
        """
        Query entries by type.

        Args:
            entry_type: Type to query
            limit: Maximum results

        Returns:
            List of matching entries
        """
        async with self._lock:
            keys = list(self._type_index.get(entry_type, set()))[:limit]
            return [
                self._entries[k]
                for k in keys
                if k in self._entries and not self._entries[k].is_expired
            ]

    async def query_by_agent(
        self,
        agent_id: str,
        limit: int = 100,
    ) -> List[MemoryEntry]:
        """
        Query entries created by agent.

        Args:
            agent_id: Agent ID to query
            limit: Maximum results

        Returns:
            List of matching entries
        """
        async with self._lock:
            keys = list(self._agent_index.get(agent_id, set()))[:limit]
            return [
                self._entries[k]
                for k in keys
                if k in self._entries and not self._entries[k].is_expired
            ]

    async def get_view(self, agent_id: str) -> MemoryView:
        """
        Get a read-only view for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            MemoryView with copy-on-write semantics
        """
        async with self._lock:
            # Create shallow copy of entries (copy-on-write)
            view_entries = {k: v for k, v in self._entries.items() if not v.is_expired}
            return MemoryView(
                agent_id=agent_id,
                entries=view_entries,
            )

    async def invalidate(self, key: str) -> bool:
        """
        Invalidate an entry.

        Args:
            key: Entry key

        Returns:
            True if entry was removed
        """
        async with self._lock:
            return await self._remove_entry(key)

    async def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate entries matching pattern.

        Args:
            pattern: Key pattern (prefix match)

        Returns:
            Number of entries removed
        """
        async with self._lock:
            keys_to_remove = [k for k in self._entries.keys() if k.startswith(pattern)]
            for key in keys_to_remove:
                await self._remove_entry(key)
            return len(keys_to_remove)

    async def _remove_entry(self, key: str) -> bool:
        """Remove entry from all indexes."""
        entry = self._entries.get(key)
        if entry:
            del self._entries[key]
            self._type_index[entry.entry_type].discard(key)
            self._agent_index[entry.creator_agent].discard(key)
            return True
        return False

    async def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        # Sort by access time
        sorted_entries = sorted(self._entries.items(), key=lambda x: x[1].accessed_at)

        # Remove oldest 10%
        evict_count = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:evict_count]:
            await self._remove_entry(key)
            self._stats["evictions"] += 1

    async def _cleanup_loop(self) -> None:
        """Background cleanup of expired entries."""
        while True:
            await asyncio.sleep(self.cleanup_interval)
            async with self._lock:
                expired_keys = [k for k, v in self._entries.items() if v.is_expired]
                for key in expired_keys:
                    await self._remove_entry(key)
                    self._stats["expirations"] += 1

    def _calculate_checksum(self, value: Any) -> str:
        """Calculate simple checksum for change detection."""
        import hashlib

        return hashlib.md5(str(value).encode()).hexdigest()[:8]

    def get_stats(self) -> Dict[str, Any]:
        """Get workspace statistics."""
        hit_rate = (self._stats["hits"] / max(1, self._stats["reads"])) * 100
        return {
            **self._stats,
            "total_entries": len(self._entries),
            "hit_rate_percent": round(hit_rate, 2),
            "namespaces": list(self._namespaces.keys()),
        }


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


@dataclass
class AgentCapability:
    """A capability that an agent provides."""

    name: str
    description: str = ""

    # Input/output types
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)

    # Performance hints
    avg_latency_ms: float = 0.0
    max_concurrent: int = 10

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInfo:
    """Information about a registered agent."""

    agent_id: str
    name: str
    agent_type: str

    # Capabilities
    capabilities: List[AgentCapability] = field(default_factory=list)

    # State
    state: AgentState = AgentState.IDLE

    # Statistics
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_response_time_ms: float = 0.0

    # Registration
    registered_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A task to be executed by an agent."""

    task_id: str
    task_type: str
    payload: Dict[str, Any]

    # Assignment
    assigned_agent: Optional[str] = None

    # Priority and timing
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: float = 60.0
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0

    # Status
    status: TaskStatus = TaskStatus.PENDING

    # Result
    result: Any = None
    error: Optional[str] = None

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get task duration."""
        if self.completed_at > 0 and self.started_at > 0:
            return (self.completed_at - self.started_at) * 1000
        return 0.0


@dataclass
class Message:
    """Inter-agent message."""

    message_id: str
    sender: str
    recipient: str

    # Content
    message_type: str
    payload: Dict[str, Any]

    # Timing
    timestamp: float = field(default_factory=time.time)

    # Response
    reply_to: Optional[str] = None
    requires_response: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStep:
    """A step in a workflow."""

    step_id: str
    task_type: str

    # Configuration
    agent_type: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Error handling
    retry_count: int = 0
    fallback_step: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """A workflow definition."""

    workflow_id: str
    name: str

    # Steps
    steps: List[WorkflowStep] = field(default_factory=list)

    # Configuration
    timeout: float = 300.0
    parallel_enabled: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of workflow execution."""

    workflow_id: str
    success: bool

    # Results
    final_result: Any = None
    step_results: Dict[str, Any] = field(default_factory=dict)

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    # Errors
    errors: List[Tuple[str, str]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get workflow duration."""
        return (self.end_time - self.start_time) * 1000


class BaseAgent(ABC):
    """Base class for agents."""

    def __init__(
        self,
        name: str,
        agent_type: str = "generic",
    ):
        """
        Initialize agent.

        Args:
            name: Agent name
            agent_type: Type of agent
        """
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.agent_type = agent_type
        self.state = AgentState.IDLE

        self._message_handlers: Dict[str, Callable] = {}

    @property
    @abstractmethod
    def capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities."""
        pass

    @abstractmethod
    async def execute(
        self,
        task: Task,
    ) -> Any:
        """Execute a task."""
        pass

    async def handle_message(
        self,
        message: Message,
    ) -> Optional[Message]:
        """Handle incoming message."""
        handler = self._message_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        return None

    def register_message_handler(
        self,
        message_type: str,
        handler: Callable[[Message], Awaitable[Optional[Message]]],
    ) -> None:
        """Register message handler."""
        self._message_handlers[message_type] = handler

    def get_info(self) -> AgentInfo:
        """Get agent info."""
        return AgentInfo(
            agent_id=self.agent_id,
            name=self.name,
            agent_type=self.agent_type,
            capabilities=self.capabilities,
            state=self.state,
        )


@dataclass
class SharedContext:
    """
    Context object that agents can share and extend.

    Implements collaborative context building where multiple
    agents contribute to a shared understanding.
    """

    context_id: str
    query: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    extracted_entities: List[str] = field(default_factory=list)
    retrieved_facts: List[str] = field(default_factory=list)
    agent_contributions: Dict[str, List[str]] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_contribution(
        self,
        agent_id: str,
        contribution_type: str,
        content: Any,
        confidence: float = 1.0,
    ) -> None:
        """Add contribution from an agent."""
        if agent_id not in self.agent_contributions:
            self.agent_contributions[agent_id] = []
        self.agent_contributions[agent_id].append(contribution_type)

        if contribution_type == "documents" and isinstance(content, list):
            self.documents.extend(content)
        elif contribution_type == "entities" and isinstance(content, list):
            self.extracted_entities.extend(content)
        elif contribution_type == "facts" and isinstance(content, list):
            self.retrieved_facts.extend(content)
        elif contribution_type == "embedding" and isinstance(content, dict):
            self.embeddings.update(content)

        self.confidence_scores[f"{agent_id}:{contribution_type}"] = confidence
        self.updated_at = time.time()

    def get_combined_context(self) -> Dict[str, Any]:
        """Get combined context from all contributions."""
        return {
            "query": self.query,
            "documents": self.documents,
            "entities": list(set(self.extracted_entities)),
            "facts": list(set(self.retrieved_facts)),
            "contributors": list(self.agent_contributions.keys()),
            "confidence": sum(self.confidence_scores.values())
            / max(1, len(self.confidence_scores)),
        }


class MemoryAwareBaseAgent(BaseAgent):
    """
    Base agent with shared memory workspace integration.

    Automatically caches and retrieves from shared workspace
    to avoid redundant computation across agents.
    """

    def __init__(
        self,
        name: str,
        agent_type: str,
        workspace: Optional[SharedMemoryWorkspace] = None,
    ):
        super().__init__(name, agent_type)
        self.workspace = workspace
        self._cache_prefix = f"{agent_type}:"

    @property
    def capabilities(self) -> List[AgentCapability]:
        """Override in subclass."""
        return []

    async def execute(self, task: Task) -> Any:
        """Override in subclass."""
        raise NotImplementedError("Subclass must implement execute")

    async def execute_with_caching(
        self,
        task: Task,
        cache_key: Optional[str] = None,
    ) -> Any:
        """
        Execute task with automatic caching.

        Args:
            task: Task to execute
            cache_key: Optional custom cache key

        Returns:
            Task result (cached or computed)
        """
        if self.workspace and cache_key:
            full_key = f"{self._cache_prefix}{cache_key}"

            # Try cache first
            cached = await self.workspace.read(full_key, self.agent_id)
            if cached is not None:
                logger.debug(f"Cache hit for {full_key}")
                return cached

            # Execute and cache
            result = await self.execute(task)

            await self.workspace.write(
                key=full_key,
                value=result,
                entry_type=self.agent_type,
                agent_id=self.agent_id,
            )

            return result

        return await self.execute(task)

    async def share_context(
        self,
        shared_context: SharedContext,
        contribution_type: str,
        content: Any,
        confidence: float = 1.0,
    ) -> None:
        """
        Share context with other agents.

        Args:
            shared_context: Shared context object
            contribution_type: Type of contribution
            content: Content to share
            confidence: Confidence score
        """
        shared_context.add_contribution(
            agent_id=self.agent_id,
            contribution_type=contribution_type,
            content=content,
            confidence=confidence,
        )

        # Also write to workspace for persistence
        if self.workspace:
            await self.workspace.write(
                key=f"context:{shared_context.context_id}:{contribution_type}",
                value=content,
                entry_type="context",
                agent_id=self.agent_id,
                metadata={"confidence": confidence},
            )

    async def get_shared_knowledge(
        self,
        knowledge_type: str,
        limit: int = 10,
    ) -> List[Any]:
        """
        Get shared knowledge from other agents.

        Args:
            knowledge_type: Type of knowledge
            limit: Maximum entries

        Returns:
            List of shared knowledge
        """
        if not self.workspace:
            return []

        entries = await self.workspace.query_by_type(knowledge_type, limit)
        return [e.value for e in entries]


class RetrieverAgent(BaseAgent):
    """Agent for document retrieval."""

    def __init__(self, name: str = "retriever"):
        super().__init__(name, "retriever")

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="retrieve",
                description="Retrieve documents from knowledge base",
                input_types=["query"],
                output_types=["documents"],
            )
        ]

    async def execute(self, task: Task) -> Any:
        """Execute retrieval task."""
        query = task.payload.get("query", "")

        # Simulated retrieval
        await asyncio.sleep(0.1)

        return {
            "query": query,
            "documents": [
                {"id": "doc1", "content": f"Retrieved content for: {query}"},
            ],
        }


class GeneratorAgent(BaseAgent):
    """Agent for response generation."""

    def __init__(self, name: str = "generator"):
        super().__init__(name, "generator")

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="generate",
                description="Generate response from context",
                input_types=["query", "documents"],
                output_types=["response"],
            )
        ]

    async def execute(self, task: Task) -> Any:
        """Execute generation task."""
        query = task.payload.get("query", "")
        documents = task.payload.get("documents", [])

        # Simulated generation
        await asyncio.sleep(0.2)

        return {
            "query": query,
            "response": f"Generated response for: {query}",
            "sources": [d.get("id") for d in documents],
        }


class MessageBus:
    """Message bus for inter-agent communication."""

    def __init__(self):
        """Initialize message bus."""
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._message_queue: asyncio.Queue = asyncio.Queue()

    def subscribe(
        self,
        topic: str,
        handler: Callable[[Message], Awaitable[None]],
    ) -> None:
        """Subscribe to topic."""
        self._subscribers[topic].append(handler)

    def unsubscribe(
        self,
        topic: str,
        handler: Callable,
    ) -> None:
        """Unsubscribe from topic."""
        if handler in self._subscribers[topic]:
            self._subscribers[topic].remove(handler)

    async def publish(
        self,
        topic: str,
        message: Message,
    ) -> None:
        """Publish message to topic."""
        handlers = self._subscribers.get(topic, [])

        tasks = [handler(message) for handler in handlers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send(
        self,
        message: Message,
    ) -> None:
        """Send direct message."""
        await self._message_queue.put(message)

    async def receive(
        self,
        timeout: float = 1.0,
    ) -> Optional[Message]:
        """Receive message from queue."""
        try:
            return await asyncio.wait_for(
                self._message_queue.get(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None


class TaskQueue:
    """Priority queue for tasks."""

    def __init__(self):
        """Initialize queue."""
        self._queues: Dict[TaskPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in TaskPriority
        }

    async def enqueue(self, task: Task) -> None:
        """Add task to queue."""
        await self._queues[task.priority].put(task)

    async def dequeue(
        self,
        timeout: float = 1.0,
    ) -> Optional[Task]:
        """Get highest priority task."""
        # Check queues in priority order
        for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
            queue = self._queues[priority]

            if not queue.empty():
                try:
                    return await asyncio.wait_for(
                        queue.get(),
                        timeout=0.01,
                    )
                except asyncio.TimeoutError:
                    continue

        return None

    @property
    def size(self) -> int:
        """Get total queue size."""
        return sum(q.qsize() for q in self._queues.values())


class AgentOrchestrator:
    """
    Main orchestrator for multi-agent systems.

    Example:
        >>> orchestrator = AgentOrchestrator()
        >>>
        >>> # Register agents
        >>> orchestrator.register_agent("retriever", RetrieverAgent())
        >>> orchestrator.register_agent("generator", GeneratorAgent())
        >>>
        >>> # Define workflow
        >>> workflow = Workflow(
        ...     workflow_id="rag_pipeline",
        ...     name="RAG Pipeline",
        ...     steps=[
        ...         WorkflowStep(step_id="retrieve", task_type="retrieve"),
        ...         WorkflowStep(step_id="generate", task_type="generate", depends_on=["retrieve"]),
        ...     ]
        ... )
        >>> orchestrator.register_workflow(workflow)
        >>>
        >>> # Execute
        >>> result = await orchestrator.execute_workflow("rag_pipeline", {"query": "What is RAG?"})
    """

    def __init__(
        self,
        max_concurrent_tasks: int = 10,
    ):
        """
        Initialize orchestrator.

        Args:
            max_concurrent_tasks: Max concurrent tasks
        """
        self.max_concurrent_tasks = max_concurrent_tasks

        # Agents
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_info: Dict[str, AgentInfo] = {}

        # Workflows
        self._workflows: Dict[str, Workflow] = {}

        # Communication
        self._message_bus = MessageBus()
        self._task_queue = TaskQueue()

        # State
        self._running = False
        self._tasks: Dict[str, Task] = {}

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)

    def register_agent(
        self,
        name: str,
        agent: BaseAgent,
    ) -> None:
        """
        Register an agent.

        Args:
            name: Agent name
            agent: Agent instance
        """
        self._agents[name] = agent
        self._agent_info[name] = agent.get_info()

        logger.info(f"Registered agent: {name} ({agent.agent_type})")

    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent.

        Args:
            name: Agent name

        Returns:
            True if unregistered
        """
        if name in self._agents:
            del self._agents[name]
            del self._agent_info[name]
            logger.info(f"Unregistered agent: {name}")
            return True
        return False

    def register_workflow(self, workflow: Workflow) -> None:
        """
        Register a workflow.

        Args:
            workflow: Workflow definition
        """
        self._workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.name}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self._agents.get(name)

    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """Get agents by type."""
        return [agent for agent in self._agents.values() if agent.agent_type == agent_type]

    def get_agents_by_capability(
        self,
        capability: str,
    ) -> List[BaseAgent]:
        """Get agents with specific capability."""
        result = []

        for agent in self._agents.values():
            for cap in agent.capabilities:
                if cap.name == capability:
                    result.append(agent)
                    break

        return result

    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float = 60.0,
    ) -> Task:
        """
        Submit a task for execution.

        Args:
            task_type: Type of task
            payload: Task payload
            priority: Task priority
            timeout: Task timeout

        Returns:
            Task instance
        """
        task = Task(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            payload=payload,
            priority=priority,
            timeout=timeout,
        )

        self._tasks[task.task_id] = task
        await self._task_queue.enqueue(task)

        return task

    async def execute_task(
        self,
        task: Task,
    ) -> Any:
        """
        Execute a single task.

        Args:
            task: Task to execute

        Returns:
            Task result
        """
        # Find suitable agent
        agents = self.get_agents_by_capability(task.task_type)

        if not agents:
            raise ValueError(f"No agent found for task type: {task.task_type}")

        # Select agent (simple round-robin for now)
        agent = agents[0]

        task.assigned_agent = agent.name
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        agent.state = AgentState.BUSY

        try:
            async with self._semaphore:
                # Execute with timeout
                result = await asyncio.wait_for(
                    agent.execute(task),
                    timeout=task.timeout,
                )

            task.result = result
            task.status = TaskStatus.COMPLETED

            # Update agent stats
            info = self._agent_info[agent.name]
            info.tasks_completed += 1

        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.error = "Task timeout"

            info = self._agent_info[agent.name]
            info.tasks_failed += 1

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)

            info = self._agent_info[agent.name]
            info.tasks_failed += 1

            logger.error(f"Task {task.task_id} failed: {e}")

        finally:
            task.completed_at = time.time()
            agent.state = AgentState.IDLE

            # Update average response time
            info = self._agent_info[agent.name]
            total_tasks = info.tasks_completed + info.tasks_failed
            if total_tasks > 0:
                info.avg_response_time_ms = (
                    info.avg_response_time_ms * (total_tasks - 1) + task.duration_ms
                ) / total_tasks
            info.last_active = time.time()

        return task.result

    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
    ) -> WorkflowResult:
        """
        Execute a workflow.

        Args:
            workflow_id: Workflow ID
            input_data: Input data

        Returns:
            WorkflowResult
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        result = WorkflowResult(
            workflow_id=workflow_id,
            success=False,
            start_time=time.time(),
        )

        # Track step results
        step_results: Dict[str, Any] = {}
        pending_steps = {step.step_id: step for step in workflow.steps}
        completed_steps: Set[str] = set()

        # Current data (passed between steps)
        current_data = input_data.copy()

        try:
            while pending_steps:
                # Find steps ready to execute
                ready_steps = []

                for step_id, step in pending_steps.items():
                    deps_met = all(dep in completed_steps for dep in step.depends_on)
                    if deps_met:
                        ready_steps.append(step)

                if not ready_steps:
                    # No progress possible - circular dependency?
                    raise RuntimeError("Workflow stuck - check dependencies")

                # Execute ready steps (possibly in parallel)
                if workflow.parallel_enabled and len(ready_steps) > 1:
                    tasks = []
                    for step in ready_steps:
                        task = Task(
                            task_id=str(uuid.uuid4()),
                            task_type=step.task_type,
                            payload={**current_data, **step.params},
                        )
                        tasks.append(self.execute_task(task))

                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for step, step_result in zip(ready_steps, results):
                        if isinstance(step_result, Exception):
                            result.errors.append((step.step_id, str(step_result)))
                        else:
                            step_results[step.step_id] = step_result
                            if isinstance(step_result, dict):
                                current_data.update(step_result)

                        completed_steps.add(step.step_id)
                        del pending_steps[step.step_id]
                else:
                    # Sequential execution
                    for step in ready_steps:
                        task = Task(
                            task_id=str(uuid.uuid4()),
                            task_type=step.task_type,
                            payload={**current_data, **step.params},
                        )

                        try:
                            step_result = await self.execute_task(task)
                            step_results[step.step_id] = step_result

                            if isinstance(step_result, dict):
                                current_data.update(step_result)

                        except Exception as e:
                            result.errors.append((step.step_id, str(e)))

                        completed_steps.add(step.step_id)
                        del pending_steps[step.step_id]

            result.success = len(result.errors) == 0
            result.final_result = current_data
            result.step_results = step_results

        except Exception as e:
            result.success = False
            result.errors.append(("workflow", str(e)))
            logger.error(f"Workflow {workflow_id} failed: {e}")

        finally:
            result.end_time = time.time()

        return result

    async def send_message(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        payload: Dict[str, Any],
    ) -> None:
        """Send message between agents."""
        message = Message(
            message_id=str(uuid.uuid4()),
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
        )

        agent = self._agents.get(recipient)
        if agent:
            await agent.handle_message(message)

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "agents": {
                name: {
                    "type": info.agent_type,
                    "state": info.state.name,
                    "tasks_completed": info.tasks_completed,
                    "tasks_failed": info.tasks_failed,
                    "avg_response_time_ms": info.avg_response_time_ms,
                }
                for name, info in self._agent_info.items()
            },
            "workflows": list(self._workflows.keys()),
            "pending_tasks": self._task_queue.size,
        }


# Convenience functions


def create_rag_workflow() -> Workflow:
    """
    Create a standard RAG workflow.

    Returns:
        Workflow for RAG pipeline
    """
    return Workflow(
        workflow_id="rag_pipeline",
        name="RAG Pipeline",
        steps=[
            WorkflowStep(
                step_id="retrieve",
                task_type="retrieve",
                agent_type="retriever",
            ),
            WorkflowStep(
                step_id="generate",
                task_type="generate",
                agent_type="generator",
                depends_on=["retrieve"],
            ),
        ],
    )


def create_orchestrator_with_agents() -> AgentOrchestrator:
    """
    Create orchestrator with default agents.

    Returns:
        Configured AgentOrchestrator
    """
    orchestrator = AgentOrchestrator()

    orchestrator.register_agent("retriever", RetrieverAgent())
    orchestrator.register_agent("generator", GeneratorAgent())
    orchestrator.register_workflow(create_rag_workflow())

    return orchestrator
