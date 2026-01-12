"""
Advanced Memory Management for Agentic RAG.

Implements state-of-the-art memory systems:
- Working Memory: Short-term scratchpad for current task
- Episodic Memory: Long-term storage of past interactions
- Semantic Memory: Embedding-based retrieval of relevant memories
- Procedural Memory: Learned patterns and successful strategies

Based on cognitive architectures and modern RAG patterns.

Example:
    >>> memory = AgentMemorySystem()
    >>> memory.store_episode(query="What is RAG?", answer="...", context=[...])
    >>> relevant = await memory.recall("retrieval augmented generation", top_k=5)
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Memory Types and Protocols
# =============================================================================


class MemoryType(Enum):
    """Types of memory in the system."""

    WORKING = auto()  # Short-term task context
    EPISODIC = auto()  # Specific past interactions
    SEMANTIC = auto()  # General knowledge
    PROCEDURAL = auto()  # Action patterns


class MemoryPriority(Enum):
    """Priority levels for memory items."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage backends."""

    async def add(self, id: str, embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Add embedding to store."""
        ...

    async def search(self, embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        ...

    async def delete(self, id: str) -> None:
        """Delete embedding from store."""
        ...


# =============================================================================
# Memory Items
# =============================================================================


@dataclass
class MemoryItem:
    """Base class for memory items."""

    id: str
    content: str
    memory_type: MemoryType
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    priority: MemoryPriority = MemoryPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    ttl: Optional[timedelta] = None

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1

    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.ttl is None:
            return False
        return datetime.utcnow() > self.created_at + self.ttl

    def get_recency_score(self, decay_factor: float = 0.9) -> float:
        """Calculate recency-based score."""
        age_hours = (datetime.utcnow() - self.accessed_at).total_seconds() / 3600
        return decay_factor**age_hours

    def get_importance_score(self) -> float:
        """Calculate importance-based score."""
        base_score = self.priority.value / 4.0
        access_bonus = min(0.3, self.access_count * 0.05)
        return base_score + access_bonus

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.name,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "priority": self.priority.name,
            "metadata": self.metadata,
        }


@dataclass
class Episode(MemoryItem):
    """
    Episodic memory: A specific interaction or event.

    Stores query, response, context, and outcome of past interactions.
    """

    query: str = ""
    response: str = ""
    context_docs: List[Dict[str, Any]] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    success: bool = True
    feedback_score: Optional[float] = None

    def __post_init__(self):
        self.memory_type = MemoryType.EPISODIC
        if not self.content:
            self.content = f"Q: {self.query}\nA: {self.response}"


@dataclass
class SemanticFact(MemoryItem):
    """
    Semantic memory: General knowledge or fact.

    Stores learned information that can be retrieved by similarity.
    """

    source: str = ""
    confidence: float = 1.0
    related_facts: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.memory_type = MemoryType.SEMANTIC


@dataclass
class Procedure(MemoryItem):
    """
    Procedural memory: A learned action pattern or strategy.

    Stores successful patterns for handling specific types of queries.
    """

    query_pattern: str = ""
    action_sequence: List[Dict[str, Any]] = field(default_factory=list)
    success_rate: float = 1.0
    use_count: int = 0

    def __post_init__(self):
        self.memory_type = MemoryType.PROCEDURAL


# =============================================================================
# Working Memory
# =============================================================================


class WorkingMemory:
    """
    Short-term working memory for current task context.

    Implements a limited-capacity scratchpad with automatic eviction.
    """

    def __init__(self, capacity: int = 10, token_limit: int = 4096):
        """Initialize working memory."""
        self.capacity = capacity
        self.token_limit = token_limit
        self._items: Deque[MemoryItem] = deque(maxlen=capacity)
        self._scratch: Dict[str, Any] = {}
        self._token_count = 0

    def add(self, item: MemoryItem) -> None:
        """Add item to working memory."""
        item.memory_type = MemoryType.WORKING
        self._items.append(item)
        self._update_token_count()

    def add_scratch(self, key: str, value: Any) -> None:
        """Add to scratchpad."""
        self._scratch[key] = value

    def get_scratch(self, key: str, default: Any = None) -> Any:
        """Get from scratchpad."""
        return self._scratch.get(key, default)

    def get_recent(self, n: int = 5) -> List[MemoryItem]:
        """Get n most recent items."""
        items = list(self._items)[-n:]
        for item in items:
            item.touch()
        return items

    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """Get formatted context string."""
        max_tokens = max_tokens or self.token_limit
        lines = []
        token_count = 0

        for item in reversed(list(self._items)):
            item_tokens = len(item.content) // 4  # Rough estimate
            if token_count + item_tokens > max_tokens:
                break
            lines.append(item.content)
            token_count += item_tokens

        return "\n".join(reversed(lines))

    def clear(self) -> None:
        """Clear working memory."""
        self._items.clear()
        self._scratch.clear()
        self._token_count = 0

    def _update_token_count(self) -> None:
        """Update token count estimate."""
        self._token_count = sum(len(item.content) // 4 for item in self._items)

    @property
    def size(self) -> int:
        """Current number of items."""
        return len(self._items)

    @property
    def token_count(self) -> int:
        """Estimated token count."""
        return self._token_count


# =============================================================================
# Episodic Memory Store
# =============================================================================


class EpisodicMemoryStore:
    """
    Long-term episodic memory with semantic retrieval.

    Stores past interactions and enables similarity-based recall.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
        max_episodes: int = 1000,
    ):
        """Initialize episodic memory."""
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._episodes: Dict[str, Episode] = {}
        self._max_episodes = max_episodes

    async def store(self, episode: Episode) -> str:
        """
        Store an episode in memory.

        Args:
            episode: Episode to store

        Returns:
            Episode ID
        """
        # Generate ID if not set
        if not episode.id:
            episode.id = self._generate_id(episode.content)

        # Generate embedding if provider available
        if self._embedding_provider and not episode.embedding:
            episode.embedding = await self._embedding_provider.embed(episode.content)

        # Store in vector store if available
        if self._vector_store and episode.embedding:
            await self._vector_store.add(
                episode.id,
                episode.embedding,
                episode.to_dict(),
            )

        # Store locally
        self._episodes[episode.id] = episode
        self._enforce_capacity()

        logger.debug(f"Stored episode: {episode.id}")
        return episode.id

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Episode]:
        """
        Recall relevant episodes based on query similarity.

        Args:
            query: Query to match against
            top_k: Number of episodes to retrieve
            min_similarity: Minimum similarity threshold
            filters: Optional metadata filters

        Returns:
            List of relevant episodes
        """
        # Use vector store if available
        if self._vector_store and self._embedding_provider:
            query_embedding = await self._embedding_provider.embed(query)
            results = await self._vector_store.search(query_embedding, top_k=top_k)

            episodes = []
            for result in results:
                episode_id = result.get("id")
                if episode_id and episode_id in self._episodes:
                    episode = self._episodes[episode_id]
                    if self._passes_filters(episode, filters):
                        episode.touch()
                        episodes.append(episode)

            return episodes

        # Fallback to simple keyword matching
        return self._keyword_search(query, top_k, filters)

    async def recall_by_id(self, episode_id: str) -> Optional[Episode]:
        """Recall specific episode by ID."""
        episode = self._episodes.get(episode_id)
        if episode:
            episode.touch()
        return episode

    async def update(self, episode_id: str, updates: Dict[str, Any]) -> bool:
        """Update an episode."""
        if episode_id not in self._episodes:
            return False

        episode = self._episodes[episode_id]
        for key, value in updates.items():
            if hasattr(episode, key):
                setattr(episode, key, value)

        return True

    async def delete(self, episode_id: str) -> bool:
        """Delete an episode."""
        if episode_id in self._episodes:
            del self._episodes[episode_id]
            if self._vector_store:
                await self._vector_store.delete(episode_id)
            return True
        return False

    def get_recent(self, n: int = 10) -> List[Episode]:
        """Get n most recent episodes."""
        sorted_episodes = sorted(
            self._episodes.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )
        return sorted_episodes[:n]

    def get_successful(self, n: int = 10) -> List[Episode]:
        """Get n most successful episodes."""
        successful = [e for e in self._episodes.values() if e.success]
        sorted_episodes = sorted(
            successful,
            key=lambda e: (e.feedback_score or 0, e.access_count),
            reverse=True,
        )
        return sorted_episodes[:n]

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content."""
        timestamp = str(time.time())
        hash_input = f"{content}{timestamp}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    def _enforce_capacity(self) -> None:
        """Enforce maximum capacity by removing old episodes."""
        if len(self._episodes) > self._max_episodes:
            # Remove oldest, lowest priority episodes
            sorted_episodes = sorted(
                self._episodes.values(),
                key=lambda e: (e.priority.value, e.access_count, e.accessed_at),
            )

            to_remove = len(self._episodes) - self._max_episodes
            for episode in sorted_episodes[:to_remove]:
                del self._episodes[episode.id]

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> List[Episode]:
        """Simple keyword-based search fallback."""
        query_terms = set(query.lower().split())
        scored_episodes = []

        for episode in self._episodes.values():
            if not self._passes_filters(episode, filters):
                continue

            content_terms = set(episode.content.lower().split())
            overlap = len(query_terms & content_terms)

            if overlap > 0:
                score = overlap / len(query_terms)
                scored_episodes.append((score, episode))

        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored_episodes[:top_k]]

    def _passes_filters(
        self,
        episode: Episode,
        filters: Optional[Dict[str, Any]],
    ) -> bool:
        """Check if episode passes metadata filters."""
        if not filters:
            return True

        for key, value in filters.items():
            if key == "success" and episode.success != value:
                return False
            if key in episode.metadata and episode.metadata[key] != value:
                return False

        return True

    @property
    def size(self) -> int:
        """Number of stored episodes."""
        return len(self._episodes)


# =============================================================================
# Semantic Memory Store
# =============================================================================


class SemanticMemoryStore:
    """
    Long-term semantic memory for general knowledge.

    Stores facts and knowledge that can be retrieved by similarity.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        max_facts: int = 5000,
    ):
        """Initialize semantic memory."""
        self._embedding_provider = embedding_provider
        self._facts: Dict[str, SemanticFact] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._max_facts = max_facts

    async def store(self, fact: SemanticFact) -> str:
        """Store a semantic fact."""
        if not fact.id:
            fact.id = hashlib.sha256(fact.content.encode()).hexdigest()[:12]

        if self._embedding_provider and not fact.embedding:
            fact.embedding = await self._embedding_provider.embed(fact.content)
            self._embeddings[fact.id] = fact.embedding

        self._facts[fact.id] = fact
        return fact.id

    async def recall(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SemanticFact]:
        """Recall facts similar to query."""
        if self._embedding_provider:
            query_embedding = await self._embedding_provider.embed(query)
            return self._similarity_search(query_embedding, top_k)

        return self._keyword_search(query, top_k)

    def _similarity_search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[SemanticFact]:
        """Search by embedding similarity."""
        scores = []

        for fact_id, embedding in self._embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            scores.append((similarity, fact_id))

        scores.sort(reverse=True)

        results = []
        for _, fact_id in scores[:top_k]:
            fact = self._facts.get(fact_id)
            if fact:
                fact.touch()
                results.append(fact)

        return results

    def _keyword_search(self, query: str, top_k: int) -> List[SemanticFact]:
        """Keyword-based search fallback."""
        query_terms = set(query.lower().split())
        scored = []

        for fact in self._facts.values():
            content_terms = set(fact.content.lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                scored.append((overlap / len(query_terms), fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    @property
    def size(self) -> int:
        """Number of stored facts."""
        return len(self._facts)


# =============================================================================
# Procedural Memory Store
# =============================================================================


class ProceduralMemoryStore:
    """
    Procedural memory for learned action patterns.

    Stores successful strategies that can be retrieved for similar queries.
    """

    def __init__(self, max_procedures: int = 500):
        """Initialize procedural memory."""
        self._procedures: Dict[str, Procedure] = {}
        self._max_procedures = max_procedures

    def store(self, procedure: Procedure) -> str:
        """Store a procedure."""
        if not procedure.id:
            procedure.id = hashlib.sha256(procedure.query_pattern.encode()).hexdigest()[:12]

        self._procedures[procedure.id] = procedure
        return procedure.id

    def recall(
        self,
        query: str,
        min_success_rate: float = 0.7,
    ) -> Optional[Procedure]:
        """Find applicable procedure for query."""
        best_match = None
        best_score = 0.0

        for procedure in self._procedures.values():
            if procedure.success_rate < min_success_rate:
                continue

            score = self._pattern_match_score(query, procedure.query_pattern)
            if score > best_score:
                best_score = score
                best_match = procedure

        if best_match:
            best_match.use_count += 1
            best_match.touch()

        return best_match

    def update_success_rate(self, procedure_id: str, success: bool) -> None:
        """Update success rate for a procedure."""
        if procedure_id in self._procedures:
            proc = self._procedures[procedure_id]
            # Exponential moving average
            alpha = 0.1
            proc.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * proc.success_rate

    def get_top_procedures(self, n: int = 10) -> List[Procedure]:
        """Get top n procedures by usage and success."""
        sorted_procs = sorted(
            self._procedures.values(),
            key=lambda p: (p.success_rate, p.use_count),
            reverse=True,
        )
        return sorted_procs[:n]

    def _pattern_match_score(self, query: str, pattern: str) -> float:
        """Score how well query matches pattern."""
        query_terms = set(query.lower().split())
        pattern_terms = set(pattern.lower().split())
        if not pattern_terms:
            return 0.0
        return len(query_terms & pattern_terms) / len(pattern_terms)

    @property
    def size(self) -> int:
        """Number of stored procedures."""
        return len(self._procedures)


# =============================================================================
# Unified Memory System
# =============================================================================


class AgentMemorySystem:
    """
    Unified memory system integrating all memory types.

    Provides a single interface for:
    - Working memory (short-term)
    - Episodic memory (interactions)
    - Semantic memory (knowledge)
    - Procedural memory (patterns)
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
        working_capacity: int = 10,
        episodic_capacity: int = 1000,
        semantic_capacity: int = 5000,
        procedural_capacity: int = 500,
    ):
        """Initialize the unified memory system."""
        self.working = WorkingMemory(capacity=working_capacity)
        self.episodic = EpisodicMemoryStore(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            max_episodes=episodic_capacity,
        )
        self.semantic = SemanticMemoryStore(
            embedding_provider=embedding_provider,
            max_facts=semantic_capacity,
        )
        self.procedural = ProceduralMemoryStore(max_procedures=procedural_capacity)

        self._embedding_provider = embedding_provider

    # -------------------------------------------------------------------------
    # Unified Operations
    # -------------------------------------------------------------------------

    async def remember(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Store content in appropriate memory.

        Args:
            content: Content to store
            memory_type: Type of memory to use
            metadata: Additional metadata
            **kwargs: Type-specific arguments

        Returns:
            Memory item ID
        """
        if memory_type == MemoryType.WORKING:
            item = MemoryItem(
                id=str(time.time()),
                content=content,
                memory_type=memory_type,
                metadata=metadata or {},
            )
            self.working.add(item)
            return item.id

        elif memory_type == MemoryType.EPISODIC:
            episode = Episode(
                id="",
                content=content,
                query=kwargs.get("query", ""),
                response=kwargs.get("response", ""),
                context_docs=kwargs.get("context_docs", []),
                tools_used=kwargs.get("tools_used", []),
                success=kwargs.get("success", True),
                metadata=metadata or {},
            )
            return await self.episodic.store(episode)

        elif memory_type == MemoryType.SEMANTIC:
            fact = SemanticFact(
                id="",
                content=content,
                source=kwargs.get("source", ""),
                confidence=kwargs.get("confidence", 1.0),
                metadata=metadata or {},
            )
            return await self.semantic.store(fact)

        elif memory_type == MemoryType.PROCEDURAL:
            procedure = Procedure(
                id="",
                content=content,
                query_pattern=kwargs.get("query_pattern", content),
                action_sequence=kwargs.get("action_sequence", []),
                success_rate=kwargs.get("success_rate", 1.0),
                metadata=metadata or {},
            )
            return self.procedural.store(procedure)

        raise ValueError(f"Unknown memory type: {memory_type}")

    async def recall(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        top_k: int = 5,
    ) -> Dict[MemoryType, List[MemoryItem]]:
        """
        Recall relevant memories across all types.

        Args:
            query: Query to match
            memory_types: Specific types to search (None = all)
            top_k: Results per type

        Returns:
            Dict mapping memory types to results
        """
        types = memory_types or [
            MemoryType.WORKING,
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
        ]

        results: Dict[MemoryType, List[MemoryItem]] = {}

        if MemoryType.WORKING in types:
            results[MemoryType.WORKING] = self.working.get_recent(top_k)

        if MemoryType.EPISODIC in types:
            results[MemoryType.EPISODIC] = await self.episodic.recall(query, top_k)

        if MemoryType.SEMANTIC in types:
            results[MemoryType.SEMANTIC] = await self.semantic.recall(query, top_k)

        if MemoryType.PROCEDURAL in types:
            proc = self.procedural.recall(query)
            results[MemoryType.PROCEDURAL] = [proc] if proc else []

        return results

    async def get_context(
        self,
        query: str,
        max_tokens: int = 2000,
        include_episodic: bool = True,
        include_semantic: bool = True,
    ) -> str:
        """
        Get combined context from all memory types.

        Args:
            query: Current query
            max_tokens: Maximum tokens in context
            include_episodic: Include episodic memories
            include_semantic: Include semantic facts

        Returns:
            Formatted context string
        """
        sections = []
        tokens_used = 0

        # Working memory (most recent context)
        working_context = self.working.get_context(max_tokens=max_tokens // 3)
        if working_context:
            sections.append(f"## Current Context\n{working_context}")
            tokens_used += len(working_context) // 4

        # Episodic memory (relevant past interactions)
        if include_episodic and tokens_used < max_tokens:
            remaining = (max_tokens - tokens_used) // 4
            episodes = await self.episodic.recall(query, top_k=3)
            if episodes:
                episode_text = "\n".join(
                    f"- Q: {e.query}\n  A: {e.response[:200]}..." for e in episodes
                )
                if len(episode_text) // 4 <= remaining:
                    sections.append(f"## Relevant Past Interactions\n{episode_text}")
                    tokens_used += len(episode_text) // 4

        # Semantic memory (relevant knowledge)
        if include_semantic and tokens_used < max_tokens:
            remaining = max_tokens - tokens_used
            facts = await self.semantic.recall(query, top_k=5)
            if facts:
                fact_text = "\n".join(f"- {f.content}" for f in facts)
                if len(fact_text) // 4 <= remaining:
                    sections.append(f"## Relevant Knowledge\n{fact_text}")

        return "\n\n".join(sections)

    def clear_working_memory(self) -> None:
        """Clear working memory only."""
        self.working.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "working_memory_size": self.working.size,
            "working_memory_tokens": self.working.token_count,
            "episodic_memory_size": self.episodic.size,
            "semantic_memory_size": self.semantic.size,
            "procedural_memory_size": self.procedural.size,
        }

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    async def store_episode(
        self,
        query: str,
        answer: str,
        context: Optional[List[Dict[str, Any]]] = None,
        success: bool = True,
        feedback_score: Optional[float] = None,
    ) -> str:
        """Convenience method to store an episode."""
        episode = Episode(
            id="",
            content=f"Q: {query}\nA: {answer}",
            query=query,
            response=answer,
            context_docs=context or [],
            success=success,
            feedback_score=feedback_score,
        )
        return await self.episodic.store(episode)

    async def store_fact(
        self,
        content: str,
        source: str = "",
        confidence: float = 1.0,
    ) -> str:
        """Convenience method to store a semantic fact."""
        fact = SemanticFact(
            id="",
            content=content,
            source=source,
            confidence=confidence,
        )
        return await self.semantic.store(fact)

    def store_procedure(
        self,
        pattern: str,
        actions: List[Dict[str, Any]],
        success_rate: float = 1.0,
    ) -> str:
        """Convenience method to store a procedure."""
        procedure = Procedure(
            id="",
            content=f"Pattern: {pattern}",
            query_pattern=pattern,
            action_sequence=actions,
            success_rate=success_rate,
        )
        return self.procedural.store(procedure)
