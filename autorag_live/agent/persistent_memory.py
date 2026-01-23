"""
Persistent agent memory with episodic recall.

State-of-the-art memory system for agentic RAG:
- Long-term memory storage beyond session
- Episodic recall of past interactions
- Semantic search over memory traces
- Memory consolidation and forgetting

Based on:
- "MemPrompt: Memory-assisted Prompt Editing" (Madaan et al., 2022)
- "Generative Agents: Interactive Simulacra" (Park et al., 2023)
- "Reflexion: Language Agents with Verbal Reinforcement" (Shinn et al., 2023)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for persistent memory."""

    # Storage
    db_path: str = "agent_memory.db"
    enable_persistence: bool = True

    # Memory capacity
    max_short_term: int = 50  # Recent interactions
    max_long_term: int = 1000  # Persistent memories
    max_episodic: int = 100  # Episodic memories

    # Consolidation
    consolidation_threshold: int = 20  # Consolidate after N interactions
    importance_threshold: float = 0.7  # Keep memories above this importance

    # Retrieval
    semantic_search_k: int = 5
    temporal_decay_factor: float = 0.95  # Recent memories weighted higher

    # Embedding
    embedding_dim: int = 768


@dataclass
class Memory:
    """A single memory entry."""

    memory_id: str
    content: str
    memory_type: str  # "interaction", "observation", "reflection", "plan"
    timestamp: float = field(default_factory=time.time)

    # Metadata
    query: Optional[str] = None
    response: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    # Importance and relevance
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    # Embedding for semantic search
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp,
            "query": self.query,
            "response": self.response,
            "context": self.context,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Memory:
        """Create from dictionary."""
        return cls(**data)


class MemoryStore:
    """
    Persistent memory storage with SQLite backend.

    Provides efficient storage and retrieval of agent memories
    across sessions.
    """

    def __init__(self, db_path: str = "agent_memory.db"):
        """
        Initialize memory store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

        # Create memories table
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                query TEXT,
                response TEXT,
                context TEXT,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                embedding BLOB
            )
        """
        )

        # Create index on timestamp for temporal queries
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON memories(timestamp DESC)
        """
        )

        # Create index on memory type
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_type
            ON memories(memory_type)
        """
        )

        self._conn.commit()
        logger.info(f"Initialized memory database at {self.db_path}")

    async def store(self, memory: Memory) -> None:
        """
        Store a memory.

        Args:
            memory: Memory to store
        """
        # Serialize embedding
        embedding_blob = None
        if memory.embedding:
            embedding_blob = json.dumps(memory.embedding)

        # Serialize context
        context_json = json.dumps(memory.context)

        await asyncio.to_thread(
            self._conn.execute,
            """
            INSERT OR REPLACE INTO memories
            (memory_id, content, memory_type, timestamp, query, response,
             context, importance, access_count, last_accessed, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                memory.memory_id,
                memory.content,
                memory.memory_type,
                memory.timestamp,
                memory.query,
                memory.response,
                context_json,
                memory.importance,
                memory.access_count,
                memory.last_accessed,
                embedding_blob,
            ),
        )

        await asyncio.to_thread(self._conn.commit)

    async def retrieve(
        self,
        memory_id: str,
    ) -> Optional[Memory]:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory if found
        """
        cursor = await asyncio.to_thread(
            self._conn.execute,
            "SELECT * FROM memories WHERE memory_id = ?",
            (memory_id,),
        )

        row = await asyncio.to_thread(cursor.fetchone)

        if row:
            return self._row_to_memory(row)

        return None

    async def retrieve_recent(
        self,
        limit: int = 10,
        memory_type: Optional[str] = None,
    ) -> List[Memory]:
        """
        Retrieve recent memories.

        Args:
            limit: Maximum number of memories
            memory_type: Filter by memory type

        Returns:
            List of memories
        """
        if memory_type:
            cursor = await asyncio.to_thread(
                self._conn.execute,
                """
                SELECT * FROM memories
                WHERE memory_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (memory_type, limit),
            )
        else:
            cursor = await asyncio.to_thread(
                self._conn.execute,
                "SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )

        rows = await asyncio.to_thread(cursor.fetchall)
        return [self._row_to_memory(row) for row in rows]

    async def retrieve_by_importance(
        self,
        min_importance: float = 0.7,
        limit: int = 100,
    ) -> List[Memory]:
        """
        Retrieve high-importance memories.

        Args:
            min_importance: Minimum importance threshold
            limit: Maximum number of memories

        Returns:
            List of important memories
        """
        cursor = await asyncio.to_thread(
            self._conn.execute,
            """
            SELECT * FROM memories
            WHERE importance >= ?
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        """,
            (min_importance, limit),
        )

        rows = await asyncio.to_thread(cursor.fetchall)
        return [self._row_to_memory(row) for row in rows]

    async def update_access(self, memory_id: str) -> None:
        """Update access count and timestamp for a memory."""
        await asyncio.to_thread(
            self._conn.execute,
            """
            UPDATE memories
            SET access_count = access_count + 1,
                last_accessed = ?
            WHERE memory_id = ?
        """,
            (time.time(), memory_id),
        )

        await asyncio.to_thread(self._conn.commit)

    async def delete_old(
        self,
        older_than_days: int = 30,
        min_importance: float = 0.5,
    ) -> int:
        """
        Delete old, low-importance memories.

        Args:
            older_than_days: Delete memories older than this
            min_importance: Keep memories above this importance

        Returns:
            Number of deleted memories
        """
        cutoff_time = time.time() - (older_than_days * 86400)

        cursor = await asyncio.to_thread(
            self._conn.execute,
            """
            DELETE FROM memories
            WHERE timestamp < ? AND importance < ?
        """,
            (cutoff_time, min_importance),
        )

        deleted = cursor.rowcount
        await asyncio.to_thread(self._conn.commit)

        logger.info(f"Deleted {deleted} old memories")
        return deleted

    def _row_to_memory(self, row: Tuple) -> Memory:
        """Convert database row to Memory object."""
        (
            memory_id,
            content,
            memory_type,
            timestamp,
            query,
            response,
            context_json,
            importance,
            access_count,
            last_accessed,
            embedding_blob,
        ) = row

        # Deserialize context
        context = json.loads(context_json) if context_json else {}

        # Deserialize embedding
        embedding = json.loads(embedding_blob) if embedding_blob else None

        return Memory(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=timestamp,
            query=query,
            response=response,
            context=context,
            importance=importance,
            access_count=access_count,
            last_accessed=last_accessed,
            embedding=embedding,
        )

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()


class PersistentAgentMemory:
    """
    Persistent memory system for agents with episodic recall.

    Provides:
    - Short-term memory (working memory for current session)
    - Long-term memory (persistent across sessions)
    - Episodic memory (important past experiences)
    - Semantic search over memories

    Example:
        >>> memory = PersistentAgentMemory()
        >>> await memory.add_interaction("What is AI?", "AI is...")
        >>> memories = await memory.recall("artificial intelligence", k=5)
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embed_fn: Optional[Any] = None,
    ):
        """
        Initialize persistent memory.

        Args:
            config: Memory configuration
            embed_fn: Function to generate embeddings
        """
        self.config = config or MemoryConfig()
        self.embed_fn = embed_fn

        # Memory stores
        self.short_term: List[Memory] = []
        self.store = MemoryStore(self.config.db_path)

        # Statistics
        self._interaction_count = 0

    async def add_interaction(
        self,
        query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
    ) -> str:
        """
        Add an interaction to memory.

        Args:
            query: User query
            response: Agent response
            context: Additional context
            importance: Importance score (auto-computed if None)

        Returns:
            Memory ID
        """
        # Compute importance if not provided
        if importance is None:
            importance = self._compute_importance(query, response)

        # Create memory
        memory_id = f"interaction_{int(time.time() * 1000)}"
        content = f"Q: {query}\nA: {response}"

        # Generate embedding
        embedding = None
        if self.embed_fn:
            embedding = await self._embed_text(content)

        memory = Memory(
            memory_id=memory_id,
            content=content,
            memory_type="interaction",
            query=query,
            response=response,
            context=context or {},
            importance=importance,
            embedding=embedding,
        )

        # Add to short-term memory
        self.short_term.append(memory)
        if len(self.short_term) > self.config.max_short_term:
            self.short_term.pop(0)

        # Store persistently
        if self.config.enable_persistence:
            await self.store.store(memory)

        self._interaction_count += 1

        # Consolidate memories periodically
        if self._interaction_count % self.config.consolidation_threshold == 0:
            await self._consolidate()

        return memory_id

    async def recall(
        self,
        query: str,
        k: int = 5,
        memory_types: Optional[List[str]] = None,
    ) -> List[Memory]:
        """
        Recall relevant memories for a query.

        Combines:
        - Semantic similarity
        - Temporal recency
        - Access frequency

        Args:
            query: Query text
            k: Number of memories to retrieve
            memory_types: Filter by memory types

        Returns:
            List of relevant memories
        """
        # Get query embedding
        query_embedding = None
        if self.embed_fn:
            query_embedding = await self._embed_text(query)

        # Retrieve candidates from long-term memory
        candidates = await self.store.retrieve_recent(limit=100)

        # Filter by type
        if memory_types:
            candidates = [m for m in candidates if m.memory_type in memory_types]

        # Add short-term memories
        candidates.extend(self.short_term)

        # Score and rank
        scored_memories = []
        for memory in candidates:
            score = self._compute_relevance(query_embedding, memory)
            scored_memories.append((score, memory))

        # Sort by relevance
        scored_memories.sort(key=lambda x: x[0], reverse=True)

        # Update access counts
        top_memories = [m for _, m in scored_memories[:k]]
        for memory in top_memories:
            await self.store.update_access(memory.memory_id)

        return top_memories

    def _compute_importance(self, query: str, response: str) -> float:
        """Compute importance score for an interaction."""
        # Heuristic: longer responses are more important
        response_length = len(response.split())

        # Questions are more important
        has_question = "?" in query

        importance = 0.5
        importance += min(0.3, response_length / 500)  # Length factor
        importance += 0.2 if has_question else 0.0

        return min(1.0, importance)

    def _compute_relevance(
        self,
        query_embedding: Optional[List[float]],
        memory: Memory,
    ) -> float:
        """
        Compute relevance score combining semantic, temporal, and access factors.

        Score = 0.5 * semantic + 0.3 * temporal + 0.2 * access
        """
        # Semantic similarity
        semantic_score = 0.5
        if query_embedding and memory.embedding:
            similarity = self._cosine_similarity(query_embedding, memory.embedding)
            semantic_score = (similarity + 1) / 2  # Normalize to [0, 1]

        # Temporal decay (recent memories weighted higher)
        age_seconds = time.time() - memory.timestamp
        age_days = age_seconds / 86400
        temporal_score = self.config.temporal_decay_factor**age_days

        # Access frequency
        access_score = min(1.0, memory.access_count / 10)

        # Combined score
        relevance = 0.5 * semantic_score + 0.3 * temporal_score + 0.2 * access_score

        # Boost by importance
        relevance *= memory.importance

        return relevance

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)

        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))

    async def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self.embed_fn:
            if asyncio.iscoroutinefunction(self.embed_fn):
                return await self.embed_fn(text)
            else:
                return await asyncio.to_thread(self.embed_fn, text)

        # Fallback: random embedding
        return list(np.random.randn(self.config.embedding_dim))

    async def _consolidate(self) -> None:
        """
        Consolidate memories by removing low-importance old memories.

        Mimics biological memory consolidation during sleep.
        """
        logger.info("Consolidating memories...")

        # Delete old, low-importance memories
        deleted = await self.store.delete_old(
            older_than_days=30, min_importance=self.config.importance_threshold
        )

        logger.info(f"Consolidated: removed {deleted} memories")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "short_term_size": len(self.short_term),
            "interaction_count": self._interaction_count,
            "db_path": str(self.config.db_path),
        }

    def close(self) -> None:
        """Close memory store."""
        self.store.close()
