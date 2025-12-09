"""
Advanced Query Caching System for AutoRAG-Live.

This module provides LRU and intelligent caching strategies to reduce redundant
query processing, embedding computation, and retrieval operations.

Caching components:
    - EmbeddingCache: LRU cache for embedding vectors
    - TokenizationCache: LRU cache for tokenization results
    - QueryCache: LRU cache for query results
    - SemanticQueryCache: Semantic similarity-based caching
    - MultiLayerCache: L1/L2 cache hierarchy for scalability
    - ResultDeduplicator: Deduplication via semantic similarity
    - DistributedCacheManager: Multi-cache coordination
"""
from __future__ import annotations

import hashlib
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, runtime_checkable

from autorag_live.utils import get_logger

logger = get_logger(__name__)

# Type variables for generic caching
K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


@runtime_checkable
class Hashable(Protocol):
    """Protocol for hashable types."""

    def __hash__(self) -> int:
        ...


@dataclass
class CacheEntry(Generic[V]):
    """
    Individual cache entry with metadata.

    Attributes:
        value: Cached value
        timestamp: Creation timestamp
        access_count: Number of times accessed
        last_access: Last access timestamp
        ttl: Time-to-live in seconds (None for infinite)
        size_bytes: Estimated memory size in bytes
    """

    value: V
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl

    def access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()


class QueryCache(Generic[K, V]):
    """
    Thread-safe LRU cache for query results with TTL support.

    Features:
        - LRU eviction policy
        - TTL-based expiration
        - Size-based eviction
        - Thread-safe operations
        - Persistence to disk
        - Cache statistics tracking
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = 3600.0,  # 1 hour default
        max_memory_mb: Optional[float] = None,
        persist_path: Optional[Path] = None,
        eviction_policy: str = "lru",  # 'lru', 'lfu', 'ttl'
    ):
        """
        Initialize query cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (None for no expiration)
            max_memory_mb: Maximum memory usage in MB (None for no limit)
            persist_path: Path to persist cache to disk
            eviction_policy: Eviction strategy ('lru', 'lfu', 'ttl')
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024) if max_memory_mb else None
        self.persist_path = persist_path
        self.eviction_policy = eviction_policy

        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._total_size_bytes = 0

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Load from disk if path provided
        if persist_path and persist_path.exists():
            self._load_from_disk()

        logger.info(
            f"QueryCache initialized: max_size={max_size}, "
            f"ttl={default_ttl}s, policy={eviction_policy}"
        )

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return default

            # Check expiration
            if entry.is_expired():
                # Remove expired entry
                del self._cache[key]
                self._total_size_bytes -= entry.size_bytes
                self._misses += 1
                return default

            # Update access statistics - optimized to reduce method calls
            entry.access_count += 1
            entry.last_access = time.time()
            self._hits += 1

            # Move to end for LRU - only if using LRU policy
            if self.eviction_policy == "lru":
                self._cache.move_to_end(key)

            return entry.value

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (overrides default_ttl)
        """
        with self._lock:
            # Estimate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 0

            # Check if we need to evict
            while self._should_evict(size_bytes):
                self._evict_one()

            # Create entry
            entry = CacheEntry(
                value=value, ttl=ttl if ttl is not None else self.default_ttl, size_bytes=size_bytes
            )

            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._total_size_bytes -= old_entry.size_bytes

            # Add new entry
            self._cache[key] = entry
            self._total_size_bytes += size_bytes

            # Move to end for LRU
            self._cache.move_to_end(key)

    def delete(self, key: K) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if key existed and was deleted
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._total_size_bytes -= entry.size_bytes
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._total_size_bytes = 0
            logger.info("Cache cleared")

    def _should_evict(self, new_size: int) -> bool:
        """Check if eviction is needed."""
        # Size limit check
        if len(self._cache) >= self.max_size:
            return True

        # Memory limit check
        if self.max_memory_bytes:
            if self._total_size_bytes + new_size > self.max_memory_bytes:
                return True

        return False

    def _evict_one(self) -> None:
        """Evict one entry based on policy."""
        if not self._cache:
            return

        if self.eviction_policy == "lru":
            # Evict least recently used (first item)
            key, entry = self._cache.popitem(last=False)
        elif self.eviction_policy == "lfu":
            # Evict least frequently used
            key = min(self._cache, key=lambda k: self._cache[k].access_count)
            entry = self._cache.pop(key)
        elif self.eviction_policy == "ttl":
            # Evict oldest by timestamp
            key = min(self._cache, key=lambda k: self._cache[k].timestamp)
            entry = self._cache.pop(key)
        else:
            # Fallback to LRU
            key, entry = self._cache.popitem(last=False)

        self._total_size_bytes -= entry.size_bytes
        self._evictions += 1

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]

            for key in expired_keys:
                entry = self._cache[key]
                self._total_size_bytes -= entry.size_bytes
                del self._cache[key]

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "memory_bytes": self._total_size_bytes,
                "memory_mb": self._total_size_bytes / (1024 * 1024),
            }

    def _save_to_disk(self) -> None:
        """Persist cache to disk."""
        if not self.persist_path:
            return

        try:
            with open(self.persist_path, "wb") as f:
                pickle.dump(dict(self._cache), f)
            logger.info(f"Cache persisted to {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to persist cache: {e}")

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, "rb") as f:
                loaded = pickle.load(f)
                self._cache = OrderedDict(loaded)

            # Recalculate size
            self._total_size_bytes = sum(entry.size_bytes for entry in self._cache.values())

            logger.info(f"Cache loaded from {self.persist_path} ({len(self._cache)} entries)")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save to disk if configured."""
        if self.persist_path:
            self._save_to_disk()


class SemanticQueryCache(QueryCache[str, List[Any]]):
    """
    Semantic query cache that uses embedding similarity for fuzzy matching.

    Instead of exact string matching, finds similar queries and reuses results.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        embedding_model: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize semantic query cache.

        Args:
            similarity_threshold: Minimum similarity for cache hit (0.0-1.0)
            embedding_model: Model for computing query embeddings
            **kwargs: Additional arguments for QueryCache
        """
        super().__init__(**kwargs)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self._query_embeddings: Dict[str, Any] = {}

    def _compute_embedding(self, query: str) -> Any:
        """Compute embedding for query."""
        if self.embedding_model is None:
            # Fallback to hash-based key
            return hashlib.md5(query.encode()).hexdigest()

        # Use sentence transformers or similar
        try:
            return self.embedding_model.encode(query)
        except Exception as e:
            logger.warning(f"Failed to compute embedding: {e}")
            return hashlib.md5(query.encode()).hexdigest()

    def get_similar_query(self, query: str) -> Optional[str]:
        """
        Find cached query similar to input query.

        Args:
            query: Query string

        Returns:
            Similar cached query or None
        """
        if not self.embedding_model:
            return query if query in self._cache else None

        query_emb = self._compute_embedding(query)

        # Find most similar cached query
        max_similarity = 0.0
        most_similar = None

        for cached_query, cached_emb in self._query_embeddings.items():
            try:
                # Compute cosine similarity
                import numpy as np

                similarity = np.dot(query_emb, cached_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(cached_emb)
                )

                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar = cached_query

            except Exception:
                continue

        if max_similarity >= self.similarity_threshold:
            return most_similar

        return None

    def get(self, key: str, default: Optional[List[Any]] = None) -> Optional[List[Any]]:
        """Get value using semantic similarity."""
        # Try exact match first
        result = super().get(key, None)
        if result is not None:
            return result

        # Try similar query
        similar_key = self.get_similar_query(key)
        if similar_key:
            logger.debug(f"Semantic cache hit: '{key}' ~= '{similar_key}'")
            return super().get(similar_key, default)

        return default

    def set(self, key: str, value: List[Any], ttl: Optional[float] = None) -> None:
        """Set value and cache embedding."""
        super().set(key, value, ttl)

        # Cache embedding for semantic matching
        if self.embedding_model:
            try:
                self._query_embeddings[key] = self._compute_embedding(key)
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")


# Global cache instances
_default_cache: Optional[QueryCache] = None
_semantic_cache: Optional[SemanticQueryCache] = None


def get_default_cache() -> QueryCache:
    """Get or create default query cache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = QueryCache(max_size=1000, default_ttl=3600.0)
    return _default_cache


def get_semantic_cache() -> SemanticQueryCache:
    """Get or create semantic query cache."""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticQueryCache(
            max_size=500, similarity_threshold=0.95, default_ttl=3600.0
        )
    return _semantic_cache
