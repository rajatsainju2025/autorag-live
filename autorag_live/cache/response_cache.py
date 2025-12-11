"""
Response Caching Layer for Agentic RAG Pipeline.

Provides intelligent caching for RAG responses with TTL,
semantic similarity matching, and automatic invalidation.
"""

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class CacheStrategy(str, Enum):
    """Cache eviction strategies."""

    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    FIFO = "fifo"


class CacheStatus(str, Enum):
    """Cache operation status."""

    HIT = "hit"
    MISS = "miss"
    EXPIRED = "expired"
    EVICTED = "evicted"


@dataclass
class CacheEntry(Generic[T]):
    """A cached entry with metadata."""

    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class CacheResult(Generic[T]):
    """Result of a cache operation."""

    status: CacheStatus
    value: Optional[T] = None
    key: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_hit(self) -> bool:
        """Check if this was a cache hit."""
        return self.status == CacheStatus.HIT


class CacheBackend(ABC, Generic[T]):
    """Abstract cache backend interface."""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry[T]]:
        """Get an entry from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set an entry in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete an entry from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get number of entries."""
        pass


class InMemoryCache(CacheBackend[T]):
    """Thread-safe in-memory cache implementation."""

    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[float] = None,
    ):
        """Initialize in-memory cache."""
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        self._order: list[str] = []

    def get(self, key: str) -> Optional[CacheEntry[T]]:
        """Get an entry from cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if entry.is_expired:
                self._remove(key)
                return None

            entry.touch()

            if self.strategy == CacheStrategy.LRU:
                if key in self._order:
                    self._order.remove(key)
                self._order.append(key)

            return entry

    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set an entry in cache."""
        with self._lock:
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict()

            entry = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl or self.default_ttl,
            )
            self._cache[key] = entry

            if key not in self._order:
                self._order.append(key)

    def delete(self, key: str) -> bool:
        """Delete an entry from cache."""
        with self._lock:
            return self._remove(key)

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._order.clear()

    def size(self) -> int:
        """Get number of entries."""
        with self._lock:
            return len(self._cache)

    def _remove(self, key: str) -> bool:
        """Remove an entry (internal)."""
        if key in self._cache:
            del self._cache[key]
            if key in self._order:
                self._order.remove(key)
            return True
        return False

    def _evict(self) -> None:
        """Evict entries based on strategy."""
        if not self._cache:
            return

        if self.strategy == CacheStrategy.LRU:
            if self._order:
                key = self._order[0]
                self._remove(key)

        elif self.strategy == CacheStrategy.LFU:
            min_count = float("inf")
            min_key = None
            for key, entry in self._cache.items():
                if entry.access_count < min_count:
                    min_count = entry.access_count
                    min_key = key
            if min_key:
                self._remove(min_key)

        elif self.strategy == CacheStrategy.FIFO:
            if self._order:
                key = self._order[0]
                self._remove(key)

        elif self.strategy == CacheStrategy.TTL:
            for key, entry in list(self._cache.items()):
                if entry.is_expired:
                    self._remove(key)
                    return
            if self._order:
                key = self._order[0]
                self._remove(key)


class ResponseCache:
    """High-level response cache for RAG pipeline."""

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
        strategy: CacheStrategy = CacheStrategy.LRU,
        hash_function: Optional[Callable[[str], str]] = None,
    ):
        """Initialize response cache."""
        self.backend = backend or InMemoryCache(
            max_size=max_size,
            strategy=strategy,
            default_ttl=default_ttl,
        )
        self.default_ttl = default_ttl
        self.hash_function = hash_function or self._default_hash
        self._stats = CacheStats()

    def _default_hash(self, key: str) -> str:
        """Default hash function using MD5."""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, query: str, context: Optional[dict[str, Any]] = None) -> CacheResult:
        """Get cached response for a query."""
        start_time = time.time()
        key = self._build_key(query, context)

        entry = self.backend.get(key)
        latency = (time.time() - start_time) * 1000

        if entry is None:
            self._stats.record_miss()
            return CacheResult(
                status=CacheStatus.MISS,
                key=key,
                latency_ms=latency,
            )

        self._stats.record_hit()
        return CacheResult(
            status=CacheStatus.HIT,
            value=entry.value,
            key=key,
            latency_ms=latency,
            metadata=entry.metadata,
        )

    def set(
        self,
        query: str,
        response: Any,
        context: Optional[dict[str, Any]] = None,
        ttl: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Cache a response for a query."""
        key = self._build_key(query, context)
        self.backend.set(key, response, ttl or self.default_ttl)
        return key

    def invalidate(self, query: str, context: Optional[dict[str, Any]] = None) -> bool:
        """Invalidate a cached response."""
        key = self._build_key(query, context)
        return self.backend.delete(key)

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all entries matching a pattern."""
        count = 0
        if isinstance(self.backend, InMemoryCache):
            with self.backend._lock:
                keys_to_remove = [
                    k for k in self.backend._cache.keys() if pattern in k
                ]
                for key in keys_to_remove:
                    self.backend._remove(key)
                    count += 1
        return count

    def _build_key(self, query: str, context: Optional[dict[str, Any]] = None) -> str:
        """Build cache key from query and context."""
        key_parts = [query]
        if context:
            key_parts.append(json.dumps(context, sort_keys=True))
        combined = "|".join(key_parts)
        return self.hash_function(combined)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._stats.to_dict()

    def clear(self) -> None:
        """Clear the cache."""
        self.backend.clear()
        self._stats.reset()


class CacheStats:
    """Statistics for cache operations."""

    def __init__(self):
        """Initialize stats."""
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._start_time = datetime.now()

    def record_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._misses += 1

    def record_eviction(self) -> None:
        """Record a cache eviction."""
        with self._lock:
            self._evictions += 1

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self._hits + self._misses
        return self._hits / max(total, 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "total_requests": total,
                "hit_rate": self.hit_rate,
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
            }

    def reset(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._start_time = datetime.now()


class SemanticCache(ResponseCache):
    """Cache with semantic similarity matching."""

    def __init__(
        self,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
        similarity_threshold: float = 0.9,
        **kwargs,
    ):
        """Initialize semantic cache."""
        super().__init__(**kwargs)
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self._embeddings: dict[str, list[float]] = {}

    def get_similar(
        self, query: str, context: Optional[dict[str, Any]] = None
    ) -> CacheResult:
        """Get cached response with semantic similarity matching."""
        exact_result = self.get(query, context)
        if exact_result.is_hit:
            return exact_result

        if not self.embedding_fn:
            return exact_result

        query_embedding = self.embedding_fn(query)

        best_match = None
        best_similarity = 0.0

        if isinstance(self.backend, InMemoryCache):
            with self.backend._lock:
                for key, embedding in self._embeddings.items():
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = key

        if best_match and best_similarity >= self.similarity_threshold:
            entry = self.backend.get(best_match)
            if entry:
                self._stats.record_hit()
                return CacheResult(
                    status=CacheStatus.HIT,
                    value=entry.value,
                    key=best_match,
                    metadata={
                        "semantic_match": True,
                        "similarity": best_similarity,
                    },
                )

        self._stats.record_miss()
        return CacheResult(
            status=CacheStatus.MISS,
            metadata={"best_similarity": best_similarity},
        )

    def set(
        self,
        query: str,
        response: Any,
        context: Optional[dict[str, Any]] = None,
        ttl: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Cache response with embedding."""
        key = super().set(query, response, context, ttl, metadata)

        if self.embedding_fn:
            self._embeddings[key] = self.embedding_fn(query)

        return key

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


class TieredCache:
    """Multi-tier caching with L1/L2 architecture."""

    def __init__(
        self,
        l1_cache: Optional[ResponseCache] = None,
        l2_cache: Optional[ResponseCache] = None,
        l1_ttl: float = 300.0,
        l2_ttl: float = 3600.0,
    ):
        """Initialize tiered cache."""
        self.l1 = l1_cache or ResponseCache(
            max_size=100,
            default_ttl=l1_ttl,
            strategy=CacheStrategy.LRU,
        )
        self.l2 = l2_cache or ResponseCache(
            max_size=10000,
            default_ttl=l2_ttl,
            strategy=CacheStrategy.LFU,
        )

    def get(self, query: str, context: Optional[dict[str, Any]] = None) -> CacheResult:
        """Get from tiered cache (L1 first, then L2)."""
        l1_result = self.l1.get(query, context)
        if l1_result.is_hit:
            l1_result.metadata["tier"] = "L1"
            return l1_result

        l2_result = self.l2.get(query, context)
        if l2_result.is_hit:
            self.l1.set(query, l2_result.value, context)
            l2_result.metadata["tier"] = "L2"
            return l2_result

        return CacheResult(
            status=CacheStatus.MISS,
            metadata={"tier": "none"},
        )

    def set(
        self,
        query: str,
        response: Any,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Set in both cache tiers."""
        self.l1.set(query, response, context)
        self.l2.set(query, response, context)

    def invalidate(self, query: str, context: Optional[dict[str, Any]] = None) -> None:
        """Invalidate from both tiers."""
        self.l1.invalidate(query, context)
        self.l2.invalidate(query, context)

    def get_stats(self) -> dict[str, Any]:
        """Get combined statistics."""
        return {
            "l1": self.l1.get_stats(),
            "l2": self.l2.get_stats(),
        }


def cached_response(
    cache: Optional[ResponseCache] = None,
    ttl: Optional[float] = None,
    key_fn: Optional[Callable[..., str]] = None,
) -> Callable:
    """Decorator for caching function responses."""
    import functools

    _cache = cache or ResponseCache()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{args}:{kwargs}"

            result = _cache.get(cache_key)
            if result.is_hit and result.value is not None:
                return result.value  # type: ignore

            value = func(*args, **kwargs)
            _cache.set(cache_key, value, ttl=ttl)
            return value

        wrapper.cache = _cache  # type: ignore
        return wrapper

    return decorator


__all__ = [
    "CacheStrategy",
    "CacheStatus",
    "CacheEntry",
    "CacheResult",
    "CacheBackend",
    "InMemoryCache",
    "ResponseCache",
    "CacheStats",
    "SemanticCache",
    "TieredCache",
    "cached_response",
]
