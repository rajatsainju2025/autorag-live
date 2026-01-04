"""Semantic query caching with fuzzy matching and embedding similarity for deduplication.

This module provides intelligent query caching that detects semantically similar
queries and reuses embeddings from cached results, reducing redundant computation.

Features:
    - Levenshtein distance-based fuzzy matching
    - Embedding-based cosine similarity lookup
    - Configurable similarity threshold
    - Thread-safe and async-safe cache operations
    - Automatic cache eviction when full (LRU)
    - Integration with existing embedding cache
    - RAG-specific caching for query-answer pairs
    - TTL-based expiration

Example:
    >>> # String-based similarity
    >>> cache = SemanticQueryCache(threshold=0.9)
    >>> similar_query = cache.find_similar("original query", all_queries)

    >>> # Embedding-based similarity
    >>> embed_cache = EmbeddingSemanticCache(embedder, threshold=0.95)
    >>> result = await embed_cache.get_or_compute("What is ML?", compute_fn)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

# Type variable for generic cache values
T = TypeVar("T")


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Levenshtein distance (minimum edit distance)
    """
    # Early exit for identical strings
    if s1 == s2:
        return 0

    len1, len2 = len(s1), len(s2)

    # Handle empty strings
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Use dynamic programming for efficiency
    # Only keep two rows to reduce memory usage
    prev_row = list(range(len2 + 1))
    curr_row = [0] * (len2 + 1)

    for i in range(1, len1 + 1):
        curr_row[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr_row[j] = min(
                curr_row[j - 1] + 1,  # Insertion
                prev_row[j] + 1,  # Deletion
                prev_row[j - 1] + cost,  # Substitution
            )
        prev_row, curr_row = curr_row, prev_row

    return prev_row[len2]


def similarity_ratio(s1: str, s2: str) -> float:
    """Compute similarity ratio (0-1) between two strings.

    Uses Levenshtein distance normalized by max string length.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity ratio from 0.0 (completely different) to 1.0 (identical)
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0  # Both empty strings are identical

    distance = levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


class SemanticQueryCache:
    """Thread-safe semantic query cache with fuzzy matching."""

    def __init__(
        self,
        max_size: int = 256,
        threshold: float = 0.85,
        ttl_seconds: Optional[float] = None,
    ):
        """Initialize semantic query cache.

        Args:
            max_size: Maximum number of cached queries
            threshold: Similarity threshold for fuzzy matching (0-1)
            ttl_seconds: Time-to-live for cache entries (None = no expiration)
        """
        self.max_size = max_size
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds

        # Cache: query -> embedding
        self.cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        # Metadata: query -> (timestamp, access_count)
        self.metadata: dict = {}
        self.lock = threading.RLock()

        self.hits = 0
        self.misses = 0
        self.fuzzy_hits = 0

    @staticmethod
    def _make_key(query: str) -> str:
        """Create a cache key from query string."""
        return hashlib.md5(query.encode()).hexdigest()

    def get_exact(self, query: str) -> Optional[np.ndarray]:
        """Get exact match from cache.

        Args:
            query: Query string

        Returns:
            Cached embedding or None
        """
        with self.lock:
            key = self._make_key(query)
            if key in self.cache:
                embedding = self.cache[key]
                self.cache.move_to_end(key)
                self.hits += 1
                return embedding

            self.misses += 1
            return None

    def find_similar(self, query: str, candidate_queries: List[str]) -> Optional[str]:
        """Find similar query from candidates using fuzzy matching.

        Args:
            query: Target query
            candidate_queries: List of candidate queries to search

        Returns:
            Most similar query if similarity > threshold, else None
        """
        if not candidate_queries:
            return None

        best_match = None
        best_score = self.threshold

        for candidate in candidate_queries:
            score = similarity_ratio(query.lower(), candidate.lower())
            if score > best_score:
                best_score = score
                best_match = candidate

        return best_match

    def get_fuzzy(self, query: str) -> Optional[np.ndarray]:
        """Get fuzzy match from cache.

        Searches all cached queries for a similar match above threshold.

        Args:
            query: Query string

        Returns:
            Embedding from similar cached query, or None
        """
        with self.lock:
            cached_queries = list(self.cache.keys())

            if not cached_queries:
                return None

            # Note: Current implementation requires tracking original query strings
            # alongside cache keys for fuzzy matching. For now, we return None.
            # TODO: Extend metadata to store both key and original query string
            # for full fuzzy matching support.

            # Fall back to None if no good match found
            return None

    def put(self, query: str, embedding: np.ndarray) -> None:
        """Cache query embedding.

        Args:
            query: Query string
            embedding: Embedding array
        """
        import time

        with self.lock:
            key = self._make_key(query)

            # Remove old entry if exists
            if key in self.cache:
                del self.cache[key]

            # Add new entry
            self.cache[key] = embedding
            self.metadata[key] = {"timestamp": time.time(), "query": query}

            # Evict oldest if over capacity
            while len(self.cache) > self.max_size:
                old_key = next(iter(self.cache))
                del self.cache[old_key]
                if old_key in self.metadata:
                    del self.metadata[old_key]

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache hit/miss stats
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                "size": len(self.cache),
                "hits": self.hits,
                "misses": self.misses,
                "fuzzy_hits": self.fuzzy_hits,
                "total": total,
                "hit_rate": hit_rate,
            }

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.metadata.clear()
            self.hits = 0
            self.misses = 0
            self.fuzzy_hits = 0


class MultiLayerCache:
    """
    Multi-layer cache hierarchy (L1: memory, L2: disk).

    Provides fast access with semantic matching and persistence.
    """

    def __init__(
        self,
        l1_size: int = 100,
        similarity_threshold: float = 0.85,
        enable_l2: bool = False,
    ):
        """Initialize multi-layer cache."""
        self.l1_cache = SemanticQueryCache(
            max_size=l1_size,
            threshold=similarity_threshold,
        )
        self.l2_cache: OrderedDict = OrderedDict()  # Disk cache (simulated)
        self.enable_l2 = enable_l2
        self.layer1_hits = 0
        self.layer2_hits = 0

    def get_exact(self, query: str) -> Optional[np.ndarray]:
        """Retrieve from multi-layer cache."""
        # Try L1 first
        result = self.l1_cache.get_exact(query)
        if result is not None:
            self.layer1_hits += 1
            return result

        # Try L2 if enabled
        if self.enable_l2:
            key = SemanticQueryCache._make_key(query)
            if key in self.l2_cache:
                self.layer2_hits += 1
                result = self.l2_cache[key]
                # Promote to L1
                self.l1_cache.put(query, result)
                return result

        return None

    def put(self, query: str, embedding: np.ndarray) -> None:
        """Store in cache layers."""
        self.l1_cache.put(query, embedding)
        if self.enable_l2:
            key = SemanticQueryCache._make_key(query)
            self.l2_cache[key] = embedding

    def clear(self) -> None:
        """Clear all layers."""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.layer1_hits = 0
        self.layer2_hits = 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "l1_hits": self.layer1_hits,
            "l2_hits": self.layer2_hits,
            "l1_size": len(self.l1_cache.cache),
            "l2_size": len(self.l2_cache),
        }


class ResultDeduplicator:
    """
    Deduplicates results across cache and retrieval operations.

    Identifies and merges similar/identical results.
    """

    def __init__(self, similarity_threshold: float = 0.9):
        """Initialize deduplicator."""
        self.similarity_threshold = similarity_threshold

    def deduplicate(self, results: List[str]) -> List[str]:
        """
        Deduplicate results based on similarity.

        Args:
            results: List of results

        Returns:
            Deduplicated results
        """
        if not results:
            return []

        unique_results = []
        used_indices = set()

        for i, result in enumerate(results):
            if i in used_indices:
                continue

            unique_results.append(result)

            # Find and mark similar results
            for j in range(i + 1, len(results)):
                if j not in used_indices:
                    dist = levenshtein_distance(result.lower(), results[j].lower())
                    max_len = max(len(result), len(results[j]))
                    similarity = 1.0 - (dist / max_len)

                    if similarity >= self.similarity_threshold:
                        used_indices.add(j)

        return unique_results


# =============================================================================
# Embedding-Based Semantic Cache (Vector Similarity)
# =============================================================================


@dataclass
class EmbeddingCacheEntry(Generic[T]):
    """
    A cache entry with embedding and metadata.

    Attributes:
        key: Cache key (query text)
        key_hash: Hash of the key
        embedding: Query embedding vector
        value: Cached value
        created_at: When entry was created
        expires_at: When entry expires
        hit_count: Number of cache hits
        last_accessed: Last access time
    """

    key: str
    key_hash: str
    embedding: List[float]
    value: T
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def touch(self) -> None:
        """Update last accessed time and hit count."""
        self.last_accessed = datetime.now()
        self.hit_count += 1


@dataclass
class EmbeddingCacheStats:
    """
    Cache statistics for embedding-based cache.

    Attributes:
        total_queries: Total queries processed
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        avg_similarity: Average similarity for hits
        entries_count: Current entries in cache
        evictions: Number of evictions
        total_latency_saved_ms: Estimated latency saved
    """

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_similarity: float = 0.0
    entries_count: int = 0
    evictions: int = 0
    total_latency_saved_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.cache_hits / self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.hit_rate,
            "avg_similarity": self.avg_similarity,
            "entries_count": self.entries_count,
            "evictions": self.evictions,
            "latency_saved_ms": self.total_latency_saved_ms,
        }


@dataclass
class EmbeddingCacheLookupResult(Generic[T]):
    """
    Result of an embedding cache lookup.

    Attributes:
        hit: Whether cache hit occurred
        value: Cached value if hit
        similarity: Similarity score if hit
        entry: The cache entry if hit
    """

    hit: bool
    value: Optional[T] = None
    similarity: float = 0.0
    entry: Optional[EmbeddingCacheEntry[T]] = None


class VectorSimilarity:
    """Calculates similarity between embedding vectors."""

    @staticmethod
    def cosine_similarity(
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """
        Calculate cosine similarity.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score (0-1)
        """
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)

        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def euclidean_distance(
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """
        Calculate Euclidean distance.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Distance (lower is more similar)
        """
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)
        return float(np.linalg.norm(arr1 - arr2))


class EmbeddingSemanticCache(Generic[T]):
    """
    Semantic cache with embedding-based similarity lookup.

    Caches results based on query semantic similarity using vector
    embeddings rather than exact string matching.

    Example:
        >>> cache = EmbeddingSemanticCache(embedder)
        >>> # First call computes and caches
        >>> result1 = await cache.get_or_compute("What is ML?", compute_fn)
        >>> # Similar query hits cache
        >>> result2 = await cache.get_or_compute("What is machine learning?", compute_fn)
    """

    def __init__(
        self,
        embedder: Optional[Callable[[str], List[float]]] = None,
        *,
        similarity_threshold: float = 0.95,
        ttl_seconds: Optional[int] = 3600,
        max_entries: int = 1000,
    ):
        """
        Initialize embedding semantic cache.

        Args:
            embedder: Function to compute embeddings
            similarity_threshold: Minimum similarity for cache hit
            ttl_seconds: Time-to-live for entries (None = no expiration)
            max_entries: Maximum cache entries
        """
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

        self._cache: OrderedDict[str, EmbeddingCacheEntry[T]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._thread_lock = threading.RLock()

        self.calculator = VectorSimilarity()
        self.stats = EmbeddingCacheStats()

    async def get(self, query: str) -> EmbeddingCacheLookupResult[T]:
        """
        Look up query in cache by embedding similarity.

        Args:
            query: Query string

        Returns:
            EmbeddingCacheLookupResult indicating hit/miss
        """
        self.stats.total_queries += 1

        # Get query embedding
        query_embedding = await self._get_embedding(query)
        if query_embedding is None:
            self.stats.cache_misses += 1
            return EmbeddingCacheLookupResult(hit=False)

        # Search for similar entries
        async with self._lock:
            entries = [e for e in self._cache.values() if not e.is_expired()]

        best_match: Optional[EmbeddingCacheEntry[T]] = None
        best_similarity = 0.0

        for entry in entries:
            similarity = self.calculator.cosine_similarity(
                query_embedding,
                entry.embedding,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        # Check if above threshold
        if best_match and best_similarity >= self.similarity_threshold:
            self.stats.cache_hits += 1
            self._update_avg_similarity(best_similarity)
            best_match.touch()

            return EmbeddingCacheLookupResult(
                hit=True,
                value=best_match.value,
                similarity=best_similarity,
                entry=best_match,
            )

        self.stats.cache_misses += 1
        return EmbeddingCacheLookupResult(hit=False, similarity=best_similarity)

    async def set(
        self,
        query: str,
        value: T,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Store value in cache.

        Args:
            query: Query string
            value: Value to cache
            ttl_seconds: Override default TTL
        """
        query_embedding = await self._get_embedding(query)
        if query_embedding is None:
            return

        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None

        entry = EmbeddingCacheEntry(
            key=query,
            key_hash=self._hash_key(query),
            embedding=query_embedding,
            value=value,
            expires_at=expires_at,
        )

        async with self._lock:
            # LRU eviction if needed
            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)
                self.stats.evictions += 1

            self._cache[entry.key_hash] = entry
            self.stats.entries_count = len(self._cache)

    async def get_or_compute(
        self,
        query: str,
        compute_fn: Callable[[], Any],
        *,
        ttl_seconds: Optional[int] = None,
    ) -> T:
        """
        Get from cache or compute and store.

        Args:
            query: Query string
            compute_fn: Function to compute value if cache miss
            ttl_seconds: Override default TTL

        Returns:
            Cached or computed value
        """
        # Try cache first
        lookup = await self.get(query)

        if lookup.hit:
            # Estimate latency saved (assume compute takes ~500ms)
            self.stats.total_latency_saved_ms += 500
            return lookup.value  # type: ignore

        # Compute value
        if asyncio.iscoroutinefunction(compute_fn):
            value = await compute_fn()
        else:
            value = compute_fn()

        # Store in cache
        await self.set(query, value, ttl_seconds=ttl_seconds)

        return value

    async def invalidate(self, query: str) -> bool:
        """
        Invalidate cache entry.

        Args:
            query: Query to invalidate

        Returns:
            True if entry was found and removed
        """
        key_hash = self._hash_key(query)
        async with self._lock:
            if key_hash in self._cache:
                del self._cache[key_hash]
                self.stats.entries_count = len(self._cache)
                return True
            return False

    async def invalidate_similar(
        self,
        query: str,
        threshold: float = 0.9,
    ) -> int:
        """
        Invalidate entries similar to query.

        Args:
            query: Reference query
            threshold: Similarity threshold

        Returns:
            Number of entries invalidated
        """
        query_embedding = await self._get_embedding(query)
        if query_embedding is None:
            return 0

        async with self._lock:
            entries = list(self._cache.values())
            to_delete = []

            for entry in entries:
                similarity = self.calculator.cosine_similarity(
                    query_embedding,
                    entry.embedding,
                )
                if similarity >= threshold:
                    to_delete.append(entry.key_hash)

            for key in to_delete:
                if key in self._cache:
                    del self._cache[key]

            self.stats.entries_count = len(self._cache)
            return len(to_delete)

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self.stats.entries_count = 0

    def get_stats(self) -> EmbeddingCacheStats:
        """Get cache statistics."""
        self.stats.entries_count = len(self._cache)
        return self.stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = EmbeddingCacheStats()
        self.stats.entries_count = len(self._cache)

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text."""
        if self.embedder is None:
            # Use simple hash-based embedding for testing
            return self._simple_embedding(text)

        try:
            if asyncio.iscoroutinefunction(self.embedder):
                return await self.embedder(text)
            return self.embedder(text)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def _simple_embedding(self, text: str) -> List[float]:
        """
        Create a simple deterministic embedding for testing.

        Not for production use - just for cache key generation.
        """
        # Normalize text
        words = text.lower().split()

        # Create bag-of-words style embedding
        embedding = [0.0] * 128

        for i, word in enumerate(words):
            # Hash word to index
            h = hash(word) % 128
            embedding[h] += 1.0 / (i + 1)

        # Normalize
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def _hash_key(self, text: str) -> str:
        """Create hash key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _update_avg_similarity(self, similarity: float) -> None:
        """Update average similarity."""
        n = self.stats.cache_hits
        if n > 0:
            self.stats.avg_similarity = self.stats.avg_similarity * (n - 1) / n + similarity / n


# =============================================================================
# RAG-Specific Cache
# =============================================================================


@dataclass
class RAGCacheEntry:
    """Cache entry for RAG results."""

    query: str
    answer: str
    sources: List[str]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGCache(EmbeddingSemanticCache[RAGCacheEntry]):
    """
    Specialized semantic cache for RAG responses.

    Caches query-answer pairs with source attribution for efficient
    retrieval of previously answered questions.

    Example:
        >>> cache = RAGCache(embedder)
        >>> await cache.cache_response(
        ...     query="What is ML?",
        ...     answer="Machine learning is...",
        ...     sources=["doc1", "doc2"]
        ... )
        >>> result = await cache.get_cached_answer("What is machine learning?")
    """

    def __init__(
        self,
        embedder: Optional[Callable[[str], List[float]]] = None,
        *,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
    ):
        """Initialize RAG cache."""
        super().__init__(
            embedder=embedder,
            similarity_threshold=similarity_threshold,
            ttl_seconds=ttl_seconds,
        )

    async def cache_response(
        self,
        query: str,
        answer: str,
        sources: List[str],
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Cache a RAG response.

        Args:
            query: Original query
            answer: Generated answer
            sources: Source document IDs
            latency_ms: Generation latency
            metadata: Additional metadata
        """
        entry = RAGCacheEntry(
            query=query,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        await self.set(query, entry)

    async def get_cached_answer(
        self,
        query: str,
    ) -> Optional[RAGCacheEntry]:
        """
        Get cached answer for query.

        Args:
            query: User query

        Returns:
            RAGCacheEntry if cache hit
        """
        lookup = await self.get(query)
        if lookup.hit:
            return lookup.value
        return None


# =============================================================================
# Convenience Functions
# =============================================================================


def create_semantic_cache(
    embedder: Optional[Callable[[str], List[float]]] = None,
    threshold: float = 0.95,
    max_entries: int = 1000,
) -> EmbeddingSemanticCache[Any]:
    """
    Create an embedding-based semantic cache.

    Args:
        embedder: Embedding function
        threshold: Similarity threshold
        max_entries: Maximum entries

    Returns:
        EmbeddingSemanticCache instance
    """
    return EmbeddingSemanticCache(
        embedder=embedder,
        similarity_threshold=threshold,
        max_entries=max_entries,
    )


def create_rag_cache(
    embedder: Optional[Callable[[str], List[float]]] = None,
    threshold: float = 0.95,
    ttl_seconds: int = 3600,
) -> RAGCache:
    """
    Create a RAG-specific cache.

    Args:
        embedder: Embedding function
        threshold: Similarity threshold
        ttl_seconds: Cache TTL

    Returns:
        RAGCache instance
    """
    return RAGCache(
        embedder=embedder,
        similarity_threshold=threshold,
        ttl_seconds=ttl_seconds,
    )
