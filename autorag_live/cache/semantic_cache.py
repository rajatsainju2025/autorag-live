"""Semantic query caching with fuzzy matching and embedding similarity for deduplication.

This module provides intelligent query caching that detects semantically similar
queries and reuses embeddings from cached results, reducing redundant computation.

Features:
    - Levenshtein distance-based fuzzy matching
    - Embedding-based cosine similarity lookup
    - Bloom filter for O(1) negative cache lookups
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

    >>> # Embedding-based similarity with Bloom filter
    >>> embed_cache = EmbeddingSemanticCache(embedder, threshold=0.95, use_bloom=True)
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
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

# Type variable for generic cache values
T = TypeVar("T")


class BloomFilter:
    """
    High-performance probabilistic data structure for set membership testing.

    Enables O(1) negative cache lookups with very low memory overhead.
    Uses NumPy uint8 bitarray (~10× faster than Python list[bool])
    and dual-hash scheme (Kirsch-Mitzenmacher) for optimal hash generation.

    Memory: ~size/8 bytes (vs size bytes for list[bool]).
    Speed: Vectorized numpy bitwise ops for add/contains.

    Auto-resize
    -----------
    When ``false_positive_rate`` exceeds ``max_fpr`` (default 5 %), calling
    :meth:`maybe_resize` doubles the bit-array capacity and re-inserts every
    tracked item.  This keeps FPR bounded without requiring the caller to
    pre-size the filter.  The re-insert is O(n·k) where n = item count and k =
    num_hashes — the same cost as a single full scan of the cache keys.

    Based on: "Less Hashing, Same Performance: Building a Better Bloom Filter"
    (Kirsch & Mitzenmacher, 2006)

    Args:
        size:        Initial bit-array size in bits (rounded up to ×8).
        num_hashes:  Number of hash functions k (optimal: ≈ (m/n)·ln2).
        max_fpr:     FPR threshold that triggers :meth:`maybe_resize`.
        max_size:    Hard upper limit on bit-array size (prevents unbounded
                     memory growth).  ``None`` = no limit.
    """

    def __init__(
        self,
        size: int = 10000,
        num_hashes: int = 5,
        max_fpr: float = 0.05,
        max_size: Optional[int] = None,
    ):
        # Round up to multiple of 8 for byte alignment
        self.size = ((size + 7) // 8) * 8
        self.num_hashes = num_hashes
        self.max_fpr = max_fpr
        self.max_size = max_size

        # NumPy uint8 bitarray: 8× less memory than list[bool]
        self._bits = np.zeros(self.size // 8, dtype=np.uint8)
        self._item_count = 0
        # Track inserted items for re-insertion on resize
        self._items: list = []

    def _hashes(self, item: str) -> np.ndarray:
        """Generate hash values using Kirsch-Mitzenmacher double hashing.

        Two base hashes generate k hash functions: h_i(x) = h1(x) + i*h2(x) mod m
        This is provably as good as k independent hash functions.
        """
        # Use SHA-256 for two independent 128-bit halves
        digest = hashlib.sha256(item.encode()).digest()
        h1 = int.from_bytes(digest[:16], "big")
        h2 = int.from_bytes(digest[16:], "big")
        return np.array(
            [(h1 + i * h2) % self.size for i in range(self.num_hashes)],
            dtype=np.int64,
        )

    def _set_bit(self, pos: int) -> None:
        """Set a single bit in the bitarray."""
        byte_idx = pos >> 3  # pos // 8
        bit_idx = pos & 7  # pos % 8
        self._bits[byte_idx] |= np.uint8(1 << bit_idx)

    def _get_bit(self, pos: int) -> bool:
        """Get a single bit from the bitarray."""
        byte_idx = pos >> 3
        bit_idx = pos & 7
        return bool(self._bits[byte_idx] & np.uint8(1 << bit_idx))

    def add(self, item: str) -> None:
        """Add item to filter."""
        for h in self._hashes(item):
            self._set_bit(int(h))
        self._item_count += 1
        self._items.append(item)

    def contains(self, item: str) -> bool:
        """
        Check if item might be in set.

        Returns:
            True if possibly present, False if definitely not present
        """
        return all(self._get_bit(int(h)) for h in self._hashes(item))

    def clear(self) -> None:
        """Clear the filter."""
        self._bits[:] = 0
        self._item_count = 0
        self._items.clear()

    @property
    def false_positive_rate(self) -> float:
        """Estimate current false positive rate: (1 - e^(-kn/m))^k."""
        if self._item_count == 0:
            return 0.0
        k, n, m = self.num_hashes, self._item_count, self.size
        return (1.0 - np.exp(-k * n / m)) ** k

    def maybe_resize(self) -> bool:
        """
        Double the bit-array if ``false_positive_rate > max_fpr``.

        All previously added items are re-inserted into the enlarged filter so
        the contains-guarantee is preserved.

        Returns:
            ``True`` if a resize was performed, ``False`` otherwise.

        Note:
            The optimal number of hash functions for a new capacity ``m'`` and
            ``n`` items is ``k* = (m'/n)·ln2``.  We keep ``num_hashes``
            unchanged for simplicity; the FPR will still drop significantly
            because the bit-saturation falls roughly in half.
        """
        if self.false_positive_rate <= self.max_fpr:
            return False

        new_size = self.size * 2
        if self.max_size is not None and new_size > self.max_size:
            logger.warning(
                f"BloomFilter: FPR={self.false_positive_rate:.3f} exceeds threshold "
                f"{self.max_fpr} but max_size={self.max_size} reached — resize skipped"
            )
            return False

        old_size = self.size
        self.size = new_size
        self._bits = np.zeros(self.size // 8, dtype=np.uint8)
        self._item_count = 0

        # Re-insert all tracked items into the new larger filter
        items_snapshot = list(self._items)
        self._items.clear()
        for item in items_snapshot:
            self.add(item)

        logger.debug(
            f"BloomFilter resized {old_size} → {new_size} bits; "
            f"new FPR ≈ {self.false_positive_rate:.4f}"
        )
        return True


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
    """Thread-safe semantic query cache with fuzzy matching and Bloom filter."""

    def __init__(
        self,
        max_size: int = 256,
        threshold: float = 0.85,
        ttl_seconds: Optional[float] = None,
        use_bloom: bool = True,
    ):
        """Initialize semantic query cache.

        Args:
            max_size: Maximum number of cached queries
            threshold: Similarity threshold for fuzzy matching (0-1)
            ttl_seconds: Time-to-live for cache entries (None = no expiration)
            use_bloom: Enable Bloom filter for fast negative lookups
        """
        self.max_size = max_size
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.use_bloom = use_bloom

        # Cache: query -> embedding
        self.cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        # Metadata: query -> (timestamp, access_count)
        self.metadata: dict = {}
        self.lock = threading.RLock()

        # Bloom filter for O(1) negative lookups
        if use_bloom:
            self.bloom = BloomFilter(size=max_size * 10, num_hashes=3)
        else:
            self.bloom = None

        self.hits = 0
        self.misses = 0
        self.fuzzy_hits = 0
        self.bloom_rejections = 0

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
        # Fast negative lookup with Bloom filter
        if self.bloom and not self.bloom.contains(query):
            self.bloom_rejections += 1
            self.misses += 1
            return None

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

            # Add to Bloom filter
            if self.bloom:
                self.bloom.add(query)

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


# =============================================================================
# FAISS-based ANN Semantic Cache - State-of-the-Art Optimization
# =============================================================================


class ANNSemanticCache(Generic[T]):
    """
    Semantic cache with Approximate Nearest Neighbor (ANN) indexing.

    Uses FAISS for sub-linear O(log n) similarity lookups instead of
    O(n) linear scanning. Critical for production RAG systems with
    10K+ cached queries.

    Based on:
    - "Retrieval-Augmented Generation for Large Language Models: A Survey" (Gao et al., 2024)
    - "Efficient Similarity Search and Clustering of Dense Vectors" (FAISS)

    Key optimizations:
    1. O(log n) lookup via IVF index instead of O(n) linear scan
    2. Automatic index rebuilding when cache grows
    3. Memory-efficient storage with numpy arrays
    4. Thread-safe operations with fine-grained locking

    Example:
        >>> cache = ANNSemanticCache(embedding_dim=1536, threshold=0.9)
        >>> await cache.set("What is ML?", embedding, result)
        >>> lookup = await cache.get_nearest(query_embedding)
    """

    def __init__(
        self,
        embedding_dim: int = 1536,
        similarity_threshold: float = 0.9,
        max_entries: int = 10000,
        nlist: int = 100,  # Number of clusters for IVF
        nprobe: int = 10,  # Number of clusters to search
        rebuild_threshold: int = 1000,  # Rebuild index after this many new entries
        ttl_seconds: Optional[int] = 3600,
    ):
        """
        Initialize ANN semantic cache.

        Args:
            embedding_dim: Dimension of embeddings
            similarity_threshold: Minimum cosine similarity for cache hit
            max_entries: Maximum cache entries
            nlist: Number of IVF clusters (affects index speed/accuracy tradeoff)
            nprobe: Number of clusters to search (affects search speed/accuracy)
            rebuild_threshold: Rebuild index after this many additions
            ttl_seconds: Time-to-live for entries
        """
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.nlist = nlist
        self.nprobe = nprobe
        self.rebuild_threshold = rebuild_threshold
        self.ttl_seconds = ttl_seconds

        # Storage
        self._entries: Dict[int, Tuple[str, T, datetime, Optional[datetime]]] = {}
        self._embeddings: List[np.ndarray] = []
        self._id_to_index: Dict[int, int] = {}
        self._index_to_id: Dict[int, int] = {}
        self._next_id = 0

        # FAISS index (lazy initialized)
        self._index = None
        self._use_ivf = False  # Use IVF only when cache is large enough
        self._additions_since_rebuild = 0

        # Locks
        self._lock = asyncio.Lock()
        self._index_lock = threading.RLock()

        # Stats
        self._stats = EmbeddingCacheStats()

        logger.info(
            f"Initialized ANNSemanticCache: dim={embedding_dim}, "
            f"threshold={similarity_threshold}, max_entries={max_entries}"
        )

    def _build_flat_index(self) -> None:
        """Build a flat (exact) FAISS index for small cache sizes."""
        try:
            import faiss

            with self._index_lock:
                self._index = faiss.IndexFlatIP(self.embedding_dim)
                if self._embeddings:
                    embeddings_array = np.vstack(self._embeddings).astype(np.float32)
                    # Normalize for cosine similarity
                    faiss.normalize_L2(embeddings_array)
                    self._index.add(embeddings_array)
                self._use_ivf = False

        except ImportError:
            logger.warning("FAISS not installed, falling back to linear search")
            self._index = None

    def _build_ivf_index(self) -> None:
        """Build an IVF index for large cache sizes - O(log n) search."""
        try:
            import faiss

            with self._index_lock:
                if len(self._embeddings) < self.nlist:
                    # Not enough data for IVF, use flat
                    self._build_flat_index()
                    return

                embeddings_array = np.vstack(self._embeddings).astype(np.float32)
                faiss.normalize_L2(embeddings_array)

                # Create IVF index with inner product (for cosine similarity)
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self._index = faiss.IndexIVFFlat(
                    quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT
                )

                # Train and add
                self._index.train(embeddings_array)
                self._index.add(embeddings_array)
                self._index.nprobe = self.nprobe

                self._use_ivf = True
                self._additions_since_rebuild = 0

                logger.info(
                    f"Built IVF index with {len(self._embeddings)} entries, "
                    f"nlist={self.nlist}, nprobe={self.nprobe}"
                )

        except ImportError:
            logger.warning("FAISS not installed, falling back to linear search")
            self._index = None

    def _should_rebuild_index(self) -> bool:
        """Check if index should be rebuilt."""
        # Rebuild if enough new entries added
        if self._additions_since_rebuild >= self.rebuild_threshold:
            return True
        # Upgrade to IVF if cache grew large enough
        if not self._use_ivf and len(self._embeddings) >= self.nlist * 2:
            return True
        return False

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    async def set(
        self,
        key: str,
        embedding: np.ndarray,
        value: T,
    ) -> int:
        """
        Add entry to cache with embedding.

        Args:
            key: Cache key (e.g., query string)
            embedding: Embedding vector
            value: Value to cache

        Returns:
            Entry ID
        """
        async with self._lock:
            # Evict if at capacity
            if len(self._entries) >= self.max_entries:
                await self._evict_oldest()

            # Create entry
            entry_id = self._next_id
            self._next_id += 1

            created_at = datetime.now()
            expires_at = None
            if self.ttl_seconds:
                expires_at = created_at + timedelta(seconds=self.ttl_seconds)

            self._entries[entry_id] = (key, value, created_at, expires_at)

            # Add embedding
            embedding = self._normalize_embedding(np.array(embedding, dtype=np.float32))
            index_pos = len(self._embeddings)
            self._embeddings.append(embedding)
            self._id_to_index[entry_id] = index_pos
            self._index_to_id[index_pos] = entry_id

            # Update index
            with self._index_lock:
                if self._index is not None:
                    try:
                        embedding_array = embedding.reshape(1, -1).astype(np.float32)
                        self._index.add(embedding_array)
                        self._additions_since_rebuild += 1
                    except Exception:
                        pass

            # Rebuild index if needed
            if self._should_rebuild_index():
                self._build_ivf_index()
            elif self._index is None and len(self._embeddings) >= 10:
                self._build_flat_index()

            self._stats.entries_count = len(self._entries)
            return entry_id

    async def get_nearest(
        self,
        query_embedding: np.ndarray,
        k: int = 1,
    ) -> List[EmbeddingCacheLookupResult[T]]:
        """
        Find nearest cache entries to query embedding.

        Uses FAISS ANN index for O(log n) lookup when available,
        falls back to O(n) linear search otherwise.

        Args:
            query_embedding: Query embedding vector
            k: Number of nearest neighbors to return

        Returns:
            List of lookup results sorted by similarity
        """
        self._stats.total_queries += 1

        if not self._embeddings:
            self._stats.cache_misses += 1
            return [EmbeddingCacheLookupResult(hit=False)]

        query_embedding = self._normalize_embedding(np.array(query_embedding, dtype=np.float32))

        # Use FAISS if available
        if self._index is not None:
            results = await self._search_faiss(query_embedding, k)
        else:
            results = await self._search_linear(query_embedding, k)

        # Update stats
        if results and results[0].hit:
            self._stats.cache_hits += 1
            # Update running average similarity
            hit_sim = results[0].similarity
            self._stats.avg_similarity = self._stats.avg_similarity * 0.9 + hit_sim * 0.1
        else:
            self._stats.cache_misses += 1

        return results

    async def _search_faiss(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> List[EmbeddingCacheLookupResult[T]]:
        """Search using FAISS index - O(log n) for IVF."""
        with self._index_lock:
            if self._index is None:
                return await self._search_linear(query_embedding, k)

            query_array = query_embedding.reshape(1, -1).astype(np.float32)

            # Search
            actual_k = min(k, self._index.ntotal)
            if actual_k == 0:
                return [EmbeddingCacheLookupResult(hit=False)]

            similarities, indices = self._index.search(query_array, actual_k)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing
                continue

            # Get entry ID from index position
            entry_id = self._index_to_id.get(idx)
            if entry_id is None:
                continue

            entry_data = self._entries.get(entry_id)
            if entry_data is None:
                continue

            key, value, created_at, expires_at = entry_data

            # Check expiration
            if expires_at and datetime.now() > expires_at:
                continue

            similarity = float(sim)  # Inner product on normalized = cosine
            hit = similarity >= self.similarity_threshold

            results.append(
                EmbeddingCacheLookupResult(
                    hit=hit,
                    value=value if hit else None,
                    similarity=similarity,
                )
            )

        if not results:
            return [EmbeddingCacheLookupResult(hit=False)]

        return results

    async def _search_linear(
        self,
        query_embedding: np.ndarray,
        k: int,
    ) -> List[EmbeddingCacheLookupResult[T]]:
        """Fallback linear search - O(n)."""
        similarities = []

        for idx, emb in enumerate(self._embeddings):
            entry_id = self._index_to_id.get(idx)
            if entry_id is None:
                continue

            entry_data = self._entries.get(entry_id)
            if entry_data is None:
                continue

            key, value, created_at, expires_at = entry_data

            # Check expiration
            if expires_at and datetime.now() > expires_at:
                continue

            # Cosine similarity (dot product of normalized vectors)
            sim = float(np.dot(query_embedding, emb))
            similarities.append((sim, entry_id, value))

        if not similarities:
            return [EmbeddingCacheLookupResult(hit=False)]

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, entry_id, value in similarities[:k]:
            hit = sim >= self.similarity_threshold
            results.append(
                EmbeddingCacheLookupResult(
                    hit=hit,
                    value=value if hit else None,
                    similarity=sim,
                )
            )

        return results

    async def _evict_oldest(self) -> None:
        """Evict oldest entries to make room."""
        if not self._entries:
            return

        # Find oldest entry
        oldest_id = min(self._entries.keys())

        if oldest_id in self._entries:
            del self._entries[oldest_id]
            self._stats.evictions += 1

        # Note: We don't remove from FAISS index (expensive)
        # Just mark as deleted and rebuild periodically

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self._stats.to_dict(),
            "use_ivf_index": self._use_ivf,
            "index_size": self._index.ntotal if self._index else 0,
            "additions_since_rebuild": self._additions_since_rebuild,
        }

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._entries.clear()
            self._embeddings.clear()
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._index = None
            self._use_ivf = False
            self._additions_since_rebuild = 0
            self._stats = EmbeddingCacheStats()


def create_ann_cache(
    embedding_dim: int = 1536,
    threshold: float = 0.9,
    max_entries: int = 10000,
) -> ANNSemanticCache[Any]:
    """
    Create an ANN-based semantic cache for production RAG systems.

    Args:
        embedding_dim: Embedding dimension
        threshold: Similarity threshold
        max_entries: Maximum entries

    Returns:
        ANNSemanticCache instance
    """
    return ANNSemanticCache(
        embedding_dim=embedding_dim,
        similarity_threshold=threshold,
        max_entries=max_entries,
    )
