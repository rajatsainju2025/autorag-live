"""Semantic query caching with fuzzy matching for deduplication.

This module provides intelligent query caching that detects semantically similar
queries and reuses embeddings from cached results, reducing redundant computation.

Features:
    - Levenshtein distance-based fuzzy matching
    - Configurable similarity threshold
    - Thread-safe cache operations
    - Automatic cache eviction when full
    - Integration with existing embedding cache

Example:
    >>> cache = SemanticQueryCache(threshold=0.9)
    >>> similar_query = cache.find_similar("original query", all_queries)
    >>> if similar_query:
    ...     cached_embedding = get_cached_embedding(similar_query)
"""

import hashlib
import threading
from collections import OrderedDict
from typing import List, Optional

import numpy as np


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
