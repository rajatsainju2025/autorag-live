"""Query result caching for repeated retrieval operations.

This module provides efficient caching of retrieval results with:
- LRU eviction policies
- TTL-based expiration
- Thread-safe operations
- Cache statistics

Example:
    >>> cache = QueryCache(max_size=1000, ttl_seconds=3600)
    >>> if (query, k) not in cache:
    ...     results = retriever.retrieve(query, k)
    ...     cache.put(query, k, results)
    ... else:
    ...     results = cache.get(query, k)
"""

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple


class QueryCache:
    """Thread-safe query result cache with LRU eviction and TTL support."""

    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: Optional[float] = None,
    ):
        """Initialize query cache.

        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cache entries (None = no expiration)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[List[str], float]] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _make_key(query: str, k: int, retriever_type: str = "") -> str:
        """Create cache key from query parameters.

        Args:
            query: Query text
            k: Number of results
            retriever_type: Type of retriever (for cache separation)

        Returns:
            Hashable cache key
        """
        key_str = f"{query}:{k}:{retriever_type}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(
        self,
        query: str,
        k: int,
        retriever_type: str = "",
    ) -> Optional[List[str]]:
        """Get cached results for query.

        Args:
            query: Query text
            k: Number of results
            retriever_type: Type of retriever

        Returns:
            Cached results or None if not found/expired
        """
        with self.lock:
            key = self._make_key(query, k, retriever_type)

            if key not in self.cache:
                self.misses += 1
                return None

            results, timestamp = self.cache[key]

            # Check expiration
            if self.ttl_seconds is not None:
                if time.time() - timestamp > self.ttl_seconds:
                    del self.cache[key]
                    self.misses += 1
                    return None

            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.hits += 1
            return results

    def put(
        self,
        query: str,
        k: int,
        results: List[str],
        retriever_type: str = "",
    ) -> None:
        """Cache results for query.

        Args:
            query: Query text
            k: Number of results
            results: Retrieved results
            retriever_type: Type of retriever
        """
        with self.lock:
            key = self._make_key(query, k, retriever_type)

            # Remove old entry if exists
            if key in self.cache:
                del self.cache[key]

            # Add new entry
            self.cache[key] = (results, time.time())

            # Evict oldest if over capacity
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
            }

    def __len__(self) -> int:
        """Return number of cached entries."""
        with self.lock:
            return len(self.cache)

    def __contains__(self, key: Tuple[str, int]) -> bool:
        """Check if query is cached."""
        with self.lock:
            query, k = key
            cache_key = self._make_key(query, k)
            if cache_key in self.cache and self.ttl_seconds is not None:
                _, timestamp = self.cache[cache_key]
                if time.time() - timestamp > self.ttl_seconds:
                    del self.cache[cache_key]
                    return False
            return cache_key in self.cache


class CacheableRetriever:
    """Base class for retrievers with built-in caching."""

    def __init__(
        self,
        base_retriever: Any,
        cache_enabled: bool = True,
        cache_size: int = 500,
        cache_ttl: Optional[float] = 3600,
    ):
        """Initialize cacheable retriever.

        Args:
            base_retriever: Base retriever instance
            cache_enabled: Whether caching is enabled
            cache_size: Max cache size
            cache_ttl: Cache TTL in seconds
        """
        self.base_retriever = base_retriever
        self.cache_enabled = cache_enabled
        self.cache = QueryCache(max_size=cache_size, ttl_seconds=cache_ttl)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve with optional caching.

        Args:
            query: Query text
            k: Number of results

        Returns:
            Retrieved results
        """
        if not self.cache_enabled:
            return self.base_retriever.retrieve(query, k)

        # Check cache
        cached = self.cache.get(query, k)
        if cached is not None:
            return cached

        # Compute and cache
        results = self.base_retriever.retrieve(query, k)
        self.cache.put(query, k, results)
        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache statistics
        """
        return self.cache.get_stats()

    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache.clear()
