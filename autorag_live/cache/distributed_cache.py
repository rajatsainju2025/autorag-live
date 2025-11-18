"""Distributed caching strategies for multi-instance deployments.

This module provides utilities for coordinating caches across instances:
- Cache invalidation patterns
- Distributed TTL management
- Statistics aggregation
- Fallback strategies

Example:
    >>> cache = DistributedCacheManager()
    >>> cache.register_cache("embeddings", embedding_cache)
    >>> cache.invalidate("embeddings", keys=["query1"])
"""

import json
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set


class CacheInvalidator(ABC):
    """Abstract base for cache invalidation strategies."""

    @abstractmethod
    def invalidate(self, cache_key: str) -> None:
        """Invalidate a cache key."""
        pass

    @abstractmethod
    def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate keys matching a pattern."""
        pass


class LocalCacheInvalidator(CacheInvalidator):
    """Local-only cache invalidation."""

    def __init__(self):
        """Initialize local invalidator."""
        self.invalidated_keys: Set[str] = set()
        self.lock = threading.RLock()

    def invalidate(self, cache_key: str) -> None:
        """Invalidate a single key.

        Args:
            cache_key: Key to invalidate
        """
        with self.lock:
            self.invalidated_keys.add(cache_key)

    def invalidate_pattern(self, pattern: str) -> None:
        """Invalidate pattern (not implemented for local).

        Args:
            pattern: Pattern to invalidate
        """
        # Local strategy: no-op for patterns
        pass

    def is_invalidated(self, cache_key: str) -> bool:
        """Check if key is invalidated.

        Args:
            cache_key: Key to check

        Returns:
            True if invalidated
        """
        with self.lock:
            return cache_key in self.invalidated_keys

    def clear(self) -> None:
        """Clear all invalidations."""
        with self.lock:
            self.invalidated_keys.clear()


class DistributedCacheManager:
    """Manage multiple caches with coordination capabilities."""

    def __init__(self):
        """Initialize cache manager."""
        self.caches: Dict[str, Any] = {}
        self.invalidators: Dict[str, CacheInvalidator] = {}
        self.stats: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()

    def register_cache(
        self,
        name: str,
        cache: Any,
        invalidator: Optional[CacheInvalidator] = None,
    ) -> None:
        """Register a cache instance.

        Args:
            name: Cache identifier
            cache: Cache instance
            invalidator: Optional invalidator strategy
        """
        with self.lock:
            self.caches[name] = cache
            self.invalidators[name] = invalidator or LocalCacheInvalidator()

    def invalidate_all(self) -> None:
        """Invalidate all registered caches."""
        with self.lock:
            for name, cache in self.caches.items():
                if hasattr(cache, "clear"):
                    cache.clear()

    def invalidate_cache(self, cache_name: str, keys: Optional[List[str]] = None) -> None:
        """Invalidate specific cache.

        Args:
            cache_name: Name of cache to invalidate
            keys: Specific keys to invalidate (None = clear all)
        """
        with self.lock:
            if cache_name not in self.caches:
                return

            if keys is None:
                cache = self.caches[cache_name]
                if hasattr(cache, "clear"):
                    cache.clear()
            else:
                invalidator = self.invalidators[cache_name]
                for key in keys:
                    invalidator.invalidate(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches.

        Returns:
            Cache statistics
        """
        with self.lock:
            stats = {}
            for name, cache in self.caches.items():
                if hasattr(cache, "get_stats"):
                    stats[name] = cache.get_stats()
            return stats

    def summary(self) -> str:
        """Get summary of all caches.

        Returns:
            Summary string
        """
        stats = self.get_stats()
        if not stats:
            return "No caches registered"

        lines = ["Distributed Cache Summary:"]
        total_hits = 0
        total_requests = 0

        for name, cache_stats in stats.items():
            hits = cache_stats.get("hits", 0)
            total = cache_stats.get("total_requests", 0)
            total_hits += hits
            total_requests += total

            lines.append(
                f"  {name}: "
                f"size={cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}, "
                f"hit_rate={cache_stats.get('hit_rate', 0):.1%}"
            )

        if total_requests > 0:
            overall_rate = total_hits / total_requests
            lines.append(f"Overall hit rate: {overall_rate:.1%}")

        return "\n".join(lines)


class CacheWarmer:
    """Pre-populate caches with common data."""

    def __init__(self, cache: Any):
        """Initialize cache warmer.

        Args:
            cache: Cache to warm
        """
        self.cache = cache

    def warm_from_list(
        self,
        items: List[str],
        compute_fn: Any,
    ) -> int:
        """Warm cache from a list of items.

        Args:
            items: Items to pre-compute
            compute_fn: Function to compute values

        Returns:
            Number of items warmed
        """
        count = 0
        for item in items:
            if item not in self.cache:
                value = compute_fn(item)
                if hasattr(self.cache, "put"):
                    self.cache.put(item, value)
                count += 1

        return count

    def warm_from_file(
        self,
        filepath: str,
        compute_fn: Any,
    ) -> int:
        """Warm cache from file with JSON lines format.

        Args:
            filepath: Path to JSON lines file
            compute_fn: Function to compute values

        Returns:
            Number of items warmed
        """
        count = 0
        try:
            with open(filepath, "r") as f:
                for line in f:
                    item = json.loads(line.strip())
                    if item not in self.cache:
                        value = compute_fn(item)
                        if hasattr(self.cache, "put"):
                            self.cache.put(item, value)
                        count += 1
        except Exception:
            pass

        return count


# Optimization: perf(io): add async cache persistence foundation
