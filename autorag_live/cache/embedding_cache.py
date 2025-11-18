"""Efficient embedding caching with LRU eviction and TTL support.

This module provides high-performance embedding caching with:
- Thread-safe LRU eviction
- Time-to-live (TTL) expiration
- Batch memoization
- Memory-efficient storage

Example:
    >>> cache = EmbeddingCache(max_size=1000, ttl_seconds=3600)
    >>> embeddings = cache.get_batch(["text1", "text2"], compute_fn)
"""

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class EmbeddingCache:
    """Thread-safe embedding cache with LRU eviction and TTL support."""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[float] = None,
    ):
        """Initialize embedding cache.

        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries (None = no expiration)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _make_key(text: str) -> str:
        """Create a cache key from text.

        Args:
            text: Text to hash

        Returns:
            Hashable cache key
        """
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text.

        Args:
            text: Text to retrieve embedding for

        Returns:
            Cached embedding or None if not found/expired
        """
        with self.lock:
            key = self._make_key(text)

            # Check expiration
            if self.ttl_seconds is not None:
                if key in self.timestamps:
                    if time.time() - self.timestamps[key] > self.ttl_seconds:
                        del self.cache[key]
                        del self.timestamps[key]
                        self.misses += 1
                        return None

            if key in self.cache:
                # Move to end (LRU)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]

            self.misses += 1
            return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Cache embedding for text.

        Args:
            text: Text to cache embedding for
            embedding: Embedding array
        """
        with self.lock:
            key = self._make_key(text)

            # Remove old entry if exists
            if key in self.cache:
                del self.cache[key]

            # Add new entry
            self.cache[key] = embedding
            self.timestamps[key] = time.time()

            # Evict oldest if over capacity
            while len(self.cache) > self.max_size:
                old_key = next(iter(self.cache))
                del self.cache[old_key]
                del self.timestamps[old_key]

    def get_batch(
        self,
        texts: List[str],
        compute_fn: Callable[[List[str]], List[np.ndarray]],
    ) -> List[np.ndarray]:
        """Get or compute embeddings for batch of texts.

        Args:
            texts: Texts to retrieve embeddings for
            compute_fn: Function to compute missing embeddings

        Returns:
            List of embeddings (cached or computed)
        """
        with self.lock:
            results: List[Optional[np.ndarray]] = []
            missing_indices: List[int] = []
            missing_texts: List[str] = []

            # Check cache for each text
            for i, text in enumerate(texts):
                embedding = self.get(text)
                if embedding is not None:
                    results.append(embedding)
                else:
                    results.append(None)
                    missing_indices.append(i)
                    missing_texts.append(text)

            # Compute missing embeddings if any
            if missing_texts:
                computed = compute_fn(missing_texts)
                for idx, text, embedding in zip(missing_indices, missing_texts, computed):
                    self.put(text, embedding)
                    results[idx] = embedding

            return [r for r in results if r is not None]

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
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
            }

    def __len__(self) -> int:
        """Return number of cached entries."""
        with self.lock:
            return len(self.cache)

    def __contains__(self, text: str) -> bool:
        """Check if text is cached."""
        with self.lock:
            key = self._make_key(text)
            if key in self.cache and self.ttl_seconds is not None:
                if time.time() - self.timestamps[key] > self.ttl_seconds:
                    del self.cache[key]
                    del self.timestamps[key]
                    return False
            return key in self.cache


# Optimization: perf(cache): add model cache manager with memory eviction
