"""
Response caching utilities for AutoRAG-Live.

Provides caching mechanisms for expensive operations like LLM calls,
embeddings, and API responses to improve performance and reduce costs.

Example:
    >>> from autorag_live.utils.response_cache import ResponseCache
    >>>
    >>> cache = ResponseCache(maxsize=100, ttl=3600)
    >>> result = cache.get_or_compute("key", expensive_function)
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T")


class ResponseCache:
    """
    Simple response caching with TTL support.

    Caches function results to avoid redundant computation.
    Useful for expensive operations like API calls or embeddings.
    """

    def __init__(
        self,
        maxsize: int = 1000,
        ttl: Optional[float] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize response cache.

        Args:
            maxsize: Maximum cache size (0 = unlimited)
            ttl: Time-to-live in seconds (None = no expiration)
            cache_dir: Optional directory for persistent cache
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: Dict[str, tuple[Any, float]] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """Create cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cached item has expired."""
        if self.ttl is None:
            return False
        return (time.time() - timestamp) > self.ttl

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value by key.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        # Check in-memory cache first
        if key in self._cache:
            value, timestamp = self._cache[key]
            if not self._is_expired(timestamp):
                return value
            else:
                del self._cache[key]

        # Check disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        value, timestamp = pickle.load(f)
                    if not self._is_expired(timestamp):
                        # Load back into memory
                        self._cache[key] = (value, timestamp)
                        return value
                    else:
                        cache_file.unlink()
                except Exception:
                    pass

        return None

    def set(self, key: str, value: Any) -> None:
        """
        Cache a value.

        Args:
            key: Cache key
            value: Value to cache
        """
        timestamp = time.time()

        # Evict oldest if over size limit
        if self.maxsize > 0 and len(self._cache) >= self.maxsize:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        # Store in memory
        self._cache[key] = (value, timestamp)

        # Store on disk if enabled
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump((value, timestamp), f)
            except Exception:
                pass

    def get_or_compute(self, key: str, compute_fn: Callable[[], T], *args: Any, **kwargs: Any) -> T:
        """
        Get cached value or compute if not found.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            *args: Arguments to compute_fn
            **kwargs: Keyword arguments to compute_fn

        Returns:
            Cached or computed value

        Example:
            >>> cache = ResponseCache()
            >>> result = cache.get_or_compute("key", lambda: expensive_op())
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        # Compute and cache
        value = compute_fn(*args, **kwargs)
        self.set(key, value)
        return value

    def invalidate(self, key: str) -> None:
        """
        Remove item from cache.

        Args:
            key: Cache key to invalidate
        """
        if key in self._cache:
            del self._cache[key]

        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.pkl"
            if cache_file.exists():
                cache_file.unlink()

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()

        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


def cached_function(
    maxsize: int = 128, ttl: Optional[float] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for caching function results.

    Args:
        maxsize: Maximum cache entries
        ttl: Time-to-live in seconds

    Returns:
        Decorated function

    Example:
        >>> @cached_function(maxsize=100, ttl=3600)
        ... def expensive_operation(x):
        ...     return x * 2
    """
    cache = ResponseCache(maxsize=maxsize, ttl=ttl)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = cache._make_key(*args, **kwargs)
            return cache.get_or_compute(key, func, *args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


# Global cache instance for convenience
_global_cache = ResponseCache(maxsize=1000, ttl=3600)


def get_global_cache() -> ResponseCache:
    """
    Get global response cache instance.

    Returns:
        Global ResponseCache instance

    Example:
        >>> cache = get_global_cache()
        >>> cache.set("key", "value")
    """
    return _global_cache


def cache_response(key: str, value: Any) -> None:
    """Cache a response globally."""
    _global_cache.set(key, value)


def get_cached_response(key: str) -> Optional[Any]:
    """Get a cached response globally."""
    return _global_cache.get(key)


def clear_global_cache() -> None:
    """Clear global cache."""
    _global_cache.clear()
