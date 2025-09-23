"""
Caching utilities for AutoRAG-Live.

This module provides caching mechanisms for expensive operations like
embeddings computation, model loading, and retrieval results.
"""

import hashlib
import json
import os
import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional, Callable, TypeVar, Union
from dataclasses import dataclass

from autorag_live.utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    hits: int = 0
    size_bytes: int = 0


class Cache:
    """Generic caching interface."""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all cache entries."""
        raise NotImplementedError

    def size(self) -> int:
        """Get number of entries in cache."""
        raise NotImplementedError


class MemoryCache(Cache):
    """In-memory cache with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self._cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with TTL check."""
        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Check TTL
        if self.default_ttl and time.time() - entry.timestamp > self.default_ttl:
            del self._cache[key]
            return None

        entry.hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache with optional TTL."""
        # Evict if at capacity (simple LRU-like)
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(),
                           key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]

        entry = CacheEntry(
            value=value,
            timestamp=time.time(),
            size_bytes=self._estimate_size(value)
        )
        self._cache[key] = entry

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def size(self) -> int:
        """Get number of entries in cache."""
        return len(self._cache)

    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate


class FileCache(Cache):
    """File-based cache for persistence."""

    def __init__(self, cache_dir: str = ".cache", max_size_mb: float = 100.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._index_file = self.cache_dir / "index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r') as f:
                    self._index = json.load(f)
            except:
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        with open(self._index_file, 'w') as f:
            json.dump(self._index, f, indent=2)

    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        if key not in self._index:
            return None

        cache_file = self.cache_dir / f"{key}.pkl"
        if not cache_file.exists():
            del self._index[key]
            self._save_index()
            return None

        try:
            with open(cache_file, 'rb') as f:
                value = pickle.load(f)
            self._index[key]['hits'] = self._index[key].get('hits', 0) + 1
            self._save_index()
            return value
        except:
            # Corrupted file, remove it
            cache_file.unlink(missing_ok=True)
            del self._index[key]
            self._save_index()
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in file cache."""
        cache_file = self.cache_dir / f"{key}.pkl"

        # Check if we need to evict
        self._evict_if_needed()

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)

            self._index[key] = {
                'timestamp': time.time(),
                'ttl': ttl,
                'hits': 0,
                'size': cache_file.stat().st_size
            }
            self._save_index()
        except Exception as e:
            logger.warning(f"Failed to cache {key}: {e}")

    def delete(self, key: str) -> None:
        """Delete value from file cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        cache_file.unlink(missing_ok=True)
        self._index.pop(key, None)
        self._save_index()

    def clear(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        self._index.clear()
        self._save_index()

    def size(self) -> int:
        """Get number of entries in cache."""
        return len(self._index)

    def _evict_if_needed(self) -> None:
        """Evict old entries if cache is too large."""
        total_size = sum(entry.get('size', 0) for entry in self._index.values())

        if total_size < self.max_size_bytes:
            return

        # Sort by access time and evict oldest
        entries = sorted(
            self._index.items(),
            key=lambda x: x[1]['timestamp']
        )

        for key, entry in entries:
            if total_size < self.max_size_bytes * 0.8:  # Keep 80% capacity
                break
            self.delete(key)
            total_size -= entry.get('size', 0)


class CacheManager:
    """Central cache management."""

    def __init__(self):
        self.caches: Dict[str, Cache] = {}
        self._setup_default_caches()

    def _setup_default_caches(self) -> None:
        """Set up default caches."""
        # Memory cache for embeddings
        self.caches['embeddings'] = MemoryCache(max_size=100, default_ttl=3600)

        # File cache for models
        self.caches['models'] = FileCache(cache_dir=".cache/models", max_size_mb=500)

        # Memory cache for retrieval results
        self.caches['retrieval'] = MemoryCache(max_size=500, default_ttl=1800)

    def get_cache(self, name: str) -> Cache:
        """Get or create a cache by name."""
        if name not in self.caches:
            self.caches[name] = MemoryCache()
        return self.caches[name]

    def clear_all(self) -> None:
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        logger.info("Cleared all caches")


# Global cache manager instance
cache_manager = CacheManager()


def cached(cache_name: str = "default", ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache = cache_manager.get_cache(cache_name)

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = {
                    'func': func.__name__,
                    'args': args,
                    'kwargs': sorted(kwargs.items())
                }
                key = hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()

            # Try cache first
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result

            # Compute result
            logger.debug(f"Cache miss for {func.__name__}, computing...")
            result = func(*args, **kwargs)

            # Cache result
            cache.set(key, result, ttl)
            return result

        return wrapper
    return decorator


def generate_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    return hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()


# Convenience functions
def get_cache(name: str) -> Cache:
    """Get a cache by name."""
    return cache_manager.get_cache(name)


def clear_cache(name: str) -> None:
    """Clear a specific cache."""
    cache_manager.get_cache(name).clear()


def clear_all_caches() -> None:
    """Clear all caches."""
    cache_manager.clear_all()