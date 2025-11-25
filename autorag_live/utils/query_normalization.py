"""Query normalization caching for consistency and performance."""

import re
from functools import lru_cache
from typing import Callable, Optional

# Pre-compile regex patterns for better performance
_WHITESPACE_PATTERN = re.compile(r"\s+")
_END_PUNCTUATION_PATTERN = re.compile(r"[?!]+$")


class QueryNormalizer:
    """Normalizes queries with caching for consistent retrieval."""

    def __init__(self, custom_normalizer: Optional[Callable[[str], str]] = None):
        """Initialize query normalizer.

        Args:
            custom_normalizer: Optional custom normalization function
        """
        self._custom_normalizer = custom_normalizer
        # Cache normalized queries with size limit
        self._normalize_cache = {}
        self._cache_size_limit = 5000

    @staticmethod
    @lru_cache(maxsize=2048)
    def _normalize_default(query: str) -> str:
        """Default normalization: lowercase and remove extra whitespace."""
        # Lowercase first for consistent caching
        normalized = query.lower().strip()
        # Remove extra whitespace using pre-compiled pattern
        normalized = _WHITESPACE_PATTERN.sub(" ", normalized)
        # Remove common stop patterns using pre-compiled pattern
        normalized = _END_PUNCTUATION_PATTERN.sub("", normalized)
        return normalized

    def normalize(self, query: str) -> str:
        """
        Normalize a query with caching and size limits.

        Args:
            query: Raw query string

        Returns:
            Normalized query string
        """
        # Check cache
        if query in self._normalize_cache:
            return self._normalize_cache[query]

        # Use custom or default normalizer
        if self._custom_normalizer:
            normalized = self._custom_normalizer(query)
        else:
            normalized = self._normalize_default(query)

        # Cache result with size limit (simple LRU-like eviction)
        if len(self._normalize_cache) >= self._cache_size_limit:
            # Remove first (oldest) entry
            self._normalize_cache.pop(next(iter(self._normalize_cache)))

        self._normalize_cache[query] = normalized
        return normalized

    def clear_cache(self):
        """Clear the normalization cache."""
        self._normalize_cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cached_queries": len(self._normalize_cache),
            "cache_limit": self._cache_size_limit,
        }


# Global instance for module-level use
_global_normalizer = QueryNormalizer()


def normalize_query(query: str) -> str:
    """Module-level query normalization function."""
    return _global_normalizer.normalize(query)


def get_query_normalizer() -> QueryNormalizer:
    """Get the global query normalizer instance."""
    return _global_normalizer
