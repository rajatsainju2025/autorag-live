"""Query normalization caching for consistency and performance."""

import re
from functools import lru_cache
from typing import Callable, Optional


class QueryNormalizer:
    """Normalizes queries with caching for consistent retrieval."""

    def __init__(self, custom_normalizer: Optional[Callable[[str], str]] = None):
        """Initialize query normalizer.

        Args:
            custom_normalizer: Optional custom normalization function
        """
        self._custom_normalizer = custom_normalizer
        # Cache normalized queries
        self._normalize_cache = {}

    @staticmethod
    @lru_cache(maxsize=1024)
    def _normalize_default(query: str) -> str:
        """Default normalization: lowercase and remove extra whitespace."""
        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", query.strip())
        # Lowercase
        normalized = normalized.lower()
        # Remove common stop patterns
        normalized = re.sub(r"[?!]+$", "", normalized)
        return normalized

    def normalize(self, query: str) -> str:
        """
        Normalize a query with caching.

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

        # Cache result
        self._normalize_cache[query] = normalized
        return normalized

    def clear_cache(self):
        """Clear the normalization cache."""
        self._normalize_cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cached_queries": len(self._normalize_cache),
            "cache_size": sum(len(k) + len(v) for k, v in self._normalize_cache.items()),
        }


# Global instance for module-level use
_global_normalizer = QueryNormalizer()


def normalize_query(query: str) -> str:
    """Module-level query normalization function."""
    return _global_normalizer.normalize(query)


def get_query_normalizer() -> QueryNormalizer:
    """Get the global query normalizer instance."""
    return _global_normalizer
