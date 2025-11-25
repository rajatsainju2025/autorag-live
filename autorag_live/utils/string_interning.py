"""
String interning utilities for memory optimization and fast string comparisons.

This module provides efficient string interning for frequently used strings
like query terms, document IDs, and configuration keys.
"""
from typing import Dict, Set


class StringInterner:
    """
    Memory-efficient string interning for frequently used strings.

    Uses Python's built-in string interning mechanism for memory efficiency
    while maintaining fast identity-based comparisons.
    """

    def __init__(self):
        self._intern_cache: Dict[str, str] = {}
        self._stats = {"hits": 0, "misses": 0, "cache_size": 0}

    def intern(self, string: str) -> str:
        """
        Intern a string for memory efficiency and fast comparisons.

        Args:
            string: String to intern

        Returns:
            Interned string (same identity for identical content)
        """
        if not isinstance(string, str):
            raise TypeError("Can only intern string objects")

        if string in self._intern_cache:
            self._stats["hits"] += 1
            return self._intern_cache[string]

        # Use sys.intern for built-in interning
        import sys

        interned = sys.intern(string)
        self._intern_cache[string] = interned
        self._stats["misses"] += 1
        self._stats["cache_size"] = len(self._intern_cache)

        return interned

    def get_stats(self) -> Dict[str, int]:
        """Get interning statistics."""
        return self._stats.copy()

    def clear(self) -> None:
        """Clear the intern cache."""
        self._intern_cache.clear()
        self._stats = {"hits": 0, "misses": 0, "cache_size": 0}


# Global string interner instance
_global_interner = StringInterner()


def intern_string(string: str) -> str:
    """
    Intern a string using the global interner.

    Args:
        string: String to intern

    Returns:
        Interned string
    """
    return _global_interner.intern(string)


class QueryStringInterner:
    """
    Specialized string interner for query processing.

    Optimized for common query patterns and terms.
    """

    def __init__(self):
        self._common_terms: Set[str] = {
            # Common English words
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "can",
            "will",
            "just",
            "should",
            "now",
            # Technical terms
            "data",
            "model",
            "algorithm",
            "query",
            "search",
            "find",
            "get",
            "information",
            "document",
            "text",
            "content",
            "result",
            "output",
        }
        self._interner = StringInterner()

        # Pre-intern common terms
        for term in self._common_terms:
            self._interner.intern(term)

    def intern_query_terms(self, query: str) -> str:
        """
        Intern query terms for efficient processing.

        Args:
            query: Query string

        Returns:
            Query with interned terms
        """
        if not query or not query.strip():
            return query

        # Split into terms and intern each one
        terms = query.lower().split()
        interned_terms = [self._interner.intern(term) for term in terms]

        return " ".join(interned_terms)

    def get_stats(self) -> Dict[str, int]:
        """Get query interning statistics."""
        return self._interner.get_stats()


# Global query interner
_query_interner = QueryStringInterner()


def intern_query(query: str) -> str:
    """
    Intern query string using the global query interner.

    Args:
        query: Query string to intern

    Returns:
        Query with interned terms
    """
    return _query_interner.intern_query_terms(query)
