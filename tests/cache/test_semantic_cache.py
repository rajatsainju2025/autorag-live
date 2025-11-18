"""Tests for semantic query cache with fuzzy matching."""

import numpy as np
import pytest

from autorag_live.cache.semantic_cache import (
    SemanticQueryCache,
    levenshtein_distance,
    similarity_ratio,
)


class TestLevenshteinDistance:
    """Test Levenshtein distance computation."""

    def test_identical_strings(self):
        """Identical strings should have distance 0."""
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        """Distance from empty string should be length of other."""
        assert levenshtein_distance("", "hello") == 5
        assert levenshtein_distance("hello", "") == 5
        assert levenshtein_distance("", "") == 0

    def test_single_character_diff(self):
        """Single character differences."""
        assert levenshtein_distance("cat", "car") == 1
        assert levenshtein_distance("hello", "hallo") == 1

    def test_insertion_deletion(self):
        """Test insertions and deletions."""
        assert levenshtein_distance("hello", "helo") == 1  # Delete l
        assert levenshtein_distance("helo", "hello") == 1  # Insert l

    def test_completely_different(self):
        """Completely different strings."""
        assert levenshtein_distance("abc", "xyz") == 3

    def test_case_sensitive(self):
        """Distance is case-sensitive."""
        assert levenshtein_distance("Hello", "hello") == 1


class TestSimilarityRatio:
    """Test similarity ratio computation."""

    def test_identical_strings(self):
        """Identical strings should have similarity 1.0."""
        assert similarity_ratio("hello", "hello") == 1.0

    def test_completely_different(self):
        """Completely different strings should have low similarity."""
        ratio = similarity_ratio("abc", "xyz")
        assert ratio < 0.4

    def test_similar_strings(self):
        """Similar strings should have high similarity."""
        ratio = similarity_ratio("hello", "hallo")
        assert ratio >= 0.8

    def test_empty_strings(self):
        """Both empty strings should have similarity 1.0."""
        assert similarity_ratio("", "") == 1.0

    def test_one_empty(self):
        """One empty string should have low similarity."""
        ratio = similarity_ratio("hello", "")
        assert ratio <= 0.3


class TestSemanticQueryCache:
    """Test SemanticQueryCache."""

    def test_initialization(self):
        """Cache should initialize correctly."""
        cache = SemanticQueryCache(max_size=100, threshold=0.9)
        assert cache.max_size == 100
        assert cache.threshold == 0.9
        assert len(cache.cache) == 0

    def test_exact_match(self):
        """Exact match should return cached embedding."""
        cache = SemanticQueryCache()
        query = "test query"
        embedding = np.array([1.0, 2.0, 3.0])

        # Put in cache
        cache.put(query, embedding)
        assert len(cache.cache) == 1

        # Get exact match
        result = cache.get_exact(query)
        assert result is not None
        assert np.allclose(result, embedding)
        assert cache.hits == 1

    def test_exact_match_miss(self):
        """Cache miss should return None."""
        cache = SemanticQueryCache()
        result = cache.get_exact("nonexistent query")
        assert result is None
        assert cache.misses == 1

    def test_lru_eviction(self):
        """LRU eviction when cache is full."""
        cache = SemanticQueryCache(max_size=3)

        # Add 4 items
        for i in range(4):
            query = f"query {i}"
            embedding = np.array([float(i)])
            cache.put(query, embedding)

        # Oldest should be evicted
        assert len(cache.cache) == 3

    def test_find_similar(self):
        """Find similar query from candidates."""
        cache = SemanticQueryCache(threshold=0.8)

        candidates = [
            "the quick brown fox",
            "the slow red fox",
            "totally different",
        ]

        # Should find similar query
        similar = cache.find_similar("the quick brown dog", candidates)
        assert similar is not None
        assert "fox" in similar

    def test_find_similar_below_threshold(self):
        """Should return None if similarity below threshold."""
        cache = SemanticQueryCache(threshold=0.99)

        candidates = ["hello world"]

        # Very high threshold, should not match
        similar = cache.find_similar("goodbye", candidates)
        assert similar is None

    def test_cache_stats(self):
        """Cache statistics should be accurate."""
        cache = SemanticQueryCache()

        # Do some operations
        cache.get_exact("query1")  # miss
        cache.put("query1", np.array([1.0]))
        cache.get_exact("query1")  # hit

        stats = cache.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert abs(stats["hit_rate"] - 0.5) < 0.01

    def test_clear(self):
        """Cache should clear properly."""
        cache = SemanticQueryCache()

        cache.put("query1", np.array([1.0]))
        cache.put("query2", np.array([2.0]))
        assert len(cache.cache) == 2

        cache.clear()
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_thread_safety(self):
        """Cache operations should be thread-safe."""
        import threading

        cache = SemanticQueryCache()

        def worker(query_id: int):
            for _ in range(10):
                query = f"query {query_id}"
                embedding = np.array([float(query_id)])
                cache.put(query, embedding)
                cache.get_exact(query)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should complete without deadlock
        assert len(cache.cache) >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
