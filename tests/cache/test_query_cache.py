"""
Tests for query caching system.
"""
import time

import pytest

from autorag_live.cache import QueryCache, SemanticQueryCache, get_default_cache


class TestQueryCache:
    """Test suite for QueryCache."""

    def test_basic_get_set(self):
        """Test basic get/set operations."""
        cache = QueryCache(max_size=10)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.set("key2", [1, 2, 3])
        assert cache.get("key2") == [1, 2, 3]

    def test_missing_key(self):
        """Test behavior for missing keys."""
        cache = QueryCache(max_size=10)

        assert cache.get("nonexistent") is None
        assert cache.get("nonexistent", "default") == "default"

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = QueryCache(max_size=3, eviction_policy="lru")

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = QueryCache(max_size=10, default_ttl=0.1)  # 100ms TTL

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.15)

        assert cache.get("key1") is None

    def test_custom_ttl(self):
        """Test custom TTL per entry."""
        cache = QueryCache(max_size=10, default_ttl=0.1)

        cache.set("key1", "value1", ttl=1.0)  # 1 second TTL
        cache.set("key2", "value2")  # Use default 100ms TTL

        time.sleep(0.15)

        assert cache.get("key1") == "value1"  # Still valid
        assert cache.get("key2") is None  # Expired

    def test_delete(self):
        """Test entry deletion."""
        cache = QueryCache(max_size=10)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("key1") is False  # Already deleted

    def test_clear(self):
        """Test cache clearing."""
        cache = QueryCache(max_size=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get_stats()["size"] == 0

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = QueryCache(max_size=10, default_ttl=0.1)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3", ttl=1.0)  # Longer TTL

        time.sleep(0.15)

        removed = cache.cleanup_expired()
        assert removed == 2

        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"

    def test_statistics(self):
        """Test cache statistics tracking."""
        cache = QueryCache(max_size=10)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss

        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2.0 / 3.0

    def test_lfu_eviction(self):
        """Test LFU (Least Frequently Used) eviction."""
        cache = QueryCache(max_size=3, eviction_policy="lfu")

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 and key3 multiple times
        cache.get("key1")
        cache.get("key1")
        cache.get("key3")

        # Add key4, should evict key2 (least frequently used)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_context_manager(self, tmp_path):
        """Test context manager with persistence."""
        cache_file = tmp_path / "cache.pkl"

        # Create and populate cache
        with QueryCache(max_size=10, persist_path=cache_file) as cache:
            cache.set("key1", "value1")
            cache.set("key2", [1, 2, 3])

        # Cache should be saved
        assert cache_file.exists()

        # Load cache
        cache2 = QueryCache(max_size=10, persist_path=cache_file)
        assert cache2.get("key1") == "value1"
        assert cache2.get("key2") == [1, 2, 3]

    def test_memory_limit(self):
        """Test memory-based eviction."""
        # Create cache with 1KB memory limit
        cache = QueryCache(max_size=1000, max_memory_mb=0.001)  # ~1KB

        # Add small entries until memory limit reached
        for i in range(100):
            cache.set(f"key{i}", "x" * 100)  # ~100 bytes each

        stats = cache.get_stats()
        # Should have evicted some entries due to memory limit
        assert stats["size"] < 100
        assert stats["evictions"] > 0


class TestSemanticQueryCache:
    """Test suite for SemanticQueryCache."""

    def test_exact_match(self):
        """Test exact query matching."""
        cache = SemanticQueryCache(max_size=10)

        cache.set("what is machine learning", ["doc1", "doc2"])
        result = cache.get("what is machine learning")

        assert result == ["doc1", "doc2"]

    def test_no_embedding_model_fallback(self):
        """Test fallback behavior without embedding model."""
        cache = SemanticQueryCache(max_size=10, embedding_model=None)

        cache.set("query1", ["result1"])
        assert cache.get("query1") == ["result1"]
        assert cache.get("query2") is None

    @pytest.mark.skip(reason="Requires sentence-transformers")
    def test_semantic_similarity(self):
        """Test semantic similarity matching (requires model)."""
        # This test requires sentence-transformers
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        cache = SemanticQueryCache(max_size=10, similarity_threshold=0.9, embedding_model=model)

        cache.set("what is AI", ["doc1", "doc2"])

        # Similar query should get cache hit
        result = cache.get("what is artificial intelligence")
        # May or may not hit depending on similarity
        # Just test it doesn't crash
        assert result is None or isinstance(result, list)


class TestGlobalCaches:
    """Test global cache instances."""

    def test_get_default_cache(self):
        """Test default cache singleton."""
        cache1 = get_default_cache()
        cache2 = get_default_cache()

        assert cache1 is cache2

        cache1.set("test", "value")
        assert cache2.get("test") == "value"


class TestConcurrency:
    """Test thread safety."""

    def test_concurrent_access(self):
        """Test concurrent get/set operations."""
        import threading

        cache = QueryCache(max_size=100)

        def worker(thread_id):
            for i in range(10):
                cache.set(f"key_{thread_id}_{i}", f"value_{thread_id}_{i}")
                cache.get(f"key_{thread_id}_{i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors
        stats = cache.get_stats()
        assert stats["hits"] > 0
