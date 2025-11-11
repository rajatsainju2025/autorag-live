"""Integration and end-to-end tests for optimizations."""


def test_embedding_cache_performance(embedding_cache, benchmark):
    """Test embedding cache performance improvement."""
    import numpy as np

    # First access - cache miss
    embedding_cache.put("text1", np.random.randn(384))
    embedding_cache.put("text2", np.random.randn(384))

    # Measure cached access
    result = benchmark(embedding_cache.get, "text1")
    assert result is not None


def test_tokenization_cache_performance(tokenization_cache, benchmark):
    """Test tokenization cache performance."""
    text = "The quick brown fox jumps over the lazy dog"

    # First access
    tokenization_cache.tokenize(text)

    # Measure cached access
    result = benchmark(tokenization_cache.tokenize, text)
    assert len(result) > 0


def test_query_cache_performance(query_cache, benchmark):
    """Test query cache performance."""
    results = ["doc1", "doc2", "doc3"]
    query_cache.put("test_query", 5, results)

    # Measure cached access
    result = benchmark(query_cache.get, "test_query", 5)
    assert result == results


def test_cache_statistics(embedding_cache):
    """Test cache statistics tracking."""
    import numpy as np

    embedding_cache.put("text1", np.random.randn(384))

    # Generate cache hits
    embedding_cache.get("text1")
    embedding_cache.get("text1")
    embedding_cache.get("text1")

    # Generate misses
    embedding_cache.get("nonexistent")
    embedding_cache.get("also_missing")

    stats = embedding_cache.get_stats()
    assert stats["hits"] == 3
    assert stats["misses"] == 2
    assert stats["hit_rate"] == 0.6
