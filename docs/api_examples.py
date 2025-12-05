"""API Examples Documentation.

Comprehensive examples for all public API functions with detailed
usage patterns and best practices.
"""

# BM25 Retrieval Examples
"""
Basic BM25 Usage:
    >>> from autorag_live.retrievers import bm25
    >>> corpus = ["Python is great", "Java is popular", "C++ is fast"]
    >>> results = bm25.bm25_retrieve("programming language", corpus, k=2)
    >>> print(results)
    ['Python is great', 'Java is popular']

Async BM25 Retrieval:
    >>> from autorag_live.retrievers.async_retrieval import bm25_retrieve_async
    >>> import asyncio
    >>> results = await bm25_retrieve_async("query", corpus, k=5)

Batch Processing:
    >>> from autorag_live.retrievers.async_retrieval import batch_retrieve_async
    >>> queries = ["query1", "query2", "query3"]
    >>> results = await batch_retrieve_async(queries, corpus, method="bm25")
"""

# Dense Retrieval Examples
"""
Dense Retrieval:
    >>> from autorag_live.retrievers import dense
    >>> corpus = ["Machine learning", "Deep learning", "AI research"]
    >>> results = dense.dense_retrieve("neural networks", corpus, k=2)

With Custom Model:
    >>> results = dense.dense_retrieve(
    ...     "query",
    ...     corpus,
    ...     k=5,
    ...     model_name="all-mpnet-base-v2"
    ... )
"""

# Circuit Breaker Examples
"""
Protecting External API Calls:
    >>> from autorag_live.utils.circuit_breaker import CircuitBreaker
    >>> breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    >>>
    >>> @breaker.protected
    ... def call_external_api():
    ...     return requests.get("https://api.example.com")
    >>>
    >>> try:
    ...     result = call_external_api()
    ... except CircuitBreakerError:
    ...     print("Circuit is open, using fallback")

Checking Circuit State:
    >>> stats = breaker.get_stats()
    >>> print(f"State: {stats['state']}, Failures: {stats['failure_count']}")
"""

# Adaptive Batching Examples
"""
Memory-Aware Batch Processing:
    >>> from autorag_live.utils.adaptive_batching import get_adaptive_batch_size
    >>> batch_size = get_adaptive_batch_size(item_size_bytes=100_000)
    >>> print(f"Optimal batch size: {batch_size}")

With Progress Tracking:
    >>> from autorag_live.utils.adaptive_batching import batch_iterator_with_progress
    >>> def progress(current, total):
    ...     print(f"Processing {current}/{total}")
    >>>
    >>> for batch in batch_iterator_with_progress(items, batch_size, progress):
    ...     process_batch(batch)
"""

# Performance Profiling Examples
"""
Query Performance Profiling:
    >>> from autorag_live.utils.query_profiler import QueryProfiler
    >>> profiler = QueryProfiler()
    >>>
    >>> with profiler.stage("embedding"):
    ...     embeddings = generate_embeddings(query)
    >>>
    >>> with profiler.stage("retrieval"):
    ...     docs = retrieve_documents(query)
    >>>
    >>> report = profiler.get_report()
    >>> print(f"Total time: {report['total_time']:.3f}s")
    >>> for stage, stats in report['stages'].items():
    ...     print(f"{stage}: {stats['total']:.3f}s")
"""

# Structured Logging Examples
"""
JSON Logging with Correlation IDs:
    >>> from autorag_live.utils.structured_logging import (
    ...     JSONFormatter, set_correlation_id
    ... )
    >>> import logging
    >>>
    >>> handler = logging.StreamHandler()
    >>> handler.setFormatter(JSONFormatter())
    >>> logger = logging.getLogger()
    >>> logger.addHandler(handler)
    >>>
    >>> correlation_id = set_correlation_id()
    >>> logger.info("Processing request")
    {"timestamp": "2025-12-05T...", "correlation_id": "...", "message": "Processing request"}
"""

# Redis Cache Examples
"""
Distributed Caching:
    >>> from autorag_live.utils.redis_cache import RedisCache
    >>> cache = RedisCache(host="localhost", prefix="autorag:")
    >>>
    >>> # Cache query results
    >>> cache.put("query:123", results, ttl=3600)
    >>> cached = cache.get("query:123")
    >>>
    >>> # Clear all cache
    >>> cache.clear()
"""

# Property-Based Testing Examples
"""
Running Property Tests:
    $ pytest tests/test_property_based.py -v

    # Run with more examples
    $ pytest tests/test_property_based.py -v --hypothesis-seed=12345

Custom Properties:
    >>> from hypothesis import given
    >>> from hypothesis import strategies as st
    >>>
    >>> @given(query=st.text(min_size=1), k=st.integers(min_value=1, max_value=10))
    >>> def test_custom_property(query, k):
    ...     results = bm25.bm25_retrieve(query, corpus, k)
    ...     assert len(results) <= k
"""

__all__ = []  # Documentation only
