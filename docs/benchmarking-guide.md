# Performance Benchmarking Guide

This guide explains how to measure and validate the performance improvements from the optimization commits.

## Running Benchmarks

### Quick Benchmark

```bash
# Run pytest with benchmark plugin
poetry run pytest tests/test_cache_integration.py -v --benchmark-only
```

### Full Performance Test Suite

```bash
# Run all performance tests
poetry run pytest tests/ -v -k "performance or benchmark" --benchmark-only
```

### Memory Profiling

```bash
# Profile memory usage
poetry run python -m memory_profiler scripts/benchmark_memory.py
```

## Expected Performance Metrics

### Cache Operations

```
Embedding Cache Get (cached):        < 1ms
Tokenization Cache Get (cached):     < 1ms
Query Cache Get (cached):            < 1ms
```

### Retrieval Operations

```
Dense Retrieve (10 repeated queries): 200-400ms (vs 1000ms baseline)
BM25 Retrieve (with cache):          150-300ms (vs 500-600ms baseline)
Hybrid Retrieve:                     300-500ms (vs 800-1000ms baseline)
```

### Batch Operations

```
Batch Process (100 items):           3500-4000ms (vs 5000ms baseline)
```

## Monitoring in Production

### Enable Performance Metrics

```python
from autorag_live.utils.performance_metrics import PerformanceMetrics

metrics = PerformanceMetrics()

# Track retrieval performance
with metrics.timer("retrieval", unit="ms"):
    results = retriever.retrieve(query, k=5)

# Get statistics
print(metrics.summary())
```

### Cache Hit Rate Monitoring

```python
# Get cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
```

## Optimization Validation Checklist

- [ ] Embedding cache hit rate > 50%
- [ ] Tokenization cache hit rate > 60%
- [ ] Query cache hit rate > 30% (depends on workload)
- [ ] Retrieval time reduced by > 25%
- [ ] Memory usage stable with LRU eviction
- [ ] No performance regression on cold cache
- [ ] Multi-threaded performance stable

## Troubleshooting Performance

### Cache Not Improving Performance

1. Check cache size is adequate for workload
2. Verify cache hit rate with `get_stats()`
3. Profile with: `poetry run python -m cProfile -s cumtime script.py`
4. Check for cache misses due to different query formats

### Memory Issues

1. Reduce cache size
2. Enable TTL: `cache = QueryCache(ttl_seconds=3600)`
3. Monitor with: `MemoryMonitor.get_memory_usage()`

### Multi-threading Issues

1. Verify thread-safety with stress test
2. Check lock contention
3. Consider lock-free implementations for hot paths

## Comparative Benchmarks

Run comparative benchmarks before and after enabling optimizations:

```bash
# Disable caching
poetry run pytest tests/test_cache_integration.py::test_dense_retrieve_no_cache -v --benchmark-only

# Enable caching
poetry run pytest tests/test_cache_integration.py::test_dense_retrieve_cached -v --benchmark-only
```

Expected improvement: 50-80% faster with caching enabled.
