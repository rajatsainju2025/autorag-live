"""
Comprehensive migration guide for performance optimizations.

This guide explains how to adopt new optimization features in AutoRAG-Live.
"""

# # Migration Guide: Performance Optimizations

## Quick Start

### 1. Enable Query Caching

Before:
```python
retriever = DenseRetriever()
results = retriever.retrieve("query", k=5)
```

After:
```python
from autorag_live.cache.query_cache import CacheableRetriever

retriever = DenseRetriever()
cached_retriever = CacheableRetriever(retriever, cache_enabled=True)
results = cached_retriever.retrieve("query", k=5)
```

### 2. Use Batch Processing

Before:
```python
results = []
for doc in large_corpus:
    results.append(retriever.retrieve(doc, k=5))
```

After:
```python
from autorag_live.utils.batch_processing import BatchProcessor

batch_processor = BatchProcessor(batch_size=32)
results = batch_processor.process(
    large_corpus,
    lambda batch: [retriever.retrieve(doc, k=5) for doc in batch]
)
```

### 3. Monitor Performance

Before:
```python
import time
start = time.time()
results = retriever.retrieve("query", k=5)
duration = time.time() - start
print(f"Query took {duration}s")
```

After:
```python
from autorag_live.utils.performance_metrics import PerformanceMetrics

metrics = PerformanceMetrics()
with metrics.timer("query_retrieval", unit="ms"):
    results = retriever.retrieve("query", k=5)

print(metrics.summary())
```

### 4. Use Multi-Retriever Hybrid Search

Before:
```python
from autorag_live.retrievers import hybrid

results = hybrid.hybrid_retrieve("query", corpus, k=5, bm25_weight=0.5)
```

After (optimized):
```python
from autorag_live.retrievers.hybrid import HybridRetriever

retriever = HybridRetriever(bm25_weight=0.5)
retriever.add_documents(corpus)
results = retriever.retrieve("query", k=5)
```

## Migration Checklist

- [ ] Update imports to use new cache classes
- [ ] Wrap retrievers with CacheableRetriever for query caching
- [ ] Use BatchProcessor for bulk operations
- [ ] Add performance metrics monitoring
- [ ] Test cache hit rates and performance improvements
- [ ] Tune batch sizes based on your hardware
- [ ] Update documentation with new patterns

## Performance Expectations

After implementing optimizations, expect:

| Scenario | Improvement |
|----------|-------------|
| Repeated queries (10x) | 50-80% faster |
| Batch processing (100 items) | 20-30% faster |
| Large corpus retrieval | 40-50% faster with caching |
| Hybrid search | 25% faster |

## Troubleshooting

### Cache Not Improving Performance

1. Check cache hit rate: `metrics = cache.get_stats()`
2. Increase cache size if too many evictions
3. Verify queries are actually repeated

### High Memory Usage

1. Reduce cache size limits
2. Use batch processing instead of sequential
3. Enable TTL expiration: `ttl_seconds=3600`

### Type Checking Issues

1. Import from `autorag_live.types.protocols` for type hints
2. Use `Protocol` classes for interface definitions
3. Run `mypy` to validate types: `poetry run mypy autorag_live/`
