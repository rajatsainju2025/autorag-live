# Performance Optimization Guide

This guide explains the performance optimizations in AutoRAG-Live.

## Retriever Optimizations

### Dense Retriever

The `DenseRetriever` class includes several performance features:

#### 1. Global Model Cache

Models are cached globally to avoid reloading:

```python
from autorag_live.retrievers import DenseRetriever

# First instantiation loads the model
retriever1 = DenseRetriever(model_name="all-MiniLM-L6-v2")
retriever1.add_documents(["doc1", "doc2", "doc3"])

# Second instantiation reuses the cached model
retriever2 = DenseRetriever(model_name="all-MiniLM-L6-v2")
retriever2.add_documents(["other docs"])  # Fast - model already loaded

# Clear cache if needed
DenseRetriever.clear_cache()
```

#### 2. Embedding Cache

Query and corpus embeddings are cached with TTL:

```python
retriever = DenseRetriever(cache_embeddings=True)
retriever.add_documents(["doc1", "doc2", "doc3"])

# First query: encodes and caches
result1 = retriever.retrieve("query", k=5)

# Second identical query: returns from cache (instant)
result2 = retriever.retrieve("query", k=5)

# Different query: encodes again
result3 = retriever.retrieve("different", k=5)
```

#### 3. Lazy Normalization

Corpus embeddings are normalized once and cached:

```python
# Efficient: normalization happens once
retriever = DenseRetriever()
retriever.add_documents(["doc1", "doc2"])  # Normalized during encoding

# All retrievals use cached normalized embeddings
retriever.retrieve("query1")
retriever.retrieve("query2")
retriever.retrieve("query3")
```

#### 4. Batch Processing

Queries are encoded in batches for efficiency:

```python
retriever = DenseRetriever(batch_size=32)  # Configurable batch size

# Batch mode (fast)
results = retriever.retrieve_batch(
    ["query1", "query2", "query3"],
    k=5
)

# Async mode
import asyncio
results = asyncio.run(retriever.retrieve_batch_async(
    ["query1", "query2", "query3"],
    k=5
))
```

#### 5. Memory-Mapped Loading

For large embeddings, use memory-mapping:

```python
retriever = DenseRetriever()
retriever.add_documents(large_document_list)

# Save with memory mapping for large corpora
retriever.save("state.pkl", use_mmap=True)

# Load efficiently
retriever_loaded = DenseRetriever()
retriever_loaded.load("state.pkl", mmap_mode="r")  # Read-only memory map
```

#### 6. Query Pre-fetching

Frequently accessed queries are pre-computed:

```python
retriever = DenseRetriever(
    enable_prefetch=True,
    prefetch_threshold=3  # Pre-fetch after 3 identical queries
)

# First 3 queries trigger pre-fetch
retriever.retrieve("popular_query")
retriever.retrieve("popular_query")
retriever.retrieve("popular_query")  # Triggers pre-fetch

# 4th+ queries are instant
result = retriever.retrieve("popular_query")  # From pre-fetch pool
```

### BM25 Retriever

BM25 includes query tokenization caching:

```python
from autorag_live.retrievers import BM25Retriever

retriever = BM25Retriever()
retriever.add_documents(["doc1", "doc2"])

# First query: tokenizes and caches
retriever.retrieve("query terms")

# Repeated queries: use cached tokens
retriever.retrieve("query terms")  # Fast
```

## Evaluation Metrics Optimizations

### Cached Discount Factors

NDCG calculations cache discount factors:

```python
from autorag_live.evals.advanced_metrics import ndcg_at_k

# First call: computes discount factors
score1 = ndcg_at_k(retrieved, relevant, k=10)

# Repeated k values: use cached factors
score2 = ndcg_at_k(retrieved, relevant, k=10)  # Cached

# Different k: new cache entry
score3 = ndcg_at_k(retrieved, relevant, k=20)  # New computation
```

### Vectorized Computations

Metrics use NumPy vectorization:

```python
# Vectorized hit rate computation
from autorag_live.evals.advanced_metrics import hit_rate_at_k

results = hit_rate_at_k(retrieved_docs, relevant_docs, k=10)
# Uses vectorized NumPy operations instead of loops
```

## Reranking Optimizations

### MMR Reranker

Similarity computations are cached:

```python
from autorag_live.rerank import MMRReranker

reranker = MMRReranker()

# First reranking: computes all similarities
reranked1 = reranker.rerank(candidates, "query")

# Cached similarity computations reduce repeated calculations
reranked2 = reranker.rerank(candidates, "query")
```

## Streaming Data Optimizations

### Fast Path for Small Batches

Streaming uses fast path for small batches:

```python
from autorag_live.data import DocumentStream

stream = DocumentStream()

# Small additions: fast path (no special processing)
stream.add_document("doc1")  # Direct append

# Large batch: optimized path
stream.add_documents(["doc2", "doc3", ...])
```

## Configuration for Performance

Optimize performance based on your use case:

```python
# High throughput
retriever = DenseRetriever(
    batch_size=64,          # Larger batches
    cache_embeddings=True,  # Cache everything
    enable_prefetch=True,   # Pre-fetch popular queries
)

# Low latency
retriever = DenseRetriever(
    batch_size=8,           # Smaller batches for faster response
    cache_embeddings=False, # Reduce memory
    enable_prefetch=False,  # Avoid pre-fetch overhead
)

# Memory constrained
retriever = DenseRetriever(
    batch_size=4,
    cache_embeddings=False,
)
retriever.save("state.pkl", use_mmap=True)  # Use memory mapping
```

## Monitoring Performance

Track performance metrics:

```python
from autorag_live.utils.performance import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.enable()

# Your code runs here
retriever.retrieve("query")

# Get performance data
stats = monitor.get_statistics()
print(f"Average time: {stats['avg_time']:.3f}s")
print(f"Peak memory: {stats['peak_memory']:.1f}MB")
```

## Best Practices

1. **Use batch operations** for multiple queries
2. **Enable caching** for repeated queries
3. **Use memory mapping** for large corpora (>1GB)
4. **Configure batch sizes** based on available memory
5. **Monitor performance** in production
6. **Clear caches** periodically to prevent memory leaks
7. **Use async retrieval** for concurrent workloads
