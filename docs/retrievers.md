# Retrievers

AutoRAG-Live supports multiple retrieval strategies that can be used individually or combined.

## 🎯 Available Retrievers

### BM25 Retrieval

BM25 (Best Matching 25) is a traditional information retrieval algorithm based on TF-IDF with document length normalization.

**Strengths:**
- Fast indexing and retrieval
- Interpretable scoring
- No training required
- Good for exact keyword matching

**Use Cases:**
- Keyword-based search
- Document ranking
- Baseline retrieval

```python
from autorag_live.retrievers import bm25

# Basic usage
results = bm25.bm25_retrieve("machine learning", corpus, k=10)

# Advanced usage with custom parameters
results = bm25.bm25_retrieve(
    query="artificial intelligence",
    corpus=documents,
    k=5,
    k1=1.5,    # Term frequency saturation
    b=0.75     # Document length normalization
)
```

### Dense Retrieval

Dense retrieval uses neural embeddings to find semantically similar documents.

**Strengths:**
- Semantic understanding
- Handles paraphrases and synonyms
- Cross-lingual retrieval
- Learns from data

**Use Cases:**
- Semantic search
- Question answering
- Content recommendation

```python
from autorag_live.retrievers import dense

# Basic usage
results = dense.dense_retrieve("AI applications", corpus, k=10)

# Advanced usage with custom model
results = dense.dense_retrieve(
    query="machine learning algorithms",
    corpus=documents,
    k=5,
    model_name="all-mpnet-base-v2",  # Better model
    device="cuda"                    # GPU acceleration
)
```

### Hybrid Retrieval

Combines BM25 and dense retrieval for optimal performance.

**Strengths:**
- Best of both worlds
- Robust to different query types
- Improved accuracy
- Handles both lexical and semantic matching

**Use Cases:**
- General-purpose retrieval
- Production systems
- High-accuracy requirements

```python
from autorag_live.retrievers import hybrid

# Basic usage (balanced weights)
results = hybrid.hybrid_retrieve("quantum computing", corpus, k=10)

# Custom weights
results = hybrid.hybrid_retrieve_weighted(
    query="neural networks",
    corpus=documents,
    k=5,
    weights=HybridWeights(bm25_weight=0.3, dense_weight=0.7)
)
```

## 🗄️ Vector Databases

### Qdrant

High-performance vector database with advanced indexing.

```python
from autorag_live.retrievers import QdrantRetriever

retriever = QdrantRetriever(
    url="http://localhost:6333",
    collection_name="my_docs",
    vector_size=384
)

# Index documents
retriever.add_documents(documents)

# Search
results = retriever.retrieve("query", k=5)
```

### FAISS

Facebook AI Similarity Search for efficient similarity search.

```python
from autorag_live.retrievers import FAISSRetriever

retriever = FAISSRetriever(
    index_type="IndexIVFFlat",
    nlist=100,
    metric="L2"
)

# Index documents
retriever.add_documents(documents)

# Search
results = retriever.retrieve("query", k=5)
```

### Elasticsearch

Full-text search engine with vector search capabilities.

```python
from autorag_live.retrievers import ElasticsearchRetriever

retriever = ElasticsearchRetriever(
    hosts=["http://localhost:9200"],
    index_name="documents",
    vector_field="embedding"
)

# Index documents
retriever.add_documents(documents)

# Search
results = retriever.retrieve("query", k=5)
```

## ⚖️ Retriever Comparison

| Retriever | Speed | Accuracy | Setup | Training |
|-----------|-------|----------|-------|----------|
| BM25 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
| Dense | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ |
| Hybrid | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ |
| Qdrant | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ |
| FAISS | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ |
| Elasticsearch | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ❌ |

## 🔧 Configuration

Configure retrievers in `config/retrieval.yaml`:

```yaml
bm25:
  k1: 1.5
  b: 0.75

dense:
  model_name: "all-MiniLM-L6-v2"
  device: "auto"
  batch_size: 32

hybrid:
  bm25_weight: 0.5
  dense_weight: 0.5
```

## 📊 Performance Monitoring

Monitor retriever performance:

```python
from autorag_live.retrievers.base import BaseRetriever

class MonitoredRetriever(BaseRetriever):
    def retrieve(self, query: str, corpus: List[str], k: int) -> List[str]:
        import time
        start_time = time.time()

        results = super().retrieve(query, corpus, k)

        elapsed = time.time() - start_time
        logger.info(f"Retrieval took {elapsed:.3f}s for query: {query}")

        return results
```

## 🎯 Best Practices

### Choosing a Retriever

- **Use BM25** for simple keyword-based search
- **Use Dense** for semantic understanding
- **Use Hybrid** for production systems
- **Use Vector DBs** for large-scale retrieval

### Optimization Tips

```python
# Pre-compute embeddings for dense retrieval
from autorag_live.retrievers import dense
embeddings = dense.precompute_embeddings(corpus)

# Use approximate indexes for speed
faiss_config = {
    "index_type": "IndexIVFPQ",  # Product quantization
    "nlist": 256,
    "m": 8,                      # Sub-quantizers
    "nbits": 8                   # Bits per sub-quantizer
}
```

### Memory Management

```python
# Use streaming for large corpora
def process_large_corpus(corpus, batch_size=1000):
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        # Process batch
        results = retriever.retrieve(query, batch, k=k)
        # Combine results...
```

## 🔍 Advanced Features

### Custom Scoring

```python
from autorag_live.retrievers.base import BaseRetriever

class CustomRetriever(BaseRetriever):
    def _score_documents(self, query: str, corpus: List[str]) -> List[float]:
        # Custom scoring logic
        scores = []
        for doc in corpus:
            # Your custom scoring
            score = len(set(query.split()) & set(doc.split())) / len(doc.split())
            scores.append(score)
        return scores
```

### Ensemble Retrieval

```python
def ensemble_retrieve(query, corpus, k=10):
    """Combine multiple retrievers."""
    bm25_results = bm25.bm25_retrieve(query, corpus, k=k*2)
    dense_results = dense.dense_retrieve(query, corpus, k=k*2)

    # Reciprocal Rank Fusion
    scores = {}
    for rank, doc in enumerate(bm25_results):
        scores[doc] = scores.get(doc, 0) + 1 / (rank + 60)
    for rank, doc in enumerate(dense_results):
        scores[doc] = scores.get(doc, 0) + 1 / (rank + 60)

    return sorted(scores.keys(), key=scores.get, reverse=True)[:k]
```

### Query Expansion

```python
from autorag_live.augment import query_rewrites

def retrieve_with_expansion(query, corpus, k=10):
    """Retrieve with query expansion."""
    expanded_queries = query_rewrites.generate_rewrites(query, num_rewrites=3)

    all_results = []
    for q in [query] + expanded_queries:
        results = hybrid.hybrid_retrieve(q, corpus, k=k)
        all_results.extend(results)

    # Remove duplicates while preserving order
    seen = set()
    unique_results = []
    for doc in all_results:
        if doc not in seen:
            unique_results.append(doc)
            seen.add(doc)

    return unique_results[:k]
```