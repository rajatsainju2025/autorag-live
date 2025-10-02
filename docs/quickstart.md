# Quick Start

Get up and running with AutoRAG-Live in minutes.

## 🎯 Basic Usage

### Simple Retrieval

```python
from autorag_live.retrievers import bm25, dense, hybrid

# Your document corpus
corpus = [
    "The sky is blue during the day.",
    "Machine learning is a subset of AI.",
    "Python is great for data science.",
    "Retrieval-Augmented Generation combines retrieval and generation.",
    "AutoRAG-Live optimizes RAG systems automatically."
]

# BM25 retrieval
query = "artificial intelligence"
bm25_results = bm25.bm25_retrieve(query, corpus, k=3)
print("BM25 Results:", bm25_results)

# Dense retrieval (requires sentence-transformers)
dense_results = dense.dense_retrieve(query, corpus, k=3)
print("Dense Results:", dense_results)

# Hybrid retrieval (combines BM25 + Dense)
hybrid_results = hybrid.hybrid_retrieve(query, corpus, k=3)
print("Hybrid Results:", hybrid_results)
```

### Disagreement Analysis

```python
from autorag_live.disagreement import metrics

# Compare retriever disagreement
jaccard_score = metrics.jaccard_at_k(bm25_results, dense_results, k=3)
kendall_tau = metrics.kendall_tau_at_k(bm25_results, dense_results, corpus)

print(f"Jaccard similarity: {jaccard_score:.3f}")
print(f"Kendall tau correlation: {kendall_tau:.3f}")
```

### Evaluation

```python
from autorag_live.evals import small

# Run the small evaluation suite
results = small.run_small_suite()
print(f"Average F1: {results['metrics']['f1']:.3f}")
print(f"Average Relevance: {results['metrics']['relevance']:.3f}")
```

## 🏗️ Advanced Usage

### Configuration Management

```python
from autorag_live.utils import ConfigManager

# Load configuration
config = ConfigManager()
retrieval_config = config.get_config('retrieval')

# Access nested configuration
bm25_config = retrieval_config['bm25']
print(f"BM25 k1 parameter: {bm25_config['k1']}")
```

### Custom Retriever

```python
from autorag_live.retrievers.base import BaseRetriever
from typing import List

class CustomRetriever(BaseRetriever):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization

    def retrieve(self, query: str, corpus: List[str], k: int) -> List[str]:
        # Custom retrieval logic
        return corpus[:k]  # Simple example

# Use custom retriever
retriever = CustomRetriever()
results = retriever.retrieve("query", corpus, k=2)
```

### Optimization Pipeline

```python
from autorag_live.pipeline import hybrid_optimizer, acceptance_policy

# Optimize hybrid weights
weights, score = hybrid_optimizer.grid_search_hybrid_weights(
    queries=["AI", "machine learning", "data science"],
    corpus=corpus,
    k=5
)

print(f"Optimal weights: BM25={weights.bm25_weight:.2f}, Dense={weights.dense_weight:.2f}")
print(f"Diversity score: {score:.3f}")

# Save optimized configuration
hybrid_optimizer.save_hybrid_config(weights)
```

## 🖥️ CLI Usage

### Basic Commands

```bash
# Show help
autorag --help

# Run evaluation
autorag eval small

# Analyze retriever disagreement
autorag disagree "your query here"
```

### Configuration

Create a configuration file `config.yaml`:

```yaml
retrieval:
  bm25:
    k1: 1.5
    b: 0.75
  dense:
    model_name: "all-MiniLM-L6-v2"
    device: "cpu"

evaluation:
  judge_type: "deterministic"
  metrics: ["exact_match", "f1", "relevance"]
```

## 📊 Monitoring and Logging

AutoRAG-Live provides comprehensive logging:

```python
import logging
from autorag_live.utils import get_logger

# Get configured logger
logger = get_logger(__name__)
logger.info("Starting retrieval process...")

# Logs are automatically configured with appropriate levels
# and can be customized via configuration
```

## 🔄 Self-Optimization Loop

AutoRAG-Live can optimize itself:

```python
from autorag_live.pipeline import acceptance_policy

# Create acceptance policy
policy = acceptance_policy.AcceptancePolicy(
    threshold=0.01,  # Minimum improvement required
    metric_key="f1"
)

# Evaluate and potentially accept changes
accepted = policy.evaluate_change(
    config_backup_paths={},  # Backup paths for config files
    runs_dir="runs"
)

if accepted:
    print("✅ Changes accepted - performance improved!")
else:
    print("❌ Changes reverted - insufficient improvement")
```

## 📈 Performance Monitoring

Track performance over time:

```python
from autorag_live.data import time_series

# Create time series tracker
tracker = time_series.TimeSeriesTracker()

# Add performance data
tracker.add_note("evaluation", {"f1": 0.85, "relevance": 0.78})
tracker.add_notes([
    {"metric": "f1", "value": 0.85, "timestamp": "2024-01-01"},
    {"metric": "relevance", "value": 0.78, "timestamp": "2024-01-01"}
])

# Analyze trends
trends = tracker.analyze_trends(days=30)
print(f"Performance trend: {trends}")
```

## 🎯 Next Steps

- [Configuration Guide](configuration.md) - Advanced configuration options
- [Core Concepts](core-concepts.md) - Understanding AutoRAG-Live's approach
- [API Reference](../api-reference.md) - Complete API documentation
- [Contributing](../contributing.md) - How to contribute to the project
