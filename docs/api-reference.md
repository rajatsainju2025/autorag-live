# API Reference

This page provides comprehensive API documentation for all AutoRAG-Live modules.

## 📚 Retrievers

### BM25 Retrieval

```python
from autorag_live.retrievers import bm25

def bm25_retrieve(query: str, corpus: List[str], k: int = 10) -> List[str]:
    """Retrieve documents using BM25 scoring.

    Args:
        query: Search query string
        corpus: List of documents to search
        k: Number of documents to retrieve

    Returns:
        List of top-k relevant documents

    Example:
        results = bm25_retrieve("machine learning", documents, k=5)
    """
```

### Dense Retrieval

```python
from autorag_live.retrievers import dense

def dense_retrieve(query: str, corpus: List[str], k: int = 10) -> List[str]:
    """Retrieve documents using dense embeddings.

    Args:
        query: Search query string
        corpus: List of documents to search
        k: Number of documents to retrieve

    Returns:
        List of top-k relevant documents

    Example:
        results = dense_retrieve("artificial intelligence", documents, k=5)
    """
```

### Hybrid Retrieval

```python
from autorag_live.retrievers import hybrid

def hybrid_retrieve(query: str, corpus: List[str], k: int = 10) -> List[str]:
    """Retrieve documents using hybrid BM25 + dense scoring.

    Args:
        query: Search query string
        corpus: List of documents to search
        k: Number of documents to retrieve

    Returns:
        List of top-k relevant documents

    Example:
        results = hybrid_retrieve("data science", documents, k=5)
    """
```

### Advanced Retrievers

```python
# Elasticsearch (optional dependency)
from autorag_live.retrievers.elasticsearch_adapter import ElasticsearchRetriever

retriever = ElasticsearchRetriever(host="localhost", port=9200)
results = retriever.retrieve("query", index_name="documents", k=10)

# FAISS (optional dependency)
from autorag_live.retrievers.faiss_adapter import FAISSRetriever

retriever = FAISSRetriever(dimension=768)
retriever.add_documents(documents, embeddings)
results = retriever.search(query_embedding, k=10)

# Qdrant (optional dependency)
from autorag_live.retrievers.qdrant_adapter import QdrantRetriever

retriever = QdrantRetriever(url="localhost:6333", collection_name="docs")
results = retriever.search("query", limit=10)
```

## 📊 Disagreement Analysis

### Metrics

```python
from autorag_live.disagreement import metrics

def jaccard_at_k(results1: List[str], results2: List[str], k: int = 10) -> float:
    """Calculate Jaccard similarity between top-k results.

    Args:
        results1: First list of retrieved documents
        results2: Second list of retrieved documents
        k: Number of top results to consider

    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """

def kendall_tau_at_k(results1: List[str], results2: List[str], k: int = 10) -> Optional[float]:
    """Calculate Kendall tau rank correlation.

    Args:
        results1: First ranking of documents
        results2: Second ranking of documents
        k: Number of top results to consider

    Returns:
        Kendall tau correlation (-1.0 to 1.0) or None if insufficient data
    """
```

### Reports

```python
from autorag_live.disagreement import report

def generate_disagreement_report(
    query: str,
    retriever_results: Dict[str, List[str]],
    disagreement_metrics: Dict[str, float],
    output_path: str
) -> None:
    """Generate HTML report for disagreement analysis.

    Args:
        query: Original search query
        retriever_results: Dictionary mapping retriever names to results
        disagreement_metrics: Dictionary of calculated metrics
        output_path: Path to save HTML report
    """
```

## 🎯 Evaluation

### Small Evaluation Suite

```python
from autorag_live.evals.small import run_small_suite

def run_small_suite(judge_type: str = "deterministic") -> Dict[str, Any]:
    """Run the small evaluation suite.

    Args:
        judge_type: Type of judge ("deterministic" or "openai")

    Returns:
        Dictionary with metrics and run information

    Example:
        results = run_small_suite("deterministic")
        print(f"EM: {results['metrics']['em']:.3f}")
        print(f"F1: {results['metrics']['f1']:.3f}")
    """
```

### Advanced Metrics

```python
from autorag_live.evals.advanced_metrics import comprehensive_evaluation

def comprehensive_evaluation(
    retrieved_docs: List[str],
    relevant_docs: List[str],
    query: str = "",
    embeddings: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, float]:
    """Run comprehensive evaluation with multiple metrics.

    Args:
        retrieved_docs: Retrieved documents
        relevant_docs: Ground truth relevant documents
        query: Original query (optional)
        embeddings: Document embeddings for advanced metrics
        **kwargs: Additional parameters

    Returns:
        Dictionary of evaluation metrics

    Metrics include:
        - ndcg@5, ndcg@10: Normalized Discounted Cumulative Gain
        - precision@5, precision@10: Precision at k
        - recall@5, recall@10: Recall at k
        - diversity: Semantic diversity of results
        - contextual_relevance: Query-document relevance
        - novelty: Novelty compared to history
        - robustness: Consistency across runs
        - fairness: Fair representation across groups
        - efficiency: Speed and throughput metrics
    """
```

### Performance Benchmarks

```python
from autorag_live.evals.performance_benchmarks import PerformanceBenchmark

class PerformanceBenchmark:
    """Performance benchmarking suite."""

    def benchmark_function(
        self,
        func: Callable,
        *args,
        iterations: int = 10,
        operation_name: str = None,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark a function's performance."""

    def save_results(self, filename: Optional[str] = None) -> str:
        """Save benchmark results to JSON file."""

    def print_summary(self) -> None:
        """Print benchmark summary."""
```

## ⚙️ Optimization

### Hybrid Optimizer

```python
from autorag_live.pipeline.hybrid_optimizer import grid_search_hybrid_weights

def grid_search_hybrid_weights(
    queries: List[str],
    corpus: List[str],
    k: int = 5,
    grid_size: int = 5
) -> Tuple[HybridWeights, float]:
    """Optimize hybrid weights using grid search.

    Args:
        queries: List of training queries
        corpus: Document corpus
        k: Number of documents to retrieve
        grid_size: Size of weight grid to search

    Returns:
        Tuple of (optimal_weights, best_diversity_score)

    Example:
        weights, score = grid_search_hybrid_weights(
            ["query1", "query2"], documents, k=5, grid_size=4
        )
        print(f"Optimal weights: BM25={weights.bm25_weight:.3f}, "
              f"Dense={weights.dense_weight:.3f}")
    """
```

### Bandit Optimization

```python
from autorag_live.pipeline.bandit_optimizer import BanditHybridOptimizer

class BanditHybridOptimizer:
    """Bandit-based optimizer for hybrid weights."""

    def __init__(
        self,
        retriever_names: List[str],
        bandit_type: str = "ucb1",
        exploration_factor: float = 2.0,
        num_random_arms: int = 10
    ):
        """Initialize bandit optimizer."""

    def suggest_weights(self) -> Dict[str, float]:
        """Suggest next weight configuration to try."""

    def update_weights(self, weights: Dict[str, float], reward: float):
        """Update with reward for weight configuration."""

    def get_best_weights(self) -> Dict[str, float]:
        """Get best weights found so far."""
```

### Acceptance Policy

```python
from autorag_live.pipeline.acceptance_policy import AcceptancePolicy, safe_config_update

class AcceptancePolicy:
    """Policy for accepting configuration changes."""

    def __init__(self, threshold: float = 0.01, metric_key: str = "f1"):
        """Initialize acceptance policy."""

    def evaluate_change(self, improvement: float) -> bool:
        """Evaluate if improvement meets threshold."""

def safe_config_update(
    update_func: Callable,
    config_files: List[str],
    policy: AcceptancePolicy
) -> bool:
    """Safely update configuration with automatic revert."""
```

## 🔧 Data Augmentation

### Synonym Mining

```python
from autorag_live.augment.synonym_miner import (
    mine_synonyms_from_disagreements,
    update_terms_from_mining
)

def mine_synonyms_from_disagreements(
    bm25_results: List[str],
    dense_results: List[str],
    hybrid_results: List[str]
) -> List[Dict[str, Any]]:
    """Mine synonyms from retriever disagreements.

    Args:
        bm25_results: BM25 retrieval results
        dense_results: Dense retrieval results
        hybrid_results: Hybrid retrieval results

    Returns:
        List of synonym mappings
    """

def update_terms_from_mining(synonyms: List[Dict[str, Any]]) -> None:
    """Update terms database with mined synonyms."""
```

## 🔄 Reranking

### Simple Reranker

```python
from autorag_live.rerank.simple import SimpleReranker

class SimpleReranker:
    """Deterministic reranker using query-document features."""

    def __init__(self):
        """Initialize reranker."""

    def rerank(
        self,
        query: str,
        docs: List[str],
        k: Optional[int] = None
    ) -> List[str]:
        """Rerank documents for query.

        Args:
            query: Search query
            docs: Documents to rerank
            k: Number of documents to return (None = all)

        Returns:
            Reranked list of documents
        """
```

## ⏰ Time-Series Retrieval

### TimeSeriesRetriever

```python
from autorag_live.data.time_series import TimeSeriesRetriever, TimeSeriesNote

class TimeSeriesRetriever:
    """Retriever for time-series documents."""

    def __init__(
        self,
        embedder: Optional[FFTEmbedder] = None,
        temporal_weight: float = 0.3,
        content_weight: float = 0.7
    ):
        """Initialize time-series retriever."""

    def add_note(
        self,
        content: str,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TimeSeriesNote:
        """Add a time-series note."""

    def search(
        self,
        query: str,
        query_time: datetime,
        top_k: int = 10,
        time_window_days: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search with temporal constraints."""
```

## 🖥️ CLI Interface

### Main Commands

```bash
# Disagreement analysis
autorag disagree "query" --k 10 --report-path report.html

# Evaluation
autorag eval --suite small --judge deterministic

# Optimization
autorag optimize --queries "query1" "query2"

# Advanced evaluation
autorag advanced-eval "query" --relevant-docs "doc1" "doc2"

# Reranking
autorag rerank "query" --k 10

# Bandit optimization
autorag bandit-optimize --iterations 20

# Time-series retrieval
autorag time-series-retrieve "query" --time-window 7d

# Retriever comparison
autorag compare-retrievers "query"

# Configuration management
autorag config show
autorag config update --bm25-weight 0.6 --dense-weight 0.4

# Self-improvement loop
autorag self-improve --iterations 5

# Performance benchmarking
autorag benchmark --components retrievers evaluation
autorag benchmark  # Run full suite
```

### CLI Options

All commands support `--help` for detailed usage information:

```bash
autorag --help          # Main help
autorag disagree --help # Command-specific help
```

## 📊 Data Structures

### HybridWeights

```python
@dataclass
class HybridWeights:
    """Weights for hybrid retrieval."""
    bm25_weight: float
    dense_weight: float
```

### BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    operation: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_time: float
    memory_usage_mb: float
    throughput: float
    metadata: Dict[str, Any]
```

### TimeSeriesNote

```python
@dataclass
class TimeSeriesNote:
    """Time-series document with temporal information."""
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
```

## 🔌 Configuration

### Environment Variables

```bash
# OpenAI API (for advanced evaluation)
export OPENAI_API_KEY="your-key-here"

# Elasticsearch
export ELASTICSEARCH_HOST="localhost"
export ELASTICSEARCH_PORT="9200"

# Qdrant
export QDRANT_URL="localhost:6333"

# Logging
export AUTORAG_LOG_LEVEL="INFO"
```

### Configuration Files

- `hybrid_config.json`: Hybrid retriever weights
- `terms.yaml`: Synonym database
- `best_runs.json`: Best evaluation results

## 🚨 Error Handling

AutoRAG-Live provides comprehensive error handling:

```python
try:
    results = hybrid_retrieve(query, corpus, k=10)
except ValueError as e:
    print(f"Invalid parameters: {e}")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 🔧 Extending AutoRAG-Live

### Custom Retriever

```python
from autorag_live.retrievers.base import BaseRetriever

class CustomRetriever(BaseRetriever):
    def retrieve(self, query: str, corpus: List[str], k: int) -> List[str]:
        # Implement custom retrieval logic
        pass
```

### Custom Metric

```python
def custom_metric(retrieved: List[str], relevant: List[str]) -> float:
    # Implement custom evaluation metric
    pass
```

### Custom Optimizer

```python
from autorag_live.pipeline.base_optimizer import BaseOptimizer

class CustomOptimizer(BaseOptimizer):
    def optimize(self, queries: List[str], corpus: List[str]) -> Dict[str, Any]:
        # Implement custom optimization logic
        pass
```

---

This API reference covers all major components. For more details, see the source code or use `help()` on any function or class.