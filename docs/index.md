# AutoRAG-Live Documentation

**A disagreement-driven, self-optimizing RAG system**

AutoRAG-Live is an advanced Retrieval-Augmented Generation (RAG) system that automatically optimizes itself through disagreement analysis and iterative refinement. It combines multiple retrieval strategies, evaluates their performance, and continuously improves through machine learning techniques.

## 🚀 Quick Start

### Installation

```bash
pip install autorag-live
# or for development
git clone https://github.com/rajatsainju2025/autorag-live
cd autorag-live
pip install -e .
```

### Basic Usage

```python
from autorag_live.retrievers import hybrid
from autorag_live.disagreement import metrics

# Your document corpus
corpus = [
    "The sky is blue during the day.",
    "Machine learning is a subset of AI.",
    "Python is great for data science."
]

# Retrieve documents
query = "artificial intelligence"
results = hybrid.hybrid_retrieve(query, corpus, k=5)

# Analyze retriever disagreement
bm25_results = bm25.bm25_retrieve(query, corpus, 5)
dense_results = dense.dense_retrieve(query, corpus, 5)
diversity = metrics.jaccard_at_k(bm25_results, dense_results)

print(f"Retrieved {len(results)} documents")
print(f"Retriever diversity: {diversity:.3f}")
```

### CLI Usage

```bash
# Run disagreement analysis
autorag disagree "your query here"

# Run evaluation
autorag eval

# Optimize hybrid weights
autorag optimize

# Run benchmarks
autorag benchmark
```

## 📚 Table of Contents

- [Core Concepts](core-concepts.md)
- [API Reference](api-reference.md)
- [CLI Commands](cli-commands.md)
- [Examples](examples.md)
- [Performance](performance.md)
- [Contributing](contributing.md)

## 🏗️ Architecture

AutoRAG-Live consists of several key components:

### Retrievers
- **BM25**: Traditional lexical retrieval using TF-IDF and document frequency
- **Dense**: Semantic retrieval using sentence transformers
- **Hybrid**: Combines BM25 and dense retrieval with configurable weights
- **Advanced**: Elasticsearch, FAISS, Qdrant adapters for production use

### Disagreement Analysis
- **Metrics**: Jaccard similarity, Kendall tau rank correlation
- **Reports**: HTML reports with visualizations
- **Optimization**: Automatic weight tuning based on disagreement

### Self-Improvement Loop
- **Evaluation**: Comprehensive metrics including NDCG, MRR, diversity
- **Optimization**: Grid search, bandit algorithms, acceptance policies
- **Augmentation**: Synonym mining from retriever disagreements

### Advanced Features
- **Time-Series Retrieval**: Temporal document ranking
- **Reranking**: Deterministic feature-based reranking
- **Performance Monitoring**: Comprehensive benchmarking suite

## 🎯 Key Features

- ✅ **Self-Optimizing**: Automatically improves through disagreement analysis
- ✅ **Modular Design**: Easy to extend with new retrievers and metrics
- ✅ **Production Ready**: Docker support, comprehensive testing, CI/CD
- ✅ **Research Oriented**: Detailed metrics, benchmarking, experimentation
- ✅ **CLI Interface**: Command-line tools for all major operations
- ✅ **Extensible**: Plugin architecture for custom components

## 📊 Performance

AutoRAG-Live provides industry-leading performance with:

- Sub-millisecond retrieval for small corpora
- Configurable trade-offs between speed and accuracy
- Memory-efficient processing with optional GPU acceleration
- Comprehensive benchmarking suite for performance tracking

## 🤝 Contributing

We welcome contributions! See our [contributing guide](contributing.md) for details on:

- Setting up a development environment
- Running tests and benchmarks
- Adding new retrievers or metrics
- Documentation standards

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/rajatsainju2025/autorag-live/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rajatsainju2025/autorag-live/discussions)
- **Documentation**: [Full API Reference](api-reference.md)

---

*AutoRAG-Live is developed by Raj Atul Sainju and contributors.*
