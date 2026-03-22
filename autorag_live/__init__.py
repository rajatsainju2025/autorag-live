"""
AutoRAG-Live: Disagreement-driven, self-optimizing RAG system.

This package provides a comprehensive framework for retrieval-augmented generation
with built-in disagreement analysis, automatic optimization, and self-improvement capabilities.
"""

import sys
import warnings
from typing import Any

__version__ = "0.1.0"

# Minimum Python version check
MIN_PYTHON = (3, 10)
if sys.version_info < MIN_PYTHON:
    sys.stderr.write(
        f"ERROR: AutoRAG-Live requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or higher. "
        f"You are using Python {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}.\n"
    )
    sys.exit(1)

# Fast module availability checks (no imports)
_HAS_NUMPY = False
_HAS_OMEGACONF = False

try:
    import importlib.util

    _HAS_NUMPY = importlib.util.find_spec("numpy") is not None
    _HAS_OMEGACONF = importlib.util.find_spec("omegaconf") is not None
except ImportError:
    pass

if not _HAS_NUMPY:
    warnings.warn(
        "NumPy is not installed. Some features may not work properly.", UserWarning, stacklevel=2
    )

if not _HAS_OMEGACONF:
    warnings.warn(
        "OmegaConf is not installed. Configuration features may not work properly.",
        UserWarning,
        stacklevel=2,
    )


# Lazy-loaded core metrics (lightweight imports)
# Hoisted to module level to avoid re-creating dict on every __getattr__ miss
_LAZY_IMPORTS = {
    "jaccard_at_k": ("disagreement.metrics", "jaccard_at_k"),
    "kendall_tau_at_k": ("disagreement.metrics", "kendall_tau_at_k"),
    "bm25_retrieve": ("retrievers.bm25", "bm25_retrieve"),
    "dense_retrieve": ("retrievers.dense", "dense_retrieve"),
    "hybrid_retrieve": ("retrievers.hybrid", "hybrid_retrieve"),
    "DenseRetriever": ("retrievers.faiss_adapter", "DenseRetriever"),
    "SentenceTransformerRetriever": (
        "retrievers.faiss_adapter",
        "SentenceTransformerRetriever",
    ),
    "create_dense_retriever": ("retrievers.faiss_adapter", "create_dense_retriever"),
    "QdrantRetriever": ("retrievers.qdrant_adapter", "QdrantRetriever"),
    "ElasticsearchRetriever": ("retrievers.elasticsearch_adapter", "ElasticsearchRetriever"),
}


def __getattr__(name: str) -> Any:
    """Lazy load commonly used functions on first access."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        try:
            module = __import__(f"autorag_live.{module_name}", fromlist=[attr_name])
            return getattr(module, attr_name)
        except (ImportError, AttributeError) as e:
            raise AttributeError(f"Cannot import {name}: {e}") from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "jaccard_at_k",
    "kendall_tau_at_k",
    "bm25_retrieve",
    "dense_retrieve",
    "hybrid_retrieve",
    "DenseRetriever",
    "SentenceTransformerRetriever",
    "create_dense_retriever",
    "QdrantRetriever",
    "ElasticsearchRetriever",
]

# Extended lazy imports for optional modules — all deferred via __getattr__
# to avoid importing heavy dependencies (evals, augment, pipeline, etc.)
# at package load time.
_LAZY_IMPORTS.update(
    {
        # evals
        "aggregate_metrics": ("evals.advanced_metrics", "aggregate_metrics"),
        "comprehensive_evaluation": ("evals.advanced_metrics", "comprehensive_evaluation"),
        "LLMJudge": ("evals.llm_judge", "LLMJudge"),
        "DeterministicJudge": ("evals.llm_judge", "DeterministicJudge"),
        "run_full_benchmark_suite": ("evals.performance_benchmarks", "run_full_benchmark_suite"),
        "run_small_suite": ("evals.small", "run_small_suite"),
        # augment
        "sample_hard_negatives": ("augment.hard_negatives", "sample_hard_negatives"),
        "rewrite_query": ("augment.query_rewrites", "rewrite_query"),
        "mine_synonyms_from_disagreements": (
            "augment.synonym_miner",
            "mine_synonyms_from_disagreements",
        ),
        "update_terms_from_mining": ("augment.synonym_miner", "update_terms_from_mining"),
        # pipeline
        "AcceptancePolicy": ("pipeline.acceptance_policy", "AcceptancePolicy"),
        "safe_config_update": ("pipeline.acceptance_policy", "safe_config_update"),
        "BanditArm": ("pipeline.bandit_optimizer", "BanditArm"),
        "BanditHybridOptimizer": ("pipeline.bandit_optimizer", "BanditHybridOptimizer"),
        "UCB1Bandit": ("pipeline.bandit_optimizer", "UCB1Bandit"),
        "HybridWeights": ("pipeline.hybrid_optimizer", "HybridWeights"),
        "grid_search_hybrid_weights": ("pipeline.hybrid_optimizer", "grid_search_hybrid_weights"),
        "load_hybrid_config": ("pipeline.hybrid_optimizer", "load_hybrid_config"),
        "save_hybrid_config": ("pipeline.hybrid_optimizer", "save_hybrid_config"),
        # disagreement
        "generate_disagreement_report": ("disagreement.report", "generate_disagreement_report"),
        # rerank
        "SimpleReranker": ("rerank.simple", "SimpleReranker"),
        # data
        "FFTEmbedder": ("data.time_series", "FFTEmbedder"),
        "TimeSeriesNote": ("data.time_series", "TimeSeriesNote"),
        "TimeSeriesRetriever": ("data.time_series", "TimeSeriesRetriever"),
    }
)

__all__ = [
    # Version
    "__version__",
    # Retrievers (fast import - core)
    "bm25_retrieve",
    "dense_retrieve",
    "hybrid_retrieve",
    "jaccard_at_k",
    "kendall_tau_at_k",
    # Lazy-loaded items (loaded on demand)
    "DenseRetriever",
    "SentenceTransformerRetriever",
    "create_dense_retriever",
    "QdrantRetriever",
    "ElasticsearchRetriever",
    "generate_disagreement_report",
    "run_small_suite",
    "comprehensive_evaluation",
    "aggregate_metrics",
    "LLMJudge",
    "DeterministicJudge",
    "run_full_benchmark_suite",
    "sample_hard_negatives",
    "rewrite_query",
    "mine_synonyms_from_disagreements",
    "update_terms_from_mining",
    "grid_search_hybrid_weights",
    "save_hybrid_config",
    "load_hybrid_config",
    "HybridWeights",
    "BanditHybridOptimizer",
    "UCB1Bandit",
    "BanditArm",
    "AcceptancePolicy",
    "safe_config_update",
    "SimpleReranker",
    "TimeSeriesRetriever",
    "TimeSeriesNote",
    "FFTEmbedder",
]
