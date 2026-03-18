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
def __getattr__(name: str) -> Any:
    """Lazy load commonly used functions on first access."""
    _lazy_imports = {
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

    if name in _lazy_imports:
        module_name, attr_name = _lazy_imports[name]
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

# Extended lazy imports for optional modules
try:
    from .evals.advanced_metrics import aggregate_metrics, comprehensive_evaluation  # noqa: F401
    from .evals.llm_judge import DeterministicJudge, LLMJudge  # noqa: F401
    from .evals.performance_benchmarks import run_full_benchmark_suite  # noqa: F401
    from .evals.small import run_small_suite  # noqa: F401
except ImportError:
    pass

try:
    from .augment.hard_negatives import sample_hard_negatives  # noqa: F401
    from .augment.query_rewrites import rewrite_query  # noqa: F401
    from .augment.synonym_miner import (  # noqa: F401
        mine_synonyms_from_disagreements,
        update_terms_from_mining,
    )
except ImportError:
    pass

try:
    from .pipeline.acceptance_policy import AcceptancePolicy, safe_config_update  # noqa: F401
    from .pipeline.bandit_optimizer import (  # noqa: F401
        BanditArm,
        BanditHybridOptimizer,
        UCB1Bandit,
    )
    from .pipeline.hybrid_optimizer import (  # noqa: F401
        HybridWeights,
        grid_search_hybrid_weights,
        load_hybrid_config,
        save_hybrid_config,
    )
except ImportError:
    pass

try:
    from .disagreement.report import generate_disagreement_report  # noqa: F401
except ImportError:
    pass

try:
    from .rerank.simple import SimpleReranker  # noqa: F401
except ImportError:
    pass

try:
    from .data.time_series import FFTEmbedder, TimeSeriesNote, TimeSeriesRetriever  # noqa: F401
except ImportError:
    pass

try:
    from . import cli

    app = cli.app  # noqa: F401
except ImportError:
    pass

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
    "save_retriever_index",
    "load_retriever_index",
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
    "app",
]
