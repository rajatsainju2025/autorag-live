"""
AutoRAG-Live: Disagreement-driven, self-optimizing RAG system.

This package provides a comprehensive framework for retrieval-augmented generation
with built-in disagreement analysis, automatic optimization, and self-improvement capabilities.
"""

__version__ = "0.1.0"

from .disagreement.metrics import jaccard_at_k, kendall_tau_at_k  # noqa: F401

# Core fast imports (always needed)
from .retrievers.bm25 import bm25_retrieve  # noqa: F401
from .retrievers.dense import dense_retrieve  # noqa: F401
from .retrievers.hybrid import hybrid_retrieve  # noqa: F401

# Optional heavy imports with try-except for better performance
try:
    from .retrievers.faiss_adapter import (  # noqa: F401
        DenseRetriever,
        SentenceTransformerRetriever,
        create_dense_retriever,
        load_retriever_index,
        save_retriever_index,
    )
except ImportError:
    pass

try:
    from .retrievers.qdrant_adapter import QdrantRetriever  # noqa: F401
except ImportError:
    pass

try:
    from .retrievers.elasticsearch_adapter import ElasticsearchRetriever  # noqa: F401
except ImportError:
    pass

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
