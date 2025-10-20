"""
AutoRAG-Live: Disagreement-driven, self-optimizing RAG system.

This package provides a comprehensive framework for retrieval-augmented generation
with built-in disagreement analysis, automatic optimization, and self-improvement capabilities.
"""

__version__ = "0.1.0"

# CLI
from . import cli

# Augmentation
from .augment.hard_negatives import sample_hard_negatives
from .augment.query_rewrites import rewrite_query
from .augment.synonym_miner import mine_synonyms_from_disagreements, update_terms_from_mining

# Data utilities
from .data.time_series import FFTEmbedder, TimeSeriesNote, TimeSeriesRetriever

# Disagreement analysis
from .disagreement.metrics import jaccard_at_k, kendall_tau_at_k
from .disagreement.report import generate_disagreement_report
from .evals.advanced_metrics import aggregate_metrics, comprehensive_evaluation
from .evals.llm_judge import DeterministicJudge, LLMJudge
from .evals.performance_benchmarks import run_full_benchmark_suite

# Evaluation
from .evals.small import run_small_suite
from .pipeline.acceptance_policy import AcceptancePolicy, safe_config_update
from .pipeline.bandit_optimizer import BanditArm, BanditHybridOptimizer, UCB1Bandit

# Pipeline optimization
from .pipeline.hybrid_optimizer import (
    HybridWeights,
    grid_search_hybrid_weights,
    load_hybrid_config,
    save_hybrid_config,
)

# Reranking
from .rerank.simple import SimpleReranker

# Core retrievers
from .retrievers.bm25 import bm25_retrieve
from .retrievers.dense import dense_retrieve
from .retrievers.elasticsearch_adapter import ElasticsearchRetriever
from .retrievers.faiss_adapter import (
    DenseRetriever,
    SentenceTransformerRetriever,
    create_dense_retriever,
    load_retriever_index,
    save_retriever_index,
)
from .retrievers.hybrid import hybrid_retrieve
from .retrievers.qdrant_adapter import QdrantRetriever

app = cli.app

__all__ = [
    # Version
    "__version__",
    # Retrievers
    "bm25_retrieve",
    "dense_retrieve",
    "hybrid_retrieve",
    "DenseRetriever",
    "SentenceTransformerRetriever",
    "create_dense_retriever",
    "save_retriever_index",
    "load_retriever_index",
    "QdrantRetriever",
    "ElasticsearchRetriever",
    # Disagreement
    "jaccard_at_k",
    "kendall_tau_at_k",
    "generate_disagreement_report",
    # Evaluation
    "run_small_suite",
    "comprehensive_evaluation",
    "aggregate_metrics",
    "LLMJudge",
    "DeterministicJudge",
    "run_full_benchmark_suite",
    # Augmentation
    "sample_hard_negatives",
    "rewrite_query",
    "mine_synonyms_from_disagreements",
    "update_terms_from_mining",
    # Pipeline
    "grid_search_hybrid_weights",
    "save_hybrid_config",
    "load_hybrid_config",
    "HybridWeights",
    "BanditHybridOptimizer",
    "UCB1Bandit",
    "BanditArm",
    "AcceptancePolicy",
    "safe_config_update",
    # Reranking
    "SimpleReranker",
    # Data
    "TimeSeriesRetriever",
    "TimeSeriesNote",
    "FFTEmbedder",
    # CLI - will be added back
    # "app"
    "app",
]
