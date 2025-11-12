"""Utility functions and helpers for AutoRAG-Live."""

from .batch_processing import BatchProcessor, ChunkIterator, estimate_optimal_batch_size
from .cache import (
    Cache,
    CacheManager,
    FileCache,
    MemoryCache,
    cached,
    clear_all_caches,
    clear_cache,
    generate_cache_key,
    get_cache,
)
from .config import ConfigManager
from .lazy_loader import (
    LazyLoader,
    get_elasticsearch_client,
    get_qdrant_client,
    get_sentence_transformer,
    lazy_import,
)
from .logging_config import (
    get_audit_logger,
    get_logger,
    get_performance_logger,
    get_structured_logger,
    logging_context,
    setup_logging,
)
from .numpy_ops import (
    batch_cosine_similarity,
    batched_dot_product,
    chunk_array,
    cosine_similarity_fast,
    ranked_top_k,
    reduce_memory_copies,
    top_k_similarity,
)
from .performance import (
    PerformanceMonitor,
    SystemMonitor,
    get_system_metrics,
    monitor_performance,
    performance_monitor,
    profile_function,
    start_system_monitoring,
    stop_system_monitoring,
    system_monitor,
)
from .performance_metrics import MemoryMonitor, PerformanceMetrics
from .schema import AutoRAGConfig, CacheConfig, EvaluationConfig, LoggingConfig, RetrievalConfig

__all__ = [
    # Cache utilities
    "Cache",
    "MemoryCache",
    "FileCache",
    "CacheManager",
    "cached",
    "generate_cache_key",
    "get_cache",
    "clear_cache",
    "clear_all_caches",
    # Configuration
    "ConfigManager",
    # Lazy loading
    "LazyLoader",
    "get_elasticsearch_client",
    "get_qdrant_client",
    "get_sentence_transformer",
    "lazy_import",
    # Logging
    "get_audit_logger",
    "get_logger",
    "get_performance_logger",
    "get_structured_logger",
    "logging_context",
    "setup_logging",
    # Performance monitoring
    "PerformanceMonitor",
    "SystemMonitor",
    "monitor_performance",
    "profile_function",
    "performance_monitor",
    "system_monitor",
    "start_system_monitoring",
    "stop_system_monitoring",
    "get_system_metrics",
    # New performance metrics
    "PerformanceMetrics",
    "MemoryMonitor",
    # Numpy operations
    "batch_cosine_similarity",
    "batched_dot_product",
    "chunk_array",
    "cosine_similarity_fast",
    "ranked_top_k",
    "reduce_memory_copies",
    "top_k_similarity",
    # Batch processing
    "BatchProcessor",
    "ChunkIterator",
    "estimate_optimal_batch_size",
    # Schema
    "AutoRAGConfig",
    "RetrievalConfig",
    "EvaluationConfig",
    "LoggingConfig",
    "CacheConfig",
]
