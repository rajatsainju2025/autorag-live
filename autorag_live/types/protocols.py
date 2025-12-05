"""Enhanced type hints and protocols for better IDE support and type safety.

This module provides custom types and protocols for:
- Retriever interface definitions
- Document and query types with validation
- Result types with metadata
- Cache key generation

Example:
    >>> def retrieve(docs: DocumentList, query: QueryText) -> RetrievalResult:
    ...     ...
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np

# Base scalar types
QueryText = str
DocumentText = str
DocumentID = Union[str, int]

# Collection types
DocumentList = List[DocumentText]
DocumentIDList = List[DocumentID]
EmbeddingVector = np.ndarray  # Shape (embedding_dim,)
EmbeddingMatrix = np.ndarray  # Shape (n_docs, embedding_dim)
SimilarityScores = np.ndarray  # Shape (n_docs,)

# Result types
RetrievalResult = List[str]
ScoredResult = Tuple[str, float]  # (document, similarity_score)
RetrievalResultWithScores = List[ScoredResult]

# Metadata types
Metadata = dict


@runtime_checkable
class Retriever(Protocol):
    """Protocol for retriever implementations."""

    def retrieve(
        self,
        query: QueryText,
        k: int = 5,
    ) -> RetrievalResult:
        """Retrieve top-k documents."""
        ...

    def add_documents(self, documents: DocumentList) -> None:
        """Add documents to retriever."""
        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = False,
    ) -> Union[EmbeddingVector, EmbeddingMatrix]:
        """Encode texts to embeddings."""
        ...


@runtime_checkable
class SimilarityFunction(Protocol):
    """Protocol for similarity computation functions."""

    def __call__(
        self,
        embeddings1: EmbeddingMatrix,
        embeddings2: EmbeddingMatrix,
    ) -> np.ndarray:
        """Compute similarity matrix."""
        ...


@runtime_checkable
class RankingFunction(Protocol):
    """Protocol for ranking/sorting functions."""

    def __call__(
        self,
        scores: SimilarityScores,
        k: int,
    ) -> Tuple[SimilarityScores, DocumentIDList]:
        """Rank and return top-k."""
        ...


@runtime_checkable
class Cache(Protocol):
    """Protocol for cache implementations with TTL support."""

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        ...

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache with optional TTL."""
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...


@runtime_checkable
class PerformanceMonitor(Protocol):
    """Protocol for performance monitoring implementations."""

    def track(
        self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track operation performance metrics."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated performance statistics."""
        ...

    def reset(self) -> None:
        """Reset all tracked metrics."""
        ...


@runtime_checkable
class ConfigManager(Protocol):
    """Protocol for configuration management."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports dot notation)."""
        ...

    def merge(self, config: Dict[str, Any]) -> None:
        """Merge configuration from dictionary."""
        ...

    def validate(self) -> bool:
        """Validate configuration schema."""
        ...
