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

from typing import List, Protocol, Tuple, Union

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


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = False,
    ) -> Union[EmbeddingVector, EmbeddingMatrix]:
        """Encode texts to embeddings."""
        ...


class SimilarityFunction(Protocol):
    """Protocol for similarity computation functions."""

    def __call__(
        self,
        embeddings1: EmbeddingMatrix,
        embeddings2: EmbeddingMatrix,
    ) -> np.ndarray:
        """Compute similarity matrix."""
        ...


class RankingFunction(Protocol):
    """Protocol for ranking/sorting functions."""

    def __call__(
        self,
        scores: SimilarityScores,
        k: int,
    ) -> Tuple[SimilarityScores, DocumentIDList]:
        """Rank and return top-k."""
        ...


class Cache(Protocol):
    """Protocol for cache implementations."""

    def get(self, key: str):
        """Get cached value."""
        ...

    def put(self, key: str, value):
        """Store value in cache."""
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...
