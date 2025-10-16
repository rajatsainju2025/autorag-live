"""
Type system for AutoRAG-Live.

This module provides comprehensive type definitions, protocols, and exceptions for
the entire AutoRAG-Live system. It ensures type safety and proper interface
implementations across the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal, TypeAlias

# Type Variables for Generics
T = TypeVar("T")  # Generic type
D = TypeVar("D", covariant=True)  # Document type (covariant)
Q = TypeVar("Q", contravariant=True)  # Query type (contravariant)
S = TypeVar("S", bound=float)  # Score type (bounded to float)

# Function Types
Scorer = Callable[[str, str], float]  # (text1, text2) -> score
QueryProcessor = Callable[[str], str]  # query -> processed_query

# Type Aliases and Custom Types
DocumentId = str  # Type alias for document IDs
QueryText = str  # Type alias for query text
DocumentText = str  # Type alias for document text
Score = float  # Type alias for scores
MetricsDict = Dict[str, float]  # Type alias for metrics
ConfigDict = Dict[str, Any]  # Type alias for configs

# Vector Types
VectorArray: TypeAlias = npt.NDArray[np.float32]
Embedding: TypeAlias = Union[List[float], VectorArray]
RetrievalResult: TypeAlias = List[Tuple[str, float]]

# Constants and Literals
DistanceMetric = Literal["cosine", "euclidean", "dot", "manhattan"]
JudgeType = Literal["deterministic", "llm"]
ModelDevice = Literal["cpu", "cuda", "mps"]
OptimizerType = Literal["bandit", "grid", "bayesian"]

# Constants and Literals
DistanceMetric = Literal["cosine", "euclidean", "dot", "manhattan"]
JudgeType = Literal["deterministic", "llm"]
ModelDevice = Literal["cpu", "cuda", "mps"]

# Remove duplicate definitions - already defined above
MetricsDict = Dict[str, float]  # Type alias for metric dictionaries
ConfigDict = Dict[str, Any]  # Type alias for configuration dictionaries


@dataclass(frozen=True)
class Document:
    """
    Immutable document representation.

    Attributes:
        id: Unique document identifier
        text: Document content
        metadata: Optional metadata dictionary
        embedding: Optional pre-computed embedding
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Embedding] = None


@dataclass(frozen=True)
class Query:
    """
    Immutable query representation.

    Attributes:
        text: Query text
        metadata: Optional metadata dictionary
        embedding: Optional pre-computed embedding
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Embedding] = None


@dataclass(frozen=True)
class ScoredDocument(Generic[D, S]):
    """
    Generic scored document container.

    Attributes:
        document: The document that was scored
        score: The relevance/similarity score
    """

    document: D
    score: S

    def __lt__(self, other: ScoredDocument[D, S]) -> bool:
        return self.score < other.score


# Protocol Definitions
@runtime_checkable
class Retriever(Protocol[D]):
    """Protocol defining the retriever interface."""

    def retrieve(self, query: str, k: int = 5, **kwargs: Any) -> List[Document]:
        """
        Retrieve the k most relevant documents for a query.

        Args:
            query: Query to find relevant documents for
            k: Number of documents to retrieve (default: 5)
            **kwargs: Additional retriever-specific parameters

        Returns:
            List of documents, sorted by relevance (highest first)

        Raises:
            RetrieverError: If retrieval fails
            ValueError: If k < 1
        """
        ...

    def add_documents(self, documents: Sequence[Document]) -> None:
        """
        Add documents to the retriever's index.

        Args:
            documents: Documents to add to the index

        Raises:
            RetrieverError: If document addition fails
            ValueError: If documents is empty
        """
        ...

    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a specific document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            The document if found, None otherwise
        """
        ...

    @property
    def document_count(self) -> int:
        """Get the number of documents in the index."""
        ...


@runtime_checkable
class DocumentStore(Protocol):
    """Protocol for document storage and retrieval."""

    def add(self, doc: Document) -> str:
        """
        Add a document and return its ID.

        Args:
            doc: Document to store

        Returns:
            Document ID
        """
        ...

    def get(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        ...

    def update(self, doc_id: str, doc: Document) -> bool:
        """Update existing document."""
        ...

    def delete(self, doc_id: str) -> bool:
        """Delete document by ID."""
        ...

    def search(self, query: str) -> List[Document]:
        """Search for documents."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Protocol defining the evaluator interface."""

    def evaluate(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """
        Evaluate predictions against references.

        Args:
            predictions: Predicted text/answers to evaluate
            references: Ground truth references
            weights: Optional metric weights
            **kwargs: Additional metric-specific parameters

        Returns:
            Evaluation metrics and details

        Raises:
            EvaluationError: If evaluation fails
            ValueError: If predictions and references have different lengths
        """
        ...


# Results and Metrics
@dataclass
class EvaluationResult:
    """
    Structured container for evaluation metrics.

    Attributes:
        metric_name: Name of the metric
        score: Primary metric score
        details: Optional detailed metrics
        timestamp: When evaluation was performed
    """

    metric_name: str
    score: float
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkResult:
    """
    Container for benchmark measurements.

    Attributes:
        operation: Name of operation being benchmarked
        iterations: Number of benchmark iterations
        total_time: Total execution time in seconds
        avg_time: Average time per iteration
        min_time: Minimum iteration time
        max_time: Maximum iteration time
        std_time: Standard deviation of times
        memory_usage_mb: Peak memory usage in MB
        throughput: Operations per second
        metadata: Additional benchmark info
    """

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


# Exception Hierarchy
class AutoRAGError(Exception):
    """Base exception class for AutoRAG-Live."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
        }


class RetrieverError(AutoRAGError):
    """
    Raised when a retriever operation fails.

    Examples:
        - Index creation failure
        - Document addition error
        - Query processing error
    """


class ConfigurationError(AutoRAGError):
    """
    Raised when there's an issue with configuration.

    Examples:
        - Missing required config
        - Invalid config values
        - Config file not found
    """


class EvaluationError(AutoRAGError):
    """
    Raised when an evaluation operation fails.

    Examples:
        - Metric computation error
        - Invalid predictions/references
        - Judge failure
    """


class OptimizerError(AutoRAGError):
    """
    Raised when an optimization operation fails.

    Examples:
        - Invalid parameter bounds
        - Optimization divergence
        - Search space error
    """


class PipelineError(AutoRAGError):
    """Errors related to pipeline execution."""


class ModelError(AutoRAGError):
    """Errors related to model operations."""


class DataError(AutoRAGError):
    """Errors related to data processing."""


class ValidationError(AutoRAGError):
    """Errors related to input/output validation."""
