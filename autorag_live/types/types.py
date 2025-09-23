"""
Common type definitions and exceptions for AutoRAG-Live.
"""
from typing import TypeVar, List, Dict, Any, Union, Optional, Tuple, Protocol
from typing_extensions import TypeAlias
from dataclasses import dataclass
from datetime import datetime

# Type variables
T = TypeVar('T')
D = TypeVar('D')  # Document type
Q = TypeVar('Q')  # Query type

# Common type aliases
DocumentId: TypeAlias = str
Score: TypeAlias = float
QueryText: TypeAlias = str
DocumentText: TypeAlias = str
RetrievalResult: TypeAlias = List[Tuple[DocumentText, Score]]
Embedding: TypeAlias = List[float]

# Custom exceptions
class AutoRAGError(Exception):
    """Base exception for all AutoRAG-Live errors."""
    pass

class RetrieverError(AutoRAGError):
    """Raised when a retriever operation fails."""
    pass

class ConfigurationError(AutoRAGError):
    """Raised when there's an issue with configuration."""
    pass

class EvaluationError(AutoRAGError):
    """Raised when an evaluation operation fails."""
    pass

# Common interfaces
class Retriever(Protocol):
    """Protocol for retriever implementations."""
    def retrieve(self, query: QueryText, k: int = 5) -> RetrievalResult:
        """Retrieve documents for a query."""
        ...

    def add_documents(self, documents: List[DocumentText]) -> None:
        """Add documents to the retriever's index."""
        ...

@dataclass
class EvaluationResult:
    """Structured container for evaluation metrics."""
    metric_name: str
    score: float
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()

@dataclass
class BenchmarkResult:
    """Container for benchmark measurements."""
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