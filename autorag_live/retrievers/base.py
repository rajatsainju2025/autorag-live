"""
Base retriever interface and abstract classes.
"""
from abc import ABC, abstractmethod
from typing import List

from ..types.types import DocumentText, QueryText, RetrievalResult
from ..utils import get_logger

logger = get_logger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    def __init__(self) -> None:
        """Initialize the retriever."""
        self._is_initialized = False

    @abstractmethod
    def retrieve(self, query: QueryText, k: int = 5) -> RetrievalResult:
        """
        Retrieve documents for a query.

        Args:
            query: The query text to search for
            k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples

        Raises:
            RetrieverError: If retrieval fails
        """
        raise NotImplementedError

    @abstractmethod
    def add_documents(self, documents: List[DocumentText]) -> None:
        """
        Add documents to the retriever's index.

        Args:
            documents: List of document texts to index

        Raises:
            RetrieverError: If indexing fails
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load retriever state from disk.

        Args:
            path: Path to load state from

        Raises:
            RetrieverError: If loading fails
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save retriever state to disk.

        Args:
            path: Path to save state to

        Raises:
            RetrieverError: If saving fails
        """
        raise NotImplementedError

    @property
    def is_initialized(self) -> bool:
        """Check if the retriever is initialized."""
        return self._is_initialized
