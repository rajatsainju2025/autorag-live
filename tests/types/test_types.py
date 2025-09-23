"""
Tests for custom types and exceptions.
"""
import pytest
from typing import List, Tuple

from autorag_live.types.types import (
    DocumentText,
    QueryText,
    RetrievalResult,
    Score,
    AutoRAGError,
    RetrieverError,
    ConfigurationError,
    EvaluationError,
    Retriever
)


def test_type_aliases():
    """Test type aliases work as expected."""
    # Document text
    doc: DocumentText = "This is a test document"
    assert isinstance(doc, str)
    
    # Query text
    query: QueryText = "test query"
    assert isinstance(query, str)
    
    # Score
    score: Score = 0.95
    assert isinstance(score, float)
    
    # Retrieval result
    result: RetrievalResult = [
        ("doc1", 0.9),
        ("doc2", 0.8)
    ]
    assert isinstance(result, list)
    assert all(isinstance(r, tuple) and len(r) == 2 for r in result)


def test_custom_exceptions():
    """Test custom exception hierarchy."""
    # Base exception
    with pytest.raises(AutoRAGError):
        raise AutoRAGError("Base error")
    
    # Retriever error
    with pytest.raises(RetrieverError):
        raise RetrieverError("Retriever failed")
    
    # Configuration error    
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("Invalid config")
    
    # Evaluation error
    with pytest.raises(EvaluationError):
        raise EvaluationError("Evaluation failed")
    
    # Check exception hierarchy
    try:
        raise RetrieverError("Test error")
    except AutoRAGError as e:
        assert isinstance(e, RetrieverError)


class MockRetriever:
    """Mock retriever for testing Protocol."""
    
    def retrieve(self, query: QueryText, k: int = 5) -> RetrievalResult:
        return [("mock doc", 1.0)]
    
    def add_documents(self, documents: List[DocumentText]) -> None:
        pass


def test_retriever_protocol():
    """Test Retriever protocol implementation."""
    # Valid implementation
    retriever = MockRetriever()
    assert isinstance(retriever.retrieve("query"), list)
    
    # Type checking - these would fail mypy but pass runtime
    result = retriever.retrieve("test")
    assert isinstance(result[0], tuple)
    assert isinstance(result[0][0], str)
    assert isinstance(result[0][1], float)