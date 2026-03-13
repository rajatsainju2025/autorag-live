"""Tests for AsyncColBERTRetriever."""

import pytest

from autorag_live.retrievers.async_colbert import AsyncColBERTRetriever


class DummySyncColBERTRetriever:
    """Dummy sync retriever for testing."""

    def retrieve(self, query: str, k: int = 10):
        """Return dummy results."""
        return [
            {"id": f"colbert-{i}", "maxsim": 0.95 - i * 0.1}
            for i in range(min(k, 3))
        ]

    def add_documents(self, docs):
        """Dummy add."""
        self.docs = docs

    def encode_queries(self, queries):
        """Return dummy encodings."""
        return [[0.1] * 128 for _ in queries]


@pytest.mark.asyncio
async def test_async_colbert_retrieve():
    """Test retrieve method."""
    sync = DummySyncColBERTRetriever()
    retriever = AsyncColBERTRetriever(sync)
    results = await retriever.retrieve("colbert query", k=2)
    assert isinstance(results, list)
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_async_colbert_add_documents():
    """Test add_documents method."""
    sync = DummySyncColBERTRetriever()
    retriever = AsyncColBERTRetriever(sync)
    docs = [{"id": "1", "text": "hello"}]
    await retriever.add_documents(docs)
    assert sync.docs == docs


@pytest.mark.asyncio
async def test_async_colbert_encode_queries():
    """Test encode_queries method."""
    sync = DummySyncColBERTRetriever()
    retriever = AsyncColBERTRetriever(sync)
    encodings = await retriever.encode_queries(["q1", "q2"])
    assert len(encodings) == 2
    assert len(encodings[0]) == 128
