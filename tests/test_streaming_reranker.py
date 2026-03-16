"""Tests for streaming reranker."""

import pytest

from autorag_live.rerank.streaming_reranker import StreamingReranker


@pytest.mark.asyncio
async def test_streaming_reranker_basic():
    """Test basic streaming rerank."""
    reranker = StreamingReranker()
    candidates = [
        {"id": "1", "text": "doc1"},
        {"id": "2", "text": "doc2"},
        {"id": "3", "text": "doc3"},
    ]

    results = []
    async for result in reranker.rerank_stream("query", candidates, k=2):
        results.append(result)

    assert len(results) == 2
    assert "rerank_score" in results[0]


@pytest.mark.asyncio
async def test_streaming_reranker_empty():
    """Test with empty candidates."""
    reranker = StreamingReranker()
    results = []
    async for result in reranker.rerank_stream("query", []):
        results.append(result)
    assert results == []


@pytest.mark.asyncio
async def test_streaming_reranker_callback():
    """Test streaming with callback."""
    reranker = StreamingReranker()
    candidates = [
        {"id": "1", "text": "doc1"},
        {"id": "2", "text": "doc2"},
    ]
    received = []

    async def on_candidate(candidate):
        received.append(candidate)

    results = await reranker.stream_with_callback("query", candidates, on_candidate)

    assert len(results) == 2
    assert len(received) == 2


@pytest.mark.asyncio
async def test_streaming_reranker_scores():
    """Test score tracking."""
    reranker = StreamingReranker()
    candidates = [
        {"id": "a", "text": "doc1"},
        {"id": "b", "text": "doc2"},
    ]

    async for _ in reranker.rerank_stream("query", candidates):
        pass

    assert reranker.get_score("a") > 0
    assert reranker.get_score("b") > 0
    assert reranker.get_score("nonexistent") == 0.0
