"""Tests for AsyncMMRReranker."""

import pytest

from autorag_live.rerank.async_mmr import AsyncMMRReranker


@pytest.mark.asyncio
async def test_async_mmr_rerank_empty():
    """Test rerank with empty candidates."""
    reranker = AsyncMMRReranker()
    results = await reranker.rerank("query", [])
    assert results == []


@pytest.mark.asyncio
async def test_async_mmr_rerank_basic():
    """Test basic rerank."""
    reranker = AsyncMMRReranker()
    candidates = [
        {"id": "1", "text": "doc1"},
        {"id": "2", "text": "doc2"},
        {"id": "3", "text": "doc3"},
    ]
    results = await reranker.rerank("query", candidates, k=2)
    assert len(results) == 2
    assert results[0]["id"] == "1"


@pytest.mark.asyncio
async def test_async_mmr_compute_similarity():
    """Test similarity computation."""
    reranker = AsyncMMRReranker()
    emb1 = [1.0, 0.0, 0.0]
    emb2 = [1.0, 0.0, 0.0]
    sim = await reranker.compute_similarity(emb1, emb2)
    assert sim > 0.3  # Dot product of identical unit vectors


@pytest.mark.asyncio
async def test_async_mmr_compute_diversity():
    """Test diversity computation."""
    reranker = AsyncMMRReranker()
    emb1 = [1.0, 0.0, 0.0]
    emb2 = [0.0, 1.0, 0.0]
    div = await reranker.compute_diversity(emb1, emb2)
    assert div > 0.9  # Should be close to 1 (orthogonal)


@pytest.mark.asyncio
async def test_async_mmr_lambda_multiplier():
    """Test lambda multiplier parameter."""
    reranker1 = AsyncMMRReranker(lambda_mult=0.3)
    reranker2 = AsyncMMRReranker(lambda_mult=0.7)
    assert reranker1._lambda_mult == 0.3
    assert reranker2._lambda_mult == 0.7
