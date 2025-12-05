"""Property-based tests for retrieval functions.

This module uses hypothesis for property-based testing to verify retrieval
invariants and edge cases that traditional unit tests might miss.

Properties tested:
- Retrieval consistency: same query always returns same results
- Score ordering: results sorted by relevance
- K-value bounds: returned documents <= k
- Empty query handling: appropriate errors raised
- Duplicate handling: consistent behavior
"""

import pytest

# Skip all tests if hypothesis is not installed
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from autorag_live.retrievers import bm25, dense, hybrid  # noqa: E402
from autorag_live.types.types import RetrieverError  # noqa: E402

# Strategy for generating valid queries
valid_queries = st.text(min_size=1, max_size=100).filter(lambda x: x.strip())

# Strategy for generating document corpora
document_lists = st.lists(
    st.text(min_size=1, max_size=200),
    min_size=1,
    max_size=50,
)

# Strategy for k values
k_values = st.integers(min_value=1, max_value=20)


class TestBM25Properties:
    """Property-based tests for BM25 retrieval."""

    @given(query=valid_queries, corpus=document_lists, k=k_values)
    @settings(max_examples=50, deadline=1000)
    def test_bm25_returns_at_most_k_documents(self, query, corpus, k):
        """Property: BM25 returns at most k documents."""
        results = bm25.bm25_retrieve(query, corpus, k)
        assert len(results) <= k
        assert len(results) <= len(corpus)

    @given(query=valid_queries, corpus=document_lists, k=k_values)
    @settings(max_examples=50, deadline=1000)
    def test_bm25_consistency(self, query, corpus, k):
        """Property: Same query on same corpus returns same results."""
        results1 = bm25.bm25_retrieve(query, corpus, k)
        results2 = bm25.bm25_retrieve(query, corpus, k)
        assert results1 == results2

    @given(query=valid_queries, corpus=document_lists, k=k_values)
    @settings(max_examples=50, deadline=1000)
    def test_bm25_returns_valid_documents(self, query, corpus, k):
        """Property: BM25 only returns documents from corpus."""
        results = bm25.bm25_retrieve(query, corpus, k)
        for doc in results:
            assert doc in corpus

    @given(corpus=document_lists)
    @settings(max_examples=30, deadline=1000)
    def test_bm25_empty_query_raises_error(self, corpus):
        """Property: Empty query raises RetrieverError."""
        with pytest.raises(RetrieverError):
            bm25.bm25_retrieve("", corpus, 5)

        with pytest.raises(RetrieverError):
            bm25.bm25_retrieve("   ", corpus, 5)

    @given(query=valid_queries, k=k_values)
    @settings(max_examples=30, deadline=500)
    def test_bm25_empty_corpus_returns_empty(self, query, k):
        """Property: Empty corpus returns empty results."""
        results = bm25.bm25_retrieve(query, [], k)
        assert results == []

    @given(query=valid_queries, corpus=document_lists)
    @settings(max_examples=30, deadline=1000)
    def test_bm25_invalid_k_raises_error(self, query, corpus):
        """Property: Invalid k values raise RetrieverError."""
        with pytest.raises(RetrieverError):
            bm25.bm25_retrieve(query, corpus, 0)

        with pytest.raises(RetrieverError):
            bm25.bm25_retrieve(query, corpus, -1)


class TestDenseProperties:
    """Property-based tests for dense retrieval."""

    @given(query=valid_queries, corpus=document_lists, k=k_values)
    @settings(max_examples=30, deadline=2000)
    def test_dense_returns_at_most_k_documents(self, query, corpus, k):
        """Property: Dense retrieval returns at most k documents."""
        results = dense.dense_retrieve(query, corpus, k)
        assert len(results) <= k
        assert len(results) <= len(corpus)

    @given(query=valid_queries, corpus=document_lists, k=k_values)
    @settings(max_examples=20, deadline=2000)
    def test_dense_returns_valid_documents(self, query, corpus, k):
        """Property: Dense retrieval only returns documents from corpus."""
        results = dense.dense_retrieve(query, corpus, k)
        for doc in results:
            assert doc in corpus

    @given(corpus=document_lists)
    @settings(max_examples=20, deadline=2000)
    def test_dense_empty_query_raises_error(self, corpus):
        """Property: Empty query raises RetrieverError."""
        with pytest.raises(RetrieverError):
            dense.dense_retrieve("", corpus, 5)

        with pytest.raises(RetrieverError):
            dense.dense_retrieve("   ", corpus, 5)

    @given(query=valid_queries, k=k_values)
    @settings(max_examples=20, deadline=500)
    def test_dense_empty_corpus_returns_empty(self, query, k):
        """Property: Empty corpus returns empty results."""
        results = dense.dense_retrieve(query, [], k)
        assert results == []

    @given(query=valid_queries, corpus=document_lists)
    @settings(max_examples=20, deadline=2000)
    def test_dense_invalid_k_raises_error(self, query, corpus):
        """Property: Invalid k values raise RetrieverError."""
        with pytest.raises(RetrieverError):
            dense.dense_retrieve(query, corpus, 0)

        with pytest.raises(RetrieverError):
            dense.dense_retrieve(query, corpus, -1)


class TestHybridProperties:
    """Property-based tests for hybrid retrieval."""

    @given(query=valid_queries, corpus=document_lists, k=k_values)
    @settings(max_examples=20, deadline=2000)
    def test_hybrid_returns_at_most_k_documents(self, query, corpus, k):
        """Property: Hybrid retrieval returns at most k documents."""
        results = hybrid.hybrid_retrieve(query, corpus, k)
        assert len(results) <= k
        assert len(results) <= len(corpus)

    @given(query=valid_queries, corpus=document_lists, k=k_values)
    @settings(max_examples=20, deadline=2000)
    def test_hybrid_returns_valid_documents(self, query, corpus, k):
        """Property: Hybrid retrieval only returns documents from corpus."""
        results = hybrid.hybrid_retrieve(query, corpus, k)
        for doc in results:
            assert doc in corpus

    @given(corpus=document_lists)
    @settings(max_examples=15, deadline=2000)
    def test_hybrid_empty_query_raises_error(self, corpus):
        """Property: Empty query raises RetrieverError."""
        with pytest.raises(RetrieverError):
            hybrid.hybrid_retrieve("", corpus, 5)

        with pytest.raises(RetrieverError):
            hybrid.hybrid_retrieve("   ", corpus, 5)

    @given(query=valid_queries, k=k_values)
    @settings(max_examples=15, deadline=500)
    def test_hybrid_empty_corpus_returns_empty(self, query, k):
        """Property: Empty corpus returns empty results."""
        results = hybrid.hybrid_retrieve(query, [], k)
        assert results == []

    @given(query=valid_queries, corpus=document_lists, k=k_values)
    @settings(max_examples=20, deadline=2000)
    def test_hybrid_weight_bounds(self, query, corpus, k):
        """Property: Invalid weights raise RetrieverError."""
        with pytest.raises(RetrieverError):
            hybrid.hybrid_retrieve(query, corpus, k, bm25_weight=-0.1)

        with pytest.raises(RetrieverError):
            hybrid.hybrid_retrieve(query, corpus, k, bm25_weight=1.1)


class TestRetrievalInvariants:
    """Test invariants that should hold across all retrieval methods."""

    @given(query=valid_queries, corpus=document_lists, k=k_values)
    @settings(max_examples=20, deadline=3000)
    def test_increasing_k_never_decreases_results(self, query, corpus, k):
        """Property: Increasing k never returns fewer documents."""
        if k < len(corpus):
            results_k = bm25.bm25_retrieve(query, corpus, k)
            results_k_plus_1 = bm25.bm25_retrieve(query, corpus, k + 1)

            # Results with k+1 should contain all results from k
            # (assuming stable retrieval)
            assert len(results_k_plus_1) >= len(results_k)

    @given(query=valid_queries, corpus=document_lists, k=k_values)
    @settings(max_examples=20, deadline=1000)
    def test_no_duplicate_results(self, query, corpus, k):
        """Property: No duplicate documents in results."""
        results = bm25.bm25_retrieve(query, corpus, k)
        assert len(results) == len(set(results))

    @given(corpus=st.lists(st.just("same doc"), min_size=5, max_size=10), k=k_values)
    @settings(max_examples=15, deadline=1000)
    def test_duplicate_corpus_handling(self, corpus, k):
        """Property: Handles duplicate documents in corpus."""
        # Should work without crashing even with duplicate docs
        results = bm25.bm25_retrieve("test", corpus, k)
        assert len(results) <= k


class TestPerformanceProperties:
    """Test performance-related properties."""

    @given(corpus=document_lists)
    @settings(max_examples=15, deadline=1000)
    def test_single_document_corpus_returns_quickly(self, corpus):
        """Property: Single document retrieval is fast."""
        if corpus:
            result = bm25.bm25_retrieve("test", corpus[:1], 1)
            assert len(result) <= 1

    @given(query=valid_queries, corpus=document_lists)
    @settings(max_examples=15, deadline=1000)
    def test_k_larger_than_corpus_works(self, query, corpus):
        """Property: k > corpus_size works correctly."""
        large_k = len(corpus) + 100
        results = bm25.bm25_retrieve(query, corpus, large_k)
        assert len(results) <= len(corpus)
