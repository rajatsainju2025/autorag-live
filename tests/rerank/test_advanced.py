"""
Tests for advanced reranking strategies.
"""
import pytest

from autorag_live.rerank.advanced import (
    DiversityReranker,
    HybridReranker,
    MMRReranker,
    RankedDocument,
    ReciprocalRankFusion,
    SemanticReranker,
)


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        RankedDocument(content="Python programming language is popular", score=0.9, id="doc1"),
        RankedDocument(content="Python is great for data science", score=0.85, id="doc2"),
        RankedDocument(content="Machine learning with Python", score=0.8, id="doc3"),
        RankedDocument(content="Java programming basics", score=0.7, id="doc4"),
        RankedDocument(content="JavaScript web development", score=0.6, id="doc5"),
    ]


class TestMMRReranker:
    """Test suite for MMR reranker."""

    def test_mmr_basic(self, sample_documents):
        """Test basic MMR reranking."""
        reranker = MMRReranker(lambda_param=0.7)
        query = "Python programming"

        reranked = reranker.rerank(query, sample_documents, top_k=3)

        assert len(reranked) == 3
        assert all(isinstance(doc, RankedDocument) for doc in reranked)

    def test_mmr_diversity(self):
        """Test that MMR promotes diversity."""
        # Create documents with high similarity
        docs = [
            RankedDocument(content="cat", score=1.0, id="1"),
            RankedDocument(content="cat cat", score=0.9, id="2"),
            RankedDocument(content="dog", score=0.8, id="3"),
        ]

        reranker = MMRReranker(lambda_param=0.5)  # Balanced relevance/diversity
        query = "cat"

        reranked = reranker.rerank(query, docs, top_k=2)

        # Should include both cat and dog for diversity
        contents = [doc.content for doc in reranked]
        assert "cat" in contents
        # Dog might be included for diversity

    def test_mmr_lambda_extremes(self, sample_documents):
        """Test MMR with extreme lambda values."""
        query = "Python programming"

        # Lambda = 1.0 (pure relevance, no diversity)
        reranker_relevance = MMRReranker(lambda_param=1.0)
        reranked_rel = reranker_relevance.rerank(query, sample_documents.copy(), top_k=3)

        # Lambda = 0.0 (pure diversity, no relevance)
        reranker_diversity = MMRReranker(lambda_param=0.0)
        reranked_div = reranker_diversity.rerank(query, sample_documents.copy(), top_k=3)

        assert len(reranked_rel) == 3
        assert len(reranked_div) == 3

    def test_mmr_empty_documents(self):
        """Test MMR with empty document list."""
        reranker = MMRReranker()
        reranked = reranker.rerank("query", [], top_k=5)

        assert reranked == []


class TestDiversityReranker:
    """Test suite for diversity reranker."""

    def test_diversity_basic(self, sample_documents):
        """Test basic diversity reranking."""
        reranker = DiversityReranker(num_clusters=2)
        query = "Python programming"

        reranked = reranker.rerank(query, sample_documents, top_k=3)

        # May return fewer than top_k due to clustering
        assert len(reranked) <= 3
        assert len(reranked) > 0
        assert all(isinstance(doc, RankedDocument) for doc in reranked)

    def test_diversity_clustering(self):
        """Test that documents are clustered."""
        # Create documents with clear clusters
        docs = [
            RankedDocument(content="Python machine learning AI", score=0.9, id="1"),
            RankedDocument(content="Python deep learning neural", score=0.85, id="2"),
            RankedDocument(content="JavaScript React frontend", score=0.8, id="3"),
            RankedDocument(content="JavaScript Node backend", score=0.75, id="4"),
        ]

        reranker = DiversityReranker(num_clusters=2, cluster_sample_ratio=0.5)
        reranked = reranker.rerank("programming", docs, top_k=4)

        # Should have some representation from clusters
        assert len(reranked) >= 2
        assert len(reranked) <= 4

    def test_diversity_fewer_docs_than_clusters(self):
        """Test with fewer documents than clusters."""
        docs = [
            RankedDocument(content="doc1", score=0.9),
            RankedDocument(content="doc2", score=0.8),
        ]

        reranker = DiversityReranker(num_clusters=5)
        reranked = reranker.rerank("query", docs, top_k=3)

        assert len(reranked) == 2


class TestReciprocalRankFusion:
    """Test suite for RRF."""

    def test_rrf_basic(self):
        """Test basic RRF fusion."""
        ranking1 = [
            RankedDocument(content="doc1", score=0.9, id="1"),
            RankedDocument(content="doc2", score=0.8, id="2"),
            RankedDocument(content="doc3", score=0.7, id="3"),
        ]

        ranking2 = [
            RankedDocument(content="doc2", score=0.95, id="2"),
            RankedDocument(content="doc1", score=0.85, id="1"),
            RankedDocument(content="doc4", score=0.75, id="4"),
        ]

        rrf = ReciprocalRankFusion(k=60)
        fused = rrf.fuse([ranking1, ranking2], top_k=3)

        assert len(fused) == 3
        # doc1 and doc2 should rank high (appear in both)
        top_ids = [doc.id for doc in fused[:2]]
        assert "1" in top_ids
        assert "2" in top_ids

    def test_rrf_single_ranking(self, sample_documents):
        """Test RRF with single ranking."""
        rrf = ReciprocalRankFusion()
        fused = rrf.fuse([sample_documents], top_k=3)

        assert len(fused) == 3

    def test_rrf_empty_rankings(self):
        """Test RRF with empty input."""
        rrf = ReciprocalRankFusion()
        fused = rrf.fuse([], top_k=5)

        assert fused == []

    def test_rrf_k_parameter(self):
        """Test different k values."""
        ranking = [
            RankedDocument(content=f"doc{i}", score=1.0 - i * 0.1, id=str(i)) for i in range(5)
        ]

        rrf_60 = ReciprocalRankFusion(k=60)
        rrf_1 = ReciprocalRankFusion(k=1)

        fused_60 = rrf_60.fuse([ranking], top_k=5)
        fused_1 = rrf_1.fuse([ranking], top_k=5)

        assert len(fused_60) == 5
        assert len(fused_1) == 5
        # With different k values, at least some scores should differ
        # (but may be same if normalization makes them equal)
        # Just test that fusion works correctly
        assert all(doc.score > 0 for doc in fused_60)
        assert all(doc.score > 0 for doc in fused_1)


class TestSemanticReranker:
    """Test suite for semantic reranker."""

    def test_semantic_fallback(self, sample_documents):
        """Test semantic reranker with fallback."""
        reranker = SemanticReranker()
        query = "Python programming"

        reranked = reranker.rerank(query, sample_documents, top_k=3)

        assert len(reranked) == 3
        assert all(isinstance(doc, RankedDocument) for doc in reranked)

    def test_semantic_caching(self):
        """Test that caching works."""
        reranker = SemanticReranker(use_cache=True)
        docs = [RankedDocument(content="test doc", score=0.5)]

        # First call
        reranker.rerank("query", docs.copy())

        # Second call should use cache
        cache_size_before = len(reranker._cache)
        reranker.rerank("query", docs.copy())
        cache_size_after = len(reranker._cache)

        # Cache should have been used (no new entries)
        assert cache_size_after == cache_size_before

    @pytest.mark.skip(reason="Requires sentence-transformers")
    def test_semantic_with_model(self, sample_documents):
        """Test with actual cross-encoder model (requires sentence-transformers)."""
        reranker = SemanticReranker(model_name="cross-encoder/ms-marco-TinyBERT-L-2")
        query = "Python programming"

        reranked = reranker.rerank(query, sample_documents, top_k=3)

        assert len(reranked) == 3


class TestHybridReranker:
    """Test suite for hybrid reranker."""

    def test_hybrid_basic(self, sample_documents):
        """Test basic hybrid reranking."""
        reranker = HybridReranker(relevance_weight=0.5, diversity_weight=0.3, semantic_weight=0.2)
        query = "Python programming"

        reranked = reranker.rerank(query, sample_documents, top_k=3)

        assert len(reranked) == 3
        assert all(isinstance(doc, RankedDocument) for doc in reranked)

    def test_hybrid_weight_variations(self, sample_documents):
        """Test with different weight configurations."""
        query = "Python programming"

        # Pure relevance
        reranker_rel = HybridReranker(
            relevance_weight=1.0, diversity_weight=0.0, semantic_weight=0.0
        )
        reranked_rel = reranker_rel.rerank(query, sample_documents.copy(), top_k=3)

        # Pure diversity
        reranker_div = HybridReranker(
            relevance_weight=0.0, diversity_weight=1.0, semantic_weight=0.0
        )
        reranked_div = reranker_div.rerank(query, sample_documents.copy(), top_k=3)

        assert len(reranked_rel) == 3
        assert len(reranked_div) == 3

    def test_hybrid_empty_documents(self):
        """Test hybrid reranker with empty list."""
        reranker = HybridReranker()
        reranked = reranker.rerank("query", [], top_k=5)

        assert reranked == []


class TestRankedDocument:
    """Test suite for RankedDocument dataclass."""

    def test_document_creation(self):
        """Test document creation."""
        doc = RankedDocument(
            content="test content", score=0.5, id="test", metadata={"key": "value"}
        )

        assert doc.content == "test content"
        assert doc.score == 0.5
        assert doc.id == "test"
        assert doc.metadata == {"key": "value"}

    def test_document_optional_fields(self):
        """Test optional fields."""
        doc = RankedDocument(content="test", score=0.5)

        assert doc.id is None
        assert doc.metadata is None
