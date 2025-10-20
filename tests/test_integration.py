"""Integration tests for autorag-live system components."""

import os
import tempfile
from datetime import datetime, timedelta

import pytest

from autorag_live.augment.synonym_miner import mine_synonyms_from_disagreements
from autorag_live.data.time_series import FFTEmbedder, TimeSeriesNote, TimeSeriesRetriever
from autorag_live.disagreement import metrics
from autorag_live.evals.advanced_metrics import comprehensive_evaluation
from autorag_live.evals.small import run_small_suite
from autorag_live.pipeline.acceptance_policy import AcceptancePolicy, safe_config_update
from autorag_live.pipeline.hybrid_optimizer import grid_search_hybrid_weights
from autorag_live.rerank.simple import SimpleReranker
from autorag_live.retrievers import bm25, dense, hybrid
from autorag_live.types.types import RetrieverError


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing."""
    return [
        "The sky is blue and beautiful during the day.",
        "The sun rises in the east and sets in the west.",
        "The sun is bright and provides light to Earth.",
        "The sun in the sky is very bright during daytime.",
        "We can see the shining sun, the bright sun in the sky.",
        "The quick brown fox jumps over the lazy dog.",
        "A lazy fox is usually sleeping in its den.",
        "The fox is a mammal that belongs to the canine family.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret visual information.",
        "Data science combines statistics, programming, and domain expertise.",
        "Python is a popular programming language for data science.",
        "Jupyter notebooks provide an interactive environment for coding.",
    ]


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        "bright sun in the sky",
        "fox jumping over dog",
        "machine learning and AI",
        "programming with Python",
        "data science techniques",
    ]


class TestRetrieverIntegration:
    """Test integration between different retrievers."""

    def test_bm25_dense_hybrid_consistency(self, sample_corpus, sample_queries):
        """Test that all retrievers work together and produce consistent results."""
        query = sample_queries[0]

        # Get results from all retrievers
        bm25_results = bm25.bm25_retrieve(query, sample_corpus, 5)
        dense_results = dense.dense_retrieve(query, sample_corpus, 5)
        hybrid_results = hybrid.hybrid_retrieve(query, sample_corpus, 5)

        # All should return lists of strings
        assert isinstance(bm25_results, list)
        assert isinstance(dense_results, list)
        assert isinstance(hybrid_results, list)

        assert len(bm25_results) <= 5
        assert len(dense_results) <= 5
        assert len(hybrid_results) <= 5

        # All results should be from the corpus
        for result in bm25_results + dense_results + hybrid_results:
            assert result in sample_corpus

    def test_retriever_diversity_calculation(self, sample_corpus, sample_queries):
        """Test disagreement metrics calculation between retrievers."""
        query = sample_queries[0]

        bm25_results = bm25.bm25_retrieve(query, sample_corpus, 5)
        dense_results = dense.dense_retrieve(query, sample_corpus, 5)
        hybrid_results = hybrid.hybrid_retrieve(query, sample_corpus, 5)

        # Calculate disagreement metrics
        jaccard_bd = metrics.jaccard_at_k(bm25_results, dense_results)
        jaccard_bh = metrics.jaccard_at_k(bm25_results, hybrid_results)
        jaccard_dh = metrics.jaccard_at_k(dense_results, hybrid_results)

        # All metrics should be between 0 and 1
        assert 0.0 <= jaccard_bd <= 1.0
        assert 0.0 <= jaccard_bh <= 1.0
        assert 0.0 <= jaccard_dh <= 1.0

        # Test Kendall tau (may be None if lists are too short)
        kendall_bd = metrics.kendall_tau_at_k(bm25_results, dense_results)
        if kendall_bd is not None:
            assert -1.0 <= kendall_bd <= 1.0


class TestEvaluationIntegration:
    """Test integration of evaluation components."""

    def test_small_suite_with_retrievers(self, sample_corpus, sample_queries, tmp_path):
        """Test small evaluation suite with retriever results."""
        # This would normally run the full evaluation suite
        # For integration testing, we'll mock the expensive parts
        # Note: run_small_suite doesn't actually use retrievers, it uses simple_qa_answer

        # This should not raise an exception
        summary = run_small_suite(runs_dir=str(tmp_path / "runs"), judge_type="deterministic")

        assert "metrics" in summary
        assert "run_id" in summary
        assert "em" in summary["metrics"]
        assert "f1" in summary["metrics"]

    def test_advanced_metrics_comprehensive(self, sample_corpus, sample_queries):
        """Test comprehensive evaluation with advanced metrics."""
        query = sample_queries[0]
        relevant_docs = sample_corpus[:3]  # First 3 docs as relevant

        # Get retriever results
        retrieved_docs = hybrid.hybrid_retrieve(query, sample_corpus, 5)

        # Run comprehensive evaluation
        metrics_dict = comprehensive_evaluation(retrieved_docs, relevant_docs, query)

        # Check that expected metrics are present
        expected_metrics = [
            "ndcg@5",
            "ndcg@10",
            "precision@5",
            "precision@10",
            "recall@5",
            "recall@10",
            "contextual_relevance",
        ]

        for metric in expected_metrics:
            assert metric in metrics_dict
            assert isinstance(metrics_dict[metric], float)
            assert 0.0 <= metrics_dict[metric] <= 1.0


class TestOptimizationIntegration:
    """Test integration of optimization components."""

    def test_hybrid_optimizer_with_acceptance_policy(self, sample_corpus, sample_queries):
        """Test hybrid optimizer working with acceptance policy."""
        # Test grid search optimization
        weights, score = grid_search_hybrid_weights(
            sample_queries[:2], sample_corpus, k=3, grid_size=3
        )

        assert hasattr(weights, "bm25_weight")
        assert hasattr(weights, "dense_weight")
        assert 0.0 <= weights.bm25_weight <= 1.0
        assert 0.0 <= weights.dense_weight <= 1.0
        assert abs(weights.bm25_weight + weights.dense_weight - 1.0) < 0.01  # Should sum to ~1
        assert isinstance(score, float)

    def test_acceptance_policy_integration(self, sample_corpus, tmp_path):
        """Test acceptance policy with file operations."""
        # Use a unique best runs file for this test
        best_runs_file = tmp_path / "test_best_runs.json"
        policy = AcceptancePolicy(threshold=0.01, best_runs_file=str(best_runs_file))

        # Create a temporary file to test with
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write('{"test": "data"}')
            temp_file = f.name

        try:
            # Test safe update
            def update_func():
                with open(temp_file, "w") as f:
                    f.write('{"test": "updated"}')

            accepted = safe_config_update(update_func, [temp_file], policy, runs_dir=str(tmp_path))

            # Should accept the first change
            assert accepted

            # Verify file was updated
            with open(temp_file, "r") as f:
                content = f.read()
                assert content == '{"test": "updated"}'

        finally:
            os.unlink(temp_file)


class TestAugmentationIntegration:
    """Test integration of data augmentation components."""

    def test_synonym_mining_workflow(self, sample_corpus, sample_queries):
        """Test the complete synonym mining workflow."""
        query = sample_queries[0]

        # Get retriever results
        bm25_results = bm25.bm25_retrieve(query, sample_corpus, 5)
        dense_results = dense.dense_retrieve(query, sample_corpus, 5)
        hybrid_results = hybrid.hybrid_retrieve(query, sample_corpus, 5)

        # Mine synonyms
        synonyms = mine_synonyms_from_disagreements(bm25_results, dense_results, hybrid_results)

        # Should return a dict (may be empty if no synonyms found)
        assert isinstance(synonyms, dict)

        # If synonyms found, they should be dictionaries with term -> synonym_list mappings
        for term, synonym_list in synonyms.items():
            assert isinstance(term, str)
            assert isinstance(synonym_list, list)
            for synonym in synonym_list:
                assert isinstance(synonym, str)


class TestRerankerIntegration:
    """Test integration of reranking components."""

    def test_simple_reranker_workflow(self, sample_corpus, sample_queries):
        """Test the complete reranking workflow."""
        query = sample_queries[0]

        # First retrieve documents
        retrieved_docs = hybrid.hybrid_retrieve(query, sample_corpus, 10)

        # Initialize reranker
        reranker = SimpleReranker()

        # Rerank documents
        reranked_docs = reranker.rerank(query, retrieved_docs)

        # Should return same number of documents
        assert len(reranked_docs) == len(retrieved_docs)

        # All documents should be from original retrieval
        assert set(reranked_docs) == set(retrieved_docs)

        # Test with different k
        reranked_subset = reranker.rerank(query, retrieved_docs, k=5)
        assert len(reranked_subset) == 5


class TestTimeSeriesIntegration:
    """Test integration of time-series components."""

    def test_time_series_retrieval_workflow(self, sample_corpus):
        """Test the complete time-series retrieval workflow."""
        # Create time-series notes
        notes = []
        base_time = datetime.now()

        for i, doc in enumerate(sample_corpus[:5]):  # Use subset for speed
            timestamp = base_time - timedelta(days=i * 2)  # Spread over 10 days
            note = TimeSeriesNote(content=doc, timestamp=timestamp, metadata={"id": f"note_{i}"})
            notes.append(note)

        # Initialize retriever
        embedder = FFTEmbedder()
        retriever = TimeSeriesRetriever(embedder=embedder)
        retriever.add_notes(notes)

        # Test search
        query = "sun bright sky"
        results = retriever.search(query=query, query_time=base_time, top_k=3, time_window_days=7)

        # Should return list of dictionaries
        assert isinstance(results, list)
        assert len(results) <= 3

        for result in results:
            assert "content" in result
            assert "timestamp" in result
            assert "combined_score" in result
            assert "temporal_score" in result
            assert "content_score" in result
            assert isinstance(result["combined_score"], float)
            assert 0.0 <= result["combined_score"] <= 1.0


class TestEndToEndIntegration:
    """Test end-to-end integration scenarios."""

    def test_complete_retrieval_pipeline(self, sample_corpus, sample_queries):
        """Test a complete retrieval pipeline from query to reranked results."""
        query = sample_queries[0]

        # Step 1: Retrieve documents
        retrieved_docs = hybrid.hybrid_retrieve(query, sample_corpus, 10)
        assert len(retrieved_docs) <= 10

        # Step 2: Rerank documents
        reranker = SimpleReranker()
        reranked_docs = reranker.rerank(query, retrieved_docs, k=5)
        assert len(reranked_docs) == 5

        # Step 3: Evaluate results
        relevant_docs = sample_corpus[:3]  # Assume first 3 are relevant
        metrics_dict = comprehensive_evaluation(reranked_docs, relevant_docs, query)

        # Should have evaluation metrics
        assert "ndcg@5" in metrics_dict
        assert "precision@5" in metrics_dict
        assert "contextual_relevance" in metrics_dict

    def test_disagreement_analysis_pipeline(self, sample_corpus, sample_queries):
        """Test the complete disagreement analysis pipeline."""
        query = sample_queries[0]

        # Get results from multiple retrievers
        bm25_results = bm25.bm25_retrieve(query, sample_corpus, 5)
        dense_results = dense.dense_retrieve(query, sample_corpus, 5)
        hybrid_results = hybrid.hybrid_retrieve(query, sample_corpus, 5)

        # Calculate disagreement metrics
        disagreement_metrics = {
            "jaccard_bm25_vs_dense": metrics.jaccard_at_k(bm25_results, dense_results),
            "jaccard_bm25_vs_hybrid": metrics.jaccard_at_k(bm25_results, hybrid_results),
            "jaccard_dense_vs_hybrid": metrics.jaccard_at_k(dense_results, hybrid_results),
        }

        # All metrics should be valid
        for name, value in disagreement_metrics.items():
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0

        # Mine synonyms from disagreements
        synonyms = mine_synonyms_from_disagreements(bm25_results, dense_results, hybrid_results)
        assert isinstance(synonyms, dict)

    def test_evaluation_pipeline_with_mocking(self, sample_corpus, tmp_path):
        """Test evaluation pipeline with mocked retrievers."""
        # Note: run_small_suite doesn't use retrievers, so no mocking needed

        # Run evaluation
        summary = run_small_suite(runs_dir=str(tmp_path / "runs"), judge_type="deterministic")

        # Verify results structure
        assert "metrics" in summary
        assert "run_id" in summary
        assert all(key in summary["metrics"] for key in ["em", "f1", "relevance", "faithfulness"])


class TestErrorHandlingIntegration:
    """Test error handling in integrated scenarios."""

    def test_empty_corpus_handling(self):
        """Test behavior with empty corpus."""
        empty_corpus = []

        # All retrievers should handle empty corpus gracefully
        bm25_results = bm25.bm25_retrieve("test query", empty_corpus, 5)
        dense_results = dense.dense_retrieve("test query", empty_corpus, 5)
        hybrid_results = hybrid.hybrid_retrieve("test query", empty_corpus, 5)

        assert bm25_results == []
        assert dense_results == []
        assert hybrid_results == []

    def test_malformed_query_handling(self, sample_corpus):
        """Test behavior with malformed queries."""
        # Empty and whitespace-only queries should raise ValueError
        with pytest.raises(ValueError, match="Query cannot be empty"):
            bm25.bm25_retrieve("", sample_corpus, 5)
        with pytest.raises(RetrieverError, match="Query cannot be empty"):
            dense.dense_retrieve("", sample_corpus, 5)
        with pytest.raises(ValueError, match="Query cannot be empty"):
            hybrid.hybrid_retrieve("", sample_corpus, 5)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            bm25.bm25_retrieve("   ", sample_corpus, 5)
        with pytest.raises(RetrieverError, match="Query cannot be empty"):
            dense.dense_retrieve("   ", sample_corpus, 5)
        with pytest.raises(ValueError, match="Query cannot be empty"):
            hybrid.hybrid_retrieve("   ", sample_corpus, 5)

        # Special characters should work (may return empty results)
        special_query = "!@#$%"
        bm25_results = bm25.bm25_retrieve(special_query, sample_corpus, 5)
        dense_results = dense.dense_retrieve(special_query, sample_corpus, 5)
        hybrid_results = hybrid.hybrid_retrieve(special_query, sample_corpus, 5)

        # Should return lists (may be empty)
        assert isinstance(bm25_results, list)
        assert isinstance(dense_results, list)
        assert isinstance(hybrid_results, list)

    def test_file_operation_error_handling(self):
        """Test error handling in file operations."""
        policy = AcceptancePolicy(threshold=0.01)

        # Test with non-existent file
        def failing_update():
            with open("/nonexistent/path/file.json", "w") as f:
                f.write("test")

        # Should reject the update due to file operation failure
        accepted = safe_config_update(failing_update, ["/nonexistent/path/file.json"], policy)
        assert not accepted


if __name__ == "__main__":
    pytest.main([__file__])
