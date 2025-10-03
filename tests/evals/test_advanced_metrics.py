"""Tests for advanced evaluation metrics."""


import numpy as np
import pytest

from autorag_live.evals.advanced_metrics import (
    aggregate_metrics,
    comprehensive_evaluation,
    contextual_relevance,
    diversity_score,
    efficiency_score,
    fairness_score,
    mean_reciprocal_rank,
    ndcg_at_k,
    novelty_score,
    robustness_score,
    semantic_coverage,
)


class TestAdvancedMetrics:
    """Test advanced evaluation metrics."""

    def test_ndcg_at_k(self):
        """Test NDCG calculation."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3", "doc5"]

        # Perfect ranking
        perfect_retrieved = ["doc1", "doc3", "doc5", "doc2", "doc4"]
        ndcg_perfect = ndcg_at_k(perfect_retrieved, relevant, k=5)
        assert ndcg_perfect > 0.8  # Should be close to 1.0

        # Worst ranking
        worst_retrieved = ["doc2", "doc4", "doc6", "doc7", "doc8"]
        ndcg_worst = ndcg_at_k(worst_retrieved, relevant, k=5)
        assert ndcg_worst == 0.0

        # Empty cases
        assert ndcg_at_k([], relevant, k=5) == 0.0
        assert ndcg_at_k(retrieved, [], k=5) == 0.0

    def test_mean_reciprocal_rank(self):
        """Test MRR calculation."""
        retrieved_lists = [
            ["doc1", "doc2", "doc3"],  # Relevant doc at position 1
            ["doc4", "doc1", "doc5"],  # Relevant doc at position 2
            ["doc6", "doc7", "doc8"],  # No relevant docs
        ]
        relevant_lists = [["doc1"], ["doc1"], ["doc1"]]

        mrr = mean_reciprocal_rank(retrieved_lists, relevant_lists)
        expected_mrr = (1.0 + 0.5 + 0.0) / 3  # (1/1 + 1/2 + 0) / 3
        assert abs(mrr - expected_mrr) < 1e-6

        # Empty case
        assert mean_reciprocal_rank([], []) == 0.0

    def test_diversity_score(self):
        """Test diversity score calculation."""
        docs = ["doc1", "doc2", "doc3"]

        # Mock embeddings with moderate similarity (not perfectly similar)
        similar_embeddings = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]])
        diversity_similar = diversity_score(docs, similar_embeddings)
        assert diversity_similar < 0.7  # Moderate diversity due to mixed similarities

        # Mock embeddings with low similarity
        diverse_embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        diversity_diverse = diversity_score(docs, diverse_embeddings)
        assert diversity_diverse > 0.9  # High diversity

        # Edge cases
        assert diversity_score([], None) == 1.0
        assert diversity_score(["doc1"], None) == 1.0

    def test_novelty_score(self):
        """Test novelty score calculation."""
        current_docs = ["doc1", "doc2", "doc3"]
        history_docs = ["doc1", "doc4", "doc5"]

        novelty = novelty_score(current_docs, history_docs)
        assert novelty == 2.0 / 3.0  # 2 out of 3 docs are novel

        # All novel
        novelty_all = novelty_score(current_docs, [])
        assert novelty_all == 1.0

        # No novel
        novelty_none = novelty_score(current_docs, current_docs + ["doc6"])
        assert novelty_none == 0.0

    def test_semantic_coverage(self):
        """Test semantic coverage calculation."""
        retrieved_docs = ["doc1", "doc2"]
        relevant_docs = ["doc3", "doc4"]

        # Mock embeddings
        embeddings = np.array(
            [
                [1.0, 0.9],  # retrieved
                [0.9, 1.0],  # retrieved
                [0.8, 0.7],  # relevant
                [0.7, 0.8],  # relevant
            ]
        )

        coverage = semantic_coverage(retrieved_docs, relevant_docs, embeddings)
        assert 0.0 <= coverage <= 1.0

        # Edge cases
        assert semantic_coverage([], [], None) == 0.0
        assert semantic_coverage(retrieved_docs, [], embeddings) == 0.0

    def test_robustness_score(self):
        """Test robustness score calculation."""
        retrieved_lists = [
            ["doc1", "doc2", "doc3"],
            ["doc1", "doc3", "doc2"],
            ["doc2", "doc1", "doc3"],
        ]
        relevant_docs = ["doc1", "doc2"]

        robustness = robustness_score(retrieved_lists, relevant_docs)
        assert 0.0 <= robustness <= 1.0

        # Single run
        robustness_single = robustness_score([retrieved_lists[0]], relevant_docs)
        assert robustness_single == 1.0

        # Empty case
        assert robustness_score([], relevant_docs) == 0.0

    def test_contextual_relevance(self):
        """Test contextual relevance calculation."""
        docs = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language",
        ]
        query = "quick brown fox"

        relevance = contextual_relevance(docs, query)
        assert 0.0 <= relevance <= 1.0

        # Query terms present
        high_relevance = contextual_relevance([query], query)
        assert high_relevance > 0.5

        # Edge cases
        assert contextual_relevance([], query) == 0.0
        assert contextual_relevance(docs, "") == 0.0

    def test_fairness_score(self):
        """Test fairness score calculation."""
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4"]
        groups = {"group1": ["doc1", "doc2"], "group2": ["doc3", "doc4"]}

        fairness = fairness_score(retrieved_docs, groups)
        assert 0.0 <= fairness <= 1.0

        # Perfect fairness
        perfect_groups = {"group1": ["doc1", "doc2"], "group2": ["doc3", "doc4"]}
        perfect_fairness = fairness_score(retrieved_docs, perfect_groups)
        assert perfect_fairness == 1.0

        # Edge cases
        assert fairness_score([], groups) == 1.0
        assert fairness_score(retrieved_docs, {}) == 1.0

    def test_efficiency_score(self):
        """Test efficiency score calculation."""
        # Fast retrieval
        efficiency_fast = efficiency_score(0.1, 100, baseline_time=1.0)
        assert efficiency_fast > 0.5

        # Slow retrieval
        efficiency_slow = efficiency_score(2.0, 50, baseline_time=1.0)
        assert efficiency_slow < 0.5

        # Edge cases
        assert efficiency_score(0, 100) == 0.0
        assert efficiency_score(1.0, 0) == 0.0

    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation."""
        retrieved_docs = ["doc1", "doc2", "doc3"]
        relevant_docs = ["doc1", "doc3"]
        query = "test query"

        # Mock embeddings
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        metrics = comprehensive_evaluation(retrieved_docs, relevant_docs, query, embeddings)

        # Check that expected metrics are present
        expected_metrics = [
            "ndcg@5",
            "ndcg@10",
            "precision@5",
            "precision@10",
            "recall@5",
            "recall@10",
            "diversity",
            "contextual_relevance",
        ]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert 0.0 <= metrics[metric] <= 1.0

    def test_comprehensive_evaluation_with_kwargs(self):
        """Test comprehensive evaluation with additional parameters."""
        retrieved_docs = ["doc1", "doc2"]
        relevant_docs = ["doc1"]
        query = "test query"

        metrics = comprehensive_evaluation(
            retrieved_docs,
            relevant_docs,
            query,
            query_history=["doc3", "doc4"],
            retrieved_docs_list=[["doc1", "doc2"], ["doc1", "doc3"]],
            document_groups={"group1": ["doc1"], "group2": ["doc2"]},
            retrieval_time=0.5,
            num_docs=2,
        )

        # Check additional metrics
        additional_metrics = ["novelty", "robustness", "fairness", "efficiency"]
        for metric in additional_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)

    def test_aggregate_metrics(self):
        """Test metrics aggregation."""
        metrics_list = [
            {"ndcg@5": 0.8, "precision@5": 0.6, "diversity": 0.7},
            {"ndcg@5": 0.9, "precision@5": 0.7, "diversity": 0.8},
            {"ndcg@5": 0.7, "precision@5": 0.5, "diversity": 0.6},
        ]

        aggregated = aggregate_metrics(metrics_list)

        # Check aggregation for each metric
        for metric_name in ["ndcg@5", "precision@5", "diversity"]:
            assert metric_name in aggregated
            stats = aggregated[metric_name]
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert "count" in stats
            assert stats["count"] == 3

        # Check mean calculation
        ndcg_stats = aggregated["ndcg@5"]
        expected_mean = (0.8 + 0.9 + 0.7) / 3
        assert abs(ndcg_stats["mean"] - expected_mean) < 1e-6

        # Empty case
        assert aggregate_metrics([]) == {}


if __name__ == "__main__":
    pytest.main([__file__])
