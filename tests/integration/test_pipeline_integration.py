"""
Integration tests for complete RAG pipelines.

Tests end-to-end workflows with multiple retrievers, rerankers, and evaluation metrics.
"""
import pytest

from autorag_live.evals.advanced_metrics import comprehensive_evaluation
from autorag_live.retrievers import bm25


@pytest.fixture
def sample_corpus():
    """Create sample corpus for testing."""
    return [
        "Python is a high-level programming language with dynamic semantics.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing helps computers understand human language.",
        "Deep learning uses neural networks with multiple layers.",
        "Information retrieval is the process of obtaining relevant information.",
        "Semantic search understands the intent behind queries.",
        "Vector databases store and query high-dimensional embeddings.",
        "RAG combines retrieval with generative AI models.",
        "Fine-tuning adapts pre-trained models to specific tasks.",
        "Evaluation metrics measure the quality of retrieval systems.",
    ]


@pytest.fixture
def sample_queries():
    """Create sample queries for testing."""
    return [
        "What is machine learning?",
        "How does semantic search work?",
        "Tell me about neural networks",
        "What is RAG?",
    ]


class TestBM25Pipeline:
    """Test pipelines with BM25 retriever."""

    def test_bm25_basic_pipeline(self, sample_corpus, sample_queries):
        """Test basic BM25 retrieval pipeline."""
        results = []
        for query in sample_queries:
            docs = bm25.bm25_retrieve(query, sample_corpus, k=3)
            results.append(docs)

        # Verify all queries returned results
        assert len(results) == len(sample_queries)
        assert all(len(docs) == 3 for docs in results)

    def test_bm25_multiple_k_values(self, sample_corpus, sample_queries):
        """Test BM25 with different k values."""
        for k in [1, 3, 5]:
            results = bm25.bm25_retrieve(sample_queries[0], sample_corpus, k=k)
            assert len(results) <= k
            assert len(results) <= len(sample_corpus)

    def test_bm25_all_queries(self, sample_corpus, sample_queries):
        """Test BM25 retrieval for all queries."""
        all_results = []
        for query in sample_queries:
            docs = bm25.bm25_retrieve(query, sample_corpus, k=5)
            all_results.append(docs)

        # Verify each query got results
        assert all(len(docs) > 0 for docs in all_results)
        # Verify results are from corpus
        for docs in all_results:
            assert all(doc in sample_corpus for doc in docs)


class TestPipelineEvaluation:
    """Test evaluation of complete pipelines."""

    def test_pipeline_metrics_computation(self, sample_corpus, sample_queries):
        """Test computing multiple metrics for a pipeline."""
        # Run retrieval
        retrieved_docs_lists = []
        for query in sample_queries:
            docs = bm25.bm25_retrieve(query, sample_corpus, k=5)
            retrieved_docs_lists.append(docs)

        # Use comprehensive_evaluation
        results = comprehensive_evaluation(
            retrieved_docs=retrieved_docs_lists[0],  # First query results
            relevant_docs=[sample_corpus[1], sample_corpus[3]],  # Example relevant docs
            queries=sample_queries[:1],
            corpus=sample_corpus,
        )

        # Verify metrics were computed
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_batch_retrieval_evaluation(self, sample_corpus, sample_queries):
        """Test batch retrieval and evaluation."""
        # Batch retrieval
        batch_results = [bm25.bm25_retrieve(q, sample_corpus, k=10) for q in sample_queries]

        # Verify batch processing
        assert len(batch_results) == len(sample_queries)
        assert all(isinstance(docs, list) for docs in batch_results)
        assert all(len(docs) > 0 for docs in batch_results)

    def test_varying_k_evaluation(self, sample_corpus, sample_queries):
        """Test evaluation with different k values."""
        k_values = [1, 3, 5, 10]

        for k in k_values:
            results = []
            for query in sample_queries:
                docs = bm25.bm25_retrieve(query, sample_corpus, k=k)
                results.append(docs)

            # Verify k constraint
            assert all(len(docs) <= k for docs in results)
            assert all(len(docs) <= len(sample_corpus) for docs in results)


class TestMultiQueryPipeline:
    """Test pipelines with multiple queries."""

    def test_sequential_queries(self, sample_corpus, sample_queries):
        """Test processing queries sequentially."""
        results = {}
        for query in sample_queries:
            docs = bm25.bm25_retrieve(query, sample_corpus, k=5)
            results[query] = docs

        # Verify all queries processed
        assert len(results) == len(sample_queries)
        for query, docs in results.items():
            assert len(docs) > 0
            assert all(doc in sample_corpus for doc in docs)

    def test_query_result_consistency(self, sample_corpus):
        """Test that same query returns same results."""
        query = "machine learning"

        # Retrieve multiple times
        results1 = bm25.bm25_retrieve(query, sample_corpus, k=5)
        results2 = bm25.bm25_retrieve(query, sample_corpus, k=5)

        # Results should be identical
        assert results1 == results2

    def test_different_queries_different_results(self, sample_corpus, sample_queries):
        """Test that different queries return different results."""
        results = [bm25.bm25_retrieve(q, sample_corpus, k=5) for q in sample_queries]

        # At least some results should differ
        assert len(set(tuple(r) for r in results)) > 1


class TestErrorHandling:
    """Test error handling in pipelines."""

    def test_empty_corpus_handling(self):
        """Test handling of empty corpus."""
        # Empty corpus returns empty list
        result = bm25.bm25_retrieve("test query", [], k=3)
        assert result == []

    def test_empty_query_handling(self, sample_corpus):
        """Test handling of empty query."""
        # Empty query should raise RetrieverError
        from autorag_live.types.types import RetrieverError

        with pytest.raises(RetrieverError, match="Query cannot be empty"):
            bm25.bm25_retrieve("", sample_corpus, k=3)

    def test_large_k_handling(self, sample_corpus, sample_queries):
        """Test handling of k larger than corpus."""
        # k > corpus size should return all docs
        results = bm25.bm25_retrieve(sample_queries[0], sample_corpus, k=1000)
        assert len(results) <= len(sample_corpus)

    def test_zero_k_handling(self, sample_corpus, sample_queries):
        """Test handling of k=0."""
        # k=0 should raise RetrieverError
        from autorag_live.types.types import RetrieverError

        with pytest.raises(RetrieverError, match="k must be positive"):
            bm25.bm25_retrieve(sample_queries[0], sample_corpus, k=0)


class TestCrossFunctionalPipeline:
    """Test cross-functional aspects of pipelines."""

    def test_corpus_coverage(self, sample_corpus, sample_queries):
        """Test how well queries cover the corpus."""
        retrieved_indices = set()

        for query in sample_queries:
            docs = bm25.bm25_retrieve(query, sample_corpus, k=5)
            for doc in docs:
                if doc in sample_corpus:
                    retrieved_indices.add(sample_corpus.index(doc))

        # Calculate coverage
        coverage = len(retrieved_indices) / len(sample_corpus)
        assert 0 <= coverage <= 1

    def test_retrieval_consistency_across_runs(self, sample_corpus, sample_queries):
        """Test retrieval consistency across multiple runs."""
        first_run = {q: bm25.bm25_retrieve(q, sample_corpus, k=5) for q in sample_queries}
        second_run = {q: bm25.bm25_retrieve(q, sample_corpus, k=5) for q in sample_queries}

        # Results should be consistent
        for query in sample_queries:
            assert first_run[query] == second_run[query]


class TestPipelinePerformance:
    """Test performance characteristics of pipelines."""

    def test_batch_vs_individual(self, sample_corpus, sample_queries):
        """Compare batch vs individual query processing."""
        # Individual processing
        individual_results = []
        for query in sample_queries:
            docs = bm25.bm25_retrieve(query, sample_corpus, k=5)
            individual_results.append(docs)

        # Batch processing
        batch_results = [bm25.bm25_retrieve(q, sample_corpus, k=5) for q in sample_queries]

        # Results should be identical
        assert individual_results == batch_results

    def test_scalability_with_corpus_size(self):
        """Test pipeline with varying corpus sizes."""
        query = "machine learning"

        for size in [10, 50, 100]:
            corpus = [f"Document {i} about various topics" for i in range(size)]
            results = bm25.bm25_retrieve(query, corpus, k=5)

            assert len(results) <= 5
            assert len(results) <= len(corpus)
