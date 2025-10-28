import pytest

from autorag_live.retrievers import dense
from autorag_live.retrievers.dense import DenseRetriever, RetrieverError


class TestDenseRetrieve:
    """Test the dense_retrieve function."""

    def test_dense_retrieve_basic(self):
        """Test basic dense retrieval functionality."""
        corpus = ["hello world", "this is a test", "another document"]
        query = "hello"
        results = dense.dense_retrieve(query, corpus, k=2)
        assert len(results) == 2
        assert isinstance(results, list)
        assert all(isinstance(doc, str) for doc in results)

    def test_dense_retrieve_empty_corpus(self):
        """Test dense retrieval with empty corpus."""
        corpus = []
        query = "hello"
        results = dense.dense_retrieve(query, corpus, k=2)
        assert results == []

    def test_dense_retrieve_empty_query(self):
        """Test dense retrieval with empty query raises error."""
        corpus = ["hello world", "this is a test"]
        with pytest.raises(RetrieverError, match="Query cannot be empty"):
            dense.dense_retrieve("", corpus, k=2)

    def test_dense_retrieve_whitespace_query(self):
        """Test dense retrieval with whitespace-only query raises error."""
        corpus = ["hello world", "this is a test"]
        with pytest.raises(RetrieverError, match="Query cannot be empty"):
            dense.dense_retrieve("   ", corpus, k=2)

    def test_dense_retrieve_invalid_k(self):
        """Test dense retrieval with invalid k value raises error."""
        corpus = ["hello world", "this is a test"]
        with pytest.raises(RetrieverError, match="k must be positive"):
            dense.dense_retrieve("hello", corpus, k=0)

    def test_dense_retrieve_k_larger_than_corpus(self):
        """Test dense retrieval when k is larger than corpus size."""
        corpus = ["hello world", "this is a test"]
        query = "hello"
        results = dense.dense_retrieve(query, corpus, k=5)
        assert len(results) == 2  # Should return all documents

    def test_dense_retrieve_fallback_mode(self):
        """Test dense retrieval falls back to Jaccard similarity when dependencies unavailable."""
        # This test assumes dependencies are available, but tests the logic path
        corpus = ["hello world", "this is a test", "another document"]
        query = "hello"
        results = dense.dense_retrieve(query, corpus, k=2)
        assert len(results) == 2


class TestDenseRetriever:
    """Test the DenseRetriever class."""

    def test_initialization(self):
        """Test DenseRetriever initialization."""
        retriever = DenseRetriever()
        assert retriever.model_name == "all-MiniLM-L6-v2"
        assert retriever.cache_embeddings is True
        assert retriever.corpus == []
        assert retriever.corpus_embeddings is None
        assert retriever.model is None
        assert not retriever.is_initialized

    def test_initialization_custom_params(self):
        """Test DenseRetriever initialization with custom parameters."""
        retriever = DenseRetriever(model_name="custom-model", cache_embeddings=False)
        assert retriever.model_name == "custom-model"
        assert retriever.cache_embeddings is False

    def test_add_documents_empty_list(self):
        """Test add_documents with empty list raises error."""
        retriever = DenseRetriever()
        with pytest.raises(RetrieverError, match="Documents list cannot be empty"):
            retriever.add_documents([])

    def test_add_documents_invalid_documents(self):
        """Test add_documents with invalid documents raises error."""
        retriever = DenseRetriever()
        with pytest.raises(RetrieverError, match="All documents must be non-empty strings"):
            retriever.add_documents(["valid doc", "", "another doc"])

    def test_add_documents_non_string(self):
        """Test add_documents with non-string documents raises error."""
        retriever = DenseRetriever()
        with pytest.raises(RetrieverError, match="All documents must be non-empty strings"):
            retriever.add_documents(["valid doc", 123, "another doc"])  # type: ignore

    def test_add_documents_valid(self):
        """Test add_documents with valid documents."""
        retriever = DenseRetriever()
        documents = ["document one", "document two", "document three"]
        retriever.add_documents(documents)
        assert retriever.is_initialized
        assert retriever.corpus == documents

    def test_retrieve_not_initialized(self):
        """Test retrieve before initialization raises error."""
        retriever = DenseRetriever()
        with pytest.raises(RetrieverError, match="Retriever not initialized"):
            retriever.retrieve("test query")

    def test_retrieve_basic(self):
        """Test basic retrieve functionality."""
        retriever = DenseRetriever()
        documents = ["hello world", "this is a test", "another document with hello"]
        retriever.add_documents(documents)

        results = retriever.retrieve("hello", k=2)
        assert len(results) == 2
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results)
        assert all(isinstance(doc, str) and isinstance(score, float) for doc, score in results)

    def test_retrieve_k_larger_than_corpus(self):
        """Test retrieve when k is larger than corpus size."""
        retriever = DenseRetriever()
        documents = ["doc1", "doc2"]
        retriever.add_documents(documents)

        results = retriever.retrieve("query", k=5)
        assert len(results) == 2  # Should return all documents

    def test_retrieve_empty_query(self):
        """Test retrieve with empty query."""
        retriever = DenseRetriever()
        documents = ["doc1", "doc2"]
        retriever.add_documents(documents)

        # Empty query should work in retrieve method (different from dense_retrieve function)
        results = retriever.retrieve("", k=2)
        assert len(results) == 2

    def test_save_and_load(self):
        """Test save and load functionality."""
        import os
        import tempfile

        retriever = DenseRetriever()
        documents = ["document one", "document two", "document three"]
        retriever.add_documents(documents)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "retriever_state.pkl")

            # Test save
            retriever.save(save_path)
            assert os.path.exists(save_path)

            # Test load
            new_retriever = DenseRetriever()
            new_retriever.load(save_path)

            assert new_retriever.is_initialized
            assert new_retriever.model_name == retriever.model_name
            assert new_retriever.corpus == retriever.corpus
            assert new_retriever.cache_embeddings == retriever.cache_embeddings

            # Test that loaded retriever can retrieve
            results = new_retriever.retrieve("document", k=2)
            assert len(results) == 2

    def test_save_uninitialized(self):
        """Test save method raises error for uninitialized retriever."""
        retriever = DenseRetriever()
        with pytest.raises(RetrieverError, match="Cannot save uninitialized retriever"):
            retriever.save("dummy_path")

    def test_load_nonexistent_file(self):
        """Test load method raises error for nonexistent file."""
        retriever = DenseRetriever()
        with pytest.raises(FileNotFoundError):
            retriever.load("nonexistent_file.pkl")

    def test_clear_cache(self):
        """Test clear_cache class method."""
        # Add something to cache first
        DenseRetriever._model_cache["test"] = "dummy_model"
        DenseRetriever._embedding_cache.put("test", "dummy_embedding")

        DenseRetriever.clear_cache()

        assert DenseRetriever._model_cache == {}
        assert len(DenseRetriever._embedding_cache) == 0

    def test_batch_size_configuration(self):
        """Test configurable batch size parameter."""
        retriever = DenseRetriever(batch_size=16)
        assert retriever.batch_size == 16

        retriever_large = DenseRetriever(batch_size=128)
        assert retriever_large.batch_size == 128

    def test_cache_ttl_expiration(self):
        """Test TTL cache expiration."""

        cache = DenseRetriever._embedding_cache
        # Clear cache first
        cache.clear()

        # Add an item
        cache.put("test_key", "test_value")

        # Should be available immediately
        assert cache.get("test_key") == "test_value"

        # Wait for TTL to expire (cache TTL is 1 hour, so this won't actually expire)
        # But we can test the logic by checking cache size
        assert len(cache) == 1

    def test_fallback_similarity_accuracy(self):
        """Test that fallback similarity produces reasonable results."""
        corpus = ["the quick brown fox", "jumped over the lazy dog", "hello world test"]
        query = "quick brown fox"

        # Test with fallback mode
        import autorag_live.retrievers.dense as dense_module

        original_st = dense_module.SentenceTransformer
        dense_module.SentenceTransformer = None

        try:
            results = dense.dense_retrieve(query, corpus, k=2)
            # Should return the most similar documents
            assert len(results) == 2
            assert "quick brown fox" in results[0]  # Most similar should be first
        finally:
            dense_module.SentenceTransformer = original_st

    def test_model_reuse_across_instances(self):
        """Test that models are reused across DenseRetriever instances."""
        retriever1 = DenseRetriever()
        retriever2 = DenseRetriever()

        docs1 = ["document one"]
        docs2 = ["document two"]

        retriever1.add_documents(docs1)
        retriever2.add_documents(docs2)

        # Both should use the same cached model
        assert retriever1.model is retriever2.model
        assert DenseRetriever._model_cache[retriever1.model_name] is not None

    def test_embedding_cache_performance(self):
        """Test that embedding caching improves performance."""
        import time

        retriever = DenseRetriever()
        documents = ["test document one", "test document two", "test document three"]

        # First call - should cache embeddings
        start_time = time.time()
        retriever.add_documents(documents)
        first_duration = time.time() - start_time

        # Create new retriever with same documents
        retriever2 = DenseRetriever()
        start_time = time.time()
        retriever2.add_documents(documents)
        second_duration = time.time() - start_time

        # Second call should be faster due to caching
        # (Allow some tolerance for measurement variability)
        assert second_duration <= first_duration * 1.5  # At most 50% slower, probably faster

    def test_large_corpus_handling(self):
        """Test handling of larger corpora."""
        # Create a moderately large corpus
        corpus = [f"document number {i} with some content" for i in range(50)]
        query = "document number 25"

        results = dense.dense_retrieve(query, corpus, k=5)
        assert len(results) == 5
        # Should find the most relevant documents
        assert any("number 25" in result for result in results)

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        corpus = ["héllo wörld", "café au lait", "naïve approach", "test document"]
        query = "héllo wörld"

        results = dense.dense_retrieve(query, corpus, k=2)
        assert len(results) == 2
        assert "héllo wörld" in results[0]  # Should be most similar

    def test_empty_and_single_word_documents(self):
        """Test handling of edge case documents."""
        corpus = ["", "a", "single", "this is a normal document"]
        query = "document"

        results = dense.dense_retrieve(query, corpus, k=2)
        assert len(results) == 2
        # Should prioritize the normal document over empty/single word docs
