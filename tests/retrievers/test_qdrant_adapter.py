"""Tests for Qdrant adapter."""

import json
import tempfile
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from autorag_live.retrievers.qdrant_adapter import QdrantRetriever
from autorag_live.types.types import RetrieverError

from autorag_live.retrievers.qdrant_adapter import QdrantRetriever
from autorag_live.types.types import RetrieverError


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock sentence transformer."""
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        # Mock encode method to return random embeddings
        def mock_encode(texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            return np.random.randn(len(texts), 384)  # Random embeddings
        
        mock_st.return_value.encode = mock_encode
        yield mock_st.return_value


@pytest.fixture
def mock_qdrant_imports():
    """Fixture to mock Qdrant imports and availability."""
    with patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True), \
         patch('autorag_live.retrievers.qdrant_adapter.QdrantClient') as mock_client_class, \
         patch('autorag_live.retrievers.qdrant_adapter.Distance') as mock_distance, \
         patch('autorag_live.retrievers.qdrant_adapter.VectorParams') as mock_vector_params:
        
        # Mock Distance enum
        mock_distance.COSINE = "cosine"
        mock_distance.EUCLID = "euclid"
        mock_distance.DOT = "dot"
        
        # Create a mock client instance
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        
        yield mock_client_class, mock_distance, mock_vector_params


class TestQdrantRetriever:
    """Test Qdrant retriever functionality."""

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', False)
    def test_qdrant_not_available_raises_error(self):
        """Test that QdrantRetriever raises error when Qdrant is not available."""
        with pytest.raises(ImportError, match="Qdrant is not installed"):
            QdrantRetriever()

    def test_qdrant_retriever_initialization(self, mock_qdrant_imports):
        """Test Qdrant retriever initialization."""
        mock_client_class, mock_distance, mock_vector_params = mock_qdrant_imports

        retriever = QdrantRetriever(
            model_name="test-model",
            collection_name="test_collection",
            host="localhost",
            port=6333
        )

        assert retriever.model_name == "test-model"
        assert retriever.collection_name == "test_collection"
        mock_client_class.assert_called_once_with(host="localhost", port=6333, api_key=None)

    def test_qdrant_retriever_with_url(self, mock_qdrant_imports):
        """Test Qdrant retriever initialization with URL."""
        mock_client, mock_distance, mock_vector_params = mock_qdrant_imports

        retriever = QdrantRetriever(
            url="https://qdrant.example.com",
            api_key="test_key"
        )

        mock_client.assert_called_once_with(url="https://qdrant.example.com", api_key="test_key")

    def test_build_index(self, mock_qdrant_imports, mock_sentence_transformer, sample_docs):
        """Test building search index from documents."""
        mock_client_class, mock_distance, mock_vector_params = mock_qdrant_imports
        mock_client = mock_client_class.return_value
        
        # Mock collection info
        mock_client.get_collection.return_value = MagicMock(
            vectors_count=0,
            config=MagicMock(params=MagicMock(vectors=MagicMock(size=384)))
        )
        
        # Mock collections list
        mock_collections = Mock()
        mock_collections.collections = []  # Empty list means collection doesn't exist
        mock_client.get_collections.return_value = mock_collections

        retriever = QdrantRetriever()
        retriever.build_index(sample_docs)
        
        # Verify documents were added
        assert mock_client.upsert.called
        call_args = mock_client.upsert.call_args[1]
        assert call_args["collection_name"] == "autorag_docs"
        assert len(call_args["points"]) == len(sample_docs)

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True)
    @patch('autorag_live.retrievers.qdrant_adapter.Distance')
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_search(self, mock_qdrant_client, mock_distance, mock_sentence_transformer):
        """Test document search."""
        # Mock Distance enum
        mock_distance.COSINE = "cosine"
        mock_distance.EUCLID = "euclid"
        mock_distance.DOT = "dot"
        
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        
        # Mock search results
        mock_client.search.return_value = [
            MagicMock(payload={"text": "doc1"}, score=0.9),
            MagicMock(payload={"text": "doc2"}, score=0.8)
        ]
        
        retriever = QdrantRetriever()
        results = retriever.search("test query", k=2)
        
        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(isinstance(score, float) for _, score in results)
        assert mock_client.search.called

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True)
    @patch('autorag_live.retrievers.qdrant_adapter.Distance')
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_error_handling(self, mock_qdrant_client, mock_distance, mock_sentence_transformer):
        """Test error handling during search."""
        # Mock Distance enum
        mock_distance.COSINE = "cosine"
        mock_distance.EUCLID = "euclid"
        mock_distance.DOT = "dot"
        
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        
        # Mock client error
        mock_client.search.side_effect = Exception("Qdrant error")
        
        retriever = QdrantRetriever()
        with pytest.raises(RetrieverError):
            retriever.search("test query")

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True)
    @patch('autorag_live.retrievers.qdrant_adapter.Distance')
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_save_and_load(self, mock_qdrant_client, mock_distance, tmp_path):
        """Test saving and loading retriever state."""
        # Mock Distance enum
        mock_distance.COSINE = "cosine"
        mock_distance.EUCLID = "euclid"
        mock_distance.DOT = "dot"
        
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        
        # Set client attributes for save/load test
        mock_client._host = "test.host"
        mock_client._port = 6333
        mock_client._url = None
        mock_client._api_key = None

        retriever = QdrantRetriever(
            collection_name="test_collection",
            host="test.host",
            port=6333
        )
        
        # Save state
        save_path = tmp_path / "retriever_state.json"
        retriever.save(str(save_path))
        
        # Load state using class method
        loaded = QdrantRetriever.from_config(str(save_path))
        assert loaded.collection_name == retriever.collection_name
        assert loaded.model_name == retriever.model_name

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True)
    @patch('autorag_live.retrievers.qdrant_adapter.Distance')
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    @pytest.mark.parametrize("distance_metric", ["cosine", "euclid", "dot"])
    def test_different_distance_metrics(self, mock_qdrant_client, mock_distance, distance_metric):
        """Test different distance metrics."""
        # Mock Distance enum
        mock_distance.COSINE = "cosine"
        mock_distance.EUCLID = "euclid"
        mock_distance.DOT = "dot"
        
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        retriever = QdrantRetriever(distance_metric=distance_metric)
        assert retriever.distance_metric == distance_metric

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True)
    @patch('autorag_live.retrievers.qdrant_adapter.VectorParams')
    @patch('autorag_live.retrievers.qdrant_adapter.Distance')
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_build_index_creates_collection(self, mock_qdrant_client, mock_distance, mock_vector_params):
        """Test building index creates Qdrant collection."""
        # Mock Distance enum
        mock_distance.COSINE = "cosine"
        mock_distance.EUCLID = "euclid"
        mock_distance.DOT = "dot"
        
        # Mock VectorParams
        mock_vector_params.return_value = Mock()
        
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        # Mock collections response
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        retriever = QdrantRetriever()
        retriever.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))

        documents = ["doc1", "doc2"]
        retriever.build_index(documents)

        # Verify collection creation was attempted
        mock_client.create_collection.assert_called_once()
        mock_client.upsert.assert_called_once()

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True)
    @patch('autorag_live.retrievers.qdrant_adapter.Distance')
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_search_returns_results(self, mock_qdrant_client, mock_distance):
        """Test search returns formatted results."""
        # Mock Distance enum
        mock_distance.COSINE = "cosine"
        mock_distance.EUCLID = "euclid"
        mock_distance.DOT = "dot"
        
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        # Mock search result
        mock_hit = Mock()
        mock_hit.payload = {"text": "test document"}
        mock_hit.score = 0.95
        mock_client.search.return_value = [mock_hit]

        retriever = QdrantRetriever()
        retriever.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))

        results = retriever.search("test query", k=1)

        assert len(results) == 1
        assert results[0] == ("test document", 0.95)
        mock_client.search.assert_called_once()

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True)
    @patch('autorag_live.retrievers.qdrant_adapter.Distance')
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_save_and_load_config(self, mock_qdrant_client, mock_distance):
        """Test saving and loading retriever configuration."""
        # Mock Distance enum
        mock_distance.COSINE = "cosine"
        mock_distance.EUCLID = "euclid"
        mock_distance.DOT = "dot"
        
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        
        # Set client attributes for save test
        mock_client._host = "localhost"
        mock_client._port = 6333
        mock_client._url = None
        mock_client._api_key = None

        retriever = QdrantRetriever(
            model_name="test-model",
            collection_name="test_collection",
            distance_metric="cosine"
        )

        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            temp_path = f.name

        try:
            retriever.save(temp_path)

            # Verify config was saved
            with open(temp_path, 'r') as f:
                config = json.load(f)

            assert config["type"] == "qdrant"
            assert config["model_name"] == "test-model"
            assert config["collection_name"] == "test_collection"
            assert config["distance_metric"] == "cosine"

            # Test loading using class method
            loaded_retriever = QdrantRetriever.from_config(temp_path)
            assert loaded_retriever.model_name == "test-model"
            assert loaded_retriever.collection_name == "test_collection"

        finally:
            import os
            os.unlink(temp_path)

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True)
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_load_invalid_config_type(self, mock_qdrant_client):
        """Test loading config with invalid type raises error."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            json.dump({"type": "invalid"}, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid config type"):
                QdrantRetriever.from_config(temp_path)
        finally:
            import os
            os.unlink(temp_path)