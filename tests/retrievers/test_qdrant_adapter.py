"""Tests for Qdrant adapter."""

import json
import tempfile
import pytest
from unittest.mock import Mock, patch

from autorag_live.retrievers.qdrant_adapter import QdrantRetriever


class TestQdrantRetriever:
    """Test Qdrant retriever functionality."""

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', False)
    def test_qdrant_not_available_raises_error(self):
        """Test that QdrantRetriever raises error when Qdrant is not available."""
        with pytest.raises(ImportError, match="Qdrant is not installed"):
            QdrantRetriever()

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True)
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_qdrant_retriever_initialization(self, mock_qdrant_client):
        """Test Qdrant retriever initialization."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        retriever = QdrantRetriever(
            model_name="test-model",
            collection_name="test_collection",
            host="localhost",
            port=6333
        )

        assert retriever.model_name == "test-model"
        assert retriever.collection_name == "test_collection"
        mock_qdrant_client.assert_called_once_with(host="localhost", port=6333, api_key=None)

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True)
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_qdrant_retriever_with_url(self, mock_qdrant_client):
        """Test Qdrant retriever initialization with URL."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        retriever = QdrantRetriever(
            url="https://qdrant.example.com",
            api_key="test_key"
        )

        mock_qdrant_client.assert_called_once_with(url="https://qdrant.example.com", api_key="test_key")

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True)
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_build_index_creates_collection(self, mock_qdrant_client):
        """Test building index creates Qdrant collection."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        # Mock collections response
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        retriever = QdrantRetriever()
        retriever.encode = Mock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        documents = ["doc1", "doc2"]
        retriever.build_index(documents)

        # Verify collection creation was attempted
        mock_client.create_collection.assert_called_once()
        mock_client.upsert.assert_called_once()

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True)
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_search_returns_results(self, mock_qdrant_client):
        """Test search returns formatted results."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        # Mock search result
        mock_hit = Mock()
        mock_hit.payload = {"text": "test document"}
        mock_hit.score = 0.95
        mock_client.search.return_value = [mock_hit]

        retriever = QdrantRetriever()
        retriever.encode = Mock(return_value=[[0.1, 0.2, 0.3]])

        results = retriever.search("test query", k=1)

        assert len(results) == 1
        assert results[0] == ("test document", 0.95)
        mock_client.search.assert_called_once()

    @patch('autorag_live.retrievers.qdrant_adapter.QDRANT_AVAILABLE', True)
    @patch('autorag_live.retrievers.qdrant_adapter.QdrantClient')
    def test_save_and_load_config(self, mock_qdrant_client):
        """Test saving and loading retriever configuration."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

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

            # Test loading
            loaded_retriever = QdrantRetriever.load(temp_path)
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
                QdrantRetriever.load(temp_path)
        finally:
            import os
            os.unlink(temp_path)