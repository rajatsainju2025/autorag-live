"""Tests for Elasticsearch adapter."""

import json
import tempfile
from unittest.mock import Mock, patch

import pytest

from autorag_live.retrievers.elasticsearch_adapter import (
    ElasticsearchRetriever,
    NumpyElasticsearchFallback,
    create_elasticsearch_retriever,
)


class TestElasticsearchRetriever:
    """Test Elasticsearch retriever functionality."""

    @patch("autorag_live.retrievers.elasticsearch_adapter.ELASTICSEARCH_AVAILABLE", False)
    def test_elasticsearch_not_available_raises_error(self):
        """Test that ElasticsearchRetriever raises error when ES is not available."""
        with pytest.raises(ImportError, match="Elasticsearch is not installed"):
            ElasticsearchRetriever()

    @patch("autorag_live.retrievers.elasticsearch_adapter.ELASTICSEARCH_AVAILABLE", True)
    @patch("autorag_live.retrievers.elasticsearch_adapter.Elasticsearch")
    def test_elasticsearch_retriever_initialization(self, mock_es):
        """Test Elasticsearch retriever initialization."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_es.return_value = mock_client

        retriever = ElasticsearchRetriever(index_name="test_index", hosts=["http://localhost:9200"])

        assert retriever.index_name == "test_index"
        mock_es.assert_called_once_with(hosts=["http://localhost:9200"], api_key=None)

    @patch("autorag_live.retrievers.elasticsearch_adapter.ELASTICSEARCH_AVAILABLE", True)
    @patch("autorag_live.retrievers.elasticsearch_adapter.Elasticsearch")
    def test_elasticsearch_retriever_cloud_init(self, mock_es):
        """Test Elasticsearch retriever initialization with cloud."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_es.return_value = mock_client

        ElasticsearchRetriever(cloud_id="test:cloud", api_key="test_key")

        mock_es.assert_called_once_with(cloud_id="test:cloud", api_key="test_key")

    @patch("autorag_live.retrievers.elasticsearch_adapter.ELASTICSEARCH_AVAILABLE", True)
    @patch("autorag_live.retrievers.elasticsearch_adapter.Elasticsearch")
    def test_connection_failure(self, mock_es):
        """Test connection failure handling."""
        mock_client = Mock()
        mock_client.ping.return_value = False
        mock_es.return_value = mock_client

        with pytest.raises(ConnectionError, match="Cannot connect to Elasticsearch"):
            ElasticsearchRetriever()

    @pytest.mark.skip(reason="Requires elasticsearch package to be installed")
    @patch("autorag_live.retrievers.elasticsearch_adapter.ELASTICSEARCH_AVAILABLE", True)
    @patch("autorag_live.retrievers.elasticsearch_adapter.Elasticsearch")
    @patch("autorag_live.retrievers.elasticsearch_adapter.bulk")
    def test_add_documents_creates_index(self, mock_bulk, mock_es):
        """Test adding documents creates index when it doesn't exist."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.indices.exists.return_value = False
        mock_client.indices.create.return_value = {"acknowledged": True}
        mock_bulk.return_value = (2, [])  # success, failed
        mock_es.return_value = mock_client

        retriever = ElasticsearchRetriever()
        retriever.encode = Mock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        documents = ["doc1", "doc2"]
        retriever.add_documents(documents)

        mock_client.indices.create.assert_called_once()
        mock_bulk.assert_called_once()

    @patch("autorag_live.retrievers.elasticsearch_adapter.ELASTICSEARCH_AVAILABLE", True)
    @patch("autorag_live.retrievers.elasticsearch_adapter.Elasticsearch")
    def test_text_search(self, mock_es):
        """Test text-based search."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.indices.exists.return_value = True
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"content": "test doc"}, "_score": 0.8},
                    {"_source": {"content": "another doc"}, "_score": 0.6},
                ]
            }
        }
        mock_es.return_value = mock_client

        retriever = ElasticsearchRetriever()
        results = retriever.search("test query", k=2, search_type="text")

        assert len(results) == 2
        assert results[0] == ("test doc", 0.8)
        assert results[1] == ("another doc", 0.6)

    @patch("autorag_live.retrievers.elasticsearch_adapter.ELASTICSEARCH_AVAILABLE", True)
    @patch("autorag_live.retrievers.elasticsearch_adapter.Elasticsearch")
    def test_vector_search(self, mock_es):
        """Test vector-based search."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.indices.exists.return_value = True
        mock_client.search.return_value = {
            "hits": {"hits": [{"_source": {"content": "vector doc"}, "_score": 1.2}]}
        }
        mock_es.return_value = mock_client

        retriever = ElasticsearchRetriever()
        import numpy as np

        retriever.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))

        results = retriever.search("test query", k=1, search_type="vector")

        assert len(results) == 1
        assert results[0] == ("vector doc", 1.2)

    @patch("autorag_live.retrievers.elasticsearch_adapter.ELASTICSEARCH_AVAILABLE", True)
    @patch("autorag_live.retrievers.elasticsearch_adapter.Elasticsearch")
    def test_hybrid_search(self, mock_es):
        """Test hybrid search."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.indices.exists.return_value = True
        mock_client.search.return_value = {
            "hits": {"hits": [{"_source": {"content": "hybrid doc"}, "_score": 1.5}]}
        }
        mock_es.return_value = mock_client

        retriever = ElasticsearchRetriever()
        import numpy as np

        retriever.encode = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))

        results = retriever.search("test query", k=1, search_type="hybrid")

        assert len(results) == 1
        assert results[0] == ("hybrid doc", 1.5)

    @patch("autorag_live.retrievers.elasticsearch_adapter.ELASTICSEARCH_AVAILABLE", True)
    @patch("autorag_live.retrievers.elasticsearch_adapter.Elasticsearch")
    def test_invalid_search_type(self, mock_es):
        """Test invalid search type raises error."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_es.return_value = mock_client

        retriever = ElasticsearchRetriever()

        with pytest.raises(ValueError, match="Unknown search type"):
            retriever.search("query", search_type="invalid")

    @patch("autorag_live.retrievers.elasticsearch_adapter.ELASTICSEARCH_AVAILABLE", True)
    @patch("autorag_live.retrievers.elasticsearch_adapter.Elasticsearch")
    def test_save_and_load_config(self, mock_es):
        """Test saving and loading retriever configuration."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        # Remove problematic attributes that can't be serialized
        del mock_client._cloud_id
        mock_client._hosts = ["http://localhost:9200"]
        mock_es.return_value = mock_client

        retriever = ElasticsearchRetriever(
            index_name="test_index", text_boost=1.5, vector_boost=2.0
        )

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            temp_path = f.name

        try:
            retriever.save(temp_path)

            # Verify config was saved
            with open(temp_path, "r") as f:
                config = json.load(f)

            assert config["type"] == "elasticsearch"
            assert config["index_name"] == "test_index"
            assert config["text_boost"] == 1.5
            assert config["vector_boost"] == 2.0

            # Test loading
            loaded_retriever = ElasticsearchRetriever.load_from_config(temp_path)
            assert loaded_retriever.index_name == "test_index"
            assert loaded_retriever.text_boost == 1.5
            assert loaded_retriever.vector_boost == 2.0

        finally:
            import os

            os.unlink(temp_path)

    @patch("autorag_live.retrievers.elasticsearch_adapter.ELASTICSEARCH_AVAILABLE", True)
    @patch("autorag_live.retrievers.elasticsearch_adapter.Elasticsearch")
    def test_load_invalid_config_type(self, mock_es):
        """Test loading config with invalid type raises error."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_es.return_value = mock_client

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            json.dump({"type": "invalid"}, f)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid config type"):
                ElasticsearchRetriever.load_from_config(temp_path)
        finally:
            import os

            os.unlink(temp_path)


class TestNumpyElasticsearchFallback:
    """Test numpy fallback retriever."""

    def test_fallback_initialization(self):
        """Test fallback retriever initialization."""
        retriever = NumpyElasticsearchFallback()
        assert retriever.documents == []
        assert retriever.embeddings is None
        assert retriever.metadata == []

    def test_fallback_add_documents(self):
        """Test adding documents to fallback retriever."""
        retriever = NumpyElasticsearchFallback()
        import numpy as np

        retriever.encode = Mock(return_value=np.array([[0.1, 0.2], [0.3, 0.4]]))

        documents = ["doc1", "doc2"]
        retriever.add_documents(documents)

        assert retriever.documents == documents
        assert retriever.embeddings is not None
        assert retriever.metadata == [{}, {}]

    def test_fallback_search(self):
        """Test searching in fallback retriever."""
        retriever = NumpyElasticsearchFallback()
        import numpy as np

        retriever.encode = Mock(
            side_effect=[
                np.array([[0.1, 0.2], [0.3, 0.4]]),  # for add_documents
                np.array([[0.5, 0.6]]),  # for search
            ]
        )

        documents = ["doc1", "doc2"]
        retriever.add_documents(documents)

        results = retriever.search("query", k=1)

        assert len(results) == 1
        assert isinstance(results[0][0], str)
        assert isinstance(results[0][1], float)

    def test_fallback_save_and_load(self):
        """Test saving and loading fallback retriever."""
        retriever = NumpyElasticsearchFallback()
        import numpy as np

        retriever.encode = Mock(return_value=np.array([[0.1, 0.2], [0.3, 0.4]]))

        documents = ["doc1", "doc2"]
        retriever.add_documents(documents)

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            temp_path = f.name

        try:
            retriever.save(temp_path)

            # Load
            loaded_retriever = NumpyElasticsearchFallback.load_from_config(temp_path)

            assert loaded_retriever.documents == documents
            assert loaded_retriever.embeddings is not None

        finally:
            import os

            os.unlink(temp_path)


class TestCreateElasticsearchRetriever:
    """Test factory function."""

    @patch("autorag_live.retrievers.elasticsearch_adapter.ELASTICSEARCH_AVAILABLE", True)
    @patch("autorag_live.retrievers.elasticsearch_adapter.Elasticsearch")
    def test_create_elasticsearch_retriever(self, mock_es):
        """Test creating Elasticsearch retriever."""
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_es.return_value = mock_client

        retriever = create_elasticsearch_retriever(index_name="test")

        assert isinstance(retriever, ElasticsearchRetriever)

    @patch("autorag_live.retrievers.elasticsearch_adapter.ELASTICSEARCH_AVAILABLE", False)
    def test_create_fallback_retriever(self):
        """Test creating fallback retriever when ES not available."""
        retriever = create_elasticsearch_retriever()

        assert isinstance(retriever, NumpyElasticsearchFallback)
