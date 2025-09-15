"""Elasticsearch adapter for hybrid search with vector and text capabilities."""

import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    Elasticsearch = None
    ELASTICSEARCH_AVAILABLE = False

from autorag_live.retrievers.faiss_adapter import DenseRetriever

logger = logging.getLogger(__name__)


class ElasticsearchRetriever(DenseRetriever):
    """Dense retriever using Elasticsearch with hybrid search capabilities."""

    def __init__(
        self,
        index_name: str = "autorag_docs",
        hosts: Optional[List[str]] = None,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_boost: float = 1.0,
        vector_boost: float = 1.0,
        **kwargs
    ):
        """Initialize Elasticsearch retriever.

        Args:
            index_name: Name of the Elasticsearch index
            hosts: List of Elasticsearch hosts
            cloud_id: Elasticsearch cloud ID for cloud deployments
            api_key: API key for authentication
            embedding_model: Sentence transformer model name
            text_boost: Boost factor for text search
            vector_boost: Boost factor for vector search
            **kwargs: Additional arguments
        """
        super().__init__(model_name=embedding_model, **kwargs)

        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError(
                "Elasticsearch is not installed. Install with: pip install elasticsearch"
            )

        self.index_name = index_name
        self.text_boost = text_boost
        self.vector_boost = vector_boost

        # Initialize Elasticsearch client
        if cloud_id:
            self.client = Elasticsearch(
                cloud_id=cloud_id,
                api_key=api_key
            )
        else:
            hosts = hosts or ["http://localhost:9200"]
            self.client = Elasticsearch(
                hosts=hosts,
                api_key=api_key
            )

        # Check connection
        if not self.client.ping():
            raise ConnectionError("Cannot connect to Elasticsearch")

        logger.info(f"Connected to Elasticsearch, using index: {index_name}")

    def _create_index_mapping(self, dimension: int) -> Dict[str, Any]:
        """Create index mapping for hybrid search."""
        mapping = {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "content_vector": {
                        "type": "dense_vector",
                        "dims": dimension,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "metadata": {
                        "type": "object",
                        "dynamic": True
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
        return mapping

    def _ensure_index_exists(self, dimension: int):
        """Ensure the index exists with correct mapping."""
        if not self.client.indices.exists(index=self.index_name):
            mapping = self._create_index_mapping(dimension)
            self.client.indices.create(
                index=self.index_name,
                body=mapping
            )
            logger.info(f"Created Elasticsearch index: {self.index_name}")
        else:
            logger.info(f"Using existing Elasticsearch index: {self.index_name}")

    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add documents to the Elasticsearch index.

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        if not documents:
            return

        # Generate embeddings
        embeddings = self.encode(documents)
        dimension = embeddings.shape[1]

        # Ensure index exists
        self._ensure_index_exists(dimension)

        # Prepare bulk actions
        actions = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            action = {
                "_index": self.index_name,
                "_id": str(i),
                "_source": {
                    "content": doc,
                    "content_vector": emb.tolist(),
                    "metadata": metadata[i] if metadata and i < len(metadata) else {}
                }
            }
            actions.append(action)

        # Bulk index
        success, failed = bulk(self.client, actions, refresh=True)

        if failed:
            logger.warning(f"Failed to index {len(failed)} documents")
        else:
            logger.info(f"Successfully indexed {success} documents")

    def build_index(self, documents: List[str]) -> None:
        """Build search index from documents."""
        self.add_documents(documents)

    def search(
        self,
        query: str,
        k: int = 10,
        search_type: str = "hybrid"
    ) -> List[Tuple[str, float]]:
        """Search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return
            search_type: Type of search ("text", "vector", "hybrid")

        Returns:
            List of (document, score) tuples
        """
        if search_type == "text":
            return self._text_search(query, k)
        elif search_type == "vector":
            return self._vector_search(query, k)
        elif search_type == "hybrid":
            return self._hybrid_search(query, k)
        else:
            raise ValueError(f"Unknown search type: {search_type}")

    def _text_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Perform text-based search."""
        search_body = {
            "query": {
                "match": {
                    "content": {
                        "query": query,
                        "boost": self.text_boost
                    }
                }
            },
            "size": k,
            "_source": ["content"]
        }

        response = self.client.search(
            index=self.index_name,
            body=search_body
        )

        results = []
        for hit in response["hits"]["hits"]:
            content = hit["_source"]["content"]
            score = hit["_score"]
            results.append((content, float(score)))

        return results

    def _vector_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Perform vector-based search."""
        # Generate query embedding
        query_emb = self.encode([query])[0]

        search_body = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                        "params": {
                            "query_vector": query_emb.tolist()
                        }
                    },
                    "boost": self.vector_boost
                }
            },
            "size": k,
            "_source": ["content"]
        }

        response = self.client.search(
            index=self.index_name,
            body=search_body
        )

        results = []
        for hit in response["hits"]["hits"]:
            content = hit["_source"]["content"]
            score = hit["_score"]
            results.append((content, float(score)))

        return results

    def _hybrid_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Perform hybrid search combining text and vector search."""
        # Generate query embedding
        query_emb = self.encode([query])[0]

        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "content": {
                                    "query": query,
                                    "boost": self.text_boost
                                }
                            }
                        },
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                                    "params": {
                                        "query_vector": query_emb.tolist()
                                    }
                                },
                                "boost": self.vector_boost
                            }
                        }
                    ]
                }
            },
            "size": k,
            "_source": ["content"]
        }

        response = self.client.search(
            index=self.index_name,
            body=search_body
        )

        results = []
        for hit in response["hits"]["hits"]:
            content = hit["_source"]["content"]
            score = hit["_score"]
            results.append((content, float(score)))

        return results

    def save(self, path: str):
        """Save retriever configuration.

        Args:
            path: Path to save configuration
        """
        config = {
            "type": "elasticsearch",
            "index_name": self.index_name,
            "model_name": self.model_name,
            "text_boost": self.text_boost,
            "vector_boost": self.vector_boost,
            "hosts": getattr(self.client, '_hosts', None),
            "cloud_id": getattr(self.client, '_cloud_id', None)
        }

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved Elasticsearch retriever config to {path}")

    @classmethod
    def load(cls, path: str) -> "ElasticsearchRetriever":
        """Load retriever from configuration.

        Args:
            path: Path to configuration file

        Returns:
            Loaded ElasticsearchRetriever instance
        """
        with open(path, 'r') as f:
            config = json.load(f)

        if config.get("type") != "elasticsearch":
            raise ValueError(f"Invalid config type: {config.get('type')}")

        return cls(
            index_name=config["index_name"],
            hosts=config.get("hosts"),
            cloud_id=config.get("cloud_id"),
            embedding_model=config["model_name"],
            text_boost=config.get("text_boost", 1.0),
            vector_boost=config.get("vector_boost", 1.0)
        )

    def delete_index(self):
        """Delete the Elasticsearch index."""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            logger.info(f"Deleted Elasticsearch index: {self.index_name}")

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        stats = self.client.indices.stats(index=self.index_name)
        return stats


class NumpyElasticsearchFallback(DenseRetriever):
    """Fallback retriever using numpy when Elasticsearch is not available."""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        """Initialize fallback retriever."""
        super().__init__(model_name=embedding_model, **kwargs)
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []

    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add documents to the retriever."""
        if not documents:
            return

        # Generate embeddings
        new_embeddings = self.encode(documents)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

        logger.info(f"Added {len(documents)} documents to numpy fallback retriever")

    def build_index(self, documents: List[str]) -> None:
        """Build search index from documents."""
        self.add_documents(documents)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search for relevant documents using cosine similarity."""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # Generate query embedding
        query_emb = self.encode([query])[0]

        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))

        return results

    def save(self, path: str):
        """Save retriever state."""
        if self.embeddings is None:
            raise ValueError("No documents added to retriever")

        data = {
            "type": "numpy_elasticsearch_fallback",
            "model_name": self.model_name,
            "documents": self.documents,
            "embeddings": self.embeddings.tolist(),
            "metadata": self.metadata
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved numpy fallback retriever to {path}")

    @classmethod
    def load(cls, path: str) -> "NumpyElasticsearchFallback":
        """Load retriever from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        if data.get("type") != "numpy_elasticsearch_fallback":
            raise ValueError(f"Invalid data type: {data.get('type')}")

        retriever = cls(embedding_model=data["model_name"])
        retriever.documents = data["documents"]
        retriever.embeddings = np.array(data["embeddings"])
        retriever.metadata = data["metadata"]

        return retriever


def create_elasticsearch_retriever(**kwargs) -> Union[ElasticsearchRetriever, NumpyElasticsearchFallback]:
    """Factory function to create Elasticsearch retriever with fallback."""
    if ELASTICSEARCH_AVAILABLE:
        return ElasticsearchRetriever(**kwargs)
    else:
        logger.warning("Elasticsearch not available, using numpy fallback")
        return NumpyElasticsearchFallback(**kwargs)