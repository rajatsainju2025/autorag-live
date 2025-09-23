"""Qdrant adapter for dense retrieval with vector database support."""

import json
import logging
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None
    Distance = None
    VectorParams = None
    QDRANT_AVAILABLE = False

from autorag_live.retrievers.faiss_adapter import DenseRetriever
from autorag_live.types.types import RetrieverError

logger = logging.getLogger(__name__)


class QdrantRetriever(DenseRetriever):
    """Dense retriever using Qdrant vector database."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "autorag_docs",
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        distance_metric: Literal["cosine", "euclid", "dot"] = "cosine"
    ) -> None:
        """Initialize Qdrant retriever.

        Args:
            model_name: Sentence transformer model name
            collection_name: Name of the Qdrant collection
            host: Qdrant server host
            port: Qdrant server port
            url: Full Qdrant URL (alternative to host/port)
            api_key: Qdrant API key for cloud instances
            distance_metric: Distance metric (cosine, euclid, dot)
        """
        super().__init__(model_name)

        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant is not installed. Install with: pip install qdrant-client"
            )

        self.collection_name = collection_name
        self.distance_metric = distance_metric

        # Initialize Qdrant client
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)  # type: ignore
        else:
            self.client = QdrantClient(host=host, port=port, api_key=api_key)  # type: ignore

        # Map distance metrics
        if QDRANT_AVAILABLE:
            distance_map = {
                "cosine": Distance.COSINE,  # type: ignore
                "euclid": Distance.EUCLID,  # type: ignore
                "dot": Distance.DOT  # type: ignore
            }
            self.qdrant_distance = distance_map.get(distance_metric, Distance.COSINE)  # type: ignore
        else:
            self.qdrant_distance = None

        # Track if collection exists
        self.collection_created = False

    def _ensure_collection(self, dimension: int):
        """Ensure collection exists with correct configuration."""
        if not self.collection_created:
            try:
                # Check if collection exists
                collections = self.client.get_collections().collections
                collection_names = [c.name for c in collections]

                if self.collection_name not in collection_names:
                    # Create collection
                    if QDRANT_AVAILABLE and self.qdrant_distance is not None:
                        self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(  # type: ignore
                                size=dimension,
                                distance=self.qdrant_distance
                            )
                        )
                    else:
                        raise RuntimeError("Qdrant not available or distance metric not set")
                    logger.info(f"Created Qdrant collection: {self.collection_name}")
                else:
                    logger.info(f"Using existing Qdrant collection: {self.collection_name}")

                self.collection_created = True

            except Exception as e:
                logger.error(f"Failed to create/access Qdrant collection: {e}")
                raise

    def build_index(self, documents: List[str]) -> None:
        """Build search index from documents."""
        if not documents:
            return

        # Generate embeddings
        embeddings = self.encode(documents)
        dimension = embeddings.shape[1]

        # Ensure collection exists
        self._ensure_collection(dimension)

        # Prepare points
        points = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            point = {
                "id": i,
                "vector": emb.tolist(),
                "payload": {
                    "text": doc
                }
            }
            points.append(point)

        # Upsert points
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        # Store documents for compatibility
        self.documents = documents

        logger.info(f"Added {len(points)} documents to Qdrant collection")

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document, score) tuples
        """
        try:
            # Generate query embedding
            query_emb = self.encode([query])[0]

            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_emb.tolist() if hasattr(query_emb, 'tolist') else query_emb,
                limit=k
            )

            # Format results
            results = []
            for hit in search_result:
                if hit.payload:
                    text = hit.payload.get("text", "")
                    score = hit.score
                    results.append((text, float(score)))

            return results
        except Exception as e:
            raise RetrieverError(f"Search failed: {str(e)}") from e

    def save(self, path: str):
        """Save retriever configuration (not the actual vectors).

        Args:
            path: Path to save configuration to
        """
        try:
            # Extract client configuration safely
            qdrant_config = {}
            if hasattr(self.client, '_host') and not hasattr(getattr(self.client, '_host', None), '__class__'):
                qdrant_config["host"] = getattr(self.client, '_host', None)
            if hasattr(self.client, '_port') and not hasattr(getattr(self.client, '_port', None), '__class__'):
                qdrant_config["port"] = getattr(self.client, '_port', None)
            if hasattr(self.client, '_url') and not hasattr(getattr(self.client, '_url', None), '__class__'):
                qdrant_config["url"] = getattr(self.client, '_url', None)
            if hasattr(self.client, '_api_key') and not hasattr(getattr(self.client, '_api_key', None), '__class__'):
                qdrant_config["api_key"] = getattr(self.client, '_api_key', None)

            config = {
                "type": "qdrant",
                "model_name": self.model_name,
                "collection_name": self.collection_name,
                "distance_metric": self.distance_metric,
                "qdrant_config": qdrant_config
            }

            with open(path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Saved Qdrant retriever config to {path}")
        except Exception as e:
            raise RetrieverError(f"Failed to save retriever config: {str(e)}") from e

    @classmethod
    def load(cls, path: str) -> "QdrantRetriever":
        """Load retriever from configuration.

        Args:
            path: Path to configuration file

        Returns:
            Loaded QdrantRetriever instance
        """
        with open(path, 'r') as f:
            config = json.load(f)

        if config.get("type") != "qdrant":
            raise ValueError(f"Invalid config type: {config.get('type')}")

        qdrant_config = config.get("qdrant_config", {})

        return cls(
            model_name=config["model_name"],
            collection_name=config["collection_name"],
            distance_metric=config["distance_metric"],
            host=qdrant_config.get("host"),
            port=qdrant_config.get("port"),
            url=qdrant_config.get("url"),
            api_key=qdrant_config.get("api_key")
        )