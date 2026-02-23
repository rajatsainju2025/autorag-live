import asyncio
import logging
from typing import Any, Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Route:
    """
    A single route definition for the Semantic Router.
    """

    def __init__(self, name: str, utterances: List[str], handler: Optional[Callable] = None):
        """
        Initialize a Route.

        Args:
            name: The unique name of the route.
            utterances: A list of example queries that should trigger this route.
            handler: An optional function or coroutine to execute when this route is selected.
        """
        self.name = name
        self.utterances = utterances
        self.handler = handler
        self.embeddings: Optional[np.ndarray] = None


class SemanticRouter:
    """
    Semantic Router.

    This class routes incoming queries to different pipelines, tools, or agents
    based on the semantic similarity between the query and predefined example
    utterances for each route. This is faster and often more reliable than using
    an LLM for intent classification.
    """

    def __init__(self, embedding_model: Any, similarity_threshold: float = 0.8):
        """
        Initialize the SemanticRouter.

        Args:
            embedding_model: The model used to generate embeddings (must have an async `embed_documents` and `embed_query` method).
            similarity_threshold: The minimum cosine similarity required to trigger a route.
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.routes: List[Route] = []
        self._is_initialized = False

    def add_route(self, route: Route):
        """
        Add a new route to the router.
        """
        self.routes.append(route)
        self._is_initialized = False  # Require re-initialization to embed new utterances

    async def initialize(self):
        """
        Embed all utterances for all routes. Must be called before routing.
        """
        if not self.routes:
            logger.warning("No routes defined. SemanticRouter will not route anything.")
            return

        logger.info("Initializing SemanticRouter: embedding route utterances...")

        for route in self.routes:
            if route.embeddings is None:
                try:
                    if hasattr(self.embedding_model, "aembed_documents"):
                        embeddings = await self.embedding_model.aembed_documents(route.utterances)
                    elif hasattr(self.embedding_model, "embed_documents"):
                        loop = asyncio.get_running_loop()
                        embeddings = await loop.run_in_executor(
                            None, self.embedding_model.embed_documents, route.utterances
                        )
                    else:
                        raise ValueError(
                            "Embedding model must have 'aembed_documents' or 'embed_documents' method."
                        )

                    route.embeddings = np.array(embeddings)
                except Exception as e:
                    logger.error(f"Failed to embed utterances for route '{route.name}': {e}")
                    raise

        self._is_initialized = True
        logger.info("SemanticRouter initialization complete.")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    async def route(self, query: str) -> Optional[Route]:
        """
        Route a query to the most semantically similar route.

        Args:
            query: The user's input query.

        Returns:
            The matching Route object, or None if no route meets the similarity threshold.
        """
        if not self._is_initialized:
            await self.initialize()

        if not self.routes:
            return None

        try:
            # Embed the query
            if hasattr(self.embedding_model, "aembed_query"):
                query_embedding = await self.embedding_model.aembed_query(query)
            elif hasattr(self.embedding_model, "embed_query"):
                loop = asyncio.get_running_loop()
                query_embedding = await loop.run_in_executor(
                    None, self.embedding_model.embed_query, query
                )
            else:
                raise ValueError(
                    "Embedding model must have 'aembed_query' or 'embed_query' method."
                )

            query_vec = np.array(query_embedding)

            best_route = None
            max_similarity = -1.0

            # Compare query embedding against all route utterances
            for route in self.routes:
                if route.embeddings is not None:
                    for utterance_vec in route.embeddings:
                        similarity = self._cosine_similarity(query_vec, utterance_vec)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_route = route

            logger.debug(
                f"Query: '{query}' -> Best Route: '{best_route.name if best_route else None}' (Similarity: {max_similarity:.4f})"
            )

            if max_similarity >= self.similarity_threshold:
                return best_route
            else:
                logger.info(
                    f"No route matched query '{query}' above threshold {self.similarity_threshold}."
                )
                return None

        except Exception as e:
            logger.error(f"Error during semantic routing: {e}")
            return None
