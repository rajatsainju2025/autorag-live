from typing import List, Optional

import numpy as np

# Try to import heavy deps; fall back to simple embedding if unavailable
try:  # pragma: no cover - import guard
    from sentence_transformers import SentenceTransformer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except Exception:  # pragma: no cover - offline fallback
    SentenceTransformer = None  # type: ignore
    cosine_similarity = None  # type: ignore


def dense_retrieve(
    query: str, corpus: List[str], k: int, model_name: str = "all-MiniLM-L6-v2"
) -> List[str]:
    """
    Retrieves top-k documents from the corpus using a dense retriever.
    """
    if not corpus:
        return []

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if k <= 0:
        raise ValueError("k must be positive")

    if SentenceTransformer is not None and cosine_similarity is not None:
        model = SentenceTransformer(model_name)
        query_embedding = model.encode([query])
        corpus_embeddings = model.encode(corpus)
        sims = cosine_similarity(query_embedding, corpus_embeddings)[0]
    else:
        # Deterministic lightweight fallback: Jaccard on tokens as a proxy
        q_set = set(query.lower().split())
        sims = []
        for doc in corpus:
            d_set = set(doc.lower().split())
            inter = len(q_set & d_set)
            union = len(q_set | d_set) or 1
            sims.append(inter / union)
        sims = np.array(sims, dtype=float)

    top_k_indices = np.argsort(sims)[-k:][::-1]
    return [corpus[i] for i in top_k_indices]


class DenseRetriever:
    """Dense retriever implementation with caching and lazy loading."""

    # Global model cache to avoid reloading the same model
    _model_cache = {}
    _embedding_cache = {}  # Cache embeddings by (model_name, text) tuples

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_embeddings: bool = True):
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self.corpus: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.model = None
        self._is_initialized = False

    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the retriever's index."""
        from ..utils import monitor_performance

        with monitor_performance("DenseRetriever.add_documents", {"num_docs": len(documents)}):
            self.corpus = documents

            if SentenceTransformer is not None and cosine_similarity is not None:
                # Lazy load model
                if self.model_name not in DenseRetriever._model_cache:
                    DenseRetriever._model_cache[self.model_name] = SentenceTransformer(
                        self.model_name
                    )

                self.model = DenseRetriever._model_cache[self.model_name]

                # Check cache for embeddings
                cache_key = (self.model_name, tuple(documents))
                if self.cache_embeddings and cache_key in DenseRetriever._embedding_cache:
                    self.corpus_embeddings = DenseRetriever._embedding_cache[cache_key]
                else:
                    # Batch encode for better performance
                    self.corpus_embeddings = self.model.encode(
                        documents, batch_size=32, show_progress_bar=False
                    )
                    if self.cache_embeddings:
                        DenseRetriever._embedding_cache[cache_key] = self.corpus_embeddings
            else:
                # Fallback mode - no embeddings needed
                self.corpus_embeddings = None

            self._is_initialized = True

    @property
    def is_initialized(self) -> bool:
        """Check if retriever is initialized."""
        return self._is_initialized

    def retrieve(self, query: str, k: int = 5) -> List[tuple]:
        """Retrieve documents for a query."""
        from ..utils import monitor_performance

        if not self.is_initialized:
            raise ValueError("Retriever not initialized. Call add_documents() first.")

        with monitor_performance("DenseRetriever.retrieve", {"query_length": len(query), "k": k}):
            if (
                self.corpus_embeddings is not None
                and self.model is not None
                and cosine_similarity is not None
            ):
                # Check query embedding cache
                query_cache_key = (self.model_name, query)
                if self.cache_embeddings and query_cache_key in DenseRetriever._embedding_cache:
                    query_embedding = DenseRetriever._embedding_cache[query_cache_key]
                else:
                    query_embedding = self.model.encode(
                        [query], batch_size=1, show_progress_bar=False
                    )[0]
                    if self.cache_embeddings:
                        DenseRetriever._embedding_cache[query_cache_key] = query_embedding

                # Compute similarities
                sims = cosine_similarity(query_embedding.reshape(1, -1), self.corpus_embeddings)[0]
            else:
                # Fallback: Jaccard similarity
                q_set = set(query.lower().split())
                sims = []
                for doc in self.corpus:
                    d_set = set(doc.lower().split())
                    inter = len(q_set & d_set)
                    union = len(q_set | d_set) or 1
                    sims.append(inter / union)
                sims = np.array(sims, dtype=float)

            # Get top-k results
            effective_k = min(k, len(sims))
            top_indices = np.argsort(sims)[-effective_k:][::-1]

            results = []
            for idx in top_indices:
                results.append((self.corpus[idx], float(sims[idx])))

            return results

    def load(self, path: str) -> None:
        """Load retriever state from disk."""
        raise NotImplementedError("Dense retriever persistence not implemented")

    def save(self, path: str) -> None:
        """Save retriever state to disk."""
        raise NotImplementedError("Dense retriever persistence not implemented")

    @classmethod
    def clear_cache(cls):
        """Clear all cached models and embeddings."""
        cls._model_cache.clear()
        cls._embedding_cache.clear()
