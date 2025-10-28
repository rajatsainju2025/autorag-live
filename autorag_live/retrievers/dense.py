import os
import pickle
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from ..types.types import DocumentText, QueryText, RetrievalResult, RetrieverError
from ..utils import get_logger
from .base import BaseRetriever

logger = get_logger(__name__)

# Try to import heavy deps; fall back to simple embedding if unavailable
try:  # pragma: no cover - import guard
    # Version checks for better compatibility
    import sentence_transformers
    import sklearn
    from sentence_transformers import SentenceTransformer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.debug(
        f"SentenceTransformers {sentence_transformers.__version__} and sklearn {sklearn.__version__} available"
    )
except ImportError as e:  # pragma: no cover - offline fallback
    logger.warning(f"Optional dependencies not available: {e}. Using fallback mode.")
    SentenceTransformer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
except Exception as e:  # pragma: no cover - unexpected error
    logger.error(f"Unexpected error importing dependencies: {e}. Using fallback mode.")
    SentenceTransformer = None  # type: ignore
    cosine_similarity = None  # type: ignore
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


class TTLCache:
    """Thread-safe TTL cache with size-based eviction."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):  # 1 hour default TTL
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[Any, float] = {}
        self.lock = threading.RLock()

    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache if it exists and hasn't expired."""
        with self.lock:
            self._cleanup_expired()
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key: Any, value: Any) -> None:
        """Put value in cache with current timestamp."""
        with self.lock:
            self._cleanup_expired()

            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]

            # Evict oldest if at capacity (after cleanup)
            while len(self.cache) >= self.max_size:
                oldest_key, _ = self.cache.popitem(last=False)
                del self.timestamps[oldest_key]

            # Add new item
            self.cache[key] = value
            self.timestamps[key] = time.time()

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key
            for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]

    def __len__(self) -> int:
        """Return number of valid (non-expired) entries."""
        with self.lock:
            self._cleanup_expired()
            return len(self.cache)

    def __contains__(self, key: Any) -> bool:
        """Check if key exists in cache (dict-like interface)."""
        with self.lock:
            self._cleanup_expired()
            return key in self.cache

    def __getitem__(self, key: Any) -> Any:
        """Get item with dict-like syntax (raises KeyError if not found)."""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set item with dict-like syntax."""
        self.put(key, value)


def dense_retrieve(
    query: str, corpus: List[str], k: int, model_name: str = "all-MiniLM-L6-v2"
) -> List[str]:
    """
    Retrieves top-k documents from the corpus using a dense retriever.
    """
    if not corpus:
        return []

    if not query or not query.strip():
        raise RetrieverError("Query cannot be empty")

    if k <= 0:
        raise RetrieverError("k must be positive")

    if SentenceTransformer is not None and cosine_similarity is not None:
        try:
            model = SentenceTransformer(model_name)
            query_embedding = model.encode([query])[0]
            corpus_embeddings = model.encode(corpus)

            # Optimized similarity computation
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            corpus_norms = corpus_embeddings / np.linalg.norm(
                corpus_embeddings, axis=1, keepdims=True
            )
            sims = np.dot(corpus_norms, query_norm)
        except Exception as e:
            logger.error(f"Error during dense retrieval with model {model_name}: {e}")
            raise RetrieverError(f"Dense retrieval failed: {e}")
    else:
        # Improved fallback: TF-IDF like scoring
        query_terms = query.lower().split()
        if not query_terms:
            sims = np.zeros(len(corpus), dtype=float)
        else:
            sims = []
            for doc in corpus:
                doc_terms = doc.lower().split()
                if not doc_terms:
                    sims.append(0.0)
                    continue

                # Calculate term frequency similarity
                query_tf = {}
                for term in query_terms:
                    query_tf[term] = query_tf.get(term, 0) + 1

                doc_tf = {}
                for term in doc_terms:
                    doc_tf[term] = doc_tf.get(term, 0) + 1

                # Compute cosine similarity between TF vectors
                common_terms = set(query_tf.keys()) & set(doc_tf.keys())
                if not common_terms:
                    sims.append(0.0)
                else:
                    # Dot product of TF vectors
                    dot_product = sum(query_tf[term] * doc_tf[term] for term in common_terms)
                    # Magnitudes
                    query_mag = sum(tf**2 for tf in query_tf.values()) ** 0.5
                    doc_mag = sum(tf**2 for tf in doc_tf.values()) ** 0.5

                    similarity = (
                        dot_product / (query_mag * doc_mag) if query_mag * doc_mag > 0 else 0.0
                    )
                    sims.append(similarity)

            sims = np.array(sims, dtype=float)

    top_k_indices = np.argsort(sims)[-k:][::-1]
    return [corpus[i] for i in top_k_indices]


class DenseRetriever(BaseRetriever):
    """Dense retriever implementation with caching and lazy loading."""

    # Global caches with TTL and size limits
    _model_cache: dict = {}  # Simple dict for models (no TTL needed)
    _embedding_cache = TTLCache(max_size=100, ttl_seconds=3600)  # 1 hour TTL, 100 items max

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True,
        batch_size: int = 32,
    ):
        super().__init__()
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self.batch_size = batch_size
        self.corpus: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None
        self._corpus_embeddings_normalized: Optional[np.ndarray] = None  # Lazy normalized cache
        self.model: Optional[Any] = None

    def add_documents(self, documents: List[DocumentText]) -> None:
        """Add documents to the retriever's index."""
        from ..utils import monitor_performance

        if not documents:
            raise RetrieverError("Documents list cannot be empty")

        if not all(isinstance(doc, str) and doc.strip() for doc in documents):
            raise RetrieverError("All documents must be non-empty strings")

        with monitor_performance("DenseRetriever.add_documents", {"num_docs": len(documents)}):
            self.corpus = documents
            self._corpus_embeddings_normalized = None  # Reset normalized cache

            if SentenceTransformer is not None and cosine_similarity is not None:
                # Lazy load model
                if self.model_name not in DenseRetriever._model_cache:
                    try:
                        DenseRetriever._model_cache[self.model_name] = SentenceTransformer(
                            self.model_name
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to load SentenceTransformer model {self.model_name}: {e}"
                        )
                        raise RetrieverError(f"Model loading failed: {e}")

                self.model = DenseRetriever._model_cache[self.model_name]

                # Check cache for embeddings
                cache_key = (self.model_name, tuple(documents))
                if self.cache_embeddings:
                    cached_embeddings = DenseRetriever._embedding_cache.get(cache_key)
                    if cached_embeddings is not None:
                        self.corpus_embeddings = cached_embeddings
                    else:
                        # Batch encode for better performance
                        if self.model is not None:
                            try:
                                self.corpus_embeddings = self.model.encode(
                                    documents,
                                    batch_size=self.batch_size,
                                    show_progress_bar=False,
                                )
                                DenseRetriever._embedding_cache.put(
                                    cache_key, self.corpus_embeddings
                                )
                            except Exception as e:
                                logger.error(f"Failed to encode documents: {e}")
                                raise RetrieverError(f"Document encoding failed: {e}")
                else:
                    # Batch encode for better performance (no caching)
                    if self.model is not None:
                        try:
                            self.corpus_embeddings = self.model.encode(
                                documents,
                                batch_size=self.batch_size,
                                show_progress_bar=False,
                            )
                        except Exception as e:
                            logger.error(f"Failed to encode documents: {e}")
                            raise RetrieverError(f"Document encoding failed: {e}")
            else:
                # Fallback mode - no embeddings needed
                self.corpus_embeddings = None

            self._is_initialized = True

    def _get_normalized_corpus_embeddings(self) -> np.ndarray:
        """Lazily compute and cache normalized corpus embeddings.

        This avoids redundant normalization on every query by caching
        the normalized embeddings after first computation.

        Returns:
            np.ndarray: Normalized corpus embeddings (L2 norm = 1 for each vector)

        Raises:
            RetrieverError: If corpus embeddings are not initialized
        """
        if self._corpus_embeddings_normalized is None:
            if self.corpus_embeddings is None:
                raise RetrieverError("Corpus embeddings not initialized")

            # Normalize once and cache
            self._corpus_embeddings_normalized = self.corpus_embeddings / np.linalg.norm(
                self.corpus_embeddings, axis=1, keepdims=True
            )

        # Type checker needs explicit assert after None check
        assert self._corpus_embeddings_normalized is not None
        return self._corpus_embeddings_normalized

    def retrieve(self, query: QueryText, k: int = 5) -> RetrievalResult:
        """Retrieve documents for a query."""
        from ..utils import monitor_performance

        if not self.is_initialized:
            raise RetrieverError("Retriever not initialized. Call add_documents() first.")

        with monitor_performance("DenseRetriever.retrieve", {"query_length": len(query), "k": k}):
            if (
                self.corpus_embeddings is not None
                and self.model is not None
                and cosine_similarity is not None
            ):
                # Check query embedding cache
                query_cache_key = (self.model_name, query)
                if self.cache_embeddings:
                    cached_query_embedding = DenseRetriever._embedding_cache.get(query_cache_key)
                    if cached_query_embedding is not None:
                        query_embedding = cached_query_embedding
                    else:
                        try:
                            query_embedding = self.model.encode(
                                [query], batch_size=1, show_progress_bar=False
                            )[0]
                            DenseRetriever._embedding_cache.put(query_cache_key, query_embedding)
                        except Exception as e:
                            logger.error(f"Failed to encode query '{query}': {e}")
                            raise RetrieverError(f"Query encoding failed: {e}")
                else:
                    try:
                        query_embedding = self.model.encode(
                            [query], batch_size=1, show_progress_bar=False
                        )[0]
                    except Exception as e:
                        logger.error(f"Failed to encode query '{query}': {e}")
                        raise RetrieverError(f"Query encoding failed: {e}")

                # Compute similarities using optimized numpy operations
                try:
                    # Normalize query embedding only (corpus already normalized via lazy cache)
                    query_norm = query_embedding / np.linalg.norm(query_embedding)
                    corpus_norms = self._get_normalized_corpus_embeddings()

                    # Compute cosine similarities using dot product (faster than sklearn)
                    sims = np.dot(corpus_norms, query_norm)
                except Exception as e:
                    logger.error(f"Failed to compute similarities: {e}")
                    raise RetrieverError(f"Similarity computation failed: {e}")
            else:
                # Fallback: TF-IDF like scoring for better similarity
                query_terms = query.lower().split()
                if not query_terms:
                    sims = np.zeros(len(self.corpus), dtype=float)
                else:
                    sims = []
                    for doc in self.corpus:
                        doc_terms = doc.lower().split()
                        if not doc_terms:
                            sims.append(0.0)
                            continue

                        # Calculate term frequency similarity
                        query_tf = {}
                        for term in query_terms:
                            query_tf[term] = query_tf.get(term, 0) + 1

                        doc_tf = {}
                        for term in doc_terms:
                            doc_tf[term] = doc_tf.get(term, 0) + 1

                        # Compute cosine similarity between TF vectors
                        common_terms = set(query_tf.keys()) & set(doc_tf.keys())
                        if not common_terms:
                            sims.append(0.0)
                        else:
                            # Dot product of TF vectors
                            dot_product = sum(
                                query_tf[term] * doc_tf[term] for term in common_terms
                            )
                            # Magnitudes
                            query_mag = sum(tf**2 for tf in query_tf.values()) ** 0.5
                            doc_mag = sum(tf**2 for tf in doc_tf.values()) ** 0.5

                            similarity = (
                                dot_product / (query_mag * doc_mag)
                                if query_mag * doc_mag > 0
                                else 0.0
                            )
                            sims.append(similarity)

                sims = np.array(sims, dtype=float)

            # Get top-k results
            effective_k = min(k, len(sims))
            top_indices = np.argsort(sims)[-effective_k:][::-1]

            results = []
            for idx in top_indices:
                results.append((self.corpus[idx], float(sims[idx])))

            return results

    def retrieve_batch(self, queries: List[QueryText], k: int = 5) -> List[RetrievalResult]:
        """Retrieve documents for multiple queries efficiently.

        Args:
            queries: List of query strings
            k: Number of documents to retrieve per query

        Returns:
            List of retrieval results, one per query

        Example:
            queries = ["query 1", "query 2", "query 3"]
            results_batch = retriever.retrieve_batch(queries, k=5)
        """
        from ..utils import monitor_performance

        if not self.is_initialized:
            raise RetrieverError("Retriever not initialized. Call add_documents() first.")

        with monitor_performance(
            "DenseRetriever.retrieve_batch",
            {"num_queries": len(queries), "k": k},
        ):
            if (
                self.corpus_embeddings is not None
                and self.model is not None
                and cosine_similarity is not None
            ):
                # Batch encode all queries at once for efficiency
                try:
                    query_embeddings = self.model.encode(
                        queries, batch_size=self.batch_size, show_progress_bar=False
                    )
                except Exception as e:
                    logger.error(f"Failed to encode queries: {e}")
                    raise RetrieverError(f"Query encoding failed: {e}")

                # Normalize query embeddings only (corpus already normalized via lazy cache)
                query_norms = query_embeddings / np.linalg.norm(
                    query_embeddings, axis=1, keepdims=True
                )
                corpus_norms = self._get_normalized_corpus_embeddings()

                # Compute similarities for all queries at once (matrix multiplication)
                # Shape: (num_queries, num_docs)
                all_sims = np.dot(query_norms, corpus_norms.T)

                # Get top-k for each query
                results_batch = []
                for query_idx, sims in enumerate(all_sims):
                    effective_k = min(k, len(sims))
                    top_indices = np.argsort(sims)[-effective_k:][::-1]

                    query_results = []
                    for idx in top_indices:
                        query_results.append((self.corpus[idx], float(sims[idx])))
                    results_batch.append(query_results)

                return results_batch
            else:
                # Fallback: process queries individually
                return [self.retrieve(query, k) for query in queries]

    def load(self, path: str, mmap_mode: Optional[Literal["r", "r+", "w+", "c"]] = "r") -> None:
        """Load retriever state from disk.

        Args:
            path: Path to the saved retriever state
            mmap_mode: Memory-map mode for loading embeddings. Options:
                      - None: Load into memory (default for non-mmap saves)
                      - 'r': Read-only memory-mapped (recommended for large files)
                      - 'r+': Read-write memory-mapped
                      - 'w+': Write mode memory-mapped
                      - 'c': Copy-on-write memory-mapped
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Retriever state file not found: {path}")

        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            # Validate state structure
            required_keys = [
                "model_name",
                "corpus",
                "cache_embeddings",
            ]
            if not all(key in state for key in required_keys):
                raise ValueError("Invalid state file: missing required keys")

            # Restore state
            self.model_name = state["model_name"]
            self.cache_embeddings = state["cache_embeddings"]
            self.corpus = state["corpus"]

            # Handle memory-mapped or regular embeddings
            if state.get("use_mmap", False) and "embeddings_path" in state:
                # Load embeddings via memory-mapping
                embeddings_path = state["embeddings_path"]
                if not os.path.exists(embeddings_path):
                    raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

                self.corpus_embeddings = np.load(embeddings_path, mmap_mode=mmap_mode)
                logger.info(f"Loaded embeddings via memory-mapping (mode={mmap_mode})")
            else:
                # Regular pickle loading
                self.corpus_embeddings = state.get("corpus_embeddings")

            self._corpus_embeddings_normalized = None  # Reset normalized cache on load

            # Recreate model if embeddings exist (lazy loading)
            if self.corpus_embeddings is not None and SentenceTransformer is not None:
                if self.model_name not in DenseRetriever._model_cache:
                    DenseRetriever._model_cache[self.model_name] = SentenceTransformer(
                        self.model_name
                    )
                self.model = DenseRetriever._model_cache[self.model_name]

            self._is_initialized = True
            logger.info(f"Loaded retriever state from {path}")

        except Exception as e:
            logger.error(f"Failed to load retriever state from {path}: {e}")
            raise RetrieverError(f"Failed to load retriever state: {e}")

    def save(self, path: str, use_mmap: bool = False) -> None:
        """Save retriever state to disk.

        Args:
            path: Path to save the retriever state
            use_mmap: If True, save embeddings separately for memory-mapped loading.
                     Useful for large corpora (> 100MB embeddings).
        """
        if not self.is_initialized:
            raise RetrieverError("Cannot save uninitialized retriever")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            embeddings_path = None
            if use_mmap and self.corpus_embeddings is not None:
                # Save embeddings separately for memory-mapped loading
                embeddings_path = path.replace(".pkl", "_embeddings.npy")
                np.save(embeddings_path, self.corpus_embeddings)

                # Save state without embeddings
                state = {
                    "model_name": self.model_name,
                    "corpus": self.corpus,
                    "corpus_embeddings": None,  # Not saved in pickle
                    "cache_embeddings": self.cache_embeddings,
                    "use_mmap": True,
                    "embeddings_path": embeddings_path,
                }
            else:
                # Traditional pickle save with embeddings
                state = {
                    "model_name": self.model_name,
                    "corpus": self.corpus,
                    "corpus_embeddings": self.corpus_embeddings,
                    "cache_embeddings": self.cache_embeddings,
                    "use_mmap": False,
                }

            with open(path, "wb") as f:
                pickle.dump(state, f)

            log_msg = f"Saved retriever state to {path}"
            if embeddings_path:
                log_msg += f" (mmap mode: {embeddings_path})"
            logger.info(log_msg)

        except Exception as e:
            logger.error(f"Failed to save retriever state to {path}: {e}")
            raise RetrieverError(f"Failed to save retriever state: {e}")

    @classmethod
    def clear_cache(cls):
        """Clear all cached models and embeddings."""
        cls._model_cache.clear()
        cls._embedding_cache.clear()
