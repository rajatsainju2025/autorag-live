from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Tuple, cast

import numpy as np

# Try to import FAISS; fall back to numpy-based implementation if not available
try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None  # type: ignore

from ..types.types import DocumentText, QueryText, RetrievalResult, RetrieverError
from ..utils import cached, get_logger, monitor_performance
from .base import BaseRetriever

logger = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """Base class for dense retrievers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self.model_name = model_name
        self.index = None
        self.documents = []
        self.embeddings = None
        self.embedding_dim = 384
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._normalized_embeddings = cast(Optional[np.ndarray], None)
        self._embedding_norms = cast(Optional[np.ndarray], None)

    @cached(cache_name="embeddings", ttl=3600)
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings. Override in subclasses."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings: List[np.ndarray] = []
        for text in texts:
            cached_embedding = self._embedding_cache.get(text)
            if cached_embedding is None:
                computed = self._compute_fallback_embedding(text)
                self._embedding_cache[text] = computed
                cached_embedding = computed
            embeddings.append(cached_embedding)

        return np.vstack(embeddings)

    def _compute_fallback_embedding(self, text: str) -> np.ndarray:
        """Create a deterministic, normalized embedding for offline fallback."""
        bytes_needed = self.embedding_dim * 4  # float32 bytes
        buffer = bytearray()
        salt = 0
        while len(buffer) < bytes_needed:
            message = f"{text}:{salt}".encode("utf-8")
            buffer.extend(hashlib.sha256(message).digest())
            salt += 1

        ints = np.frombuffer(buffer[:bytes_needed], dtype=np.uint32)
        floats = (ints.astype(np.float32) / np.float32(2**31)) - np.float32(1.0)
        vector = floats.astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector
        return vector / norm

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize rows of a matrix, guarding against zero vectors."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return matrix / norms

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a 2D vector array with shape (1, dim)."""
        norms = np.linalg.norm(vector, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return vector / norms

    def build_index(self, documents: List[str]) -> None:
        """Build search index from documents."""
        with monitor_performance("DenseRetriever.build_index", {"num_docs": len(documents)}):
            self.documents = documents
            # Ensure a float32 contiguous view for downstream libraries
            self.embeddings = np.ascontiguousarray(self.encode(documents).astype(np.float32))
            self._normalized_embeddings = None
            self._embedding_norms = None

            if FAISS_AVAILABLE and self.embeddings is not None:
                # Use FAISS for efficient search
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # type: ignore[attr-defined]  # Inner product (cosine)
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(self.embeddings)  # type: ignore[attr-defined]
                self.index.add(self.embeddings)
            else:
                # Fallback to numpy-based search
                self.index = "numpy"
                self._ensure_normalized_embeddings()

            self._is_initialized = True

    def retrieve(self, query: QueryText, k: int = 5) -> RetrievalResult:
        """Retrieve documents for a query."""
        if not self.is_initialized:
            raise RetrieverError("Retriever not initialized. Call build_index() first.")

        with monitor_performance("DenseRetriever.retrieve", {"query_length": len(query), "k": k}):
            return self.search(query, k)

    def add_documents(self, documents: List[DocumentText]) -> None:
        """Add documents to the retriever's index."""
        if not self.is_initialized:
            self.build_index(documents)
        else:
            # Rebuild index with new documents
            all_docs = self.documents + documents
            self.build_index(all_docs)

    def load(self, path: str) -> None:
        """Load retriever state from disk."""
        try:
            loaded = load_retriever_index(path)
            self.model_name = loaded.model_name
            self.index = loaded.index
            self.documents = loaded.documents
            self.embeddings = loaded.embeddings
            self._is_initialized = True
        except Exception as e:
            raise RetrieverError(f"Failed to load retriever: {e}")

    def save(self, path: str) -> None:
        """Save retriever state to disk."""
        try:
            save_retriever_index(self, path)
        except Exception as e:
            raise RetrieverError(f"Failed to save retriever: {e}")

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for most similar documents."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_embedding = self.encode([query])

        if FAISS_AVAILABLE and hasattr(self.index, "search"):  # type: ignore
            # FAISS search
            faiss.normalize_L2(query_embedding)  # type: ignore[attr-defined]
            scores, indices = self.index.search(query_embedding.astype("float32"), k)  # type: ignore
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))
            return results
        else:
            # Numpy fallback
            if self.embeddings is not None:
                self._ensure_normalized_embeddings()
                if self._normalized_embeddings is None:
                    return []
                assert self._normalized_embeddings is not None

                # Cosine similarity using cached normalized matrices
                query_norm = self._normalize_vector(query_embedding)
                similarities = np.dot(self._normalized_embeddings, query_norm.T).flatten()

                # Get top k
                top_indices = np.argsort(similarities)[-k:][::-1]
                results: List[Tuple[str, float]] = []
                for idx in top_indices:
                    results.append((self.documents[idx], float(similarities[idx])))
                return results
            return []

    def _ensure_normalized_embeddings(self) -> None:
        """Ensure cached normalized embeddings and norms are available for numpy fallback."""
        if self.embeddings is None:
            self._normalized_embeddings = None
            self._embedding_norms = None
            return

        if self._normalized_embeddings is not None and self._embedding_norms is not None:
            return

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=False)
        if norms.size == 0:
            self._embedding_norms = np.array([], dtype=np.float32)
            self._normalized_embeddings = self.embeddings.copy()
            return

        norms = np.where(norms == 0.0, 1.0, norms)
        self._embedding_norms = norms.astype(np.float32)
        self._normalized_embeddings = self.embeddings / norms[:, np.newaxis]


class SentenceTransformerRetriever(DenseRetriever):
    """Dense retriever using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.model = None

        # Try to load sentence-transformers model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(model_name)
        except ImportError:
            logger.warning("sentence-transformers not available. Using deterministic fallback.")

    @cached(cache_name="embeddings", ttl=3600)
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using sentence-transformers or fallback."""
        if self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            # Deterministic fallback
            return super().encode(texts)


def create_dense_retriever(
    retriever_type: str = "sentence-transformer", **kwargs
) -> DenseRetriever:
    """Factory function for dense retrievers."""
    if retriever_type == "sentence-transformer":
        return SentenceTransformerRetriever(**kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def save_retriever_index(retriever: DenseRetriever, path: str) -> None:
    """Save retriever index and documents."""
    os.makedirs(path, exist_ok=True)

    # Save documents
    with open(os.path.join(path, "documents.txt"), "w") as f:
        for doc in retriever.documents:
            f.write(doc + "\n")

    # Save embeddings if available
    if retriever.embeddings is not None:
        np.save(os.path.join(path, "embeddings.npy"), retriever.embeddings)

    # Save FAISS index if available
    if FAISS_AVAILABLE and hasattr(retriever.index, "write"):  # type: ignore
        faiss.write_index(retriever.index, os.path.join(path, "faiss.index"))  # type: ignore

    # Save config
    config = {
        "model_name": retriever.model_name,
        "retriever_type": retriever.__class__.__name__,
        "num_documents": len(retriever.documents),
        "faiss_available": FAISS_AVAILABLE,
    }
    import json

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_retriever_index(path: str) -> DenseRetriever:
    """Load retriever index and documents."""
    import json

    # Load config
    with open(os.path.join(path, "config.json"), "r") as f:
        config = json.load(f)

    # Create retriever
    retriever = create_dense_retriever(
        retriever_type="sentence-transformer", model_name=config["model_name"]
    )

    # Load documents
    with open(os.path.join(path, "documents.txt"), "r") as f:
        retriever.documents = [line.strip() for line in f if line.strip()]

    # Load embeddings
    embeddings_path = os.path.join(path, "embeddings.npy")
    if os.path.exists(embeddings_path):
        retriever.embeddings = np.load(embeddings_path)

    # Load FAISS index
    faiss_path = os.path.join(path, "faiss.index")
    if FAISS_AVAILABLE and os.path.exists(faiss_path):
        retriever.index = faiss.read_index(faiss_path)  # type: ignore
    else:
        retriever.index = "numpy"

    return retriever
