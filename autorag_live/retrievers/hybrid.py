from typing import List

from . import bm25, dense


def hybrid_retrieve(query: str, corpus: List[str], k: int, bm25_weight: float = 0.5) -> List[str]:
    """
    Retrieves top-k documents from the corpus using a hybrid of BM25 and dense retrieval.
    """
    if not corpus:
        return []

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if k <= 0:
        raise ValueError("k must be positive")

    if not (0.0 <= bm25_weight <= 1.0):
        raise ValueError("bm25_weight must be between 0.0 and 1.0")

    # Placeholder implementation
    bm25_results = bm25.bm25_retrieve(query, corpus, k)
    dense_results = dense.dense_retrieve(query, corpus, k)

    # Simple interleaving for now
    results = []
    for i in range(k):
        if i % 2 == 0 and bm25_results:
            results.append(bm25_results.pop(0))
        elif dense_results:
            results.append(dense_results.pop(0))
    return results


class HybridRetriever:
    """Hybrid retriever combining BM25 and dense retrieval with optimized performance."""

    def __init__(
        self,
        bm25_weight: float = 0.5,
        dense_model_name: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True,
    ):
        if not (0.0 <= bm25_weight <= 1.0):
            raise ValueError("bm25_weight must be between 0.0 and 1.0")

        self.bm25_weight = bm25_weight
        self.dense_weight = 1.0 - bm25_weight

        # Initialize retrievers
        self.bm25_retriever = bm25.BM25Retriever()
        self.dense_retriever = dense.DenseRetriever(
            model_name=dense_model_name, cache_embeddings=cache_embeddings
        )

        self._is_initialized = False

    def add_documents(self, documents: List[str]) -> None:
        """Add documents to both retrievers."""
        from ..utils import monitor_performance

        with monitor_performance("HybridRetriever.add_documents", {"num_docs": len(documents)}):
            self.bm25_retriever.add_documents(documents)
            self.dense_retriever.add_documents(documents)
            self._is_initialized = True

    @property
    def is_initialized(self) -> bool:
        """Check if retriever is initialized."""
        return self._is_initialized

    def retrieve(self, query: str, k: int = 5) -> List[tuple]:
        """Retrieve documents using weighted combination of BM25 and dense retrieval."""

        from ..utils import monitor_performance

        if not self.is_initialized:
            raise ValueError("Retriever not initialized. Call add_documents() first.")

        with monitor_performance("HybridRetriever.retrieve", {"query_length": len(query), "k": k}):
            # Get results from both retrievers
            bm25_results = self.bm25_retriever.retrieve(query, k * 2)  # Get more candidates
            dense_results = self.dense_retriever.retrieve(query, k * 2)

            # Create score dictionaries for efficient lookup
            bm25_scores = {doc: score for doc, score in bm25_results}
            dense_scores = {doc: score for doc, score in dense_results}

            # Combine scores for all unique documents
            all_docs = set(bm25_scores.keys()) | set(dense_scores.keys())
            combined_scores = {}

            for doc in all_docs:
                bm25_score = bm25_scores.get(doc, 0.0)
                dense_score = dense_scores.get(doc, 0.0)
                combined_scores[doc] = (
                    self.bm25_weight * bm25_score + self.dense_weight * dense_score
                )

            # Sort by combined score and return top-k
            sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_docs[:k]

    def load(self, path: str) -> None:
        """Load retriever state from disk."""
        raise NotImplementedError("Hybrid retriever persistence not implemented")

    def save(self, path: str) -> None:
        """Save retriever state to disk."""
        raise NotImplementedError("Hybrid retriever persistence not implemented")
