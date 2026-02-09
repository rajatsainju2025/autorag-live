from typing import Dict, List, Optional, Tuple

import numpy as np

from ..types.types import RetrieverError
from . import bm25, dense


def hybrid_retrieve(query: str, corpus: List[str], k: int, bm25_weight: float = 0.5) -> List[str]:
    """Retrieve top-k documents using hybrid BM25 and dense retrieval.

    Uses equal contribution strategy for fast, balanced results.

    Args:
        query: The search query string.
        corpus: List of document strings to search.
        k: Number of top documents to retrieve.
        bm25_weight: Weight for BM25 scores (0.0 to 1.0), dense weight = 1 - bm25_weight.

    Returns:
        List of top-k documents sorted by combined relevance score.

    Raises:
        RetrieverError: If query is empty, k is invalid, or weight is out of range.
    """
    if not corpus:
        return []

    if not query or not query.strip():
        raise RetrieverError("Query cannot be empty", context={"query_length": len(query)})

    if k <= 0:
        raise RetrieverError(f"k must be positive, got {k}", context={"k": k})

    if not (0.0 <= bm25_weight <= 1.0):
        raise RetrieverError(
            f"bm25_weight must be between 0.0 and 1.0, got {bm25_weight}",
            context={"bm25_weight": bm25_weight},
        )

    dense_weight = 1.0 - bm25_weight

    # Get results from both retrievers (2x k for flexibility)
    bm25_results = bm25.bm25_retrieve(query, corpus, min(k * 2, len(corpus)))
    dense_results = dense.dense_retrieve(query, corpus, min(k * 2, len(corpus)))

    # Score each document based on position in both rankings
    scores: Dict[str, float] = {}

    # Assign scores based on ranking position (inverse ranking)
    # Lower position = higher score
    for i, doc in enumerate(bm25_results):
        if doc not in scores:
            scores[doc] = 0.0
        scores[doc] += bm25_weight * (1.0 - i / len(bm25_results))

    for i, doc in enumerate(dense_results):
        if doc not in scores:
            scores[doc] = 0.0
        scores[doc] += dense_weight * (1.0 - i / len(dense_results))

    # Return top-k by combined score
    top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [doc for doc, _ in top_docs]


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

            # Normalize BM25 scores to [0, 1] range for fair combination
            if bm25_scores:
                max_bm25 = max(bm25_scores.values())
                if max_bm25 > 0:
                    bm25_scores = {doc: score / max_bm25 for doc, score in bm25_scores.items()}

            # Optimize: Use dict union operator for Python 3.9+ or faster iteration
            combined_scores = {}

            # Process BM25 scores first
            for doc, bm25_score in bm25_scores.items():
                dense_score = dense_scores.get(doc, 0.0)
                combined_scores[doc] = (
                    self.bm25_weight * bm25_score + self.dense_weight * dense_score
                )

            # Process remaining dense-only documents
            for doc, dense_score in dense_scores.items():
                if doc not in combined_scores:
                    combined_scores[doc] = self.dense_weight * dense_score

            # Sort by combined score and return top-k
            sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_docs[:k]

    def load(self, path: str) -> None:
        """Load retriever state from disk."""
        raise NotImplementedError("Hybrid retriever persistence not implemented")

    def save(self, path: str) -> None:
        """Save retriever state to disk."""
        raise NotImplementedError("Hybrid retriever persistence not implemented")


class AdaptiveFusionRetriever(HybridRetriever):
    """
    Hybrid retriever with online adaptive weight learning.

    Dynamically adjusts BM25/dense fusion weights based on per-query
    feedback using an exponential moving average (EMA) of relevance signals.

    State-of-the-art RAG systems don't use fixed fusion weights — they adapt
    based on query characteristics and retrieval quality feedback.

    Methods:
    1. **EMA weight update**: Smooth weight adaptation from relevance feedback
    2. **Query-type conditioned weights**: Different optimal weights per query type
    3. **Reciprocal Rank Fusion (RRF)**: Alternative to linear combination

    Based on:
    - "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
      (Cormack et al., 2009)
    - Hybrid search tuning patterns from Vespa, Weaviate, Pinecone best practices

    Example:
        >>> retriever = AdaptiveFusionRetriever(initial_bm25_weight=0.5)
        >>> retriever.add_documents(corpus)
        >>> results = retriever.retrieve("what is quantum computing?", k=10)
        >>> # After evaluating results, provide feedback
        >>> retriever.update_weights(relevance_score=0.85, query_type="factoid")
    """

    def __init__(
        self,
        initial_bm25_weight: float = 0.5,
        learning_rate: float = 0.05,
        rrf_k: int = 60,
        use_rrf: bool = True,
        dense_model_name: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True,
    ):
        """
        Initialize adaptive fusion retriever.

        Args:
            initial_bm25_weight: Starting BM25 weight (adapted over time)
            learning_rate: EMA learning rate for weight updates (0-1)
            rrf_k: RRF constant (default 60, per Cormack et al.)
            use_rrf: Use Reciprocal Rank Fusion instead of linear combination
            dense_model_name: Sentence-transformer model name
            cache_embeddings: Whether to cache embeddings
        """
        super().__init__(
            bm25_weight=initial_bm25_weight,
            dense_model_name=dense_model_name,
            cache_embeddings=cache_embeddings,
        )
        self.learning_rate = learning_rate
        self.rrf_k = rrf_k
        self.use_rrf = use_rrf

        # Per-query-type adaptive weights (EMA)
        self._type_weights: Dict[str, float] = {
            "factoid": 0.6,  # BM25 better for exact keyword matches
            "definition": 0.5,
            "explanation": 0.3,  # Dense better for semantic queries
            "comparison": 0.4,
            "multi_hop": 0.3,
            "default": initial_bm25_weight,
        }

        # Tracking for analysis
        self._update_count = 0
        self._weight_history: List[Tuple[float, float]] = []  # (bm25_w, relevance)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        query_type: Optional[str] = None,
    ) -> List[tuple]:
        """
        Retrieve with adaptive fusion weights.

        Args:
            query: Search query
            k: Number of results
            query_type: Optional query type for conditioned weights

        Returns:
            List of (document, score) tuples
        """
        from ..utils import monitor_performance

        if not self.is_initialized:
            raise ValueError("Retriever not initialized. Call add_documents() first.")

        with monitor_performance(
            "AdaptiveFusionRetriever.retrieve",
            {"query_length": len(query), "k": k, "use_rrf": self.use_rrf},
        ):
            # Select adaptive weight for query type
            if query_type and query_type in self._type_weights:
                active_bm25_weight = self._type_weights[query_type]
            else:
                active_bm25_weight = self.bm25_weight

            # Get results from both retrievers
            bm25_results = self.bm25_retriever.retrieve(query, k * 2)
            dense_results = self.dense_retriever.retrieve(query, k * 2)

            if self.use_rrf:
                return self._rrf_fusion(bm25_results, dense_results, k)
            else:
                return self._weighted_fusion(bm25_results, dense_results, active_bm25_weight, k)

    def _rrf_fusion(
        self,
        bm25_results: List[tuple],
        dense_results: List[tuple],
        k: int,
    ) -> List[tuple]:
        """
        Reciprocal Rank Fusion (Cormack et al., 2009).

        RRF(d) = sum_r ( 1 / (rrf_k + rank_r(d)) )

        Provably outperforms linear combination for most retrieval tasks.
        """
        rrf_scores: Dict[str, float] = {}

        for rank, (doc, _score) in enumerate(bm25_results):
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, (doc, _score) in enumerate(dense_results):
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:k]

    def _weighted_fusion(
        self,
        bm25_results: List[tuple],
        dense_results: List[tuple],
        bm25_weight: float,
        k: int,
    ) -> List[tuple]:
        """Linear weighted combination with min-max normalization."""
        dense_weight = 1.0 - bm25_weight

        bm25_scores = {doc: score for doc, score in bm25_results}
        dense_scores = {doc: score for doc, score in dense_results}

        # Min-max normalize both score sets to [0, 1]
        for scores in [bm25_scores, dense_scores]:
            if scores:
                vals = list(scores.values())
                min_v, max_v = min(vals), max(vals)
                rng = max_v - min_v
                if rng > 0:
                    for doc in scores:
                        scores[doc] = (scores[doc] - min_v) / rng

        # Combine
        all_docs = set(bm25_scores.keys()) | set(dense_scores.keys())
        combined = {}
        for doc in all_docs:
            combined[doc] = bm25_weight * bm25_scores.get(
                doc, 0.0
            ) + dense_weight * dense_scores.get(doc, 0.0)

        sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:k]

    def update_weights(
        self,
        relevance_score: float,
        query_type: Optional[str] = None,
    ) -> None:
        """
        Update fusion weights based on relevance feedback (EMA).

        Call this after evaluating retrieval quality. Higher relevance
        reinforces current weights; lower relevance shifts toward
        the opposite direction.

        Args:
            relevance_score: Quality of last retrieval (0-1)
            query_type: Query type for conditioned weight update
        """
        self._update_count += 1
        self._weight_history.append((self.bm25_weight, relevance_score))

        # EMA update: if relevance is high, reinforce; if low, shift
        # We nudge bm25_weight toward 0.5 + (relevance - 0.5) * direction
        lr = self.learning_rate

        if query_type and query_type in self._type_weights:
            old_w = self._type_weights[query_type]
            # Gradient-free adaptation: shift weight based on feedback
            delta = lr * (relevance_score - 0.5) * (1.0 - old_w if old_w < 0.5 else old_w)
            new_w = np.clip(old_w + delta, 0.05, 0.95)
            self._type_weights[query_type] = float(new_w)
        else:
            old_w = self.bm25_weight
            delta = lr * (relevance_score - 0.5) * (1.0 - old_w if old_w < 0.5 else old_w)
            self.bm25_weight = float(np.clip(old_w + delta, 0.05, 0.95))
            self.dense_weight = 1.0 - self.bm25_weight

    def get_weight_stats(self) -> Dict[str, object]:
        """Get adaptive weight statistics."""
        return {
            "current_bm25_weight": self.bm25_weight,
            "type_weights": dict(self._type_weights),
            "update_count": self._update_count,
            "use_rrf": self.use_rrf,
            "rrf_k": self.rrf_k,
        }
