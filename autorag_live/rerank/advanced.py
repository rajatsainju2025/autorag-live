"""
Advanced reranking strategies for AutoRAG-Live.

This module implements state-of-the-art reranking algorithms including:
- Maximal Marginal Relevance (MMR) for diversity
- Cross-encoder semantic reranking
- Diversity-aware reranking
- Reciprocal Rank Fusion (RRF)
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from autorag_live.utils import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)  # type: ignore[call-arg]
class RankedDocument:
    """
    Document with ranking metadata.

    Attributes:
        content: Document text
        score: Initial relevance score
        id: Optional document identifier
        metadata: Additional metadata
    """

    content: str
    score: float
    id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MMRReranker:
    """
    Maximal Marginal Relevance (MMR) reranker for diversity.

    MMR balances relevance and diversity by selecting documents that are
    relevant to the query while being dissimilar to already selected documents.

    Formula:
        MMR = argmax_d [λ * Sim(q, d) - (1-λ) * max(Sim(d, d_i) for d_i in selected)]
    """

    def __init__(
        self,
        lambda_param: float = 0.7,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """
        Initialize MMR reranker.

        Args:
            lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
            similarity_fn: Custom similarity function (default: cosine on term vectors)
        """
        self.lambda_param = lambda_param
        self.similarity_fn = similarity_fn or self._default_similarity
        self._similarity_cache: Dict[Tuple[str, str], float] = {}

    def _default_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity on term frequency vectors."""
        key = (text1, text2) if text1 <= text2 else (text2, text1)
        if key in self._similarity_cache:
            return self._similarity_cache[key]

        tokens1 = text1.lower().split()
        tokens2 = text2.lower().split()

        if not tokens1 or not tokens2:
            sim = 0.0
        else:
            # Use Counter for efficient sparse vector operations
            c1 = Counter(tokens1)
            c2 = Counter(tokens2)

            # Dot product only on intersection
            intersection = set(c1.keys()) & set(c2.keys())
            dot = sum(c1[t] * c2[t] for t in intersection)

            norm1 = math.sqrt(sum(v * v for v in c1.values()))
            norm2 = math.sqrt(sum(v * v for v in c2.values()))

            if norm1 == 0 or norm2 == 0:
                sim = 0.0
            else:
                sim = float(dot / (norm1 * norm2))

        self._similarity_cache[key] = sim
        return sim

    def rerank(
        self, query: str, documents: List[RankedDocument], top_k: Optional[int] = None
    ) -> List[RankedDocument]:
        """
        Rerank documents using MMR.

        Args:
            query: Query text
            documents: Documents to rerank
            top_k: Number of documents to return (default: all)

        Returns:
            Reranked documents with updated scores
        """
        if not documents:
            return []

        top_k = top_k or len(documents)
        selected: List[RankedDocument] = []
        remaining = documents.copy()

        # Pre-compute relevance scores
        relevance_scores = {
            doc.content: self.similarity_fn(query, doc.content) for doc in documents
        }

        # Track max similarity to selected docs for each remaining doc
        # Initialize with 0.0
        max_sim_to_selected = {doc.content: 0.0 for doc in documents}

        while len(selected) < top_k and remaining:
            mmr_scores = []

            for doc in remaining:
                # MMR score
                mmr = (
                    self.lambda_param * relevance_scores[doc.content]
                    - (1 - self.lambda_param) * max_sim_to_selected[doc.content]
                )
                mmr_scores.append((doc, mmr))

            # Select document with highest MMR score
            best_doc, best_score = max(mmr_scores, key=lambda x: x[1])
            best_doc.score = best_score  # Update score to MMR score
            selected.append(best_doc)
            remaining.remove(best_doc)

            # Update max similarity for remaining docs
            for doc in remaining:
                sim = self.similarity_fn(doc.content, best_doc.content)
                if sim > max_sim_to_selected[doc.content]:
                    max_sim_to_selected[doc.content] = sim

        logger.debug(f"MMR reranked {len(documents)} → {len(selected)} documents")
        return selected


class DiversityReranker:
    """
    Diversity-aware reranker using clustering and stratified sampling.

    Groups similar documents and ensures representation from each cluster.
    """

    def __init__(
        self,
        num_clusters: int = 5,
        cluster_sample_ratio: float = 0.5,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """
        Initialize diversity reranker.

        Args:
            num_clusters: Number of clusters to create
            cluster_sample_ratio: Ratio of top docs to sample per cluster
            similarity_fn: Custom similarity function
        """
        self.num_clusters = num_clusters
        self.cluster_sample_ratio = cluster_sample_ratio
        self.similarity_fn = similarity_fn or self._default_similarity
        self._similarity_cache: Dict[Tuple[str, str], float] = {}

    def _default_similarity(self, text1: str, text2: str) -> float:
        """Simple token overlap similarity."""
        key = (text1, text2) if text1 <= text2 else (text2, text1)
        if key in self._similarity_cache:
            return self._similarity_cache[key]

        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            sim = 0.0
        else:
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            sim = intersection / union if union > 0 else 0.0

        self._similarity_cache[key] = sim
        return sim

    def _cluster_documents(self, documents: List[RankedDocument]) -> List[List[RankedDocument]]:
        """Simple k-means-like clustering of documents."""
        if len(documents) <= self.num_clusters:
            return [[doc] for doc in documents]

        # Initialize clusters with top K documents
        clusters: List[List[RankedDocument]] = [
            [documents[i]] for i in range(min(self.num_clusters, len(documents)))
        ]

        # Assign remaining documents to nearest cluster
        for doc in documents[self.num_clusters :]:
            # Find most similar cluster
            max_sim = -1.0
            best_cluster_idx = 0

            for idx, cluster in enumerate(clusters):
                # Average similarity to cluster
                avg_sim = sum(self.similarity_fn(doc.content, c.content) for c in cluster) / len(
                    cluster
                )

                if avg_sim > max_sim:
                    max_sim = avg_sim
                    best_cluster_idx = idx

            clusters[best_cluster_idx].append(doc)

        return clusters

    def rerank(
        self, query: str, documents: List[RankedDocument], top_k: Optional[int] = None
    ) -> List[RankedDocument]:
        """
        Rerank documents for diversity.

        Args:
            query: Query text (unused but kept for interface consistency)
            documents: Documents to rerank
            top_k: Number of documents to return

        Returns:
            Diversified document list
        """
        if not documents:
            return []

        top_k = top_k or len(documents)

        # Cluster documents
        clusters = self._cluster_documents(documents)

        # Sample from each cluster proportionally
        reranked = []
        docs_per_cluster = max(1, int(top_k / len(clusters)))

        for cluster in clusters:
            # Sort cluster by original score
            cluster_sorted = sorted(cluster, key=lambda d: d.score, reverse=True)

            # Take top docs from cluster
            sample_size = min(docs_per_cluster, len(cluster_sorted))
            reranked.extend(cluster_sorted[:sample_size])

        # Sort by original scores and take top_k
        reranked_sorted = sorted(reranked, key=lambda d: d.score, reverse=True)[:top_k]

        logger.debug(
            f"Diversity reranking: {len(documents)} docs → {len(clusters)} clusters → {len(reranked_sorted)} final"
        )
        return reranked_sorted


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple rankings.

    RRF formula:
        RRF(d) = sum_r (1 / (k + rank_r(d)))

    Where:
        - d is a document
        - r is a retrieval system/ranker
        - k is a constant (typically 60)
        - rank_r(d) is the rank of d in system r
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF combiner.

        Args:
            k: Constant for rank normalization (typical value: 60)
        """
        self.k = k

    def fuse(
        self, rankings: List[List[RankedDocument]], top_k: Optional[int] = None
    ) -> List[RankedDocument]:
        """
        Fuse multiple rankings using RRF.

        Args:
            rankings: List of ranked document lists from different systems
            top_k: Number of documents to return

        Returns:
            Fused ranking
        """
        if not rankings:
            return []

        # Collect all unique documents
        doc_scores: Dict[str, Tuple[RankedDocument, float]] = {}

        for ranking in rankings:
            for rank, doc in enumerate(ranking, start=1):
                # Use content as key (or id if available)
                doc_key = doc.id if doc.id else doc.content

                # RRF score contribution
                rrf_score = 1.0 / (self.k + rank)

                if doc_key in doc_scores:
                    existing_doc, existing_score = doc_scores[doc_key]
                    doc_scores[doc_key] = (existing_doc, existing_score + rrf_score)
                else:
                    doc_scores[doc_key] = (doc, rrf_score)

        # Sort by RRF score
        fused = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)

        # Update scores and return
        result = []
        for doc, rrf_score in fused:
            doc.score = rrf_score
            result.append(doc)

        top_k = top_k or len(result)
        logger.debug(
            f"RRF fused {len(rankings)} rankings → {len(result)} unique docs → top {top_k}"
        )
        return result[:top_k]


class SemanticReranker:
    """
    Semantic reranker using cross-encoder models.

    Note: Requires sentence-transformers with cross-encoder models.
    Falls back to simple similarity if model not available.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        use_cache: bool = True,
    ):
        """
        Initialize semantic reranker.

        Args:
            model_name: Cross-encoder model name
            batch_size: Batch size for inference
            use_cache: Whether to cache model predictions
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_cache = use_cache
        self._model = None
        self._cache: Dict[Tuple[str, str], float] = {}

    def _load_model(self) -> None:
        """Lazy load cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)  # type: ignore
                logger.info(f"Loaded cross-encoder model: {self.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, using fallback similarity")
                self._model = "fallback"  # type: ignore
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self._model = "fallback"  # type: ignore

    def _fallback_score(self, query: str, doc: str) -> float:
        """Fallback similarity scoring."""
        q_tokens = set(query.lower().split())
        d_tokens = set(doc.lower().split())

        if not q_tokens or not d_tokens:
            return 0.0

        intersection = len(q_tokens & d_tokens)
        union = len(q_tokens | d_tokens)

        return intersection / union if union > 0 else 0.0

    def rerank(
        self, query: str, documents: List[RankedDocument], top_k: Optional[int] = None
    ) -> List[RankedDocument]:
        """
        Rerank documents using semantic cross-encoder.

        Args:
            query: Query text
            documents: Documents to rerank
            top_k: Number of documents to return

        Returns:
            Semantically reranked documents
        """
        if not documents:
            return []

        self._load_model()
        top_k = top_k or len(documents)

        # Prepare query-document pairs
        pairs = [(query, doc.content) for doc in documents]

        # Score pairs
        if self._model == "fallback":
            scores = [self._fallback_score(query, doc.content) for doc in documents]
        else:
            # Check cache
            if self.use_cache:
                scores = []
                for q, d in pairs:
                    cache_key = (q, d)
                    if cache_key in self._cache:
                        scores.append(self._cache[cache_key])
                    else:
                        score = float(self._model.predict([cache_key])[0])  # type: ignore
                        self._cache[cache_key] = score
                        scores.append(score)
            else:
                # Batch prediction
                scores = [float(s) for s in self._model.predict(pairs)]  # type: ignore

        # Update scores and sort
        for doc, score in zip(documents, scores):
            doc.score = score

        reranked = sorted(documents, key=lambda d: d.score, reverse=True)[:top_k]

        logger.debug(f"Semantic reranking: {len(documents)} docs → top {top_k}")
        return reranked


class HybridReranker:
    """
    Hybrid reranker combining multiple strategies.

    Combines relevance, diversity, and semantic scoring.
    """

    def __init__(
        self,
        relevance_weight: float = 0.5,
        diversity_weight: float = 0.3,
        semantic_weight: float = 0.2,
        mmr_lambda: float = 0.7,
    ):
        """
        Initialize hybrid reranker.

        Args:
            relevance_weight: Weight for original relevance scores
            diversity_weight: Weight for diversity (MMR)
            semantic_weight: Weight for semantic similarity
            mmr_lambda: Lambda parameter for MMR
        """
        self.relevance_weight = relevance_weight
        self.diversity_weight = diversity_weight
        self.semantic_weight = semantic_weight

        self.mmr_reranker = MMRReranker(lambda_param=mmr_lambda)
        self.semantic_reranker = SemanticReranker()

    def rerank(
        self, query: str, documents: List[RankedDocument], top_k: Optional[int] = None
    ) -> List[RankedDocument]:
        """
        Rerank using hybrid strategy.

        Args:
            query: Query text
            documents: Documents to rerank
            top_k: Number of documents to return

        Returns:
            Hybrid-reranked documents
        """
        if not documents:
            return []

        top_k = top_k or len(documents)

        # Get original relevance scores
        relevance_scores = {doc.content: doc.score for doc in documents}

        # Get MMR scores
        mmr_docs = self.mmr_reranker.rerank(query, documents.copy(), top_k=None)
        mmr_scores = {doc.content: doc.score for doc in mmr_docs}

        # Get semantic scores
        semantic_docs = self.semantic_reranker.rerank(query, documents.copy(), top_k=None)
        semantic_scores = {doc.content: doc.score for doc in semantic_docs}

        # Normalize scores to [0, 1]
        def normalize(scores: Dict[str, float]) -> Dict[str, float]:
            values = list(scores.values())
            if not values:
                return scores
            min_val, max_val = min(values), max(values)
            if max_val == min_val:
                return {k: 1.0 for k in scores}
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

        rel_norm = normalize(relevance_scores)
        mmr_norm = normalize(mmr_scores)
        sem_norm = normalize(semantic_scores)

        # Combine scores
        for doc in documents:
            content = doc.content
            combined_score = (
                self.relevance_weight * rel_norm.get(content, 0.0)
                + self.diversity_weight * mmr_norm.get(content, 0.0)
                + self.semantic_weight * sem_norm.get(content, 0.0)
            )
            doc.score = combined_score

        # Sort and return top_k
        reranked = sorted(documents, key=lambda d: d.score, reverse=True)[:top_k]

        logger.debug(f"Hybrid reranking: {len(documents)} docs → top {top_k}")
        return reranked
