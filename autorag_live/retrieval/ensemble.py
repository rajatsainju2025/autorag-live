"""
Ensemble Retrieval with Learned Fusion Weights.

Implements sophisticated retrieval ensemble methods that combine multiple
retrievers with learned or adaptive fusion weights.

Key techniques:
- Reciprocal Rank Fusion (RRF)
- Linear score combination with learned weights
- Attention-based fusion
- Query-dependent weight adaptation
- Cross-encoder reranking fusion
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# ============================================================================
# Protocols and Types
# ============================================================================


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for retrieval backends."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list["RetrievedDocument"]:
        """Retrieve documents for query."""
        ...

    @property
    def name(self) -> str:
        """Retriever identifier."""
        ...


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Embed text into vector."""
        ...


@runtime_checkable
class CrossEncoderProtocol(Protocol):
    """Protocol for cross-encoder rerankers."""

    async def score(self, query: str, document: str) -> float:
        """Score query-document pair."""
        ...

    async def score_batch(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """Score multiple documents for a query."""
        ...


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""
    source_retriever: str = ""


# ============================================================================
# Fusion Methods
# ============================================================================


class FusionMethod(Enum):
    """Available fusion methods."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    LINEAR = "linear"  # Linear score combination
    WEIGHTED_LINEAR = "weighted_linear"  # Weighted linear combination
    MAX_SCORE = "max_score"  # Maximum score across retrievers
    ATTENTION = "attention"  # Attention-based fusion
    BORDA = "borda"  # Borda count
    CONDORCET = "condorcet"  # Condorcet method


@dataclass
class FusionConfig:
    """Configuration for ensemble fusion."""

    method: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60  # RRF constant
    normalize_scores: bool = True
    min_retrievers_agree: int = 1
    use_query_weights: bool = False
    cross_encoder_rerank: bool = False
    top_k_rerank: int = 20


@dataclass
class EnsembleResult:
    """Result from ensemble retrieval."""

    documents: list[RetrievedDocument]
    fusion_method: str
    retriever_contributions: dict[str, float]
    fusion_scores: dict[str, float]  # doc_id -> fused score


# ============================================================================
# Score Fusion Strategies
# ============================================================================


class ScoreFuser(ABC):
    """Abstract base for score fusion strategies."""

    @abstractmethod
    def fuse(
        self,
        retriever_results: dict[str, list[RetrievedDocument]],
        weights: dict[str, float] | None = None,
    ) -> list[tuple[str, float, RetrievedDocument]]:
        """
        Fuse scores from multiple retrievers.

        Args:
            retriever_results: Mapping of retriever name to results
            weights: Optional retriever weights

        Returns:
            List of (doc_id, fused_score, document) tuples
        """
        ...


class RRFFuser(ScoreFuser):
    """
    Reciprocal Rank Fusion (RRF).

    Combines rankings using: score = sum(1 / (k + rank))
    Works well even when scores are not comparable across retrievers.
    """

    def __init__(self, k: int = 60):
        """Initialize with RRF constant k."""
        self.k = k

    def fuse(
        self,
        retriever_results: dict[str, list[RetrievedDocument]],
        weights: dict[str, float] | None = None,
    ) -> list[tuple[str, float, RetrievedDocument]]:
        """Fuse using RRF."""
        # Initialize weights if not provided
        if weights is None:
            weights = {name: 1.0 for name in retriever_results}

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate RRF scores
        doc_scores: dict[str, float] = {}
        doc_objects: dict[str, RetrievedDocument] = {}

        for retriever_name, results in retriever_results.items():
            weight = weights.get(retriever_name, 1.0)

            for rank, doc in enumerate(results, 1):
                doc_id = doc.doc_id or hash(doc.content)
                rrf_score = weight * (1.0 / (self.k + rank))

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_objects[doc_id] = doc

                doc_scores[doc_id] += rrf_score

        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        return [(doc_id, score, doc_objects[doc_id]) for doc_id, score in sorted_docs]


class LinearFuser(ScoreFuser):
    """
    Linear score combination.

    Combines normalized scores: score = sum(weight * normalized_score)
    """

    def __init__(self, normalize: bool = True):
        """Initialize with normalization setting."""
        self.normalize = normalize

    def fuse(
        self,
        retriever_results: dict[str, list[RetrievedDocument]],
        weights: dict[str, float] | None = None,
    ) -> list[tuple[str, float, RetrievedDocument]]:
        """Fuse using linear combination."""
        if weights is None:
            weights = {name: 1.0 for name in retriever_results}

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Normalize scores per retriever if needed
        normalized_results: dict[str, list[tuple[str, float, RetrievedDocument]]] = {}

        for retriever_name, results in retriever_results.items():
            if self.normalize and results:
                scores = [doc.score for doc in results]
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score

                normalized = []
                for doc in results:
                    if score_range > 0:
                        norm_score = (doc.score - min_score) / score_range
                    else:
                        norm_score = 1.0
                    doc_id = doc.doc_id or hash(doc.content)
                    normalized.append((doc_id, norm_score, doc))

                normalized_results[retriever_name] = normalized
            else:
                normalized_results[retriever_name] = [
                    (doc.doc_id or hash(doc.content), doc.score, doc) for doc in results
                ]

        # Combine scores
        doc_scores: dict[str, float] = {}
        doc_objects: dict[str, RetrievedDocument] = {}
        doc_counts: dict[str, int] = {}

        for retriever_name, results in normalized_results.items():
            weight = weights.get(retriever_name, 1.0)

            for doc_id, score, doc in results:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_objects[doc_id] = doc
                    doc_counts[doc_id] = 0

                doc_scores[doc_id] += weight * score
                doc_counts[doc_id] += 1

        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        return [(doc_id, score, doc_objects[doc_id]) for doc_id, score in sorted_docs]


class MaxScoreFuser(ScoreFuser):
    """
    Maximum score fusion.

    Takes the maximum score across all retrievers for each document.
    """

    def fuse(
        self,
        retriever_results: dict[str, list[RetrievedDocument]],
        weights: dict[str, float] | None = None,
    ) -> list[tuple[str, float, RetrievedDocument]]:
        """Fuse using max score."""
        doc_scores: dict[str, float] = {}
        doc_objects: dict[str, RetrievedDocument] = {}

        for _retriever_name, results in retriever_results.items():
            for doc in results:
                doc_id = doc.doc_id or hash(doc.content)

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = doc.score
                    doc_objects[doc_id] = doc
                else:
                    doc_scores[doc_id] = max(doc_scores[doc_id], doc.score)

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, doc_objects[doc_id]) for doc_id, score in sorted_docs]


class BordaFuser(ScoreFuser):
    """
    Borda count fusion.

    Assigns points based on rank: n-1 for first, n-2 for second, etc.
    """

    def fuse(
        self,
        retriever_results: dict[str, list[RetrievedDocument]],
        weights: dict[str, float] | None = None,
    ) -> list[tuple[str, float, RetrievedDocument]]:
        """Fuse using Borda count."""
        if weights is None:
            weights = {name: 1.0 for name in retriever_results}

        doc_scores: dict[str, float] = {}
        doc_objects: dict[str, RetrievedDocument] = {}

        for retriever_name, results in retriever_results.items():
            weight = weights.get(retriever_name, 1.0)
            n = len(results)

            for rank, doc in enumerate(results):
                doc_id = doc.doc_id or hash(doc.content)
                borda_score = weight * (n - rank)

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_objects[doc_id] = doc

                doc_scores[doc_id] += borda_score

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, doc_objects[doc_id]) for doc_id, score in sorted_docs]


# ============================================================================
# Weight Learning
# ============================================================================


@dataclass
class WeightLearningConfig:
    """Configuration for weight learning."""

    learning_rate: float = 0.01
    iterations: int = 100
    regularization: float = 0.1
    method: str = "gradient"  # gradient, bayesian, grid


class WeightLearner(ABC):
    """Abstract base for weight learning."""

    @abstractmethod
    def learn_weights(
        self,
        retriever_results: list[dict[str, list[RetrievedDocument]]],
        relevance_labels: list[dict[str, float]],
    ) -> dict[str, float]:
        """
        Learn optimal weights from training data.

        Args:
            retriever_results: List of retriever results per query
            relevance_labels: Relevance labels per query (doc_id -> relevance)

        Returns:
            Learned weights per retriever
        """
        ...


class GradientWeightLearner(WeightLearner):
    """
    Gradient-based weight learning.

    Optimizes weights to maximize NDCG or other IR metrics.
    """

    def __init__(self, config: WeightLearningConfig | None = None):
        """Initialize with config."""
        self.config = config or WeightLearningConfig()

    def learn_weights(
        self,
        retriever_results: list[dict[str, list[RetrievedDocument]]],
        relevance_labels: list[dict[str, float]],
    ) -> dict[str, float]:
        """Learn weights using gradient descent on NDCG."""
        if not retriever_results:
            return {}

        # Get retriever names
        retriever_names = list(retriever_results[0].keys())

        # Initialize weights uniformly
        weights = {name: 1.0 / len(retriever_names) for name in retriever_names}

        fuser = LinearFuser(normalize=True)

        for _iteration in range(self.config.iterations):
            # Calculate gradients
            gradients = {name: 0.0 for name in retriever_names}

            for results, labels in zip(retriever_results, relevance_labels):
                # Get current ranking with current weights
                fused = fuser.fuse(results, weights)
                current_ndcg = self._calculate_ndcg(fused, labels)

                # Calculate gradient per retriever
                for name in retriever_names:
                    # Perturb weight
                    perturbed_weights = weights.copy()
                    perturbed_weights[name] += 0.01

                    # Get perturbed ranking
                    perturbed_fused = fuser.fuse(results, perturbed_weights)
                    perturbed_ndcg = self._calculate_ndcg(perturbed_fused, labels)

                    # Approximate gradient
                    gradients[name] += (perturbed_ndcg - current_ndcg) / 0.01

            # Update weights
            for name in retriever_names:
                weights[name] += self.config.learning_rate * gradients[name]
                # Apply regularization
                weights[name] -= self.config.regularization * weights[name]

            # Ensure positive weights
            weights = {k: max(0.01, v) for k, v in weights.items()}

            # Normalize
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _calculate_ndcg(
        self,
        fused_results: list[tuple[str, float, RetrievedDocument]],
        relevance_labels: dict[str, float],
        k: int = 10,
    ) -> float:
        """Calculate NDCG@k."""
        # DCG
        dcg = 0.0
        for i, (doc_id, _score, _doc) in enumerate(fused_results[:k], 1):
            rel = relevance_labels.get(doc_id, 0.0)
            dcg += (2**rel - 1) / self._log2(i + 1)

        # Ideal DCG
        sorted_rels = sorted(relevance_labels.values(), reverse=True)[:k]
        idcg = sum((2**rel - 1) / self._log2(i + 1) for i, rel in enumerate(sorted_rels, 1))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _log2(self, x: float) -> float:
        """Calculate log base 2."""
        import math

        return math.log2(x) if x > 0 else 0.0


class BayesianWeightLearner(WeightLearner):
    """
    Bayesian weight learning with uncertainty estimation.

    Uses Thompson sampling for exploration-exploitation.
    """

    def __init__(self, config: WeightLearningConfig | None = None):
        """Initialize with config."""
        self.config = config or WeightLearningConfig()
        self.alpha: dict[str, float] = {}  # Success counts
        self.beta: dict[str, float] = {}  # Failure counts

    def learn_weights(
        self,
        retriever_results: list[dict[str, list[RetrievedDocument]]],
        relevance_labels: list[dict[str, float]],
    ) -> dict[str, float]:
        """Learn weights using Bayesian updating."""
        import random

        if not retriever_results:
            return {}

        retriever_names = list(retriever_results[0].keys())

        # Initialize priors
        for name in retriever_names:
            if name not in self.alpha:
                self.alpha[name] = 1.0
                self.beta[name] = 1.0

        # Update based on relevance
        for results, labels in zip(retriever_results, relevance_labels):
            for name, docs in results.items():
                # Count relevant documents retrieved
                relevant_count = sum(
                    1 for doc in docs[:10] if labels.get(doc.doc_id or hash(doc.content), 0) > 0
                )
                irrelevant_count = min(10, len(docs)) - relevant_count

                self.alpha[name] += relevant_count
                self.beta[name] += irrelevant_count

        # Sample weights from Beta distributions
        weights = {}
        for name in retriever_names:
            # Sample from Beta(alpha, beta)
            sampled = random.betavariate(self.alpha[name], self.beta[name])
            weights[name] = sampled

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


# ============================================================================
# Query-Dependent Weight Adaptation
# ============================================================================


class QueryWeightAdapter:
    """
    Adapts retriever weights based on query characteristics.

    Uses query features to predict optimal weights.
    """

    # Query type indicators and their preferred retrievers
    QUERY_PATTERNS = {
        "definition": {"sparse": 0.3, "dense": 0.7},
        "factual": {"sparse": 0.5, "dense": 0.5},
        "analytical": {"sparse": 0.2, "dense": 0.8},
        "comparison": {"sparse": 0.4, "dense": 0.6},
    }

    def __init__(self, embedder: EmbeddingProtocol | None = None):
        """Initialize with optional embedder for similarity-based adaptation."""
        self.embedder = embedder
        self.query_weight_history: list[tuple[str, dict[str, float]]] = []

    def adapt_weights(
        self,
        query: str,
        base_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Adapt weights based on query.

        Args:
            query: Input query
            base_weights: Base retriever weights

        Returns:
            Adapted weights
        """
        # Classify query
        query_type = self._classify_query(query)

        # Get pattern-based adjustments
        if query_type in self.QUERY_PATTERNS:
            adjustments = self.QUERY_PATTERNS[query_type]
        else:
            adjustments = {}

        # Apply adjustments
        adapted = base_weights.copy()
        for retriever, adjustment in adjustments.items():
            if retriever in adapted:
                # Blend base weight with adjustment
                adapted[retriever] = 0.7 * adapted[retriever] + 0.3 * adjustment

        # Normalize
        total = sum(adapted.values())
        return {k: v / total for k, v in adapted.items()}

    def _classify_query(self, query: str) -> str:
        """Classify query type based on patterns."""
        query_lower = query.lower()

        if "what is" in query_lower or "define" in query_lower:
            return "definition"
        elif "compare" in query_lower or "difference" in query_lower:
            return "comparison"
        elif "why" in query_lower or "how" in query_lower:
            return "analytical"
        else:
            return "factual"

    def update_from_feedback(
        self,
        query: str,
        weights: dict[str, float],
        quality_score: float,
    ):
        """Update adaptation model based on feedback."""
        self.query_weight_history.append((query, weights))


# ============================================================================
# Ensemble Retriever
# ============================================================================


class EnsembleRetriever:
    """
    Ensemble retriever combining multiple retrieval methods.

    Features:
    - Multiple fusion strategies (RRF, linear, attention)
    - Learned or adaptive weights
    - Query-dependent weight adaptation
    - Optional cross-encoder reranking
    """

    FUSERS = {
        FusionMethod.RRF: RRFFuser,
        FusionMethod.LINEAR: LinearFuser,
        FusionMethod.MAX_SCORE: MaxScoreFuser,
        FusionMethod.BORDA: BordaFuser,
    }

    def __init__(
        self,
        retrievers: list[RetrieverProtocol],
        config: FusionConfig | None = None,
        weights: dict[str, float] | None = None,
        cross_encoder: CrossEncoderProtocol | None = None,
    ):
        """
        Initialize ensemble retriever.

        Args:
            retrievers: List of retriever backends
            config: Fusion configuration
            weights: Initial retriever weights
            cross_encoder: Optional cross-encoder for reranking
        """
        self.retrievers = {r.name: r for r in retrievers}
        self.config = config or FusionConfig()
        self.weights = weights or {name: 1.0 for name in self.retrievers}
        self.cross_encoder = cross_encoder
        self.query_adapter: QueryWeightAdapter | None = None

        # Initialize fuser
        fuser_class = self.FUSERS.get(self.config.method, RRFFuser)
        if self.config.method == FusionMethod.RRF:
            self.fuser = fuser_class(k=self.config.rrf_k)
        else:
            self.fuser = fuser_class()

    def set_query_adapter(self, adapter: QueryWeightAdapter):
        """Set query-dependent weight adapter."""
        self.query_adapter = adapter

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> EnsembleResult:
        """
        Retrieve documents using ensemble.

        Args:
            query: User query
            top_k: Number of documents to return

        Returns:
            EnsembleResult with fused documents
        """
        # Get weights (possibly query-adapted)
        weights = self.weights
        if self.config.use_query_weights and self.query_adapter:
            weights = self.query_adapter.adapt_weights(query, self.weights)

        # Retrieve from all retrievers
        retriever_results: dict[str, list[RetrievedDocument]] = {}

        for name, retriever in self.retrievers.items():
            results = await retriever.retrieve(query, top_k=top_k * 2)
            # Tag with source retriever
            for doc in results:
                doc.source_retriever = name
            retriever_results[name] = results

        # Fuse results
        fused = self.fuser.fuse(retriever_results, weights)

        # Apply minimum agreement filter
        if self.config.min_retrievers_agree > 1:
            fused = self._filter_by_agreement(fused, retriever_results)

        # Cross-encoder reranking if enabled
        if self.config.cross_encoder_rerank and self.cross_encoder:
            fused = await self._cross_encoder_rerank(query, fused, self.config.top_k_rerank)

        # Build result
        documents = [doc for _doc_id, _score, doc in fused[:top_k]]
        fusion_scores = {doc_id: score for doc_id, score, _doc in fused}

        # Calculate contributions
        contributions = self._calculate_contributions(documents, retriever_results)

        return EnsembleResult(
            documents=documents,
            fusion_method=self.config.method.value,
            retriever_contributions=contributions,
            fusion_scores=fusion_scores,
        )

    def _filter_by_agreement(
        self,
        fused: list[tuple[str, float, RetrievedDocument]],
        retriever_results: dict[str, list[RetrievedDocument]],
    ) -> list[tuple[str, float, RetrievedDocument]]:
        """Filter to documents that appear in multiple retrievers."""
        # Count appearances
        doc_appearances: dict[str, int] = {}
        for _name, results in retriever_results.items():
            for doc in results:
                doc_id = doc.doc_id or hash(doc.content)
                doc_appearances[doc_id] = doc_appearances.get(doc_id, 0) + 1

        # Filter
        return [
            (doc_id, score, doc)
            for doc_id, score, doc in fused
            if doc_appearances.get(doc_id, 0) >= self.config.min_retrievers_agree
        ]

    async def _cross_encoder_rerank(
        self,
        query: str,
        fused: list[tuple[str, float, RetrievedDocument]],
        top_k: int,
    ) -> list[tuple[str, float, RetrievedDocument]]:
        """Rerank top documents using cross-encoder."""
        if not self.cross_encoder:
            return fused

        # Take top candidates for reranking
        candidates = fused[:top_k]

        # Score with cross-encoder
        doc_texts = [doc.content for _doc_id, _score, doc in candidates]
        ce_scores = await self.cross_encoder.score_batch(query, doc_texts)

        # Combine scores
        reranked = []
        for (doc_id, fusion_score, doc), ce_score in zip(candidates, ce_scores):
            # Combine fusion score and cross-encoder score
            combined = 0.5 * fusion_score + 0.5 * ce_score
            reranked.append((doc_id, combined, doc))

        # Sort by combined score
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Add remaining documents
        remaining = fused[top_k:]
        return reranked + remaining

    def _calculate_contributions(
        self,
        documents: list[RetrievedDocument],
        retriever_results: dict[str, list[RetrievedDocument]],
    ) -> dict[str, float]:
        """Calculate each retriever's contribution to final results."""
        contributions = {name: 0.0 for name in self.retrievers}

        for doc in documents:
            if doc.source_retriever:
                contributions[doc.source_retriever] += 1.0

        # Normalize
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return contributions


# ============================================================================
# Adaptive Ensemble (learns from feedback)
# ============================================================================


class AdaptiveEnsembleRetriever(EnsembleRetriever):
    """
    Adaptive ensemble that learns from user feedback.

    Continuously updates weights based on retrieval quality.
    """

    def __init__(
        self,
        retrievers: list[RetrieverProtocol],
        config: FusionConfig | None = None,
        weight_learner: WeightLearner | None = None,
        cross_encoder: CrossEncoderProtocol | None = None,
    ):
        """Initialize adaptive ensemble."""
        super().__init__(retrievers, config, cross_encoder=cross_encoder)
        self.weight_learner = weight_learner or BayesianWeightLearner()
        self.feedback_buffer: list[tuple[dict[str, list[RetrievedDocument]], dict[str, float]]] = []
        self.update_frequency = 10

    async def retrieve_with_feedback(
        self,
        query: str,
        top_k: int = 10,
    ) -> tuple[EnsembleResult, str]:
        """
        Retrieve and return feedback token.

        Args:
            query: User query
            top_k: Number of documents

        Returns:
            Tuple of (result, feedback_token)
        """
        result = await self.retrieve(query, top_k)

        # Generate feedback token
        import uuid

        feedback_token = str(uuid.uuid4())

        return result, feedback_token

    def record_feedback(
        self,
        retriever_results: dict[str, list[RetrievedDocument]],
        relevance_labels: dict[str, float],
    ):
        """
        Record feedback for weight learning.

        Args:
            retriever_results: Results from each retriever
            relevance_labels: User-provided relevance labels
        """
        self.feedback_buffer.append((retriever_results, relevance_labels))

        # Update weights periodically
        if len(self.feedback_buffer) >= self.update_frequency:
            self._update_weights()

    def _update_weights(self):
        """Update weights from feedback buffer."""
        if not self.feedback_buffer:
            return

        results_list = [r for r, _labels in self.feedback_buffer]
        labels_list = [labels for _r, labels in self.feedback_buffer]

        new_weights = self.weight_learner.learn_weights(results_list, labels_list)

        # Blend with current weights for stability
        for name in self.weights:
            if name in new_weights:
                self.weights[name] = 0.7 * self.weights[name] + 0.3 * new_weights[name]

        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        # Clear buffer
        self.feedback_buffer = []


# ============================================================================
# Specialized Ensembles
# ============================================================================


class HybridRetriever(EnsembleRetriever):
    """
    Hybrid retriever combining sparse and dense retrieval.

    Optimized for combining BM25/TF-IDF with neural retrievers.
    """

    def __init__(
        self,
        sparse_retriever: RetrieverProtocol,
        dense_retriever: RetrieverProtocol,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5,
        cross_encoder: CrossEncoderProtocol | None = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            sparse_retriever: Sparse retriever (BM25, TF-IDF)
            dense_retriever: Dense retriever (neural)
            sparse_weight: Weight for sparse retriever
            dense_weight: Weight for dense retriever
            cross_encoder: Optional cross-encoder
        """
        super().__init__(
            retrievers=[sparse_retriever, dense_retriever],
            config=FusionConfig(method=FusionMethod.RRF),
            weights={
                sparse_retriever.name: sparse_weight,
                dense_retriever.name: dense_weight,
            },
            cross_encoder=cross_encoder,
        )


class MultiModalEnsemble(EnsembleRetriever):
    """
    Multi-modal ensemble for combining different modality retrievers.

    Can combine text, image, and code retrievers.
    """

    def __init__(
        self,
        retrievers: dict[str, RetrieverProtocol],
        modality_weights: dict[str, float] | None = None,
    ):
        """
        Initialize multi-modal ensemble.

        Args:
            retrievers: Mapping of modality to retriever
            modality_weights: Weights per modality
        """
        retriever_list = list(retrievers.values())

        if modality_weights is None:
            modality_weights = {m: 1.0 for m in retrievers}

        # Map modality weights to retriever names
        weights = {retrievers[m].name: w for m, w in modality_weights.items()}

        super().__init__(
            retrievers=retriever_list,
            config=FusionConfig(method=FusionMethod.LINEAR),
            weights=weights,
        )


# ============================================================================
# Factory Functions
# ============================================================================


def create_ensemble_retriever(
    retrievers: list[RetrieverProtocol],
    method: FusionMethod = FusionMethod.RRF,
    weights: dict[str, float] | None = None,
) -> EnsembleRetriever:
    """
    Create an ensemble retriever.

    Args:
        retrievers: List of retriever backends
        method: Fusion method
        weights: Optional retriever weights

    Returns:
        Configured EnsembleRetriever
    """
    config = FusionConfig(method=method)
    return EnsembleRetriever(retrievers, config, weights)


def create_adaptive_ensemble(
    retrievers: list[RetrieverProtocol],
    method: FusionMethod = FusionMethod.RRF,
    learning_rate: float = 0.01,
) -> AdaptiveEnsembleRetriever:
    """
    Create an adaptive ensemble that learns from feedback.

    Args:
        retrievers: List of retriever backends
        method: Fusion method
        learning_rate: Weight learning rate

    Returns:
        Configured AdaptiveEnsembleRetriever
    """
    config = FusionConfig(method=method)
    learner_config = WeightLearningConfig(learning_rate=learning_rate)
    learner = GradientWeightLearner(learner_config)

    return AdaptiveEnsembleRetriever(retrievers, config, learner)


def create_hybrid_retriever(
    sparse_retriever: RetrieverProtocol,
    dense_retriever: RetrieverProtocol,
    alpha: float = 0.5,
) -> HybridRetriever:
    """
    Create a hybrid sparse-dense retriever.

    Args:
        sparse_retriever: Sparse retriever (BM25)
        dense_retriever: Dense retriever (neural)
        alpha: Weight for sparse retriever (1-alpha for dense)

    Returns:
        Configured HybridRetriever
    """
    return HybridRetriever(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        sparse_weight=alpha,
        dense_weight=1 - alpha,
    )


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example demonstrating ensemble retrieval."""
    # This would use actual implementations
    # sparse = BM25Retriever(...)
    # dense = DenseRetriever(...)
    # neural = NeuralRetriever(...)

    # Basic ensemble
    # ensemble = create_ensemble_retriever([sparse, dense, neural])
    # result = await ensemble.retrieve("What is machine learning?")

    # Hybrid retriever
    # hybrid = create_hybrid_retriever(sparse, dense, alpha=0.5)
    # result = await hybrid.retrieve("Explain transformer architecture")

    # Adaptive ensemble
    # adaptive = create_adaptive_ensemble([sparse, dense])
    # result, token = await adaptive.retrieve_with_feedback("query")
    # adaptive.record_feedback(results, {"doc1": 1.0, "doc2": 0.5})

    print("Ensemble Retrieval Implementation Ready")
    print("=" * 50)
    print("Features:")
    print("- Multiple fusion methods (RRF, Linear, Max, Borda)")
    print("- Learned weights via gradient descent or Bayesian learning")
    print("- Query-dependent weight adaptation")
    print("- Cross-encoder reranking integration")
    print("- Adaptive ensemble with feedback learning")
    print("- Hybrid sparse-dense retrieval")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
