"""
Active Learning for RAG Implementation.

Implements active learning strategies to continuously improve
RAG systems through human feedback and uncertainty sampling.

Key features:
1. Uncertainty sampling for retrieval
2. Query-by-committee for diverse sampling
3. Human feedback integration
4. Retrieval model fine-tuning loop
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# ============================================================================
# Protocols
# ============================================================================


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM providers."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for retrieval backends."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list["RetrievedDocument"]:
        """Retrieve documents for query."""
        ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts."""
        ...


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""


# ============================================================================
# Feedback Types
# ============================================================================


class FeedbackType(Enum):
    """Types of human feedback."""

    RELEVANT = "relevant"  # Document is relevant
    NOT_RELEVANT = "not_relevant"  # Document is not relevant
    PARTIALLY_RELEVANT = "partially_relevant"  # Somewhat relevant
    CORRECT_ANSWER = "correct_answer"  # Answer is correct
    INCORRECT_ANSWER = "incorrect_answer"  # Answer is wrong
    MISSING_INFO = "missing_info"  # Missing important information


class SamplingStrategy(Enum):
    """Active learning sampling strategies."""

    UNCERTAINTY = "uncertainty"  # Low confidence predictions
    DIVERSITY = "diversity"  # Maximize diversity
    QUERY_BY_COMMITTEE = "query_by_committee"  # Disagreement-based
    EXPECTED_MODEL_CHANGE = "expected_model_change"
    RANDOM = "random"  # Random baseline


@dataclass
class FeedbackInstance:
    """A single feedback instance."""

    query: str
    document: RetrievedDocument
    feedback_type: FeedbackType
    annotator_id: str
    timestamp: datetime
    notes: str = ""
    confidence: float | None = None


@dataclass
class QuerySample:
    """A query sample for annotation."""

    query: str
    documents: list[RetrievedDocument]
    sampling_reason: str
    uncertainty_score: float
    priority: float


@dataclass
class ActiveLearningMetrics:
    """Metrics from active learning cycle."""

    total_samples: int
    annotated_samples: int
    model_improvement: float
    precision_delta: float
    recall_delta: float
    annotation_agreement: float


# ============================================================================
# Uncertainty Estimators
# ============================================================================


class UncertaintyEstimator(ABC):
    """Abstract base for uncertainty estimation."""

    @abstractmethod
    async def estimate(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> float:
        """Estimate uncertainty for a query-documents pair."""
        ...


class ScoreBasedUncertainty(UncertaintyEstimator):
    """Uncertainty based on retrieval scores."""

    def __init__(
        self,
        score_threshold: float = 0.5,
        margin_threshold: float = 0.1,
    ):
        """
        Initialize score-based uncertainty.

        Args:
            score_threshold: Scores below this are uncertain
            margin_threshold: Small margins indicate uncertainty
        """
        self.score_threshold = score_threshold
        self.margin_threshold = margin_threshold

    async def estimate(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> float:
        """
        Estimate uncertainty from scores.

        High uncertainty if:
        - Top score is low
        - Margin between top scores is small
        """
        if not documents:
            return 1.0

        scores = [d.score for d in documents]
        top_score = max(scores)

        # Score-based uncertainty
        score_uncertainty = 1.0 - top_score

        # Margin-based uncertainty (small margin = uncertain)
        if len(scores) >= 2:
            sorted_scores = sorted(scores, reverse=True)
            margin = sorted_scores[0] - sorted_scores[1]
            margin_uncertainty = 1.0 - min(margin / self.margin_threshold, 1.0)
        else:
            margin_uncertainty = 0.5

        # Combine
        return 0.5 * score_uncertainty + 0.5 * margin_uncertainty


class EmbeddingUncertainty(UncertaintyEstimator):
    """Uncertainty based on embedding distances."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        distance_threshold: float = 0.5,
    ):
        """Initialize embedding-based uncertainty."""
        self.embedder = embedder
        self.distance_threshold = distance_threshold

    async def estimate(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> float:
        """Estimate uncertainty from embedding distances."""
        if not documents:
            return 1.0

        # Embed query and documents
        texts = [query] + [d.content[:500] for d in documents]
        embeddings = await self.embedder.embed(texts)

        query_emb = embeddings[0]
        doc_embs = embeddings[1:]

        # Calculate distances
        distances = []
        for doc_emb in doc_embs:
            dist = self._cosine_distance(query_emb, doc_emb)
            distances.append(dist)

        # High distance = high uncertainty
        avg_distance = sum(distances) / len(distances)
        return min(avg_distance / self.distance_threshold, 1.0)

    def _cosine_distance(self, emb1: list[float], emb2: list[float]) -> float:
        """Calculate cosine distance."""
        import math

        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))

        if norm1 == 0 or norm2 == 0:
            return 1.0

        similarity = dot_product / (norm1 * norm2)
        return 1.0 - similarity


class LLMUncertainty(UncertaintyEstimator):
    """LLM-based uncertainty estimation."""

    UNCERTAINTY_PROMPT = """Rate how confident you are that these documents answer the question.

Question: {query}

Documents:
{documents}

Rate confidence from 0 to 10 (10 = very confident these are the right documents):
Confidence:"""

    def __init__(self, llm: LLMProtocol):
        """Initialize LLM-based uncertainty."""
        self.llm = llm

    async def estimate(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> float:
        """Estimate uncertainty using LLM."""
        doc_text = "\n".join(f"[{i}] {d.content[:200]}" for i, d in enumerate(documents, 1))

        prompt = self.UNCERTAINTY_PROMPT.format(
            query=query,
            documents=doc_text,
        )

        response = await self.llm.generate(prompt, max_tokens=10, temperature=0.0)

        try:
            confidence = float("".join(c for c in response if c.isdigit() or c == "."))
            confidence = min(10, max(0, confidence)) / 10
        except ValueError:
            confidence = 0.5

        # Uncertainty = 1 - confidence
        return 1.0 - confidence


# ============================================================================
# Query-by-Committee
# ============================================================================


class QueryByCommittee:
    """
    Query-by-committee sampling.

    Uses multiple retrievers and measures disagreement.
    """

    def __init__(
        self,
        retrievers: list[RetrieverProtocol],
    ):
        """
        Initialize with committee of retrievers.

        Args:
            retrievers: List of different retrievers
        """
        self.retrievers = retrievers

    async def estimate_disagreement(
        self,
        query: str,
        top_k: int = 10,
    ) -> tuple[float, list[list[RetrievedDocument]]]:
        """
        Estimate disagreement among committee members.

        Returns:
            Tuple of (disagreement_score, all_results)
        """
        all_results = []

        for retriever in self.retrievers:
            results = await retriever.retrieve(query, top_k=top_k)
            all_results.append(results)

        disagreement = self._calculate_disagreement(all_results)
        return disagreement, all_results

    def _calculate_disagreement(
        self,
        all_results: list[list[RetrievedDocument]],
    ) -> float:
        """
        Calculate disagreement score.

        Based on Jaccard distance between result sets.
        """
        if len(all_results) < 2:
            return 0.0

        total_disagreement = 0.0
        num_pairs = 0

        for i in range(len(all_results)):
            for j in range(i + 1, len(all_results)):
                set_i = {d.content[:100] for d in all_results[i]}
                set_j = {d.content[:100] for d in all_results[j]}

                intersection = len(set_i & set_j)
                union = len(set_i | set_j)

                if union > 0:
                    jaccard = intersection / union
                    total_disagreement += 1.0 - jaccard
                    num_pairs += 1

        return total_disagreement / max(num_pairs, 1)


# ============================================================================
# Sample Selector
# ============================================================================


@dataclass
class SamplerConfig:
    """Configuration for active learning sampler."""

    strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY
    batch_size: int = 10
    uncertainty_threshold: float = 0.7
    diversity_weight: float = 0.3


class ActiveSampler:
    """
    Select samples for annotation using active learning.

    Prioritizes uncertain or diverse samples.
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        uncertainty_estimator: UncertaintyEstimator,
        config: SamplerConfig | None = None,
    ):
        """
        Initialize active sampler.

        Args:
            retriever: Retrieval backend
            uncertainty_estimator: Uncertainty estimation method
            config: Sampling configuration
        """
        self.retriever = retriever
        self.uncertainty_estimator = uncertainty_estimator
        self.config = config or SamplerConfig()

    async def select_samples(
        self,
        candidate_queries: list[str],
        top_k: int = 5,
    ) -> list[QuerySample]:
        """
        Select samples for annotation.

        Args:
            candidate_queries: Pool of candidate queries
            top_k: Documents per query

        Returns:
            Selected samples for annotation
        """
        samples = []

        for query in candidate_queries:
            # Retrieve documents
            documents = await self.retriever.retrieve(query, top_k=top_k)

            # Estimate uncertainty
            uncertainty = await self.uncertainty_estimator.estimate(query, documents)

            sample = QuerySample(
                query=query,
                documents=documents,
                sampling_reason=self.config.strategy.value,
                uncertainty_score=uncertainty,
                priority=uncertainty,
            )
            samples.append(sample)

        # Sort by uncertainty/priority
        samples.sort(key=lambda s: s.priority, reverse=True)

        # Apply threshold and batch size
        selected = [s for s in samples if s.uncertainty_score >= self.config.uncertainty_threshold][
            : self.config.batch_size
        ]

        # If not enough uncertain samples, add more
        if len(selected) < self.config.batch_size:
            remaining = [s for s in samples if s not in selected]
            selected.extend(remaining[: self.config.batch_size - len(selected)])

        return selected


# ============================================================================
# Feedback Store
# ============================================================================


class FeedbackStore:
    """
    Store and manage feedback instances.

    In-memory store for simplicity; extend for persistence.
    """

    def __init__(self):
        """Initialize feedback store."""
        self.feedback: list[FeedbackInstance] = []
        self._query_index: dict[str, list[int]] = {}

    def add_feedback(self, feedback: FeedbackInstance) -> None:
        """Add feedback instance."""
        idx = len(self.feedback)
        self.feedback.append(feedback)

        # Index by query
        if feedback.query not in self._query_index:
            self._query_index[feedback.query] = []
        self._query_index[feedback.query].append(idx)

    def get_feedback_for_query(
        self,
        query: str,
    ) -> list[FeedbackInstance]:
        """Get all feedback for a query."""
        indices = self._query_index.get(query, [])
        return [self.feedback[i] for i in indices]

    def get_positive_pairs(
        self,
    ) -> list[tuple[str, str]]:
        """Get (query, positive_doc) pairs."""
        pairs = []
        for fb in self.feedback:
            if fb.feedback_type == FeedbackType.RELEVANT:
                pairs.append((fb.query, fb.document.content))
        return pairs

    def get_negative_pairs(
        self,
    ) -> list[tuple[str, str]]:
        """Get (query, negative_doc) pairs."""
        pairs = []
        for fb in self.feedback:
            if fb.feedback_type == FeedbackType.NOT_RELEVANT:
                pairs.append((fb.query, fb.document.content))
        return pairs

    def export_training_data(
        self,
    ) -> list[dict[str, Any]]:
        """Export data for model training."""
        data = []
        for fb in self.feedback:
            data.append(
                {
                    "query": fb.query,
                    "document": fb.document.content,
                    "label": fb.feedback_type.value,
                    "timestamp": fb.timestamp.isoformat(),
                }
            )
        return data


# ============================================================================
# Active Learning Loop
# ============================================================================


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning loop."""

    min_samples_per_cycle: int = 50
    improvement_threshold: float = 0.01
    max_cycles: int = 10
    evaluation_split: float = 0.2


class ActiveLearningLoop:
    """
    Main active learning loop.

    Iteratively selects samples, collects feedback, and improves model.
    """

    def __init__(
        self,
        sampler: ActiveSampler,
        feedback_store: FeedbackStore,
        config: ActiveLearningConfig | None = None,
    ):
        """
        Initialize active learning loop.

        Args:
            sampler: Sample selection component
            feedback_store: Feedback storage
            config: Loop configuration
        """
        self.sampler = sampler
        self.feedback_store = feedback_store
        self.config = config or ActiveLearningConfig()

        self.cycle_history: list[ActiveLearningMetrics] = []
        self.current_cycle = 0

    async def select_for_annotation(
        self,
        query_pool: list[str],
        num_samples: int | None = None,
    ) -> list[QuerySample]:
        """
        Select samples for human annotation.

        Args:
            query_pool: Pool of candidate queries
            num_samples: Number of samples to select

        Returns:
            Selected samples
        """
        if num_samples:
            self.sampler.config.batch_size = num_samples

        samples = await self.sampler.select_samples(query_pool)
        return samples

    def record_feedback(
        self,
        sample: QuerySample,
        doc_index: int,
        feedback_type: FeedbackType,
        annotator_id: str,
        notes: str = "",
    ) -> None:
        """
        Record human feedback for a sample.

        Args:
            sample: The query sample
            doc_index: Index of the document being rated
            feedback_type: Type of feedback
            annotator_id: ID of the annotator
            notes: Optional notes
        """
        if doc_index >= len(sample.documents):
            raise ValueError(f"Invalid doc_index: {doc_index}")

        feedback = FeedbackInstance(
            query=sample.query,
            document=sample.documents[doc_index],
            feedback_type=feedback_type,
            annotator_id=annotator_id,
            timestamp=datetime.now(),
            notes=notes,
        )

        self.feedback_store.add_feedback(feedback)

    def get_training_data(
        self,
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Get training data from accumulated feedback.

        Returns:
            Dict with 'positive' and 'negative' query-doc pairs
        """
        return {
            "positive": self.feedback_store.get_positive_pairs(),
            "negative": self.feedback_store.get_negative_pairs(),
        }

    def compute_cycle_metrics(
        self,
        eval_results: dict[str, float],
    ) -> ActiveLearningMetrics:
        """
        Compute metrics for current cycle.

        Args:
            eval_results: Evaluation results from updated model

        Returns:
            Cycle metrics
        """
        previous_precision = 0.0
        previous_recall = 0.0

        if self.cycle_history:
            previous_precision = self.cycle_history[-1].precision_delta
            previous_recall = self.cycle_history[-1].recall_delta

        metrics = ActiveLearningMetrics(
            total_samples=len(self.feedback_store.feedback),
            annotated_samples=self.config.min_samples_per_cycle,
            model_improvement=eval_results.get("improvement", 0.0),
            precision_delta=eval_results.get("precision", 0.0) - previous_precision,
            recall_delta=eval_results.get("recall", 0.0) - previous_recall,
            annotation_agreement=eval_results.get("agreement", 1.0),
        )

        self.cycle_history.append(metrics)
        self.current_cycle += 1

        return metrics


# ============================================================================
# Factory Functions
# ============================================================================


def create_uncertainty_sampler(
    retriever: RetrieverProtocol,
    batch_size: int = 10,
    threshold: float = 0.7,
) -> ActiveSampler:
    """
    Create uncertainty-based active sampler.

    Args:
        retriever: Retrieval backend
        batch_size: Samples per batch
        threshold: Uncertainty threshold

    Returns:
        Configured ActiveSampler
    """
    uncertainty = ScoreBasedUncertainty()
    config = SamplerConfig(
        strategy=SamplingStrategy.UNCERTAINTY,
        batch_size=batch_size,
        uncertainty_threshold=threshold,
    )
    return ActiveSampler(retriever, uncertainty, config)


def create_llm_uncertainty_sampler(
    retriever: RetrieverProtocol,
    llm: LLMProtocol,
    batch_size: int = 10,
) -> ActiveSampler:
    """
    Create LLM-based uncertainty sampler.

    Args:
        retriever: Retrieval backend
        llm: LLM for uncertainty estimation
        batch_size: Samples per batch

    Returns:
        Configured ActiveSampler
    """
    uncertainty = LLMUncertainty(llm)
    config = SamplerConfig(
        strategy=SamplingStrategy.UNCERTAINTY,
        batch_size=batch_size,
    )
    return ActiveSampler(retriever, uncertainty, config)


def create_active_learning_loop(
    retriever: RetrieverProtocol,
    min_samples: int = 50,
) -> ActiveLearningLoop:
    """
    Create active learning loop.

    Args:
        retriever: Retrieval backend
        min_samples: Minimum samples per cycle

    Returns:
        Configured ActiveLearningLoop
    """
    uncertainty = ScoreBasedUncertainty()
    sampler_config = SamplerConfig(batch_size=min_samples)
    sampler = ActiveSampler(retriever, uncertainty, sampler_config)

    store = FeedbackStore()
    loop_config = ActiveLearningConfig(min_samples_per_cycle=min_samples)

    return ActiveLearningLoop(sampler, store, loop_config)


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example demonstrating active learning."""
    print("Active Learning for RAG Implementation Ready")
    print("=" * 50)
    print("\nFeatures:")
    print("- Uncertainty sampling strategies:")
    print("  - Score-based uncertainty")
    print("  - Embedding distance uncertainty")
    print("  - LLM-based confidence estimation")
    print("- Query-by-committee (retriever ensemble)")
    print("- Feedback collection and storage")
    print("- Training data export")
    print("- Active learning loop with metrics")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
