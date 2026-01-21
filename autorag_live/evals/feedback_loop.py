"""
Feedback Learning Loop Module.

Implements online learning from user feedback to continuously
improve RAG system performance over time.

Key Features:
1. Feedback collection and storage
2. Signal extraction from implicit/explicit feedback
3. Retriever fine-tuning with feedback
4. Reranker adaptation
5. A/B testing framework

Example:
    >>> feedback_loop = FeedbackLoop(retriever)
    >>> feedback_loop.record_feedback(query, response, rating=5)
    >>> await feedback_loop.apply_updates()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Protocols and Interfaces
# =============================================================================


class RetrieverProtocol(Protocol):
    """Protocol for retriever interface."""

    async def retrieve(self, query: str, k: int = 5) -> List[Any]:
        """Retrieve relevant documents."""
        ...


class RerankerProtocol(Protocol):
    """Protocol for reranker interface."""

    async def rerank(
        self,
        query: str,
        documents: List[Any],
    ) -> List[Any]:
        """Rerank documents."""
        ...


# =============================================================================
# Data Structures
# =============================================================================


class FeedbackType(str, Enum):
    """Types of feedback."""

    EXPLICIT_RATING = "explicit_rating"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CLICK = "click"
    DWELL_TIME = "dwell_time"
    COPY = "copy"
    REGENERATE = "regenerate"
    EDIT = "edit"


class FeedbackSignal(str, Enum):
    """Derived feedback signals."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class FeedbackRecord:
    """
    A recorded feedback instance.

    Attributes:
        query: Original query
        response: Generated response
        documents: Retrieved documents
        feedback_type: Type of feedback
        value: Feedback value
        signal: Derived signal
        timestamp: When feedback was recorded
        metadata: Additional metadata
    """

    query: str
    response: str
    documents: List[str]
    feedback_type: FeedbackType
    value: Any
    signal: FeedbackSignal = FeedbackSignal.NEUTRAL
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    query_id: str = ""

    def __post_init__(self):
        """Derive signal from feedback."""
        if not self.query_id:
            self.query_id = hashlib.md5(f"{self.query}{self.timestamp}".encode()).hexdigest()[:12]

        self.signal = self._derive_signal()

    def _derive_signal(self) -> FeedbackSignal:
        """Derive signal from feedback type and value."""
        if self.feedback_type == FeedbackType.EXPLICIT_RATING:
            if isinstance(self.value, (int, float)):
                if self.value >= 4:
                    return FeedbackSignal.POSITIVE
                elif self.value <= 2:
                    return FeedbackSignal.NEGATIVE
            return FeedbackSignal.NEUTRAL

        elif self.feedback_type in (FeedbackType.THUMBS_UP, FeedbackType.COPY):
            return FeedbackSignal.POSITIVE

        elif self.feedback_type in (
            FeedbackType.THUMBS_DOWN,
            FeedbackType.REGENERATE,
        ):
            return FeedbackSignal.NEGATIVE

        elif self.feedback_type == FeedbackType.DWELL_TIME:
            if isinstance(self.value, (int, float)):
                if self.value > 30:  # seconds
                    return FeedbackSignal.POSITIVE
                elif self.value < 5:
                    return FeedbackSignal.NEGATIVE
            return FeedbackSignal.NEUTRAL

        elif self.feedback_type == FeedbackType.CLICK:
            return FeedbackSignal.POSITIVE

        return FeedbackSignal.NEUTRAL


@dataclass
class TrainingPair:
    """
    A training pair for model updates.

    Attributes:
        query: Query text
        positive_docs: Positively labeled documents
        negative_docs: Negatively labeled documents
        weight: Sample weight
    """

    query: str
    positive_docs: List[str]
    negative_docs: List[str]
    weight: float = 1.0


@dataclass
class FeedbackStats:
    """
    Statistics about feedback.

    Attributes:
        total_feedback: Total feedback count
        positive_rate: Rate of positive feedback
        negative_rate: Rate of negative feedback
        avg_rating: Average explicit rating
    """

    total_feedback: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    avg_rating: float = 0.0
    ratings_count: int = 0
    ratings_sum: float = 0.0

    @property
    def positive_rate(self) -> float:
        """Get positive feedback rate."""
        return self.positive_count / self.total_feedback if self.total_feedback > 0 else 0.0

    @property
    def negative_rate(self) -> float:
        """Get negative feedback rate."""
        return self.negative_count / self.total_feedback if self.total_feedback > 0 else 0.0

    def update(self, record: FeedbackRecord) -> None:
        """Update stats with new record."""
        self.total_feedback += 1

        if record.signal == FeedbackSignal.POSITIVE:
            self.positive_count += 1
        elif record.signal == FeedbackSignal.NEGATIVE:
            self.negative_count += 1
        else:
            self.neutral_count += 1

        if record.feedback_type == FeedbackType.EXPLICIT_RATING:
            if isinstance(record.value, (int, float)):
                self.ratings_sum += record.value
                self.ratings_count += 1
                self.avg_rating = self.ratings_sum / self.ratings_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_feedback": self.total_feedback,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "positive_rate": self.positive_rate,
            "negative_rate": self.negative_rate,
            "avg_rating": self.avg_rating,
        }


# =============================================================================
# Feedback Storage
# =============================================================================


class FeedbackStore(ABC):
    """Abstract base class for feedback storage."""

    @abstractmethod
    def add(self, record: FeedbackRecord) -> None:
        """Add feedback record."""
        pass

    @abstractmethod
    def get_by_query(self, query: str) -> List[FeedbackRecord]:
        """Get feedback for query."""
        pass

    @abstractmethod
    def get_all(self) -> List[FeedbackRecord]:
        """Get all feedback records."""
        pass

    @abstractmethod
    def get_recent(self, n: int) -> List[FeedbackRecord]:
        """Get most recent feedback."""
        pass


class InMemoryFeedbackStore(FeedbackStore):
    """In-memory feedback storage."""

    def __init__(self, max_size: int = 10000):
        """Initialize store."""
        self.max_size = max_size
        self._records: List[FeedbackRecord] = []
        self._query_index: Dict[str, List[int]] = defaultdict(list)

    def add(self, record: FeedbackRecord) -> None:
        """Add feedback record."""
        idx = len(self._records)
        self._records.append(record)

        # Index by query
        query_key = record.query.lower().strip()
        self._query_index[query_key].append(idx)

        # Evict old records if over limit
        if len(self._records) > self.max_size:
            self._evict_oldest(self.max_size // 10)

    def get_by_query(self, query: str) -> List[FeedbackRecord]:
        """Get feedback for query."""
        query_key = query.lower().strip()
        indices = self._query_index.get(query_key, [])
        return [self._records[i] for i in indices if i < len(self._records)]

    def get_all(self) -> List[FeedbackRecord]:
        """Get all records."""
        return list(self._records)

    def get_recent(self, n: int) -> List[FeedbackRecord]:
        """Get most recent records."""
        return self._records[-n:]

    def _evict_oldest(self, count: int) -> None:
        """Evict oldest records."""
        self._records = self._records[count:]
        # Rebuild index
        self._query_index.clear()
        for i, record in enumerate(self._records):
            query_key = record.query.lower().strip()
            self._query_index[query_key].append(i)


# =============================================================================
# Training Data Generation
# =============================================================================


class TrainingDataGenerator:
    """
    Generates training data from feedback.

    Creates positive/negative pairs for model updates.
    """

    def __init__(
        self,
        min_confidence: float = 0.7,
        balance_ratio: float = 1.0,
    ):
        """
        Initialize generator.

        Args:
            min_confidence: Minimum confidence for including samples
            balance_ratio: Ratio of negative to positive samples
        """
        self.min_confidence = min_confidence
        self.balance_ratio = balance_ratio

    def generate_pairs(
        self,
        records: List[FeedbackRecord],
    ) -> List[TrainingPair]:
        """
        Generate training pairs from feedback records.

        Args:
            records: Feedback records

        Returns:
            List of training pairs
        """
        # Group by query
        by_query: Dict[str, List[FeedbackRecord]] = defaultdict(list)
        for record in records:
            by_query[record.query].append(record)

        pairs = []

        for query, query_records in by_query.items():
            positive_docs = []
            negative_docs = []

            for record in query_records:
                if record.signal == FeedbackSignal.POSITIVE:
                    positive_docs.extend(record.documents)
                elif record.signal == FeedbackSignal.NEGATIVE:
                    negative_docs.extend(record.documents)

            if positive_docs:
                # Deduplicate
                positive_docs = list(set(positive_docs))
                negative_docs = list(set(negative_docs))

                # Balance if needed
                if negative_docs and self.balance_ratio > 0:
                    max_neg = int(len(positive_docs) * self.balance_ratio)
                    negative_docs = negative_docs[:max_neg]

                pairs.append(
                    TrainingPair(
                        query=query,
                        positive_docs=positive_docs,
                        negative_docs=negative_docs,
                        weight=len(query_records) / len(records),
                    )
                )

        return pairs

    def generate_contrastive_pairs(
        self,
        records: List[FeedbackRecord],
    ) -> List[Tuple[str, str, str]]:
        """
        Generate contrastive triplets (query, positive, negative).

        Args:
            records: Feedback records

        Returns:
            List of (query, positive_doc, negative_doc) triplets
        """
        pairs = self.generate_pairs(records)
        triplets = []

        for pair in pairs:
            for pos in pair.positive_docs:
                for neg in pair.negative_docs:
                    triplets.append((pair.query, pos, neg))

        return triplets


# =============================================================================
# Model Adapters
# =============================================================================


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    @abstractmethod
    async def update(self, training_data: List[TrainingPair]) -> Dict[str, Any]:
        """Update model with training data."""
        pass


class RetrievalAdapter(ModelAdapter):
    """
    Adapter for retrieval model updates.

    Updates retrieval scores based on feedback.
    """

    def __init__(self, learning_rate: float = 0.1):
        """Initialize adapter."""
        self.learning_rate = learning_rate
        self._score_adjustments: Dict[str, Dict[str, float]] = defaultdict(dict)

    async def update(
        self,
        training_data: List[TrainingPair],
    ) -> Dict[str, Any]:
        """Update retrieval scores."""
        updated_pairs = 0

        for pair in training_data:
            query_key = pair.query.lower().strip()

            # Boost positive docs
            for doc in pair.positive_docs:
                doc_key = hashlib.md5(doc[:100].encode()).hexdigest()
                current = self._score_adjustments[query_key].get(doc_key, 0.0)
                self._score_adjustments[query_key][doc_key] = (
                    current + self.learning_rate * pair.weight
                )
                updated_pairs += 1

            # Penalize negative docs
            for doc in pair.negative_docs:
                doc_key = hashlib.md5(doc[:100].encode()).hexdigest()
                current = self._score_adjustments[query_key].get(doc_key, 0.0)
                self._score_adjustments[query_key][doc_key] = (
                    current - self.learning_rate * pair.weight
                )
                updated_pairs += 1

        return {
            "updated_pairs": updated_pairs,
            "total_adjustments": sum(len(v) for v in self._score_adjustments.values()),
        }

    def get_score_adjustment(self, query: str, doc: str) -> float:
        """Get score adjustment for query-doc pair."""
        query_key = query.lower().strip()
        doc_key = hashlib.md5(doc[:100].encode()).hexdigest()
        return self._score_adjustments[query_key].get(doc_key, 0.0)


class PromptAdapter(ModelAdapter):
    """
    Adapter for prompt optimization.

    Learns from feedback to improve prompts.
    """

    def __init__(self):
        """Initialize adapter."""
        self._successful_patterns: List[str] = []
        self._failed_patterns: List[str] = []

    async def update(
        self,
        training_data: List[TrainingPair],
    ) -> Dict[str, Any]:
        """Update prompt patterns."""
        for pair in training_data:
            if pair.positive_docs:
                self._successful_patterns.append(pair.query)
            if pair.negative_docs and not pair.positive_docs:
                self._failed_patterns.append(pair.query)

        return {
            "successful_patterns": len(self._successful_patterns),
            "failed_patterns": len(self._failed_patterns),
        }


# =============================================================================
# A/B Testing
# =============================================================================


@dataclass
class Experiment:
    """
    An A/B test experiment.

    Attributes:
        name: Experiment name
        variants: Variant configurations
        traffic_split: Traffic allocation per variant
        metrics: Collected metrics per variant
    """

    name: str
    variants: List[str]
    traffic_split: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, FeedbackStats] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    is_active: bool = True

    def __post_init__(self):
        """Initialize metrics and traffic split."""
        if not self.traffic_split:
            split = 1.0 / len(self.variants)
            self.traffic_split = {v: split for v in self.variants}

        for variant in self.variants:
            if variant not in self.metrics:
                self.metrics[variant] = FeedbackStats()

    def assign_variant(self, user_id: str) -> str:
        """Assign user to variant."""
        # Deterministic assignment based on user_id hash
        hash_val = int(hashlib.md5(f"{self.name}{user_id}".encode()).hexdigest(), 16)
        rand = (hash_val % 10000) / 10000.0

        cumulative = 0.0
        for variant, split in self.traffic_split.items():
            cumulative += split
            if rand < cumulative:
                return variant

        return self.variants[-1]

    def record_feedback(
        self,
        variant: str,
        record: FeedbackRecord,
    ) -> None:
        """Record feedback for variant."""
        if variant in self.metrics:
            self.metrics[variant].update(record)

    def get_results(self) -> Dict[str, Any]:
        """Get experiment results."""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "variants": {v: stats.to_dict() for v, stats in self.metrics.items()},
        }


class ABTestingFramework:
    """
    A/B testing framework for RAG systems.

    Manages experiments and tracks metrics.
    """

    def __init__(self):
        """Initialize framework."""
        self._experiments: Dict[str, Experiment] = {}

    def create_experiment(
        self,
        name: str,
        variants: List[str],
        traffic_split: Optional[Dict[str, float]] = None,
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            variants: Variant names
            traffic_split: Optional traffic allocation

        Returns:
            Created experiment
        """
        experiment = Experiment(
            name=name,
            variants=variants,
            traffic_split=traffic_split or {},
        )
        self._experiments[name] = experiment
        return experiment

    def get_experiment(self, name: str) -> Optional[Experiment]:
        """Get experiment by name."""
        return self._experiments.get(name)

    def assign_variant(
        self,
        experiment_name: str,
        user_id: str,
    ) -> Optional[str]:
        """Assign user to experiment variant."""
        experiment = self._experiments.get(experiment_name)
        if experiment and experiment.is_active:
            return experiment.assign_variant(user_id)
        return None

    def record_feedback(
        self,
        experiment_name: str,
        variant: str,
        record: FeedbackRecord,
    ) -> None:
        """Record feedback for experiment."""
        experiment = self._experiments.get(experiment_name)
        if experiment:
            experiment.record_feedback(variant, record)

    def get_all_results(self) -> Dict[str, Any]:
        """Get results for all experiments."""
        return {name: exp.get_results() for name, exp in self._experiments.items()}


# =============================================================================
# Main Feedback Loop
# =============================================================================


class FeedbackLoop:
    """
    Main feedback learning loop.

    Collects feedback and applies updates to improve RAG performance.

    Example:
        >>> loop = FeedbackLoop()
        >>> loop.record_feedback(
        ...     query="What is ML?",
        ...     response="Machine learning is...",
        ...     documents=["doc1", "doc2"],
        ...     feedback_type=FeedbackType.THUMBS_UP,
        ... )
        >>> await loop.apply_updates()
    """

    def __init__(
        self,
        store: Optional[FeedbackStore] = None,
        adapters: Optional[List[ModelAdapter]] = None,
        update_interval: int = 100,
        min_samples: int = 10,
    ):
        """
        Initialize feedback loop.

        Args:
            store: Feedback storage
            adapters: Model adapters
            update_interval: Feedback count between updates
            min_samples: Minimum samples for update
        """
        self.store = store or InMemoryFeedbackStore()
        self.adapters = adapters or [RetrievalAdapter()]
        self.update_interval = update_interval
        self.min_samples = min_samples

        self.generator = TrainingDataGenerator()
        self.ab_testing = ABTestingFramework()
        self.stats = FeedbackStats()

        self._feedback_since_update = 0
        self._update_lock = asyncio.Lock()

    def record_feedback(
        self,
        query: str,
        response: str,
        documents: List[str],
        feedback_type: FeedbackType,
        value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackRecord:
        """
        Record user feedback.

        Args:
            query: Query text
            response: Generated response
            documents: Retrieved documents
            feedback_type: Type of feedback
            value: Feedback value
            metadata: Additional metadata

        Returns:
            Created feedback record
        """
        record = FeedbackRecord(
            query=query,
            response=response,
            documents=documents,
            feedback_type=feedback_type,
            value=value,
            metadata=metadata or {},
        )

        self.store.add(record)
        self.stats.update(record)
        self._feedback_since_update += 1

        logger.info(f"Recorded feedback: {feedback_type.value} " f"(signal={record.signal.value})")

        return record

    def record_rating(
        self,
        query: str,
        response: str,
        documents: List[str],
        rating: int,
    ) -> FeedbackRecord:
        """Convenience method for rating feedback."""
        return self.record_feedback(
            query=query,
            response=response,
            documents=documents,
            feedback_type=FeedbackType.EXPLICIT_RATING,
            value=rating,
        )

    def record_thumbs_up(
        self,
        query: str,
        response: str,
        documents: List[str],
    ) -> FeedbackRecord:
        """Convenience method for thumbs up."""
        return self.record_feedback(
            query=query,
            response=response,
            documents=documents,
            feedback_type=FeedbackType.THUMBS_UP,
            value=True,
        )

    def record_thumbs_down(
        self,
        query: str,
        response: str,
        documents: List[str],
    ) -> FeedbackRecord:
        """Convenience method for thumbs down."""
        return self.record_feedback(
            query=query,
            response=response,
            documents=documents,
            feedback_type=FeedbackType.THUMBS_DOWN,
            value=True,
        )

    async def apply_updates(
        self,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply model updates from feedback.

        Args:
            force: Force update regardless of thresholds

        Returns:
            Update results
        """
        async with self._update_lock:
            if not force:
                if self._feedback_since_update < self.update_interval:
                    return {"skipped": True, "reason": "insufficient_feedback"}

            records = self.store.get_all()
            if len(records) < self.min_samples:
                return {"skipped": True, "reason": "insufficient_samples"}

            # Generate training data
            training_data = self.generator.generate_pairs(records)

            if not training_data:
                return {"skipped": True, "reason": "no_training_data"}

            # Apply updates to each adapter
            results = {}
            for adapter in self.adapters:
                adapter_name = type(adapter).__name__
                try:
                    result = await adapter.update(training_data)
                    results[adapter_name] = result
                except Exception as e:
                    logger.error(f"Adapter update failed: {e}")
                    results[adapter_name] = {"error": str(e)}

            self._feedback_since_update = 0

            return {
                "skipped": False,
                "training_pairs": len(training_data),
                "adapters": results,
            }

    async def maybe_update(self) -> Optional[Dict[str, Any]]:
        """Apply updates if threshold reached."""
        if self._feedback_since_update >= self.update_interval:
            return await self.apply_updates()
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        return {
            "overall": self.stats.to_dict(),
            "feedback_since_update": self._feedback_since_update,
            "total_records": len(self.store.get_all()),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_feedback_loop(
    update_interval: int = 100,
) -> FeedbackLoop:
    """
    Create a feedback loop.

    Args:
        update_interval: Updates between model refreshes

    Returns:
        FeedbackLoop instance
    """
    return FeedbackLoop(update_interval=update_interval)


async def collect_and_update(
    loop: FeedbackLoop,
    query: str,
    response: str,
    documents: List[str],
    rating: int,
) -> Dict[str, Any]:
    """
    Collect feedback and apply updates.

    Args:
        loop: Feedback loop
        query: Query text
        response: Response text
        documents: Retrieved documents
        rating: User rating

    Returns:
        Update results
    """
    loop.record_rating(query, response, documents, rating)
    result = await loop.maybe_update()
    return result or {"status": "recorded"}


# =============================================================================
# OPTIMIZATION 8: Online Contextual Bandit for RAG Component Selection
# Based on: "Contextual Bandits for Adaptive RAG" and
# "LinUCB: A Contextual Bandit Approach" (Li et al., 2010)
#
# Implements online learning for:
# 1. Retriever selection based on query context
# 2. Reranker adaptation with feedback
# 3. Exploration-exploitation balance using UCB
# 4. Non-stationary adaptation for concept drift
# =============================================================================


@dataclass
class Arm:
    """An arm (option) in the bandit problem."""

    name: str
    component: Any  # The actual retriever/reranker/etc
    feature_dim: int = 10

    # LinUCB parameters
    A: Optional[List[List[float]]] = None  # d x d matrix
    b: Optional[List[float]] = None  # d x 1 vector
    theta: Optional[List[float]] = None  # Learned weights

    # Statistics
    pulls: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0

    def __post_init__(self):
        """Initialize LinUCB matrices."""
        if self.A is None:
            # Identity matrix
            self.A = [
                [1.0 if i == j else 0.0 for j in range(self.feature_dim)]
                for i in range(self.feature_dim)
            ]
        if self.b is None:
            self.b = [0.0] * self.feature_dim
        if self.theta is None:
            self.theta = [0.0] * self.feature_dim


@dataclass
class BanditContext:
    """Context features for bandit decision."""

    features: List[float]
    query: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BanditDecision:
    """A decision made by the bandit."""

    arm: str
    predicted_reward: float
    ucb_score: float
    exploration_bonus: float
    context_features: List[float] = field(default_factory=list)


@dataclass
class BanditReward:
    """Reward feedback for a bandit decision."""

    arm: str
    reward: float
    context: BanditContext
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextualBandit(ABC):
    """Abstract base class for contextual bandits."""

    @abstractmethod
    def select(self, context: BanditContext) -> BanditDecision:
        """Select an arm given context."""
        pass

    @abstractmethod
    def update(self, reward: BanditReward) -> None:
        """Update from reward feedback."""
        pass


class LinUCB(ContextualBandit):
    """
    Linear Upper Confidence Bound bandit.

    Implements LinUCB algorithm for contextual bandits with
    linear reward models and UCB-style exploration.

    Example:
        >>> bandit = LinUCB(alpha=1.0, feature_dim=10)
        >>> bandit.add_arm("dense_retriever", dense_retriever)
        >>> bandit.add_arm("sparse_retriever", sparse_retriever)
        >>> decision = bandit.select(context)
        >>> # Use decision.arm
        >>> bandit.update(BanditReward(arm=decision.arm, reward=0.8, context=context))
    """

    def __init__(
        self,
        alpha: float = 1.0,
        feature_dim: int = 10,
        decay_rate: float = 0.0,
    ):
        """
        Initialize LinUCB bandit.

        Args:
            alpha: Exploration parameter (higher = more exploration)
            feature_dim: Dimension of context features
            decay_rate: Discount for non-stationarity (0 = none)
        """
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.decay_rate = decay_rate

        self.arms: Dict[str, Arm] = {}

        # Statistics
        self._stats = {
            "total_selections": 0,
            "selections_per_arm": defaultdict(int),
            "rewards_per_arm": defaultdict(list),
        }

    def add_arm(self, name: str, component: Any) -> None:
        """Add an arm to the bandit."""
        self.arms[name] = Arm(
            name=name,
            component=component,
            feature_dim=self.feature_dim,
        )

    def select(self, context: BanditContext) -> BanditDecision:
        """
        Select arm using LinUCB.

        Args:
            context: Context features

        Returns:
            BanditDecision with selected arm
        """
        if not self.arms:
            raise ValueError("No arms registered")

        best_arm = None
        best_ucb = float("-inf")
        best_pred = 0.0
        best_bonus = 0.0

        x = context.features
        if len(x) != self.feature_dim:
            # Pad or truncate
            x = (x + [0.0] * self.feature_dim)[: self.feature_dim]

        for arm_name, arm in self.arms.items():
            # Compute A^-1
            A_inv = self._matrix_inverse(arm.A)

            # Compute theta = A^-1 * b
            theta = self._matrix_vector_mult(A_inv, arm.b)

            # Predicted reward: theta^T * x
            pred_reward = sum(t * xi for t, xi in zip(theta, x))

            # UCB bonus: alpha * sqrt(x^T * A^-1 * x)
            Ax = self._matrix_vector_mult(A_inv, x)
            variance = sum(xi * axi for xi, axi in zip(x, Ax))
            bonus = self.alpha * math.sqrt(max(0, variance))

            ucb = pred_reward + bonus

            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm_name
                best_pred = pred_reward
                best_bonus = bonus

        self._stats["total_selections"] += 1
        self._stats["selections_per_arm"][best_arm] += 1

        return BanditDecision(
            arm=best_arm,
            predicted_reward=best_pred,
            ucb_score=best_ucb,
            exploration_bonus=best_bonus,
            context_features=x,
        )

    def update(self, reward: BanditReward) -> None:
        """
        Update arm parameters from reward.

        Args:
            reward: Reward feedback
        """
        if reward.arm not in self.arms:
            return

        arm = self.arms[reward.arm]
        x = reward.context.features
        if len(x) != self.feature_dim:
            x = (x + [0.0] * self.feature_dim)[: self.feature_dim]

        r = reward.reward

        # Apply decay for non-stationarity
        if self.decay_rate > 0:
            decay = 1 - self.decay_rate
            arm.A = [[decay * a for a in row] for row in arm.A]
            arm.b = [decay * b for b in arm.b]

            # Re-add identity for stability
            for i in range(self.feature_dim):
                arm.A[i][i] += self.decay_rate

        # Update A = A + x * x^T
        for i in range(self.feature_dim):
            for j in range(self.feature_dim):
                arm.A[i][j] += x[i] * x[j]

        # Update b = b + r * x
        for i in range(self.feature_dim):
            arm.b[i] += r * x[i]

        # Update statistics
        arm.pulls += 1
        arm.total_reward += r
        arm.avg_reward = arm.total_reward / arm.pulls

        self._stats["rewards_per_arm"][reward.arm].append(r)

    def _matrix_inverse(self, A: List[List[float]]) -> List[List[float]]:
        """Compute matrix inverse using Gaussian elimination."""
        n = len(A)
        # Augment with identity
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]

        # Forward elimination
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]

            # Check for singularity
            if abs(aug[i][i]) < 1e-10:
                aug[i][i] = 1e-10

            # Eliminate column
            for k in range(i + 1, n):
                factor = aug[k][i] / aug[i][i]
                for j in range(2 * n):
                    aug[k][j] -= factor * aug[i][j]

        # Back substitution
        for i in range(n - 1, -1, -1):
            for k in range(i - 1, -1, -1):
                factor = aug[k][i] / aug[i][i]
                for j in range(2 * n):
                    aug[k][j] -= factor * aug[i][j]

        # Normalize and extract inverse
        result = []
        for i in range(n):
            row = []
            for j in range(n, 2 * n):
                row.append(aug[i][j] / aug[i][i])
            result.append(row)

        return result

    def _matrix_vector_mult(self, A: List[List[float]], x: List[float]) -> List[float]:
        """Multiply matrix by vector."""
        return [sum(a * xi for a, xi in zip(row, x)) for row in A]

    def get_component(self, arm: str) -> Any:
        """Get component for an arm."""
        return self.arms[arm].component if arm in self.arms else None

    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        return {
            **self._stats,
            "arm_stats": {
                name: {"pulls": arm.pulls, "avg_reward": arm.avg_reward}
                for name, arm in self.arms.items()
            },
        }


class RAGComponentBandit:
    """
    Bandit-based component selection for RAG pipelines.

    Uses LinUCB to select between multiple retrievers, rerankers,
    or other components based on query context.
    """

    def __init__(
        self,
        feature_extractor: Optional[Callable[[str], List[float]]] = None,
        alpha: float = 1.0,
        feature_dim: int = 10,
    ):
        """
        Initialize RAG component bandit.

        Args:
            feature_extractor: Function to extract features from query
            alpha: Exploration parameter
            feature_dim: Feature dimension
        """
        self.feature_extractor = feature_extractor or self._default_features
        self.feature_dim = feature_dim

        # Separate bandits for different component types
        self.retriever_bandit = LinUCB(alpha=alpha, feature_dim=feature_dim)
        self.reranker_bandit = LinUCB(alpha=alpha, feature_dim=feature_dim)

    def add_retriever(self, name: str, retriever: Any) -> None:
        """Add a retriever option."""
        self.retriever_bandit.add_arm(name, retriever)

    def add_reranker(self, name: str, reranker: Any) -> None:
        """Add a reranker option."""
        self.reranker_bandit.add_arm(name, reranker)

    def select_retriever(self, query: str) -> Tuple[str, Any]:
        """
        Select retriever for query.

        Args:
            query: User query

        Returns:
            Tuple of (retriever_name, retriever_component)
        """
        features = self.feature_extractor(query)
        context = BanditContext(features=features, query=query)

        decision = self.retriever_bandit.select(context)
        component = self.retriever_bandit.get_component(decision.arm)

        return decision.arm, component

    def select_reranker(self, query: str) -> Tuple[str, Any]:
        """
        Select reranker for query.

        Args:
            query: User query

        Returns:
            Tuple of (reranker_name, reranker_component)
        """
        features = self.feature_extractor(query)
        context = BanditContext(features=features, query=query)

        decision = self.reranker_bandit.select(context)
        component = self.reranker_bandit.get_component(decision.arm)

        return decision.arm, component

    def update_retriever(self, query: str, retriever_name: str, reward: float) -> None:
        """Update retriever selection from feedback."""
        features = self.feature_extractor(query)
        context = BanditContext(features=features, query=query)

        self.retriever_bandit.update(
            BanditReward(arm=retriever_name, reward=reward, context=context)
        )

    def update_reranker(self, query: str, reranker_name: str, reward: float) -> None:
        """Update reranker selection from feedback."""
        features = self.feature_extractor(query)
        context = BanditContext(features=features, query=query)

        self.reranker_bandit.update(BanditReward(arm=reranker_name, reward=reward, context=context))

    def _default_features(self, query: str) -> List[float]:
        """Extract default features from query."""
        # Simple feature extraction
        words = query.lower().split()

        features = [
            len(query) / 100,  # Normalized length
            len(words) / 20,  # Normalized word count
            query.count("?") / 3,  # Question marks
            query.count(",") / 5,  # Complexity indicator
            1.0 if any(w in query.lower() for w in ["how", "why", "explain"]) else 0.0,
            1.0 if any(w in query.lower() for w in ["what", "who", "when", "where"]) else 0.0,
            1.0 if any(w in query.lower() for w in ["compare", "difference", "vs"]) else 0.0,
            sum(1 for c in query if c.isupper()) / max(1, len(query)),  # Caps ratio
            len(set(words)) / max(1, len(words)),  # Vocabulary diversity
            1.0 if len(words) > 10 else 0.0,  # Long query indicator
        ]

        return features[: self.feature_dim]

    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        return {
            "retriever_bandit": self.retriever_bandit.get_stats(),
            "reranker_bandit": self.reranker_bandit.get_stats(),
        }


class EpsilonGreedy(ContextualBandit):
    """
    Simple epsilon-greedy bandit for comparison.

    Explores with probability epsilon, otherwise exploits best arm.
    """

    def __init__(self, epsilon: float = 0.1):
        """Initialize epsilon-greedy bandit."""
        self.epsilon = epsilon
        self.arms: Dict[str, Arm] = {}

    def add_arm(self, name: str, component: Any) -> None:
        """Add an arm."""
        self.arms[name] = Arm(name=name, component=component)

    def select(self, context: BanditContext) -> BanditDecision:
        """Select arm using epsilon-greedy."""
        if random.random() < self.epsilon:
            # Explore
            arm_name = random.choice(list(self.arms.keys()))
        else:
            # Exploit
            arm_name = max(self.arms.keys(), key=lambda a: self.arms[a].avg_reward)

        arm = self.arms[arm_name]
        return BanditDecision(
            arm=arm_name,
            predicted_reward=arm.avg_reward,
            ucb_score=arm.avg_reward,
            exploration_bonus=0.0,
        )

    def update(self, reward: BanditReward) -> None:
        """Update arm statistics."""
        if reward.arm not in self.arms:
            return

        arm = self.arms[reward.arm]
        arm.pulls += 1
        arm.total_reward += reward.reward
        arm.avg_reward = arm.total_reward / arm.pulls


def create_rag_bandit(
    alpha: float = 1.0,
    feature_dim: int = 10,
) -> RAGComponentBandit:
    """
    Create a RAG component bandit.

    Args:
        alpha: Exploration parameter
        feature_dim: Feature dimension

    Returns:
        Configured RAGComponentBandit
    """
    return RAGComponentBandit(alpha=alpha, feature_dim=feature_dim)
