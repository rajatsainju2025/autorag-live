"""Feedback Collection Module for AutoRAG-Live.

Collect and process user feedback for RAG improvement:
- Explicit feedback (thumbs up/down, ratings)
- Implicit feedback (clicks, dwell time)
- Detailed annotations
- Aggregation and analysis
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback."""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    CLICK = "click"
    DWELL_TIME = "dwell_time"
    ANNOTATION = "annotation"
    CORRECTION = "correction"
    FLAG = "flag"


class FeedbackCategory(Enum):
    """Categories for negative feedback."""

    IRRELEVANT = "irrelevant"
    INACCURATE = "inaccurate"
    INCOMPLETE = "incomplete"
    OUTDATED = "outdated"
    CONFUSING = "confusing"
    HALLUCINATION = "hallucination"
    OFF_TOPIC = "off_topic"
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"
    OTHER = "other"


@dataclass
class FeedbackItem:
    """Represents a single feedback item."""

    feedback_id: str
    feedback_type: FeedbackType
    timestamp: datetime = field(default_factory=datetime.now)

    # Context
    query: str = ""
    answer: str = ""
    document_ids: list[str] = field(default_factory=list)

    # Feedback data
    value: float | None = None  # For ratings (0-1)
    is_positive: bool | None = None
    category: FeedbackCategory | None = None
    comment: str = ""
    correction: str = ""

    # Metadata
    user_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "answer": self.answer,
            "document_ids": self.document_ids,
            "value": self.value,
            "is_positive": self.is_positive,
            "category": self.category.value if self.category else None,
            "comment": self.comment,
            "correction": self.correction,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeedbackItem":
        """Create from dictionary."""
        return cls(
            feedback_id=data["feedback_id"],
            feedback_type=FeedbackType(data["feedback_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            query=data.get("query", ""),
            answer=data.get("answer", ""),
            document_ids=data.get("document_ids", []),
            value=data.get("value"),
            is_positive=data.get("is_positive"),
            category=FeedbackCategory(data["category"]) if data.get("category") else None,
            comment=data.get("comment", ""),
            correction=data.get("correction", ""),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class FeedbackSummary:
    """Summary statistics for feedback."""

    total_feedback: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    average_rating: float = 0.0
    category_counts: dict[str, int] = field(default_factory=dict)
    document_scores: dict[str, float] = field(default_factory=dict)
    recent_feedback: list[FeedbackItem] = field(default_factory=list)

    @property
    def positive_rate(self) -> float:
        """Get positive feedback rate."""
        total = self.positive_count + self.negative_count
        return self.positive_count / total if total > 0 else 0.5


class BaseFeedbackStore(ABC):
    """Abstract base class for feedback storage."""

    @abstractmethod
    def save(self, feedback: FeedbackItem) -> bool:
        """Save feedback item.

        Args:
            feedback: Feedback to save

        Returns:
            Success status
        """
        pass

    @abstractmethod
    def get(self, feedback_id: str) -> FeedbackItem | None:
        """Get feedback by ID.

        Args:
            feedback_id: Feedback identifier

        Returns:
            Feedback item or None
        """
        pass

    @abstractmethod
    def query(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[FeedbackItem]:
        """Query feedback items.

        Args:
            filters: Filter criteria
            limit: Maximum results

        Returns:
            List of feedback items
        """
        pass

    @abstractmethod
    def get_document_feedback(self, doc_id: str) -> list[FeedbackItem]:
        """Get all feedback for a document.

        Args:
            doc_id: Document identifier

        Returns:
            List of feedback items
        """
        pass


class InMemoryFeedbackStore(BaseFeedbackStore):
    """In-memory feedback storage."""

    def __init__(self) -> None:
        """Initialize in-memory store."""
        self._feedback: dict[str, FeedbackItem] = {}
        self._document_index: dict[str, list[str]] = {}

    def save(self, feedback: FeedbackItem) -> bool:
        """Save feedback to memory."""
        self._feedback[feedback.feedback_id] = feedback

        # Index by document
        for doc_id in feedback.document_ids:
            if doc_id not in self._document_index:
                self._document_index[doc_id] = []
            if feedback.feedback_id not in self._document_index[doc_id]:
                self._document_index[doc_id].append(feedback.feedback_id)

        return True

    def get(self, feedback_id: str) -> FeedbackItem | None:
        """Get feedback by ID."""
        return self._feedback.get(feedback_id)

    def query(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[FeedbackItem]:
        """Query feedback items."""
        items = list(self._feedback.values())

        if filters:
            # Apply filters
            if "feedback_type" in filters:
                ft = filters["feedback_type"]
                items = [i for i in items if i.feedback_type.value == ft]

            if "is_positive" in filters:
                pos = filters["is_positive"]
                items = [i for i in items if i.is_positive == pos]

            if "user_id" in filters:
                uid = filters["user_id"]
                items = [i for i in items if i.user_id == uid]

            if "since" in filters:
                since = filters["since"]
                items = [i for i in items if i.timestamp >= since]

        # Sort by timestamp descending
        items.sort(key=lambda x: x.timestamp, reverse=True)

        return items[:limit]

    def get_document_feedback(self, doc_id: str) -> list[FeedbackItem]:
        """Get feedback for document."""
        feedback_ids = self._document_index.get(doc_id, [])
        return [self._feedback[fid] for fid in feedback_ids if fid in self._feedback]


class FileFeedbackStore(BaseFeedbackStore):
    """File-based feedback storage."""

    def __init__(self, storage_path: str | Path) -> None:
        """Initialize file-based store.

        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self.storage_path / "index.json"
        self._index: dict[str, str] = {}  # feedback_id -> filename
        self._load_index()

    def _load_index(self) -> None:
        """Load feedback index."""
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text())
            except Exception as e:
                logger.warning(f"Failed to load feedback index: {e}")
                self._index = {}

    def _save_index(self) -> None:
        """Save feedback index."""
        try:
            self._index_path.write_text(json.dumps(self._index))
        except Exception as e:
            logger.error(f"Failed to save feedback index: {e}")

    def save(self, feedback: FeedbackItem) -> bool:
        """Save feedback to file."""
        try:
            filename = f"{feedback.feedback_id}.json"
            filepath = self.storage_path / filename

            filepath.write_text(json.dumps(feedback.to_dict(), indent=2))

            self._index[feedback.feedback_id] = filename
            self._save_index()

            return True
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return False

    def get(self, feedback_id: str) -> FeedbackItem | None:
        """Get feedback by ID."""
        filename = self._index.get(feedback_id)
        if not filename:
            return None

        filepath = self.storage_path / filename
        if not filepath.exists():
            return None

        try:
            data = json.loads(filepath.read_text())
            return FeedbackItem.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load feedback {feedback_id}: {e}")
            return None

    def query(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[FeedbackItem]:
        """Query feedback items."""
        items: list[FeedbackItem] = []

        for feedback_id in self._index:
            feedback = self.get(feedback_id)
            if feedback:
                items.append(feedback)

        # Apply filters (same as in-memory implementation)
        if filters:
            if "feedback_type" in filters:
                ft = filters["feedback_type"]
                items = [i for i in items if i.feedback_type.value == ft]

            if "is_positive" in filters:
                pos = filters["is_positive"]
                items = [i for i in items if i.is_positive == pos]

            if "user_id" in filters:
                uid = filters["user_id"]
                items = [i for i in items if i.user_id == uid]

        items.sort(key=lambda x: x.timestamp, reverse=True)
        return items[:limit]

    def get_document_feedback(self, doc_id: str) -> list[FeedbackItem]:
        """Get feedback for document."""
        items: list[FeedbackItem] = []

        for feedback in self.query(limit=10000):
            if doc_id in feedback.document_ids:
                items.append(feedback)

        return items


class FeedbackCollector:
    """Collects and processes user feedback."""

    def __init__(
        self,
        store: BaseFeedbackStore | None = None,
    ) -> None:
        """Initialize feedback collector.

        Args:
            store: Feedback storage backend
        """
        self.store = store or InMemoryFeedbackStore()

    def _generate_id(self, content: str) -> str:
        """Generate feedback ID."""
        timestamp = datetime.now().isoformat()
        content_hash = hashlib.md5(f"{timestamp}:{content}".encode()).hexdigest()[:8]
        return f"fb_{content_hash}"

    def collect_thumbs(
        self,
        is_positive: bool,
        query: str,
        answer: str,
        document_ids: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> FeedbackItem:
        """Collect thumbs up/down feedback.

        Args:
            is_positive: Thumbs up (True) or down (False)
            query: User query
            answer: Generated answer
            document_ids: Related document IDs
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Created feedback item
        """
        feedback = FeedbackItem(
            feedback_id=self._generate_id(query),
            feedback_type=FeedbackType.THUMBS_UP if is_positive else FeedbackType.THUMBS_DOWN,
            query=query,
            answer=answer,
            document_ids=document_ids or [],
            value=1.0 if is_positive else 0.0,
            is_positive=is_positive,
            user_id=user_id,
            session_id=session_id,
        )

        self.store.save(feedback)
        return feedback

    def collect_rating(
        self,
        rating: float,
        query: str,
        answer: str,
        document_ids: list[str] | None = None,
        comment: str = "",
        user_id: str | None = None,
    ) -> FeedbackItem:
        """Collect numerical rating feedback.

        Args:
            rating: Rating value (0-5 or 0-1)
            query: User query
            answer: Generated answer
            document_ids: Related document IDs
            comment: Optional comment
            user_id: User identifier

        Returns:
            Created feedback item
        """
        # Normalize rating to 0-1
        if rating > 1:
            rating = rating / 5.0

        feedback = FeedbackItem(
            feedback_id=self._generate_id(query),
            feedback_type=FeedbackType.RATING,
            query=query,
            answer=answer,
            document_ids=document_ids or [],
            value=rating,
            is_positive=rating >= 0.6,
            comment=comment,
            user_id=user_id,
        )

        self.store.save(feedback)
        return feedback

    def collect_annotation(
        self,
        query: str,
        answer: str,
        category: FeedbackCategory,
        comment: str = "",
        correction: str = "",
        document_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> FeedbackItem:
        """Collect detailed annotation feedback.

        Args:
            query: User query
            answer: Generated answer
            category: Feedback category
            comment: Additional comment
            correction: Suggested correction
            document_ids: Related document IDs
            user_id: User identifier

        Returns:
            Created feedback item
        """
        feedback = FeedbackItem(
            feedback_id=self._generate_id(query),
            feedback_type=FeedbackType.ANNOTATION,
            query=query,
            answer=answer,
            document_ids=document_ids or [],
            is_positive=False,  # Annotations usually indicate issues
            category=category,
            comment=comment,
            correction=correction,
            user_id=user_id,
        )

        self.store.save(feedback)
        return feedback

    def collect_click(
        self,
        query: str,
        document_id: str,
        position: int,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> FeedbackItem:
        """Collect click feedback (implicit).

        Args:
            query: Search query
            document_id: Clicked document ID
            position: Position in results
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Created feedback item
        """
        feedback = FeedbackItem(
            feedback_id=self._generate_id(f"{query}:{document_id}"),
            feedback_type=FeedbackType.CLICK,
            query=query,
            document_ids=[document_id],
            is_positive=True,
            user_id=user_id,
            session_id=session_id,
            metadata={"position": position},
        )

        self.store.save(feedback)
        return feedback

    def collect_dwell_time(
        self,
        query: str,
        document_id: str,
        dwell_seconds: float,
        user_id: str | None = None,
    ) -> FeedbackItem:
        """Collect dwell time feedback (implicit).

        Args:
            query: Search query
            document_id: Viewed document ID
            dwell_seconds: Time spent on document
            user_id: User identifier

        Returns:
            Created feedback item
        """
        # Convert dwell time to satisfaction score
        # Assumption: 30+ seconds is good engagement
        value = min(1.0, dwell_seconds / 60.0)

        feedback = FeedbackItem(
            feedback_id=self._generate_id(f"{query}:{document_id}:dwell"),
            feedback_type=FeedbackType.DWELL_TIME,
            query=query,
            document_ids=[document_id],
            value=value,
            is_positive=dwell_seconds >= 10,
            user_id=user_id,
            metadata={"dwell_seconds": dwell_seconds},
        )

        self.store.save(feedback)
        return feedback

    def flag_content(
        self,
        query: str,
        answer: str,
        reason: str,
        document_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> FeedbackItem:
        """Flag content for review.

        Args:
            query: User query
            answer: Flagged answer
            reason: Reason for flagging
            document_ids: Related document IDs
            user_id: User identifier

        Returns:
            Created feedback item
        """
        feedback = FeedbackItem(
            feedback_id=self._generate_id(f"flag:{query}"),
            feedback_type=FeedbackType.FLAG,
            query=query,
            answer=answer,
            document_ids=document_ids or [],
            is_positive=False,
            comment=reason,
            user_id=user_id,
        )

        self.store.save(feedback)
        return feedback

    def get_summary(
        self,
        filters: dict[str, Any] | None = None,
    ) -> FeedbackSummary:
        """Get feedback summary statistics.

        Args:
            filters: Optional filters

        Returns:
            Feedback summary
        """
        all_feedback = self.store.query(filters, limit=10000)

        summary = FeedbackSummary(total_feedback=len(all_feedback))

        if not all_feedback:
            return summary

        ratings: list[float] = []
        doc_feedback: dict[str, list[float]] = {}

        for fb in all_feedback:
            # Count positive/negative
            if fb.is_positive is True:
                summary.positive_count += 1
            elif fb.is_positive is False:
                summary.negative_count += 1
            else:
                summary.neutral_count += 1

            # Collect ratings
            if fb.value is not None:
                ratings.append(fb.value)

            # Count categories
            if fb.category:
                cat = fb.category.value
                summary.category_counts[cat] = summary.category_counts.get(cat, 0) + 1

            # Aggregate document scores
            for doc_id in fb.document_ids:
                if doc_id not in doc_feedback:
                    doc_feedback[doc_id] = []
                if fb.value is not None:
                    doc_feedback[doc_id].append(fb.value)
                elif fb.is_positive is not None:
                    doc_feedback[doc_id].append(1.0 if fb.is_positive else 0.0)

        # Calculate averages
        if ratings:
            summary.average_rating = sum(ratings) / len(ratings)

        for doc_id, scores in doc_feedback.items():
            if scores:
                summary.document_scores[doc_id] = sum(scores) / len(scores)

        # Recent feedback
        summary.recent_feedback = all_feedback[:10]

        return summary

    def get_document_score(self, doc_id: str) -> float:
        """Get aggregated feedback score for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Aggregated score (0-1)
        """
        feedback_items = self.store.get_document_feedback(doc_id)

        if not feedback_items:
            return 0.5  # Neutral for no feedback

        scores: list[float] = []
        for fb in feedback_items:
            if fb.value is not None:
                scores.append(fb.value)
            elif fb.is_positive is not None:
                scores.append(1.0 if fb.is_positive else 0.0)

        return sum(scores) / len(scores) if scores else 0.5


# Convenience functions


def create_feedback_collector(
    storage_path: str | Path | None = None,
) -> FeedbackCollector:
    """Create a feedback collector.

    Args:
        storage_path: Optional path for persistent storage

    Returns:
        Feedback collector instance
    """
    if storage_path:
        store = FileFeedbackStore(storage_path)
    else:
        store = InMemoryFeedbackStore()

    return FeedbackCollector(store)


def collect_thumbs_feedback(
    is_positive: bool,
    query: str,
    answer: str,
) -> FeedbackItem:
    """Quick function to collect thumbs feedback.

    Args:
        is_positive: Thumbs up or down
        query: User query
        answer: Generated answer

    Returns:
        Feedback item
    """
    collector = FeedbackCollector()
    return collector.collect_thumbs(is_positive, query, answer)


def collect_rating_feedback(
    rating: float,
    query: str,
    answer: str,
) -> FeedbackItem:
    """Quick function to collect rating feedback.

    Args:
        rating: Rating (0-5 or 0-1)
        query: User query
        answer: Generated answer

    Returns:
        Feedback item
    """
    collector = FeedbackCollector()
    return collector.collect_rating(rating, query, answer)
