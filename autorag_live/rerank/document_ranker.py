"""Document Ranking Module for AutoRAG-Live.

Multi-signal document ranking with configurable weights:
- Semantic relevance
- BM25/lexical matching
- Recency scoring
- Source authority
- User feedback signals
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RankingSignal(Enum):
    """Available ranking signals."""

    SEMANTIC = "semantic"
    BM25 = "bm25"
    RECENCY = "recency"
    AUTHORITY = "authority"
    FEEDBACK = "feedback"
    LENGTH = "length"
    POSITION = "position"
    CUSTOM = "custom"


@dataclass
class RankingDocument:
    """Document with ranking information."""

    doc_id: str
    content: str
    base_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Individual signal scores
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    recency_score: float = 0.0
    authority_score: float = 0.0
    feedback_score: float = 0.0
    length_score: float = 0.0
    position_score: float = 0.0
    custom_scores: dict[str, float] = field(default_factory=dict)

    # Final ranking score
    final_score: float = 0.0

    def get_score(self, signal: RankingSignal) -> float:
        """Get score for a specific signal."""
        score_map = {
            RankingSignal.SEMANTIC: self.semantic_score,
            RankingSignal.BM25: self.bm25_score,
            RankingSignal.RECENCY: self.recency_score,
            RankingSignal.AUTHORITY: self.authority_score,
            RankingSignal.FEEDBACK: self.feedback_score,
            RankingSignal.LENGTH: self.length_score,
            RankingSignal.POSITION: self.position_score,
        }
        return score_map.get(signal, 0.0)


@dataclass
class RankingConfig:
    """Configuration for document ranking."""

    # Signal weights (should sum to 1.0 for normalized scoring)
    weights: dict[RankingSignal, float] = field(
        default_factory=lambda: {
            RankingSignal.SEMANTIC: 0.4,
            RankingSignal.BM25: 0.3,
            RankingSignal.RECENCY: 0.1,
            RankingSignal.AUTHORITY: 0.1,
            RankingSignal.FEEDBACK: 0.1,
        }
    )

    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Recency parameters
    recency_decay: float = 0.1  # Decay factor per day
    recency_date_field: str = "date"

    # Authority configuration
    authority_sources: dict[str, float] = field(default_factory=dict)
    default_authority: float = 0.5

    # Length preferences
    ideal_length: int = 500
    length_tolerance: float = 0.3

    # Position bias (for initial ranking)
    position_decay: float = 0.05


class BaseRankingSignal(ABC):
    """Abstract base class for ranking signals."""

    signal_type: RankingSignal

    @abstractmethod
    def compute(
        self,
        doc: RankingDocument,
        query: str,
        config: RankingConfig,
    ) -> float:
        """Compute signal score for a document.

        Args:
            doc: Document to score
            query: Search query
            config: Ranking configuration

        Returns:
            Signal score (0-1)
        """
        pass


class SemanticSignal(BaseRankingSignal):
    """Semantic relevance signal (uses base score)."""

    signal_type = RankingSignal.SEMANTIC

    def compute(
        self,
        doc: RankingDocument,
        query: str,
        config: RankingConfig,
    ) -> float:
        """Use base semantic score."""
        # Normalize to 0-1 if needed
        score = doc.base_score
        if score > 1.0:
            score = min(1.0, score / 100.0)  # Handle percentage scores
        return max(0.0, min(1.0, score))


class BM25Signal(BaseRankingSignal):
    """BM25 lexical matching signal."""

    signal_type = RankingSignal.BM25

    def __init__(self, corpus_stats: dict[str, Any] | None = None) -> None:
        """Initialize BM25 signal.

        Args:
            corpus_stats: Pre-computed corpus statistics
        """
        self.corpus_stats = corpus_stats or {}
        self._idf_cache: dict[str, float] = {}

    def compute(
        self,
        doc: RankingDocument,
        query: str,
        config: RankingConfig,
    ) -> float:
        """Compute BM25 score."""
        k1 = config.bm25_k1
        b = config.bm25_b

        # Tokenize
        query_terms = self._tokenize(query)
        doc_terms = self._tokenize(doc.content)

        if not query_terms or not doc_terms:
            return 0.0

        # Document length normalization
        doc_len = len(doc_terms)
        avg_doc_len = self.corpus_stats.get("avg_doc_len", doc_len)
        total_docs = self.corpus_stats.get("total_docs", 1)

        # Term frequency in document
        doc_term_freq = Counter(doc_terms)

        score = 0.0
        for term in query_terms:
            tf = doc_term_freq.get(term, 0)
            if tf == 0:
                continue

            # IDF calculation
            idf = self._compute_idf(term, total_docs)

            # BM25 term score
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += idf * (numerator / denominator)

        # Normalize score
        max_possible = len(query_terms) * 10  # Rough upper bound
        return min(1.0, score / max_possible) if max_possible > 0 else 0.0

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text."""
        # Simple tokenization
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _compute_idf(self, term: str, total_docs: int) -> float:
        """Compute IDF for term."""
        if term in self._idf_cache:
            return self._idf_cache[term]

        doc_freq = self.corpus_stats.get("doc_freq", {}).get(term, 1)
        idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        self._idf_cache[term] = idf

        return idf


class RecencySignal(BaseRankingSignal):
    """Recency-based scoring signal."""

    signal_type = RankingSignal.RECENCY

    def compute(
        self,
        doc: RankingDocument,
        query: str,
        config: RankingConfig,
    ) -> float:
        """Compute recency score with exponential decay."""
        date_value = doc.metadata.get(config.recency_date_field)

        if not date_value:
            return 0.5  # Neutral score for unknown date

        try:
            if isinstance(date_value, str):
                # Try common date formats
                for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"]:
                    try:
                        doc_date = datetime.strptime(date_value[:10], fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return 0.5
            elif isinstance(date_value, datetime):
                doc_date = date_value
            else:
                return 0.5

            # Calculate days since document date
            days_old = (datetime.now() - doc_date).days
            days_old = max(0, days_old)

            # Exponential decay
            score = math.exp(-config.recency_decay * days_old / 30)  # Decay per month

            return max(0.0, min(1.0, score))

        except Exception as e:
            logger.debug(f"Error computing recency for {doc.doc_id}: {e}")
            return 0.5


class AuthoritySignal(BaseRankingSignal):
    """Source authority signal."""

    signal_type = RankingSignal.AUTHORITY

    def compute(
        self,
        doc: RankingDocument,
        query: str,
        config: RankingConfig,
    ) -> float:
        """Compute authority score based on source."""
        source = doc.metadata.get("source", "")

        if not source:
            return config.default_authority

        # Check configured authorities
        for pattern, authority in config.authority_sources.items():
            if pattern.lower() in source.lower():
                return authority

        return config.default_authority


class FeedbackSignal(BaseRankingSignal):
    """User feedback signal."""

    signal_type = RankingSignal.FEEDBACK

    def __init__(self, feedback_store: dict[str, float] | None = None) -> None:
        """Initialize feedback signal.

        Args:
            feedback_store: Document ID to feedback score mapping
        """
        self.feedback_store = feedback_store or {}

    def compute(
        self,
        doc: RankingDocument,
        query: str,
        config: RankingConfig,
    ) -> float:
        """Compute feedback-based score."""
        # Check for stored feedback
        if doc.doc_id in self.feedback_store:
            return self.feedback_store[doc.doc_id]

        # Check metadata for feedback
        feedback = doc.metadata.get("feedback_score")
        if feedback is not None:
            return float(feedback)

        # Check click/view data
        clicks = doc.metadata.get("clicks", 0)
        views = doc.metadata.get("views", 1)

        if views > 0:
            ctr = clicks / views
            return min(1.0, ctr * 2)  # Scale CTR

        return 0.5  # Neutral for no feedback


class LengthSignal(BaseRankingSignal):
    """Document length preference signal."""

    signal_type = RankingSignal.LENGTH

    def compute(
        self,
        doc: RankingDocument,
        query: str,
        config: RankingConfig,
    ) -> float:
        """Compute length-based score."""
        doc_len = len(doc.content)
        ideal_len = config.ideal_length
        tolerance = config.length_tolerance

        # Calculate distance from ideal length
        distance = abs(doc_len - ideal_len) / ideal_len

        if distance <= tolerance:
            return 1.0

        # Decay score as distance increases
        excess_distance = distance - tolerance
        score = 1.0 - (excess_distance / (1 + excess_distance))

        return max(0.0, min(1.0, score))


class PositionSignal(BaseRankingSignal):
    """Original position bias signal."""

    signal_type = RankingSignal.POSITION

    def compute(
        self,
        doc: RankingDocument,
        query: str,
        config: RankingConfig,
    ) -> float:
        """Compute position-based score."""
        position = doc.metadata.get("original_position", 0)

        # Exponential decay from first position
        score = math.exp(-config.position_decay * position)

        return max(0.0, min(1.0, score))


class DocumentRanker:
    """Multi-signal document ranker."""

    def __init__(
        self,
        config: RankingConfig | None = None,
    ) -> None:
        """Initialize document ranker.

        Args:
            config: Ranking configuration
        """
        self.config = config or RankingConfig()

        # Initialize signal computers
        self._signals: dict[RankingSignal, BaseRankingSignal] = {
            RankingSignal.SEMANTIC: SemanticSignal(),
            RankingSignal.BM25: BM25Signal(),
            RankingSignal.RECENCY: RecencySignal(),
            RankingSignal.AUTHORITY: AuthoritySignal(),
            RankingSignal.FEEDBACK: FeedbackSignal(),
            RankingSignal.LENGTH: LengthSignal(),
            RankingSignal.POSITION: PositionSignal(),
        }

        self._custom_signals: dict[str, Callable[[RankingDocument, str], float]] = {}

    def register_signal(
        self,
        signal_type: RankingSignal,
        signal: BaseRankingSignal,
    ) -> None:
        """Register a custom signal implementation.

        Args:
            signal_type: Signal type
            signal: Signal implementation
        """
        self._signals[signal_type] = signal

    def add_custom_signal(
        self,
        name: str,
        scorer: Callable[[RankingDocument, str], float],
        weight: float = 0.1,
    ) -> None:
        """Add a custom scoring function.

        Args:
            name: Signal name
            scorer: Scoring function (doc, query) -> score
            weight: Signal weight
        """
        self._custom_signals[name] = scorer
        self.config.weights[RankingSignal.CUSTOM] = (
            self.config.weights.get(RankingSignal.CUSTOM, 0) + weight
        )

    def rank(
        self,
        documents: list[RankingDocument],
        query: str,
        top_k: int | None = None,
    ) -> list[RankingDocument]:
        """Rank documents using multi-signal scoring.

        Args:
            documents: Documents to rank
            query: Search query
            top_k: Number of top documents to return

        Returns:
            Ranked documents
        """
        if not documents:
            return []

        # Compute scores for each document
        for i, doc in enumerate(documents):
            # Store original position
            doc.metadata["original_position"] = i

            # Compute each signal
            for signal_type, signal in self._signals.items():
                if signal_type in self.config.weights:
                    try:
                        score = signal.compute(doc, query, self.config)
                        self._set_signal_score(doc, signal_type, score)
                    except Exception as e:
                        logger.warning(f"Signal {signal_type.value} failed: {e}")
                        self._set_signal_score(doc, signal_type, 0.5)

            # Compute custom signals
            for name, scorer in self._custom_signals.items():
                try:
                    doc.custom_scores[name] = scorer(doc, query)
                except Exception as e:
                    logger.warning(f"Custom signal {name} failed: {e}")
                    doc.custom_scores[name] = 0.5

            # Compute final weighted score
            doc.final_score = self._compute_final_score(doc)

        # Sort by final score
        ranked = sorted(documents, key=lambda d: d.final_score, reverse=True)

        if top_k:
            ranked = ranked[:top_k]

        return ranked

    def _set_signal_score(
        self,
        doc: RankingDocument,
        signal_type: RankingSignal,
        score: float,
    ) -> None:
        """Set signal score on document."""
        if signal_type == RankingSignal.SEMANTIC:
            doc.semantic_score = score
        elif signal_type == RankingSignal.BM25:
            doc.bm25_score = score
        elif signal_type == RankingSignal.RECENCY:
            doc.recency_score = score
        elif signal_type == RankingSignal.AUTHORITY:
            doc.authority_score = score
        elif signal_type == RankingSignal.FEEDBACK:
            doc.feedback_score = score
        elif signal_type == RankingSignal.LENGTH:
            doc.length_score = score
        elif signal_type == RankingSignal.POSITION:
            doc.position_score = score

    def _compute_final_score(self, doc: RankingDocument) -> float:
        """Compute weighted final score."""
        total_weight = sum(self.config.weights.values())
        if total_weight == 0:
            return doc.base_score

        weighted_sum = 0.0

        for signal_type, weight in self.config.weights.items():
            if signal_type == RankingSignal.CUSTOM:
                # Average custom signal scores
                if doc.custom_scores:
                    custom_avg = sum(doc.custom_scores.values()) / len(doc.custom_scores)
                    weighted_sum += weight * custom_avg
            else:
                score = doc.get_score(signal_type)
                weighted_sum += weight * score

        return weighted_sum / total_weight

    def explain_ranking(self, doc: RankingDocument) -> dict[str, Any]:
        """Explain ranking for a document.

        Args:
            doc: Ranked document

        Returns:
            Explanation dictionary
        """
        explanation: dict[str, Any] = {
            "doc_id": doc.doc_id,
            "final_score": doc.final_score,
            "signals": {},
        }

        for signal_type, weight in self.config.weights.items():
            if signal_type == RankingSignal.CUSTOM:
                explanation["signals"]["custom"] = {
                    "weight": weight,
                    "scores": doc.custom_scores,
                }
            else:
                score = doc.get_score(signal_type)
                contribution = score * weight
                explanation["signals"][signal_type.value] = {
                    "score": score,
                    "weight": weight,
                    "contribution": contribution,
                }

        return explanation


# Convenience functions


def rank_documents(
    documents: list[dict[str, Any]],
    query: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Rank documents using default configuration.

    Args:
        documents: List of document dictionaries with 'content' and 'score'
        query: Search query
        top_k: Number of results

    Returns:
        Ranked documents
    """
    # Convert to RankingDocument
    ranking_docs = [
        RankingDocument(
            doc_id=doc.get("id", f"doc_{i}"),
            content=doc.get("content", ""),
            base_score=doc.get("score", 0.0),
            metadata=doc.get("metadata", {}),
        )
        for i, doc in enumerate(documents)
    ]

    ranker = DocumentRanker()
    ranked = ranker.rank(ranking_docs, query, top_k)

    # Convert back to dictionaries
    return [
        {
            "id": doc.doc_id,
            "content": doc.content,
            "score": doc.final_score,
            "metadata": doc.metadata,
        }
        for doc in ranked
    ]


def create_ranker(
    semantic_weight: float = 0.4,
    bm25_weight: float = 0.3,
    recency_weight: float = 0.15,
    authority_weight: float = 0.15,
) -> DocumentRanker:
    """Create a document ranker with custom weights.

    Args:
        semantic_weight: Weight for semantic similarity
        bm25_weight: Weight for BM25 matching
        recency_weight: Weight for recency
        authority_weight: Weight for source authority

    Returns:
        Configured ranker
    """
    config = RankingConfig(
        weights={
            RankingSignal.SEMANTIC: semantic_weight,
            RankingSignal.BM25: bm25_weight,
            RankingSignal.RECENCY: recency_weight,
            RankingSignal.AUTHORITY: authority_weight,
        }
    )

    return DocumentRanker(config)


def simple_rerank(
    documents: list[str],
    scores: list[float],
    query: str,
) -> list[tuple[str, float]]:
    """Simple reranking of documents.

    Args:
        documents: Document contents
        scores: Original scores
        query: Search query

    Returns:
        List of (document, new_score) tuples
    """
    ranking_docs = [
        RankingDocument(
            doc_id=f"doc_{i}",
            content=doc,
            base_score=scores[i] if i < len(scores) else 0.0,
        )
        for i, doc in enumerate(documents)
    ]

    ranker = DocumentRanker()
    ranked = ranker.rank(ranking_docs, query)

    return [(doc.content, doc.final_score) for doc in ranked]
