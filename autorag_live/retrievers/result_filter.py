"""Result Filtering Module for AutoRAG-Live.

Provides filtering and deduplication strategies for retrieval results:
- Semantic deduplication
- Relevance threshold filtering
- Diversity-based selection
- Source-based filtering
- Configurable filter chains
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""

    content: str
    score: float
    doc_id: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int | None = None

    @property
    def content_hash(self) -> str:
        """Get hash of content for deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()

    def __eq__(self, other: object) -> bool:
        """Check equality based on content hash."""
        if not isinstance(other, RetrievalResult):
            return False
        return self.content_hash == other.content_hash


class FilterStrategy(Enum):
    """Available filtering strategies."""

    THRESHOLD = "threshold"
    TOP_K = "top_k"
    PERCENTILE = "percentile"
    EXACT_DEDUP = "exact_dedup"
    SEMANTIC_DEDUP = "semantic_dedup"
    DIVERSITY = "diversity"
    SOURCE = "source"
    LENGTH = "length"
    RECENCY = "recency"


@dataclass
class FilterConfig:
    """Configuration for result filtering."""

    # Threshold filtering
    min_score: float = 0.0
    max_score: float = 1.0

    # Top-k filtering
    top_k: int = 10

    # Percentile filtering
    percentile: float = 0.8

    # Deduplication
    similarity_threshold: float = 0.85

    # Diversity
    diversity_weight: float = 0.3

    # Length filtering
    min_length: int = 10
    max_length: int = 10000

    # Source filtering
    allowed_sources: list[str] = field(default_factory=list)
    blocked_sources: list[str] = field(default_factory=list)


class BaseFilter(ABC):
    """Abstract base class for result filters."""

    filter_strategy: FilterStrategy

    @abstractmethod
    def filter(
        self,
        results: list[RetrievalResult],
        config: FilterConfig,
    ) -> list[RetrievalResult]:
        """Filter results.

        Args:
            results: Results to filter
            config: Filter configuration

        Returns:
            Filtered results
        """
        pass


class ThresholdFilter(BaseFilter):
    """Filter results by score threshold."""

    filter_strategy = FilterStrategy.THRESHOLD

    def filter(
        self,
        results: list[RetrievalResult],
        config: FilterConfig,
    ) -> list[RetrievalResult]:
        """Filter by score threshold."""
        return [
            r
            for r in results
            if config.min_score <= r.score <= config.max_score
        ]


class TopKFilter(BaseFilter):
    """Select top-k results by score."""

    filter_strategy = FilterStrategy.TOP_K

    def filter(
        self,
        results: list[RetrievalResult],
        config: FilterConfig,
    ) -> list[RetrievalResult]:
        """Select top-k results."""
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        return sorted_results[: config.top_k]


class PercentileFilter(BaseFilter):
    """Filter results above a score percentile."""

    filter_strategy = FilterStrategy.PERCENTILE

    def filter(
        self,
        results: list[RetrievalResult],
        config: FilterConfig,
    ) -> list[RetrievalResult]:
        """Filter by percentile."""
        if not results:
            return []

        scores = sorted([r.score for r in results])
        cutoff_index = int(len(scores) * (1 - config.percentile))
        cutoff_score = scores[cutoff_index] if cutoff_index < len(scores) else 0

        return [r for r in results if r.score >= cutoff_score]


class ExactDeduplicationFilter(BaseFilter):
    """Remove exact duplicate results."""

    filter_strategy = FilterStrategy.EXACT_DEDUP

    def filter(
        self,
        results: list[RetrievalResult],
        config: FilterConfig,
    ) -> list[RetrievalResult]:
        """Remove exact duplicates based on content hash."""
        seen_hashes: set[str] = set()
        unique_results: list[RetrievalResult] = []

        for result in results:
            if result.content_hash not in seen_hashes:
                seen_hashes.add(result.content_hash)
                unique_results.append(result)

        return unique_results


class SemanticDeduplicationFilter(BaseFilter):
    """Remove semantically similar results."""

    filter_strategy = FilterStrategy.SEMANTIC_DEDUP

    def __init__(
        self,
        similarity_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        """Initialize semantic deduplication filter.

        Args:
            similarity_fn: Custom similarity function
        """
        self.similarity_fn = similarity_fn or self._default_similarity

    def _default_similarity(self, text1: str, text2: str) -> float:
        """Compute simple word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def filter(
        self,
        results: list[RetrievalResult],
        config: FilterConfig,
    ) -> list[RetrievalResult]:
        """Remove semantically similar results."""
        if not results:
            return []

        unique_results: list[RetrievalResult] = []

        for result in results:
            is_duplicate = False
            for existing in unique_results:
                similarity = self.similarity_fn(result.content, existing.content)
                if similarity >= config.similarity_threshold:
                    # Keep the one with higher score
                    if result.score > existing.score:
                        unique_results.remove(existing)
                        unique_results.append(result)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(result)

        return unique_results


class DiversityFilter(BaseFilter):
    """Select diverse results using MMR-like approach."""

    filter_strategy = FilterStrategy.DIVERSITY

    def __init__(
        self,
        similarity_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        """Initialize diversity filter.

        Args:
            similarity_fn: Custom similarity function
        """
        self.similarity_fn = similarity_fn or self._default_similarity

    def _default_similarity(self, text1: str, text2: str) -> float:
        """Compute simple word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def filter(
        self,
        results: list[RetrievalResult],
        config: FilterConfig,
    ) -> list[RetrievalResult]:
        """Select diverse results using MMR scoring.

        MMR = λ * relevance - (1-λ) * max_similarity_to_selected
        """
        if not results:
            return []

        if len(results) <= config.top_k:
            return results

        selected: list[RetrievalResult] = []
        remaining = list(results)

        # Select first result (highest relevance)
        remaining.sort(key=lambda x: x.score, reverse=True)
        selected.append(remaining.pop(0))

        # Iteratively select diverse results
        lambda_param = 1 - config.diversity_weight

        while len(selected) < config.top_k and remaining:
            best_mmr_score = float("-inf")
            best_candidate = None
            best_idx = -1

            for idx, candidate in enumerate(remaining):
                # Compute relevance component
                relevance = candidate.score

                # Compute max similarity to selected results
                max_sim = max(
                    self.similarity_fn(candidate.content, s.content)
                    for s in selected
                )

                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate
                    best_idx = idx

            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break

        return selected


class SourceFilter(BaseFilter):
    """Filter results by source."""

    filter_strategy = FilterStrategy.SOURCE

    def filter(
        self,
        results: list[RetrievalResult],
        config: FilterConfig,
    ) -> list[RetrievalResult]:
        """Filter by allowed/blocked sources."""
        filtered: list[RetrievalResult] = []

        for result in results:
            source = result.source or ""

            # Check blocked sources
            if config.blocked_sources:
                is_blocked = any(
                    blocked.lower() in source.lower()
                    for blocked in config.blocked_sources
                )
                if is_blocked:
                    continue

            # Check allowed sources
            if config.allowed_sources:
                is_allowed = any(
                    allowed.lower() in source.lower()
                    for allowed in config.allowed_sources
                )
                if not is_allowed:
                    continue

            filtered.append(result)

        return filtered


class LengthFilter(BaseFilter):
    """Filter results by content length."""

    filter_strategy = FilterStrategy.LENGTH

    def filter(
        self,
        results: list[RetrievalResult],
        config: FilterConfig,
    ) -> list[RetrievalResult]:
        """Filter by content length."""
        return [
            r
            for r in results
            if config.min_length <= len(r.content) <= config.max_length
        ]


class RecencyFilter(BaseFilter):
    """Filter results by recency metadata."""

    filter_strategy = FilterStrategy.RECENCY

    def __init__(
        self,
        date_field: str = "date",
        prefer_recent: bool = True,
    ) -> None:
        """Initialize recency filter.

        Args:
            date_field: Metadata field containing date
            prefer_recent: Prefer more recent documents
        """
        self.date_field = date_field
        self.prefer_recent = prefer_recent

    def filter(
        self,
        results: list[RetrievalResult],
        config: FilterConfig,
    ) -> list[RetrievalResult]:
        """Sort by recency and select top-k."""
        # Extract results with valid dates
        results_with_dates: list[tuple[RetrievalResult, Any]] = []
        results_without_dates: list[RetrievalResult] = []

        for result in results:
            date_value = result.metadata.get(self.date_field)
            if date_value:
                results_with_dates.append((result, date_value))
            else:
                results_without_dates.append(result)

        # Sort by date
        results_with_dates.sort(
            key=lambda x: str(x[1]),
            reverse=self.prefer_recent,
        )

        # Combine: dated results first, then undated
        sorted_results = [r for r, _ in results_with_dates] + results_without_dates

        return sorted_results[: config.top_k]


class FilterChain:
    """Chain multiple filters together."""

    def __init__(self, filters: list[BaseFilter] | None = None) -> None:
        """Initialize filter chain.

        Args:
            filters: List of filters to apply in order
        """
        self.filters = filters or []

    def add(self, filter_instance: BaseFilter) -> "FilterChain":
        """Add a filter to the chain.

        Args:
            filter_instance: Filter to add

        Returns:
            Self for chaining
        """
        self.filters.append(filter_instance)
        return self

    def apply(
        self,
        results: list[RetrievalResult],
        config: FilterConfig,
    ) -> list[RetrievalResult]:
        """Apply all filters in sequence.

        Args:
            results: Results to filter
            config: Filter configuration

        Returns:
            Filtered results
        """
        current_results = results

        for filter_instance in self.filters:
            if not current_results:
                break

            try:
                current_results = filter_instance.filter(current_results, config)
                logger.debug(
                    f"After {filter_instance.filter_strategy.value}: "
                    f"{len(current_results)} results"
                )
            except Exception as e:
                logger.warning(
                    f"Filter {filter_instance.filter_strategy.value} failed: {e}"
                )

        return current_results


class ResultFilterPipeline:
    """Complete result filtering pipeline."""

    def __init__(
        self,
        config: FilterConfig | None = None,
    ) -> None:
        """Initialize filtering pipeline.

        Args:
            config: Filter configuration
        """
        self.config = config or FilterConfig()
        self._filter_registry: dict[FilterStrategy, BaseFilter] = {
            FilterStrategy.THRESHOLD: ThresholdFilter(),
            FilterStrategy.TOP_K: TopKFilter(),
            FilterStrategy.PERCENTILE: PercentileFilter(),
            FilterStrategy.EXACT_DEDUP: ExactDeduplicationFilter(),
            FilterStrategy.SEMANTIC_DEDUP: SemanticDeduplicationFilter(),
            FilterStrategy.DIVERSITY: DiversityFilter(),
            FilterStrategy.SOURCE: SourceFilter(),
            FilterStrategy.LENGTH: LengthFilter(),
        }

    def register_filter(
        self,
        strategy: FilterStrategy,
        filter_instance: BaseFilter,
    ) -> None:
        """Register a custom filter.

        Args:
            strategy: Filter strategy
            filter_instance: Filter instance
        """
        self._filter_registry[strategy] = filter_instance

    def create_chain(
        self,
        strategies: list[FilterStrategy],
    ) -> FilterChain:
        """Create a filter chain from strategies.

        Args:
            strategies: List of filter strategies

        Returns:
            Configured filter chain
        """
        filters = []
        for strategy in strategies:
            if strategy in self._filter_registry:
                filters.append(self._filter_registry[strategy])
            else:
                logger.warning(f"Unknown filter strategy: {strategy}")

        return FilterChain(filters)

    def filter(
        self,
        results: list[RetrievalResult],
        strategies: list[FilterStrategy] | None = None,
    ) -> list[RetrievalResult]:
        """Filter results using specified strategies.

        Args:
            results: Results to filter
            strategies: Filter strategies to apply

        Returns:
            Filtered results
        """
        if not strategies:
            # Default chain: dedup -> threshold -> top-k
            strategies = [
                FilterStrategy.EXACT_DEDUP,
                FilterStrategy.THRESHOLD,
                FilterStrategy.TOP_K,
            ]

        chain = self.create_chain(strategies)
        return chain.apply(results, self.config)

    def deduplicate(
        self,
        results: list[RetrievalResult],
        semantic: bool = False,
    ) -> list[RetrievalResult]:
        """Deduplicate results.

        Args:
            results: Results to deduplicate
            semantic: Use semantic deduplication

        Returns:
            Deduplicated results
        """
        if semantic:
            strategy = FilterStrategy.SEMANTIC_DEDUP
        else:
            strategy = FilterStrategy.EXACT_DEDUP

        return self.filter(results, [strategy])

    def select_diverse(
        self,
        results: list[RetrievalResult],
        top_k: int | None = None,
        diversity_weight: float | None = None,
    ) -> list[RetrievalResult]:
        """Select diverse results.

        Args:
            results: Results to select from
            top_k: Number of results to select
            diversity_weight: Weight for diversity vs relevance

        Returns:
            Diverse selection of results
        """
        config = FilterConfig(
            top_k=top_k or self.config.top_k,
            diversity_weight=diversity_weight or self.config.diversity_weight,
        )

        original_config = self.config
        self.config = config

        try:
            return self.filter(results, [FilterStrategy.DIVERSITY])
        finally:
            self.config = original_config


# Convenience functions


def filter_results(
    results: list[RetrievalResult],
    min_score: float = 0.0,
    top_k: int = 10,
    deduplicate: bool = True,
) -> list[RetrievalResult]:
    """Filter retrieval results with common settings.

    Args:
        results: Results to filter
        min_score: Minimum score threshold
        top_k: Maximum number of results
        deduplicate: Remove duplicates

    Returns:
        Filtered results
    """
    config = FilterConfig(min_score=min_score, top_k=top_k)
    pipeline = ResultFilterPipeline(config)

    strategies = []
    if deduplicate:
        strategies.append(FilterStrategy.EXACT_DEDUP)
    strategies.extend([FilterStrategy.THRESHOLD, FilterStrategy.TOP_K])

    return pipeline.filter(results, strategies)


def deduplicate_results(
    results: list[RetrievalResult],
    semantic: bool = False,
    threshold: float = 0.85,
) -> list[RetrievalResult]:
    """Deduplicate retrieval results.

    Args:
        results: Results to deduplicate
        semantic: Use semantic deduplication
        threshold: Similarity threshold for semantic dedup

    Returns:
        Deduplicated results
    """
    config = FilterConfig(similarity_threshold=threshold)
    pipeline = ResultFilterPipeline(config)
    return pipeline.deduplicate(results, semantic=semantic)


def select_diverse_results(
    results: list[RetrievalResult],
    top_k: int = 5,
    diversity_weight: float = 0.3,
) -> list[RetrievalResult]:
    """Select diverse results using MMR.

    Args:
        results: Results to select from
        top_k: Number of results
        diversity_weight: Diversity weight (0-1)

    Returns:
        Diverse selection
    """
    pipeline = ResultFilterPipeline()
    return pipeline.select_diverse(results, top_k, diversity_weight)


def create_result(
    content: str,
    score: float,
    source: str | None = None,
    **metadata: Any,
) -> RetrievalResult:
    """Create a retrieval result.

    Args:
        content: Result content
        score: Relevance score
        source: Source identifier
        **metadata: Additional metadata

    Returns:
        Retrieval result
    """
    return RetrievalResult(
        content=content,
        score=score,
        source=source,
        metadata=metadata,
    )
