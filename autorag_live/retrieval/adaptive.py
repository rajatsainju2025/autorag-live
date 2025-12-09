"""
Adaptive retrieval strategy engine for dynamic query-specific optimization.

Selects and optimizes retrieval strategies based on query analysis and
performance metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""

    KEYWORD = "keyword"  # BM25-based
    SEMANTIC = "semantic"  # Dense embeddings
    HYBRID = "hybrid"  # Combination of keyword and semantic
    GRAPH = "graph"  # Knowledge graph-based
    RERANKING = "reranking"  # Apply reranker to results


@dataclass
class StrategyMetrics:
    """Performance metrics for a retrieval strategy."""

    strategy: RetrievalStrategy
    num_uses: int = 0
    avg_latency: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    success_rate: float = 1.0
    cost: float = 0.0

    def get_score(self) -> float:
        """Calculate composite score for strategy."""
        # Balance precision, recall, and latency
        effectiveness = (
            (self.avg_precision * 0.4) + (self.avg_recall * 0.4) + (self.success_rate * 0.2)
        )

        # Normalize latency (assume 100ms is baseline)
        latency_factor = min(1.0, 100 / max(self.avg_latency, 1))

        # Cost factor (assume $0.01 per call is baseline)
        cost_factor = max(0.0, 1.0 - (self.cost / 0.01))

        return effectiveness * 0.6 + latency_factor * 0.2 + cost_factor * 0.2


@dataclass
class RetrievalPlan:
    """Plan for retrieving documents."""

    primary_strategy: RetrievalStrategy
    fallback_strategies: List[RetrievalStrategy] = field(default_factory=list)
    rerank_enabled: bool = False
    expected_quality: float = 0.5
    estimated_latency: float = 0.0
    confidence: float = 0.5
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "primary_strategy": self.primary_strategy.value,
            "fallback_strategies": [s.value for s in self.fallback_strategies],
            "rerank_enabled": self.rerank_enabled,
            "expected_quality": self.expected_quality,
            "estimated_latency": self.estimated_latency,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


class QueryAnalyzer:
    """Analyzes queries to determine optimal retrieval strategies."""

    def __init__(self):
        """Initialize query analyzer."""
        self.logger = logging.getLogger("QueryAnalyzer")

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query characteristics.

        Args:
            query: Input query

        Returns:
            Analysis with strategy recommendations
        """
        analysis = {
            "query_length": len(query.split()),
            "has_entities": self._detect_entities(query),
            "has_relationships": self._detect_relationships(query),
            "requires_reasoning": self._requires_reasoning(query),
            "specificity": self._assess_specificity(query),
            "domain": self._detect_domain(query),
        }

        return analysis

    def _detect_entities(self, query: str) -> bool:
        """Detect if query contains named entities."""
        # Simple heuristic: capitalized words
        words = query.split()
        capitalized = sum(1 for w in words if w[0].isupper())
        return capitalized >= 1

    def _detect_relationships(self, query: str) -> bool:
        """Detect if query asks about relationships."""
        relationship_words = [
            "relation",
            "connection",
            "link",
            "associate",
            "compare",
            "difference",
            "similarity",
        ]
        return any(w in query.lower() for w in relationship_words)

    def _requires_reasoning(self, query: str) -> bool:
        """Detect if query requires reasoning."""
        reasoning_words = ["why", "how", "analyze", "explain", "reason"]
        return any(w in query.lower() for w in reasoning_words)

    def _assess_specificity(self, query: str) -> float:
        """Assess query specificity (0-1)."""
        # More specific queries have more words and entities
        words = query.split()
        word_count_score = min(1.0, len(words) / 10)

        # Check for qualifying adjectives/adverbs
        qualifiers = ["specific", "particular", "exact", "detailed"]
        qualifier_count = sum(1 for q in qualifiers if q in query.lower())

        return min(1.0, (word_count_score * 0.7) + (qualifier_count * 0.3))

    def _detect_domain(self, query: str) -> str:
        """Detect query domain."""
        domains = {
            "technical": ["code", "algorithm", "system", "API"],
            "medical": ["disease", "treatment", "symptom", "health"],
            "legal": ["law", "contract", "court", "regulation"],
            "scientific": ["study", "research", "data", "experiment"],
        }

        for domain, keywords in domains.items():
            if any(k in query.lower() for k in keywords):
                return domain

        return "general"


class AdaptiveRetrievalEngine:
    """
    Dynamically selects and optimizes retrieval strategies.

    Adapts based on query characteristics and performance metrics.
    """

    def __init__(self):
        """Initialize adaptive retrieval engine."""
        self.logger = logging.getLogger("AdaptiveRetrievalEngine")
        self.analyzer = QueryAnalyzer()

        # Initialize strategy metrics
        self.strategy_metrics: Dict[RetrievalStrategy, StrategyMetrics] = {
            strategy: StrategyMetrics(strategy) for strategy in RetrievalStrategy
        }

        # Default strategies for different query types
        self.strategy_preferences = {
            "entity-heavy": [
                RetrievalStrategy.SEMANTIC,
                RetrievalStrategy.KEYWORD,
            ],
            "relationship-heavy": [
                RetrievalStrategy.GRAPH,
                RetrievalStrategy.HYBRID,
            ],
            "reasoning-heavy": [
                RetrievalStrategy.SEMANTIC,
                RetrievalStrategy.RERANKING,
            ],
            "specific": [RetrievalStrategy.KEYWORD, RetrievalStrategy.SEMANTIC],
            "general": [
                RetrievalStrategy.HYBRID,
                RetrievalStrategy.SEMANTIC,
            ],
        }

    def plan_retrieval(self, query: str) -> RetrievalPlan:
        """
        Plan retrieval strategy for query.

        Args:
            query: Input query

        Returns:
            RetrievalPlan with recommended strategy
        """
        # Analyze query
        analysis = self.analyzer.analyze(query)

        # Determine query type
        query_type = self._classify_query(analysis)

        # Get strategy preferences
        preferred_strategies = self.strategy_preferences.get(query_type, [RetrievalStrategy.HYBRID])

        # Select primary strategy
        primary_strategy = self._select_best_strategy(preferred_strategies)

        # Get fallback strategies
        fallback_strategies = preferred_strategies[1:]

        # Determine if reranking should be enabled
        rerank_enabled = analysis["has_relationships"] or analysis["requires_reasoning"]

        # Create plan
        plan = RetrievalPlan(
            primary_strategy=primary_strategy,
            fallback_strategies=fallback_strategies,
            rerank_enabled=rerank_enabled,
            expected_quality=self._estimate_quality(primary_strategy, analysis),
            estimated_latency=self._estimate_latency(primary_strategy),
            confidence=analysis["specificity"],
            reasoning=self._explain_choice(primary_strategy, query_type, analysis),
        )

        return plan

    def record_performance(
        self,
        strategy: RetrievalStrategy,
        latency: float,
        precision: float,
        recall: float,
        success: bool,
        cost: float = 0.0,
    ) -> None:
        """
        Record strategy performance for adaptation.

        Args:
            strategy: Strategy used
            latency: Query latency in ms
            precision: Precision score (0-1)
            recall: Recall score (0-1)
            success: Whether retrieval succeeded
            cost: Cost of retrieval
        """
        metrics = self.strategy_metrics[strategy]
        old_uses = metrics.num_uses

        metrics.num_uses += 1
        metrics.avg_latency = (metrics.avg_latency * old_uses + latency) / metrics.num_uses
        metrics.avg_precision = (metrics.avg_precision * old_uses + precision) / metrics.num_uses
        metrics.avg_recall = (metrics.avg_recall * old_uses + recall) / metrics.num_uses
        metrics.success_rate = (
            metrics.success_rate * old_uses + (1.0 if success else 0.0)
        ) / metrics.num_uses
        metrics.cost += cost

        self.logger.info(
            f"Recorded {strategy.value} performance: "
            f"latency={latency:.1f}ms, "
            f"precision={precision:.2f}, recall={recall:.2f}"
        )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all strategy metrics."""
        return {
            strategy.value: {
                "uses": metrics.num_uses,
                "avg_latency": round(metrics.avg_latency, 2),
                "avg_precision": round(metrics.avg_precision, 3),
                "avg_recall": round(metrics.avg_recall, 3),
                "success_rate": round(metrics.success_rate, 3),
                "score": round(metrics.get_score(), 3),
            }
            for strategy, metrics in self.strategy_metrics.items()
        }

    def _classify_query(self, analysis: Dict[str, Any]) -> str:
        """Classify query type based on analysis."""
        if analysis["has_relationships"]:
            return "relationship-heavy"
        elif analysis["requires_reasoning"]:
            return "reasoning-heavy"
        elif analysis["has_entities"]:
            return "entity-heavy"
        elif analysis["specificity"] > 0.7:
            return "specific"
        else:
            return "general"

    def _select_best_strategy(
        self, preferred_strategies: List[RetrievalStrategy]
    ) -> RetrievalStrategy:
        """Select best strategy from preferred list."""
        # Score each strategy
        scores = [(s, self.strategy_metrics[s].get_score()) for s in preferred_strategies]

        # Return highest scoring
        best = max(scores, key=lambda x: x[1])
        return best[0]

    def _estimate_quality(
        self,
        strategy: RetrievalStrategy,
        analysis: Dict[str, Any],
    ) -> float:
        """Estimate expected quality for strategy."""
        metrics = self.strategy_metrics[strategy]
        base_quality = (metrics.avg_precision * 0.5) + (metrics.avg_recall * 0.5)

        # Adjust based on query characteristics
        if strategy == RetrievalStrategy.KEYWORD:
            if analysis["specificity"] > 0.7:
                return base_quality + 0.15
            else:
                return base_quality - 0.1

        elif strategy == RetrievalStrategy.SEMANTIC:
            if analysis["has_relationships"]:
                return base_quality + 0.1
            else:
                return base_quality

        elif strategy == RetrievalStrategy.GRAPH:
            if analysis["has_relationships"]:
                return base_quality + 0.2
            else:
                return base_quality - 0.2

        return base_quality

    def _estimate_latency(self, strategy: RetrievalStrategy) -> float:
        """Estimate latency for strategy."""
        latency_map = {
            RetrievalStrategy.KEYWORD: 10,
            RetrievalStrategy.SEMANTIC: 50,
            RetrievalStrategy.HYBRID: 80,
            RetrievalStrategy.GRAPH: 100,
            RetrievalStrategy.RERANKING: 20,
        }

        return latency_map.get(strategy, 50)

    def _explain_choice(
        self,
        strategy: RetrievalStrategy,
        query_type: str,
        analysis: Dict[str, Any],
    ) -> str:
        """Explain why strategy was chosen."""
        reasons = []

        if strategy == RetrievalStrategy.KEYWORD:
            reasons.append(f"Specific query ({analysis['specificity']:.1%})")

        elif strategy == RetrievalStrategy.SEMANTIC:
            reasons.append("Good for semantic matching")

        elif strategy == RetrievalStrategy.HYBRID:
            reasons.append("Balanced approach for variety")

        elif strategy == RetrievalStrategy.GRAPH:
            reasons.append("Relationship-heavy query detected")

        return f"Selected {strategy.value}: {'; '.join(reasons)}"
