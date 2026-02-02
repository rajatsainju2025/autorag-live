"""
Intelligent query routing for optimized agentic RAG processing.

Routes queries to optimal processing paths based on:
- Query complexity and type
- Available resources
- Latency requirements
- Cost optimization
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from autorag_live.utils import get_logger

logger = get_logger(__name__)


class QueryType(Enum):
    """Types of queries."""

    FACTOID = "factoid"  # Simple fact lookup
    DEFINITION = "definition"  # Term definition
    EXPLANATION = "explanation"  # How/why questions
    COMPARISON = "comparison"  # Compare entities
    AGGREGATION = "aggregation"  # Statistical/summary
    MULTI_HOP = "multi_hop"  # Requires multiple steps
    CONVERSATIONAL = "conversational"  # Follow-up question


class RoutingStrategy(Enum):
    """Routing strategies."""

    FASTEST = "fastest"  # Minimize latency
    BALANCED = "balanced"  # Balance latency and quality
    QUALITY = "quality"  # Maximize quality
    COST_OPTIMIZED = "cost_optimized"  # Minimize cost


@dataclass
class ProcessingPath:
    """Processing path configuration."""

    path_id: str
    name: str
    description: str
    supported_types: List[QueryType]
    estimated_latency_ms: float
    estimated_cost: float
    quality_score: float  # 0-1
    required_resources: List[str] = field(default_factory=list)

    def score(self, strategy: RoutingStrategy) -> float:
        """Score path for given strategy."""
        if strategy == RoutingStrategy.FASTEST:
            return 1.0 / (self.estimated_latency_ms + 1)
        elif strategy == RoutingStrategy.QUALITY:
            return self.quality_score
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            return 1.0 / (self.estimated_cost + 0.1)
        else:  # BALANCED
            return (
                0.4 * self.quality_score
                + 0.4 / (self.estimated_latency_ms / 1000 + 1)
                + 0.2 / (self.estimated_cost + 0.1)
            )


@dataclass
class RoutingDecision:
    """Routing decision with metadata."""

    query: str
    query_type: QueryType
    selected_path: ProcessingPath
    strategy: RoutingStrategy
    confidence: float
    reasoning: str
    timestamp: float = field(default_factory=time.time)


class QueryClassifier:
    """Classify queries to determine type."""

    def __init__(self):
        self.patterns = {
            QueryType.FACTOID: ["what is", "who is", "when did", "where is"],
            QueryType.DEFINITION: ["define", "definition of", "meaning of"],
            QueryType.EXPLANATION: ["explain", "how does", "why does", "how to"],
            QueryType.COMPARISON: ["compare", "difference", "versus", "vs", "better"],
            QueryType.AGGREGATION: ["how many", "total", "sum", "average", "count"],
            QueryType.MULTI_HOP: [
                "and then",
                "after that",
                "step by step",
                "relationship",
            ],
        }

    def classify(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify query type.

        Args:
            query: User query

        Returns:
            Tuple of (type, confidence)
        """
        query_lower = query.lower()
        word_count = len(query.split())

        # Check patterns
        scores: Dict[QueryType, float] = {}

        for qtype, patterns in self.patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                scores[qtype] = score

        # Heuristics
        if not scores:
            # Default based on length and structure
            if word_count < 6 and "?" in query:
                scores[QueryType.FACTOID] = 0.5
            elif word_count > 15:
                scores[QueryType.EXPLANATION] = 0.5
            else:
                scores[QueryType.CONVERSATIONAL] = 0.5

        # Get best match
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            confidence = min(1.0, best_type[1] / 3.0)
            return best_type[0], confidence

        return QueryType.CONVERSATIONAL, 0.3


class ProcessingPathRegistry:
    """Registry of available processing paths."""

    def __init__(self):
        self.paths: Dict[str, ProcessingPath] = {}
        self._initialize_default_paths()

    def _initialize_default_paths(self) -> None:
        """Initialize default processing paths."""
        # Fast path: Minimal processing
        self.register_path(
            ProcessingPath(
                path_id="fast_retrieval",
                name="Fast Retrieval",
                description="Quick BM25 retrieval with simple synthesis",
                supported_types=[QueryType.FACTOID, QueryType.DEFINITION],
                estimated_latency_ms=100,
                estimated_cost=0.1,
                quality_score=0.7,
                required_resources=["bm25"],
            )
        )

        # Standard path: Balanced
        self.register_path(
            ProcessingPath(
                path_id="standard_rag",
                name="Standard RAG",
                description="Hybrid retrieval with reasoning and synthesis",
                supported_types=[
                    QueryType.FACTOID,
                    QueryType.DEFINITION,
                    QueryType.EXPLANATION,
                ],
                estimated_latency_ms=300,
                estimated_cost=0.5,
                quality_score=0.85,
                required_resources=["bm25", "dense", "llm"],
            )
        )

        # Deep path: High quality
        self.register_path(
            ProcessingPath(
                path_id="deep_reasoning",
                name="Deep Reasoning",
                description="Multi-step retrieval with chain-of-thought",
                supported_types=[
                    QueryType.EXPLANATION,
                    QueryType.COMPARISON,
                    QueryType.MULTI_HOP,
                ],
                estimated_latency_ms=800,
                estimated_cost=1.5,
                quality_score=0.95,
                required_resources=["bm25", "dense", "hybrid", "llm", "reasoner"],
            )
        )

        # Conversational path
        self.register_path(
            ProcessingPath(
                path_id="conversational",
                name="Conversational",
                description="Context-aware conversation handling",
                supported_types=[QueryType.CONVERSATIONAL],
                estimated_latency_ms=200,
                estimated_cost=0.3,
                quality_score=0.8,
                required_resources=["memory", "llm"],
            )
        )

    def register_path(self, path: ProcessingPath) -> None:
        """Register a processing path."""
        self.paths[path.path_id] = path
        logger.debug(f"Registered path: {path.name}")

    def get_compatible_paths(self, query_type: QueryType) -> List[ProcessingPath]:
        """Get paths compatible with query type."""
        return [path for path in self.paths.values() if query_type in path.supported_types]


class IntelligentRouter:
    """
    Intelligent query router for agentic RAG.

    Routes queries to optimal processing paths based on
    query characteristics and routing strategy.
    """

    def __init__(
        self,
        default_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
    ):
        """
        Initialize router.

        Args:
            default_strategy: Default routing strategy
        """
        self.default_strategy = default_strategy
        self.classifier = QueryClassifier()
        self.path_registry = ProcessingPathRegistry()

        # Statistics
        self.total_queries = 0
        self.routing_stats: Dict[str, int] = {}

    def route(
        self,
        query: str,
        strategy: Optional[RoutingStrategy] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Route query to optimal processing path.

        Args:
            query: User query
            strategy: Routing strategy (uses default if None)
            constraints: Optional constraints (e.g., max_latency_ms)

        Returns:
            Routing decision
        """
        strategy = strategy or self.default_strategy

        # Classify query
        query_type, confidence = self.classifier.classify(query)

        # Get compatible paths
        compatible_paths = self.path_registry.get_compatible_paths(query_type)

        if not compatible_paths:
            logger.warning(f"No compatible paths for {query_type}, using standard")
            compatible_paths = [self.path_registry.paths.get("standard_rag")]

        # Apply constraints
        if constraints:
            compatible_paths = self._apply_constraints(compatible_paths, constraints)

        # Select best path
        best_path = max(compatible_paths, key=lambda p: p.score(strategy))

        # Create decision
        decision = RoutingDecision(
            query=query,
            query_type=query_type,
            selected_path=best_path,
            strategy=strategy,
            confidence=confidence,
            reasoning=self._generate_reasoning(query_type, best_path, strategy),
        )

        # Update stats
        self.total_queries += 1
        self.routing_stats[best_path.path_id] = self.routing_stats.get(best_path.path_id, 0) + 1

        logger.info(
            f"Routed query to {best_path.name} "
            f"(type={query_type.value}, confidence={confidence:.2f})"
        )

        return decision

    def _apply_constraints(
        self,
        paths: List[ProcessingPath],
        constraints: Dict[str, Any],
    ) -> List[ProcessingPath]:
        """Apply constraints to filter paths."""
        filtered = paths

        # Max latency constraint
        if "max_latency_ms" in constraints:
            max_latency = constraints["max_latency_ms"]
            filtered = [p for p in filtered if p.estimated_latency_ms <= max_latency]

        # Max cost constraint
        if "max_cost" in constraints:
            max_cost = constraints["max_cost"]
            filtered = [p for p in filtered if p.estimated_cost <= max_cost]

        # Min quality constraint
        if "min_quality" in constraints:
            min_quality = constraints["min_quality"]
            filtered = [p for p in filtered if p.quality_score >= min_quality]

        if not filtered:
            logger.warning("No paths match constraints, using all paths")
            return paths

        return filtered

    def _generate_reasoning(
        self,
        query_type: QueryType,
        path: ProcessingPath,
        strategy: RoutingStrategy,
    ) -> str:
        """Generate human-readable reasoning for routing decision."""
        return (
            f"Query classified as {query_type.value} with {strategy.value} strategy. "
            f"Selected {path.name} for optimal balance of "
            f"latency ({path.estimated_latency_ms}ms), "
            f"quality ({path.quality_score:.2f}), "
            f"and cost ({path.estimated_cost:.2f})."
        )

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        stats = {
            "total_queries": self.total_queries,
            "by_path": {},
        }

        for path_id, count in self.routing_stats.items():
            path = self.path_registry.paths.get(path_id)
            if path:
                stats["by_path"][path_id] = {
                    "name": path.name,
                    "count": count,
                    "percentage": (count / self.total_queries * 100)
                    if self.total_queries > 0
                    else 0,
                }

        return stats

    def register_custom_path(self, path: ProcessingPath) -> None:
        """Register a custom processing path."""
        self.path_registry.register_path(path)


class AdaptiveRouter(IntelligentRouter):
    """
    Adaptive router that learns from query patterns.

    Adjusts routing decisions based on historical performance.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path_performance: Dict[str, List[float]] = {}

    def record_performance(
        self,
        path_id: str,
        latency_ms: float,
        quality_score: float,
    ) -> None:
        """
        Record actual performance of a path.

        Args:
            path_id: Path identifier
            latency_ms: Actual latency
            quality_score: Actual quality score
        """
        if path_id not in self.path_performance:
            self.path_performance[path_id] = []

        # Record combined metric
        combined_score = quality_score / (latency_ms / 100 + 1)
        self.path_performance[path_id].append(combined_score)

        # Keep only recent history
        if len(self.path_performance[path_id]) > 100:
            self.path_performance[path_id] = self.path_performance[path_id][-100:]

        logger.debug(f"Recorded performance for {path_id}: {combined_score:.3f}")

    def get_path_performance(self, path_id: str) -> float:
        """Get average performance score for path."""
        if path_id not in self.path_performance:
            return 1.0

        scores = self.path_performance[path_id]
        return sum(scores) / len(scores) if scores else 1.0


# Example usage
def example_routing():
    """Example of intelligent routing."""
    router = IntelligentRouter()

    queries = [
        "What is machine learning?",
        "Explain how neural networks work",
        "Compare supervised and unsupervised learning",
        "How many layers in a typical CNN?",
    ]

    for query in queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        print("=" * 50)

        # Route with different strategies
        for strategy in [
            RoutingStrategy.FASTEST,
            RoutingStrategy.BALANCED,
            RoutingStrategy.QUALITY,
        ]:
            decision = router.route(query, strategy=strategy)
            print(
                f"\n{strategy.value.upper()}: {decision.selected_path.name} "
                f"(latency={decision.selected_path.estimated_latency_ms}ms, "
                f"quality={decision.selected_path.quality_score:.2f})"
            )

    # Stats
    print("\n" + "=" * 50)
    print("Routing Statistics")
    print("=" * 50)
    stats = router.get_routing_stats()
    print(f"Total queries: {stats['total_queries']}")
    for path_id, path_stats in stats["by_path"].items():
        print(
            f"  {path_stats['name']}: {path_stats['count']} " f"({path_stats['percentage']:.1f}%)"
        )


if __name__ == "__main__":
    example_routing()
