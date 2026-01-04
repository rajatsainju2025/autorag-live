"""
Dynamic Retrieval Router Module.

Implements query-aware dynamic retrieval strategy selection using
rule-based and ML-based routing approaches.

Key Features:
1. Query complexity classification
2. Multi-strategy routing (dense, sparse, hybrid, graph)
3. Adaptive strategy selection based on feedback
4. Strategy ensemble with dynamic weighting
5. Performance-based learning

Example:
    >>> router = DynamicRetrievalRouter()
    >>> strategy = await router.route(query)
    >>> docs = await router.retrieve(query)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from autorag_live.core.protocols import BaseLLM, Document, Message, RetrievalResult

logger = logging.getLogger(__name__)


# =============================================================================
# Strategy Types
# =============================================================================


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""

    SPARSE = "sparse"  # BM25, keyword-based
    DENSE = "dense"  # Embedding-based
    HYBRID = "hybrid"  # Combination of sparse + dense
    GRAPH = "graph"  # Knowledge graph traversal
    MULTI_HOP = "multi_hop"  # Multi-step reasoning retrieval
    ENSEMBLE = "ensemble"  # Weighted combination


class QueryComplexity(str, Enum):
    """Query complexity levels."""

    SIMPLE = "simple"  # Direct lookup
    MODERATE = "moderate"  # Some reasoning needed
    COMPLEX = "complex"  # Multi-hop reasoning
    ANALYTICAL = "analytical"  # Deep analysis needed


class QueryIntent(str, Enum):
    """Query intent classification."""

    FACTUAL = "factual"  # What/when/where questions
    EXPLANATORY = "explanatory"  # How/why questions
    COMPARATIVE = "comparative"  # Compare X and Y
    PROCEDURAL = "procedural"  # How to do something
    CREATIVE = "creative"  # Generate/create something


@dataclass
class QueryFeatures:
    """
    Extracted features from a query.

    Attributes:
        length: Number of tokens
        entity_count: Number of entities detected
        has_temporal: Contains time references
        has_comparison: Contains comparison terms
        has_negation: Contains negation
        question_words: Question words found
        keywords: Important keywords
        complexity_score: Computed complexity (0-1)
    """

    length: int = 0
    entity_count: int = 0
    has_temporal: bool = False
    has_comparison: bool = False
    has_negation: bool = False
    question_words: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    complexity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "length": self.length,
            "entity_count": self.entity_count,
            "has_temporal": self.has_temporal,
            "has_comparison": self.has_comparison,
            "has_negation": self.has_negation,
            "question_words": self.question_words,
            "keywords": self.keywords,
            "complexity_score": self.complexity_score,
        }


@dataclass
class RoutingDecision:
    """
    Routing decision for a query.

    Attributes:
        primary_strategy: Main strategy to use
        fallback_strategies: Backup strategies
        confidence: Confidence in decision
        reasoning: Explanation for routing
        features: Extracted query features
        weights: Strategy weights for ensemble
    """

    primary_strategy: RetrievalStrategy
    fallback_strategies: List[RetrievalStrategy] = field(default_factory=list)
    confidence: float = 0.5
    reasoning: str = ""
    features: Optional[QueryFeatures] = None
    weights: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_strategy": self.primary_strategy.value,
            "fallback_strategies": [s.value for s in self.fallback_strategies],
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "weights": self.weights,
        }


@dataclass
class StrategyMetrics:
    """
    Performance metrics for a retrieval strategy.

    Attributes:
        strategy: The strategy
        total_calls: Number of times used
        avg_latency_ms: Average execution time
        avg_precision: Average precision at k
        avg_recall: Average recall
        success_rate: Success rate
        last_scores: Recent relevance scores
    """

    strategy: RetrievalStrategy
    total_calls: int = 0
    avg_latency_ms: float = 0.0
    avg_precision: float = 0.5
    avg_recall: float = 0.5
    success_rate: float = 1.0
    last_scores: List[float] = field(default_factory=list)

    def update(
        self,
        latency_ms: float,
        precision: float,
        recall: float,
        success: bool,
    ) -> None:
        """Update metrics with new observation."""
        # Exponential moving average
        alpha = 0.2
        self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms
        self.avg_precision = alpha * precision + (1 - alpha) * self.avg_precision
        self.avg_recall = alpha * recall + (1 - alpha) * self.avg_recall
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate

        self.total_calls += 1

        # Keep last N scores
        self.last_scores.append(precision)
        if len(self.last_scores) > 100:
            self.last_scores = self.last_scores[-100:]

    def get_score(self) -> float:
        """Calculate overall strategy score."""
        # Balance effectiveness and efficiency
        effectiveness = 0.6 * self.avg_precision + 0.4 * self.avg_recall
        efficiency = 1.0 / (1.0 + self.avg_latency_ms / 1000)  # Normalize latency
        reliability = self.success_rate

        return effectiveness * 0.5 + efficiency * 0.2 + reliability * 0.3


# =============================================================================
# Query Analyzer
# =============================================================================


class QueryAnalyzer:
    """
    Analyzes queries to extract features for routing.

    Uses rule-based heuristics and optionally LLM-based classification.
    """

    # Question word patterns
    FACTUAL_WORDS = {"what", "which", "who", "when", "where", "name", "list"}
    EXPLANATORY_WORDS = {"how", "why", "explain", "describe"}
    COMPARISON_WORDS = {"compare", "versus", "vs", "difference", "better", "worse", "similar"}

    # Complexity indicators
    MULTI_HOP_INDICATORS = {
        "and then",
        "after that",
        "following",
        "leads to",
        "causes",
        "results in",
        "therefore",
    }
    TEMPORAL_PATTERNS = [
        r"\b(before|after|during|since|until|when)\b",
        r"\b\d{4}\b",  # Year
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    ]

    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize analyzer."""
        self.llm = llm

    def extract_features(self, query: str) -> QueryFeatures:
        """
        Extract features from query.

        Args:
            query: Input query

        Returns:
            QueryFeatures with extracted features
        """
        query_lower = query.lower()
        words = query_lower.split()

        features = QueryFeatures(
            length=len(words),
            question_words=[w for w in words if w in self.FACTUAL_WORDS | self.EXPLANATORY_WORDS],
        )

        # Entity detection (simple heuristic: capitalized words)
        entities = re.findall(r"\b[A-Z][a-z]+\b", query)
        features.entity_count = len(entities)
        features.keywords = entities[:5]

        # Temporal detection
        features.has_temporal = any(
            re.search(pattern, query_lower) for pattern in self.TEMPORAL_PATTERNS
        )

        # Comparison detection
        features.has_comparison = any(word in query_lower for word in self.COMPARISON_WORDS)

        # Negation detection
        features.has_negation = bool(
            re.search(r"\b(not|no|never|without|don't|doesn't)\b", query_lower)
        )

        # Compute complexity score
        features.complexity_score = self._compute_complexity(features, query_lower)

        return features

    def _compute_complexity(self, features: QueryFeatures, query_lower: str) -> float:
        """Compute complexity score from features."""
        score = 0.0

        # Length contribution
        score += min(features.length / 20, 0.3)

        # Entity contribution
        score += min(features.entity_count * 0.1, 0.2)

        # Feature contributions
        if features.has_temporal:
            score += 0.1
        if features.has_comparison:
            score += 0.15
        if features.has_negation:
            score += 0.1

        # Multi-hop indicators
        if any(indicator in query_lower for indicator in self.MULTI_HOP_INDICATORS):
            score += 0.2

        return min(score, 1.0)

    def classify_complexity(self, features: QueryFeatures) -> QueryComplexity:
        """Classify query complexity from features."""
        score = features.complexity_score

        if score < 0.25:
            return QueryComplexity.SIMPLE
        elif score < 0.5:
            return QueryComplexity.MODERATE
        elif score < 0.75:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.ANALYTICAL

    def classify_intent(self, features: QueryFeatures, query: str) -> QueryIntent:
        """Classify query intent."""
        query_lower = query.lower()

        # Check for comparison
        if features.has_comparison:
            return QueryIntent.COMPARATIVE

        # Check question words
        first_word = query_lower.split()[0] if query_lower else ""

        if first_word in ("how", "what's the process", "steps"):
            if "work" in query_lower or "does" in query_lower:
                return QueryIntent.EXPLANATORY
            return QueryIntent.PROCEDURAL

        if first_word in ("why", "explain"):
            return QueryIntent.EXPLANATORY

        if first_word in self.FACTUAL_WORDS or "?" in query:
            return QueryIntent.FACTUAL

        if any(word in query_lower for word in ["create", "generate", "write", "compose"]):
            return QueryIntent.CREATIVE

        return QueryIntent.FACTUAL

    async def analyze_with_llm(self, query: str) -> Tuple[QueryComplexity, QueryIntent]:
        """
        Use LLM for more accurate classification.

        Args:
            query: Input query

        Returns:
            (complexity, intent) tuple
        """
        if not self.llm:
            features = self.extract_features(query)
            return self.classify_complexity(features), self.classify_intent(features, query)

        prompt = f"""Analyze this query and classify it:

Query: {query}

Classify:
1. Complexity: simple, moderate, complex, or analytical
2. Intent: factual, explanatory, comparative, procedural, or creative

Respond in JSON format:
{{"complexity": "<level>", "intent": "<type>"}}"""

        result = await self.llm.generate([Message.user(prompt)], temperature=0.0)

        try:
            import json

            data = json.loads(result.content)
            complexity = QueryComplexity(data.get("complexity", "moderate"))
            intent = QueryIntent(data.get("intent", "factual"))
            return complexity, intent
        except Exception:
            features = self.extract_features(query)
            return self.classify_complexity(features), self.classify_intent(features, query)


# =============================================================================
# Routing Logic
# =============================================================================


class RoutingPolicy:
    """
    Policy for mapping query characteristics to strategies.

    Encapsulates the routing logic that can be customized or learned.
    """

    # Default mappings
    COMPLEXITY_TO_STRATEGY: Dict[QueryComplexity, RetrievalStrategy] = {
        QueryComplexity.SIMPLE: RetrievalStrategy.SPARSE,
        QueryComplexity.MODERATE: RetrievalStrategy.HYBRID,
        QueryComplexity.COMPLEX: RetrievalStrategy.MULTI_HOP,
        QueryComplexity.ANALYTICAL: RetrievalStrategy.ENSEMBLE,
    }

    INTENT_TO_STRATEGY: Dict[QueryIntent, RetrievalStrategy] = {
        QueryIntent.FACTUAL: RetrievalStrategy.SPARSE,
        QueryIntent.EXPLANATORY: RetrievalStrategy.DENSE,
        QueryIntent.COMPARATIVE: RetrievalStrategy.HYBRID,
        QueryIntent.PROCEDURAL: RetrievalStrategy.DENSE,
        QueryIntent.CREATIVE: RetrievalStrategy.DENSE,
    }

    def __init__(self):
        """Initialize policy."""
        self._strategy_metrics: Dict[RetrievalStrategy, StrategyMetrics] = {
            strategy: StrategyMetrics(strategy=strategy) for strategy in RetrievalStrategy
        }

    def get_strategy(
        self,
        features: QueryFeatures,
        complexity: QueryComplexity,
        intent: QueryIntent,
    ) -> RoutingDecision:
        """
        Get routing decision based on query analysis.

        Args:
            features: Extracted query features
            complexity: Query complexity
            intent: Query intent

        Returns:
            RoutingDecision with strategy selection
        """
        # Get base strategies
        complexity_strategy = self.COMPLEXITY_TO_STRATEGY.get(complexity, RetrievalStrategy.HYBRID)
        intent_strategy = self.INTENT_TO_STRATEGY.get(intent, RetrievalStrategy.DENSE)

        # Combine signals
        if complexity_strategy == intent_strategy:
            primary = complexity_strategy
            confidence = 0.9
        elif features.complexity_score > 0.6:
            # Trust complexity for complex queries
            primary = complexity_strategy
            confidence = 0.7
        else:
            # Balance both signals
            primary = intent_strategy
            confidence = 0.6

        # Override for specific patterns
        if features.entity_count > 3:
            primary = RetrievalStrategy.GRAPH
            confidence = 0.75

        if features.has_comparison:
            primary = RetrievalStrategy.HYBRID
            confidence = 0.8

        # Get fallbacks (strategies with good historical performance)
        fallbacks = self._get_fallback_strategies(primary)

        # Compute ensemble weights
        weights = self._compute_weights(features)

        reasoning = (
            f"Query complexity: {complexity.value}, intent: {intent.value}. "
            f"Features: {features.entity_count} entities, "
            f"complexity score: {features.complexity_score:.2f}"
        )

        return RoutingDecision(
            primary_strategy=primary,
            fallback_strategies=fallbacks,
            confidence=confidence,
            reasoning=reasoning,
            features=features,
            weights=weights,
        )

    def _get_fallback_strategies(
        self,
        primary: RetrievalStrategy,
        max_fallbacks: int = 2,
    ) -> List[RetrievalStrategy]:
        """Get fallback strategies based on performance."""
        candidates = [s for s in RetrievalStrategy if s != primary]

        # Sort by historical performance
        scored = [(s, self._strategy_metrics[s].get_score()) for s in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [s for s, _ in scored[:max_fallbacks]]

    def _compute_weights(self, features: QueryFeatures) -> Dict[str, float]:
        """Compute strategy weights for ensemble."""
        weights = {}

        # Base weights from performance
        total_score = sum(m.get_score() for m in self._strategy_metrics.values())

        for strategy, metrics in self._strategy_metrics.items():
            base_weight = metrics.get_score() / total_score if total_score > 0 else 0.25
            weights[strategy.value] = base_weight

        # Adjust based on features
        if features.complexity_score > 0.5:
            weights[RetrievalStrategy.DENSE.value] *= 1.2
            weights[RetrievalStrategy.SPARSE.value] *= 0.8

        if features.entity_count > 2:
            weights[RetrievalStrategy.GRAPH.value] *= 1.3

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def update_metrics(
        self,
        strategy: RetrievalStrategy,
        latency_ms: float,
        precision: float,
        recall: float,
        success: bool,
    ) -> None:
        """Update strategy metrics with feedback."""
        self._strategy_metrics[strategy].update(latency_ms, precision, recall, success)

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all strategy metrics."""
        return {
            s.value: {
                "score": m.get_score(),
                "calls": m.total_calls,
                "precision": m.avg_precision,
                "latency_ms": m.avg_latency_ms,
            }
            for s, m in self._strategy_metrics.items()
        }


# =============================================================================
# Dynamic Router
# =============================================================================


class DynamicRetrievalRouter:
    """
    Dynamic router that selects retrieval strategies based on query analysis.

    Features:
    - Query feature extraction and classification
    - ML-based or rule-based routing
    - Performance-based strategy adaptation
    - Ensemble retrieval support

    Example:
        >>> router = DynamicRetrievalRouter(retrievers=my_retrievers)
        >>> decision = await router.route("What is machine learning?")
        >>> docs = await router.retrieve("What is machine learning?")
    """

    def __init__(
        self,
        retrievers: Optional[Dict[RetrievalStrategy, Callable]] = None,
        llm: Optional[BaseLLM] = None,
        use_llm_classification: bool = False,
    ):
        """
        Initialize router.

        Args:
            retrievers: Dict mapping strategies to retriever functions
            llm: LLM for classification (optional)
            use_llm_classification: Use LLM for query classification
        """
        self.retrievers = retrievers or {}
        self.analyzer = QueryAnalyzer(llm if use_llm_classification else None)
        self.policy = RoutingPolicy()
        self.use_llm = use_llm_classification and llm is not None

    def register_retriever(
        self,
        strategy: RetrievalStrategy,
        retriever: Callable[[str, int], List[Document]],
    ) -> None:
        """Register a retriever for a strategy."""
        self.retrievers[strategy] = retriever

    async def route(self, query: str) -> RoutingDecision:
        """
        Route query to appropriate strategy.

        Args:
            query: User query

        Returns:
            RoutingDecision with selected strategy
        """
        # Extract features
        features = self.analyzer.extract_features(query)

        # Classify
        if self.use_llm:
            complexity, intent = await self.analyzer.analyze_with_llm(query)
        else:
            complexity = self.analyzer.classify_complexity(features)
            intent = self.analyzer.classify_intent(features, query)

        # Get routing decision
        decision = self.policy.get_strategy(features, complexity, intent)

        logger.info(
            f"Routed query to {decision.primary_strategy.value} "
            f"(confidence: {decision.confidence:.2f})"
        )

        return decision

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        strategy: Optional[RetrievalStrategy] = None,
    ) -> RetrievalResult:
        """
        Retrieve documents using appropriate strategy.

        Args:
            query: User query
            k: Number of documents
            strategy: Override strategy selection

        Returns:
            RetrievalResult with documents
        """
        import asyncio
        import time

        # Get routing decision
        if strategy:
            decision = RoutingDecision(primary_strategy=strategy)
        else:
            decision = await self.route(query)

        selected_strategy = decision.primary_strategy

        # Check if retriever is registered
        if selected_strategy not in self.retrievers:
            # Fall back to available retriever
            if self.retrievers:
                selected_strategy = next(iter(self.retrievers.keys()))
            else:
                return RetrievalResult(
                    documents=[],
                    query=query,
                    metadata={"error": "No retriever available"},
                )

        # Execute retrieval
        retriever = self.retrievers[selected_strategy]
        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(retriever):
                docs = await retriever(query, k)
            else:
                docs = retriever(query, k)

            latency_ms = (time.time() - start_time) * 1000

            # Update metrics
            precision = self._estimate_precision(docs)
            self.policy.update_metrics(
                selected_strategy,
                latency_ms,
                precision,
                precision,  # Use precision as proxy for recall
                success=True,
            )

            return RetrievalResult(
                documents=docs if isinstance(docs, list) else docs.documents,
                query=query,
                latency_ms=latency_ms,
                metadata={
                    "strategy": selected_strategy.value,
                    "routing_decision": decision.to_dict(),
                },
            )

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            self.policy.update_metrics(
                selected_strategy,
                latency_ms=(time.time() - start_time) * 1000,
                precision=0.0,
                recall=0.0,
                success=False,
            )

            # Try fallback
            for fallback in decision.fallback_strategies:
                if fallback in self.retrievers:
                    try:
                        retriever = self.retrievers[fallback]
                        if asyncio.iscoroutinefunction(retriever):
                            docs = await retriever(query, k)
                        else:
                            docs = retriever(query, k)
                        return RetrievalResult(
                            documents=docs if isinstance(docs, list) else docs.documents,
                            query=query,
                            metadata={"strategy": fallback.value, "fallback": True},
                        )
                    except Exception:
                        continue

            return RetrievalResult(
                documents=[],
                query=query,
                metadata={"error": str(e)},
            )

    async def retrieve_ensemble(
        self,
        query: str,
        k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve using weighted ensemble of strategies.

        Args:
            query: User query
            k: Documents per strategy

        Returns:
            Combined retrieval result
        """
        import asyncio

        decision = await self.route(query)
        weights = decision.weights

        # Run all available retrievers
        all_docs: List[Document] = []
        tasks = []

        for strategy, retriever in self.retrievers.items():
            weight = weights.get(strategy.value, 0.25)
            if weight > 0.1:  # Skip very low weight strategies
                tasks.append((strategy, retriever, weight))

        for strategy, retriever, weight in tasks:
            try:
                if asyncio.iscoroutinefunction(retriever):
                    docs = await retriever(query, k)
                else:
                    docs = retriever(query, k)

                # Apply weight to scores
                for doc in docs if isinstance(docs, list) else docs.documents:
                    doc.score = doc.score * weight
                    all_docs.append(doc)
            except Exception as e:
                logger.warning(f"Strategy {strategy.value} failed: {e}")

        # Deduplicate and sort by score
        seen_ids = set()
        unique_docs = []
        for doc in sorted(all_docs, key=lambda d: d.score, reverse=True):
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                unique_docs.append(doc)
                if len(unique_docs) >= k:
                    break

        return RetrievalResult(
            documents=unique_docs,
            query=query,
            metadata={
                "strategy": "ensemble",
                "weights": weights,
            },
        )

    def _estimate_precision(self, docs: List[Document]) -> float:
        """Estimate precision from document scores."""
        if not docs:
            return 0.0
        avg_score = sum(d.score for d in docs) / len(docs)
        # Assume scores are normalized 0-1
        return min(max(avg_score, 0.0), 1.0)

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get strategy performance metrics."""
        return self.policy.get_metrics()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_router(
    retrievers: Dict[str, Callable],
    llm: Optional[BaseLLM] = None,
) -> DynamicRetrievalRouter:
    """
    Create a router with retrievers.

    Args:
        retrievers: Dict mapping strategy names to retriever functions
        llm: Optional LLM for classification

    Returns:
        Configured DynamicRetrievalRouter
    """
    router = DynamicRetrievalRouter(llm=llm)

    for name, retriever in retrievers.items():
        try:
            strategy = RetrievalStrategy(name)
            router.register_retriever(strategy, retriever)
        except ValueError:
            logger.warning(f"Unknown strategy: {name}")

    return router
