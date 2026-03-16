# Routing Guide for AutoRAG-Live

This guide explains how to use the routing components in AutoRAG-Live to intelligently direct queries to appropriate retrievers based on intent, semantics, and system state.

## Table of Contents

1. [Routing Overview](#routing-overview)
2. [Core Routing Components](#core-routing-components)
3. [Intent Classification](#intent-classification)
4. [Semantic Routing](#semantic-routing)
5. [Intelligent Routing](#intelligent-routing)
6. [Load Balancing](#load-balancing)
7. [Custom Routing Strategies](#custom-routing-strategies)
8. [Best Practices](#best-practices)

## Routing Overview

Routing determines which retriever(s) handle each incoming query. AutoRAG-Live provides multiple routing strategies:

- **Intent-Based**: Classify query intent and route to specialized handlers
- **Semantic**: Route based on semantic similarity to known query types
- **Intelligent**: Combine intent, semantics, and system metrics
- **Load-Balanced**: Distribute queries across retrievers based on availability

### Why Routing Matters

In large RAG systems:
- Different queries benefit from different retrieval strategies
- System resources are finite (connections, memory, processing)
- Query intent varies (factual, reasoning, creative, etc.)
- Specialized retrievers excel at specific domain queries

### Example: Multi-Retriever System

```
Query: "Explain machine learning"
  ↓
Routing Layer
  ├→ Intent: EDUCATIONAL → Semantic Retriever
  ├→ Domain: AI/ML → Specialized ML Index
  ├→ Complexity: Medium → Medium-K Retrieval
  ↓
Results merged → Ranked → Returned to LLM
```

## Core Routing Components

### IntentClassifier

Classifies queries into intent categories:

```python
from autorag_live.routing import IntentClassifier

classifier = IntentClassifier()

# Define intents
intents = [
    "factual_question",  # "What is?"
    "reasoning",         # "Why does?"
    "comparison",        # "Difference between?"
    "creative",          # "Write a story about..."
    "technical",         # Code snippets, architecture
]

# Classify a query
query = "How does attention mechanism work?"
intent = classifier.classify(query, intents)
# Returns: "reasoning"
```

**How It Works**:

1. Converts query to embedding
2. Compares with intent prototypes
3. Returns most similar intent
4. Provides confidence score

**Use Cases**:
- Route factual queries to keyword-based retrievers
- Route reasoning queries to semantic retrievers
- Route technical queries to code-indexed retrievers

### SemanticRouter

Routes based on semantic similarity to query templates:

```python
from autorag_live.routing import SemanticRouter

router = SemanticRouter()

# Define routing rules with query templates
routes = {
    "medical": [
        "What are symptoms of?",
        "How to treat?",
        "What causes?",
    ],
    "legal": [
        "What is the law?",
        "Is it legal?",
        "How do I?",
    ],
    "technical": [
        "How do I implement?",
        "What is the best way?",
        "What is the difference?",
    ],
}

# Add routes
for domain, templates in routes.items():
    router.add_route(domain, templates)

# Route a query
query = "What is diabetes?"
route = router.route(query)
# Returns: "medical" (most similar to medical templates)
```

**Advantages**:
- No labeled training data needed
- Easy to add new domains
- Semantic understanding of query meaning
- Handles paraphrases

### IntelligentRouter

Combines multiple signals for smart routing:

```python
from autorag_live.routing import IntelligentRouter

intelligent_router = IntelligentRouter()

# Register retrievers with capabilities
intelligent_router.register_retriever(
    name="keyword_retriever",
    retriever=keyword_search,
    capabilities=["fast", "exact_match"],
    throughput=1000,  # Queries per second
)

intelligent_router.register_retriever(
    name="semantic_retriever",
    retriever=semantic_search,
    capabilities=["semantic", "paraphrase_robust"],
    throughput=100,
)

intelligent_router.register_retriever(
    name="hybrid_retriever",
    retriever=hybrid_search,
    capabilities=["semantic", "exact_match", "fast"],
    throughput=500,
)

# Route query considering multiple factors
query = "machine learning definition"

decision = intelligent_router.route(
    query=query,
    factors={
        "latency_target": 0.1,  # 100ms max
        "quality_target": 0.9,  # 90% relevance
        "cost_weight": 0.5,     # Balance cost vs quality
    }
)

# Returns routing decision with reasoning
print(decision)
# {
#     "recommended_retriever": "hybrid_retriever",
#     "confidence": 0.95,
#     "reasoning": "Balances speed and quality requirements",
#     "fallback": "semantic_retriever",
# }
```

### LoadBalancer

Distributes queries across retrievers:

```python
from autorag_live.routing import LoadBalancer

load_balancer = LoadBalancer(strategy="least_loaded")

# Register retrievers
load_balancer.add_retriever("retriever_1", max_concurrent=50)
load_balancer.add_retriever("retriever_2", max_concurrent=50)
load_balancer.add_retriever("retriever_3", max_concurrent=50)

# Route queries based on current load
for query in queries:
    selected_retriever = load_balancer.route(query)
    # Automatically selects least loaded retriever
    results = selected_retriever.retrieve(query)
```

**Strategies**:
- `least_loaded`: Distributes to retriever with fewest active requests
- `round_robin`: Cycles through retrievers in order
- `random`: Randomly selects retriever
- `weighted`: Distributes proportional to retriever weights

## Intent Classification

### Built-in Intent Categories

```python
from autorag_live.routing import IntentClassifier

classifier = IntentClassifier()

# Access standard intents
standard_intents = {
    "factual": "Direct factual questions",
    "why": "Causal reasoning questions",
    "how": "Procedural/instructional questions",
    "compare": "Comparison questions",
    "definition": "Definition/explanation requests",
    "creative": "Creative/generative requests",
    "code": "Programming/technical questions",
    "list": "Enumeration/listing requests",
}
```

### Implementing Custom Intent Classifier

```python
from typing import List
from autorag_live.routing import IntentClassifier

class DomainIntentClassifier(IntentClassifier):
    """Domain-specific intent classifier."""

    def __init__(self, domain: str):
        super().__init__()
        self.domain = domain

        # Domain-specific intents
        if domain == "medical":
            self.intents = [
                "symptom_inquiry",
                "treatment_question",
                "diagnosis_question",
                "prevention_query",
            ]
        elif domain == "legal":
            self.intents = [
                "law_query",
                "contract_question",
                "compliance_check",
                "rights_inquiry",
            ]

    def classify(self, query: str, intents: List[str] = None) -> str:
        """Classify query with domain context."""
        if intents is None:
            intents = self.intents

        return super().classify(query, intents)

# Usage
medical_classifier = DomainIntentClassifier("medical")
intent = medical_classifier.classify("What causes headaches?")
# Returns: "symptom_inquiry"
```

### Multi-Level Intent Hierarchy

```python
class HierarchicalIntentClassifier:
    """Route using intent hierarchy."""

    def __init__(self):
        self.hierarchy = {
            "question": {
                "factual": ["what", "who", "when", "where"],
                "reasoning": ["why", "how"],
                "hypothetical": ["what if", "suppose"],
            },
            "statement": {
                "assertion": ["confirm", "agree"],
                "disagreement": ["disagree", "but"],
            },
            "request": {
                "action": ["do", "make", "create"],
                "information": ["list", "summarize", "explain"],
            },
        }

    def classify(self, query: str) -> tuple:
        """Return (primary_intent, secondary_intent)."""
        # First level: Determine query type
        primary = self._detect_primary_intent(query)

        # Second level: Determine sub-intent
        secondary = self._detect_secondary_intent(query, primary)

        return (primary, secondary)

    def _detect_primary_intent(self, query: str) -> str:
        # Implementation
        pass

    def _detect_secondary_intent(self, query: str, primary: str) -> str:
        # Implementation
        pass

# Usage
classifier = HierarchicalIntentClassifier()
primary, secondary = classifier.classify("Why does machine learning work?")
# Returns: ("question", "reasoning")
```

## Semantic Routing

### Dynamic Route Discovery

```python
from autorag_live.routing import SemanticRouter
import numpy as np

class AdaptiveSemanticRouter(SemanticRouter):
    """Learns routing patterns from query/result pairs."""

    def __init__(self):
        super().__init__()
        self.route_performance = {}  # Track retriever performance

    def record_result(self, query: str, route: str, metrics: dict):
        """Record outcome of routing decision."""
        if route not in self.route_performance:
            self.route_performance[route] = []

        self.route_performance[route].append({
            "query": query,
            "latency": metrics.get("latency", 0),
            "relevance": metrics.get("relevance", 0),
            "user_satisfied": metrics.get("user_satisfied", False),
        })

    def optimize_routes(self):
        """Adjust routes based on performance history."""
        # Identify poor-performing routes
        for route, results in self.route_performance.items():
            avg_relevance = np.mean([r["relevance"] for r in results])

            if avg_relevance < 0.7:
                logger.warning(f"Route {route} underperforming: {avg_relevance}")
                # Could adjust route weights or templates

    def route(self, query: str, use_history: bool = True) -> str:
        """Route with optional consideration of history."""
        if use_history:
            # Check if similar queries were routed successfully
            similar_queries = self._find_similar_queries(query)
            if similar_queries:
                best_route = self._recommend_route(similar_queries)
                return best_route

        return super().route(query)

# Usage
router = AdaptiveSemanticRouter()

# Process queries and record results
for query in queries:
    route = router.route(query)
    results = retrieve(route, query)
    metrics = evaluate(results)
    router.record_result(query, route, metrics)

# Optimize based on history
router.optimize_routes()
```

## Intelligent Routing

### Multi-Factor Routing Decision

```python
from dataclasses import dataclass
from enum import Enum

class QueryComplexity(Enum):
    SIMPLE = "simple"      # Single fact lookup
    MODERATE = "moderate"  # Multiple facts/reasoning
    COMPLEX = "complex"    # Multi-step reasoning

@dataclass
class RoutingContext:
    """Context for routing decision."""
    query: str
    intent: str
    complexity: QueryComplexity
    domain: str
    user_preference: str = "balanced"  # balanced, fast, quality
    system_load: float = 0.5  # 0-1, 0=empty, 1=full

class ContextAwareRouter:
    """Route based on rich query context."""

    def __init__(self):
        self.retrievers = {}
        self.routing_rules = {}

    def route(self, context: RoutingContext) -> str:
        """Multi-factor routing decision."""

        # Factor 1: Intent-based preference
        intent_preference = self._get_intent_preference(context.intent)

        # Factor 2: Complexity-based requirement
        complexity_requirement = self._get_complexity_requirement(context.complexity)

        # Factor 3: Domain-based specialization
        domain_specialist = self._find_domain_specialist(context.domain)

        # Factor 4: System load and user preference
        load_aware = self._adjust_for_load(
            context.user_preference,
            context.system_load
        )

        # Combine factors
        scores = {}
        for retriever in self.retrievers:
            score = (
                0.3 * intent_preference.get(retriever, 0) +
                0.25 * complexity_requirement.get(retriever, 0) +
                0.25 * (1.0 if retriever == domain_specialist else 0.5) +
                0.2 * load_aware.get(retriever, 0)
            )
            scores[retriever] = score

        return max(scores, key=scores.get)

    def _get_intent_preference(self, intent: str) -> dict:
        # Return retriever scores for this intent
        pass

    def _get_complexity_requirement(self, complexity: QueryComplexity) -> dict:
        # Return retriever suitability for complexity
        pass

    def _find_domain_specialist(self, domain: str) -> str:
        # Return best retriever for domain
        pass

    def _adjust_for_load(self, user_preference: str, system_load: float) -> dict:
        # Adjust scores based on load and preference
        pass

# Usage
router = ContextAwareRouter()

context = RoutingContext(
    query="Explain transformer attention",
    intent="educational",
    complexity=QueryComplexity.MODERATE,
    domain="machine_learning",
    user_preference="quality",
    system_load=0.3,
)

best_retriever = router.route(context)
```

## Load Balancing

### Health-Aware Load Balancing

```python
from autoqa.routing import LoadBalancer
import asyncio
from datetime import datetime, timedelta

class HealthAwareLoadBalancer(LoadBalancer):
    """Load balance considering retriever health."""

    def __init__(self):
        super().__init__()
        self.health_status = {}  # retriever -> health info
        self.last_health_check = {}

    async def check_retriever_health(self, retriever_name: str) -> bool:
        """Check if retriever is healthy."""
        try:
            # Try simple query
            await asyncio.wait_for(
                self.retrievers[retriever_name].retrieve("test"),
                timeout=1.0
            )
            self.health_status[retriever_name] = "healthy"
            return True
        except Exception as e:
            logger.warning(f"Retriever {retriever_name} unhealthy: {e}")
            self.health_status[retriever_name] = "unhealthy"
            return False

    async def route_with_health_check(self, query: str) -> str:
        """Route with periodic health checks."""
        # Check health every minute
        now = datetime.now()

        healthy_retrievers = []
        for name, status in self.health_status.items():
            last_check = self.last_health_check.get(name, now)

            if now - last_check > timedelta(minutes=1):
                # Time for health check
                if await self.check_retriever_health(name):
                    self.last_health_check[name] = now
                    healthy_retrievers.append(name)
            elif status == "healthy":
                healthy_retrievers.append(name)

        # Route to healthy retrievers only
        if not healthy_retrievers:
            raise Exception("No healthy retrievers available")

        return self.route(query, candidates=healthy_retrievers)

# Usage
lb = HealthAwareLoadBalancer()

async def retrieve_with_health_check(query):
    retriever = await lb.route_with_health_check(query)
    return await retriever.retrieve(query)
```

## Custom Routing Strategies

### Canary Routing

Route small percentage of traffic to test new retrievers:

```python
import random

class CanaryRouter:
    """Route with canary deployments."""

    def __init__(self, stable_retriever, canary_retriever, canary_percent=5):
        self.stable = stable_retriever
        self.canary = canary_retriever
        self.canary_percent = canary_percent
        self.canary_results = []

    def route(self, query: str) -> tuple:
        """Return (retriever, is_canary)."""
        if random.random() < self.canary_percent / 100:
            return self.canary, True
        else:
            return self.stable, False

    def evaluate_canary(self) -> dict:
        """Evaluate canary retriever performance."""
        if not self.canary_results:
            return {"status": "no_data"}

        avg_quality = sum(
            r["quality"] for r in self.canary_results
        ) / len(self.canary_results)

        return {
            "quality": avg_quality,
            "traffic": len(self.canary_results),
            "ready_for_promotion": avg_quality > 0.95,
        }

# Usage
canary_router = CanaryRouter(
    stable_retriever=semantic_search,
    canary_retriever=new_ml_retriever,
    canary_percent=10,  # 10% traffic to new retriever
)

retriever, is_canary = canary_router.route(query)
results = retriever.retrieve(query)

if is_canary:
    canary_router.canary_results.append({
        "query": query,
        "quality": evaluate_quality(results),
    })

# Check if canary is ready
status = canary_router.evaluate_canary()
if status["ready_for_promotion"]:
    logger.info("Canary retriever ready for production")
```

### A/B Testing Router

Compare two retriever strategies:

```python
import hashlib

class ABTestRouter:
    """Route users to A or B variant consistently."""

    def __init__(self, retriever_a, retriever_b):
        self.a = retriever_a
        self.b = retriever_b
        self.results = {"a": [], "b": []}

    def route(self, user_id: str, query: str) -> tuple:
        """Return (retriever, variant) based on user."""
        # Consistent hash ensures same user always gets same variant
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        variant = "a" if user_hash % 2 == 0 else "b"

        retriever = self.a if variant == "a" else self.b
        return retriever, variant

    def record_result(self, user_id: str, variant: str, metrics: dict):
        """Record metrics for analysis."""
        self.results[variant].append({
            "user_id": user_id,
            "metrics": metrics,
        })

    def get_stats(self) -> dict:
        """Compare A vs B performance."""
        return {
            "a_quality": self._avg_metric("a", "relevance"),
            "b_quality": self._avg_metric("b", "relevance"),
            "a_latency": self._avg_metric("a", "latency"),
            "b_latency": self._avg_metric("b", "latency"),
            "significant": self._is_statistically_significant(),
        }

    def _avg_metric(self, variant: str, metric: str) -> float:
        if not self.results[variant]:
            return 0.0
        return sum(
            r["metrics"][metric]
            for r in self.results[variant]
        ) / len(self.results[variant])

    def _is_statistically_significant(self) -> bool:
        # Statistical test implementation
        pass
```

## Best Practices

### 1. Monitor Routing Decisions

```python
from collections import defaultdict
import logging

class MonitoredRouter:
    """Track and log routing patterns."""

    def __init__(self, base_router):
        self.router = base_router
        self.routing_history = defaultdict(int)
        self.logger = logging.getLogger("routing")

    def route(self, query: str) -> str:
        selected = self.router.route(query)
        self.routing_history[selected] += 1

        self.logger.info(
            f"Routed query to {selected}",
            extra={"query": query[:50]}
        )

        return selected

    def get_distribution(self) -> dict:
        """Get routing distribution."""
        total = sum(self.routing_history.values())
        return {
            name: count / total
            for name, count in self.routing_history.items()
        }

# Usage
monitored_router = MonitoredRouter(intelligent_router)
distribution = monitored_router.get_distribution()
logger.info(f"Routing distribution: {distribution}")
```

### 2. Implement Fallbacks

```python
async def route_with_fallback(query: str, routers: List[Router]) -> str:
    """Try multiple routers in sequence."""
    for router in routers:
        try:
            selected = router.route(query)
            if await verify_retriever_available(selected):
                return selected
        except Exception as e:
            logger.warning(f"Router {router} failed: {e}")
            continue

    # Final fallback
    return default_retriever
```

### 3. Cache Routing Decisions

```python
from functools import lru_cache

class CachedRouter:
    """Cache routing decisions for similar queries."""

    def __init__(self, base_router):
        self.router = base_router
        self._cache = {}

    def route(self, query: str) -> str:
        # Hash query for cache lookup
        query_hash = hash(query)

        if query_hash in self._cache:
            return self._cache[query_hash]

        selected = self.router.route(query)
        self._cache[query_hash] = selected

        return selected

    def clear_cache(self):
        self._cache.clear()
```

## Conclusion

AutoRAG-Live routing provides flexible, composable strategies for directing queries to optimal retrievers. Use:

- **Intent Classification** for query type understanding
- **Semantic Routing** for domain awareness
- **Intelligent Routing** for multi-factor decisions
- **Load Balancing** for resource management
- **Custom Strategies** for specialized requirements

Combine multiple routing approaches for robust, scalable RAG systems.
