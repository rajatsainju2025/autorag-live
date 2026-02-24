"""
Advanced routing and task decomposition for agentic RAG.

Enables intelligent routing to specialized sub-agents based on query
characteristics and multi-step task decomposition.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

try:
    import numpy as np
except ImportError:  # numpy is optional for pure-keyword routing
    np = None  # type: ignore[assignment]


class QueryType(Enum):
    """Query classification types."""

    FACTUAL = "factual"  # "What is X?"
    PROCEDURAL = "procedural"  # "How do I...?"
    COMPARATIVE = "comparative"  # "Compare X and Y"
    ANALYTICAL = "analytical"  # "Why is X?"
    CREATIVE = "creative"  # "Generate idea for..."
    RETRIEVAL = "retrieval"  # Simple lookup


class QueryDomain(Enum):
    """Query domain classification."""

    GENERAL = "general"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    SCIENTIFIC = "scientific"


@dataclass
class QueryAnalysis:
    """Analysis of a query."""

    original_query: str
    query_type: QueryType
    domain: QueryDomain
    complexity: float  # 0-1 scale
    requires_multi_step: bool
    requires_reasoning: bool
    requires_external_knowledge: bool
    entity_count: int = 0
    key_phrases: List[str] = field(default_factory=list)
    confidence: float = 0.5

    def needs_decomposition(self) -> bool:
        """Check if query needs task decomposition."""
        return self.requires_multi_step and self.complexity > 0.6 and len(self.key_phrases) > 2


@dataclass
class Task:
    """Single task in decomposed workflow."""

    description: str
    task_type: str  # "retrieve", "reason", "synthesize", "evaluate"
    dependencies: List[str] = field(default_factory=list)
    estimated_complexity: float = 0.5
    required_resources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Decision for query routing."""

    original_query: str
    analysis: QueryAnalysis
    recommended_agent: str  # e.g., "semantic_retriever", "reasoning_engine"
    confidence: float
    fallback_agents: List[str] = field(default_factory=list)
    reasoning: str = ""
    parallel_tasks: List[Task] = field(default_factory=list)


class QueryClassifier:
    """Classifies queries into types and domains."""

    def __init__(self):
        """Initialize query classifier."""
        self.logger = logging.getLogger("QueryClassifier")

        # Type indicators
        self.type_indicators = {
            QueryType.FACTUAL: [
                "what",
                "when",
                "where",
                "who",
                "which",
                "name",
                "list",
                "define",
                "explain",
            ],
            QueryType.PROCEDURAL: [
                "how",
                "steps",
                "instructions",
                "guide",
                "process",
                "teach",
                "create",
                "build",
            ],
            QueryType.COMPARATIVE: [
                "compare",
                "difference",
                "contrast",
                "versus",
                "vs",
                "better",
                "worse",
                "similar",
            ],
            QueryType.ANALYTICAL: [
                "why",
                "cause",
                "effect",
                "reason",
                "impact",
                "consequence",
                "result",
            ],
            QueryType.CREATIVE: [
                "generate",
                "create",
                "invent",
                "suggest",
                "brainstorm",
                "idea",
                "novel",
                "unique",
            ],
        }

        # Domain indicators
        self.domain_indicators = {
            QueryDomain.TECHNICAL: [
                "code",
                "programming",
                "software",
                "api",
                "database",
                "algorithm",
                "system",
            ],
            QueryDomain.MEDICAL: [
                "disease",
                "treatment",
                "medicine",
                "symptom",
                "health",
                "clinical",
                "doctor",
            ],
            QueryDomain.LEGAL: [
                "law",
                "legal",
                "court",
                "contract",
                "attorney",
                "statute",
                "regulation",
            ],
            QueryDomain.FINANCIAL: [
                "money",
                "finance",
                "investment",
                "stock",
                "bank",
                "economy",
                "budget",
            ],
            QueryDomain.SCIENTIFIC: [
                "research",
                "study",
                "theory",
                "hypothesis",
                "experiment",
                "data",
                "scientific",
                "science",
            ],
        }

    def classify(self, query: str) -> QueryAnalysis:
        """Classify a query."""
        query_lower = query.lower()
        words = set(query_lower.split())

        # Determine query type
        query_type = self._classify_type(words)

        # Determine domain
        domain = self._classify_domain(words)

        # Assess complexity
        complexity = self._assess_complexity(query)

        # Check multi-step requirements
        requires_multi_step = any(
            word in query_lower for word in ["and", "then", "next", "after", "also"]
        )

        # Check reasoning requirement
        requires_reasoning = any(
            word in query_lower for word in ["why", "how", "analyze", "explain", "reason"]
        )

        # Extract key phrases
        key_phrases = self._extract_key_phrases(query)

        # Count entities
        entity_count = len(key_phrases)

        analysis = QueryAnalysis(
            original_query=query,
            query_type=query_type,
            domain=domain,
            complexity=complexity,
            requires_multi_step=requires_multi_step,
            requires_reasoning=requires_reasoning,
            requires_external_knowledge=True,
            entity_count=entity_count,
            key_phrases=key_phrases,
            confidence=0.7 + (entity_count * 0.05),
        )

        return analysis

    def _classify_type(self, words: set) -> QueryType:
        """Classify query type from words."""
        for query_type, indicators in self.type_indicators.items():
            if any(ind in words for ind in indicators):
                return query_type

        return QueryType.RETRIEVAL

    def _classify_domain(self, words: set) -> QueryDomain:
        """Classify query domain from words."""
        for domain, indicators in self.domain_indicators.items():
            if any(ind in words for ind in indicators):
                return domain

        return QueryDomain.GENERAL

    def _assess_complexity(self, query: str) -> float:
        """Assess query complexity."""
        # Heuristics
        word_count = len(query.split())
        question_count = query.count("?")
        clause_count = query.count(",")
        special_chars = sum(1 for c in query if c in "():[]{}@#$%")

        complexity = min(
            1.0,
            (word_count / 50) * 0.3
            + (question_count / 3) * 0.2
            + (clause_count / 3) * 0.3
            + (special_chars / 10) * 0.2,
        )

        return complexity

    def _extract_key_phrases(self, query: str) -> List[str]:
        """Extract key phrases from query."""
        # Simple noun/concept extraction
        phrases = []

        # Split by common delimiters
        parts = query.replace("and", "|").replace("or", "|").split("|")

        for part in parts:
            part = part.strip()
            if len(part) > 3:  # At least 3 chars
                phrases.append(part)

        return phrases[:5]  # Limit to 5 phrases


# =============================================================================
# Embedding-based semantic routing (fallback to keyword classifier)
# =============================================================================


class SemanticQueryRouter:
    """
    Embedding-cosine-similarity router for query type and domain classification.

    The pure keyword-based ``QueryClassifier`` fails on paraphrased / domain-
    shifted queries that are common in agentic multi-hop reasoning.  This class
    pre-computes centroid embeddings for each ``QueryType`` and ``QueryDomain``
    from a small set of representative seed phrases, then classifies unseen
    queries by finding the nearest centroid.

    Falls back gracefully to the keyword classifier when an embedding model
    is unavailable (no API key, no ``sentence-transformers``, etc.).

    Design
    ------
    * Centroid vectors are computed once at construction time and cached in
      memory — inference is a single matrix-vector dot product (O(C·D) where
      C = number of classes, D = embedding dimension ≈ 384).
    * Optional *confidence blending*: the final classification can blend the
      keyword score and the cosine score, weighted by ``blend_alpha``.

    Args:
        embed_fn:    Callable ``(texts: List[str]) → np.ndarray`` of shape
                     ``(n, d)``.  If ``None`` the keyword fallback is used.
        blend_alpha: Weight of embedding score in the final decision (0–1).
                     ``1.0`` = pure embedding; ``0.0`` = pure keyword.
    """

    # Seed phrases per QueryType — diverse enough to span the vocabulary
    _TYPE_SEEDS: Dict[str, List[str]] = {
        "factual": [
            "What is X?",
            "When did Y happen?",
            "Who invented Z?",
            "Define the term",
            "List the top results",
        ],
        "procedural": [
            "How do I build this?",
            "Step-by-step guide",
            "Explain the process",
            "Instructions to create",
            "How to configure",
        ],
        "comparative": [
            "Compare A and B",
            "Difference between X and Y",
            "Which is better",
            "Contrast the approaches",
            "Pros and cons",
        ],
        "analytical": [
            "Why does this happen?",
            "Analyze the root cause",
            "What is the impact of",
            "Explain the reason for",
            "Consequences of",
        ],
        "creative": [
            "Generate an idea for",
            "Brainstorm solutions",
            "Suggest a novel approach",
            "Create a plan for",
            "Invent a new way",
        ],
        "retrieval": [
            "Find the document about",
            "Search for information on",
            "Retrieve the record",
            "Look up the entry",
        ],
    }

    _DOMAIN_SEEDS: Dict[str, List[str]] = {
        "technical": [
            "API endpoint returns 500",
            "SQL query optimization",
            "Kubernetes deployment",
            "Python type hints",
        ],
        "medical": [
            "Symptoms of diabetes",
            "Drug interactions",
            "Clinical trial phase",
            "Patient diagnosis",
        ],
        "legal": [
            "Contract breach liability",
            "Statute of limitations",
            "Intellectual property rights",
            "Regulatory compliance",
        ],
        "financial": [
            "Portfolio diversification",
            "Earnings per share",
            "Interest rate risk",
            "Balance sheet analysis",
        ],
        "scientific": [
            "Hypothesis testing methodology",
            "Peer-reviewed study results",
            "Experimental control group",
            "Quantum entanglement",
        ],
        "general": [
            "General knowledge question",
            "Common everyday topic",
            "Miscellaneous information",
        ],
    }

    def __init__(
        self,
        embed_fn: Optional[Any] = None,
        blend_alpha: float = 0.7,
    ) -> None:
        self._embed_fn = embed_fn
        self.blend_alpha = blend_alpha
        self._keyword_clf = QueryClassifier()
        self.logger = logging.getLogger("SemanticQueryRouter")

        self._type_centroids: Optional[np.ndarray] = None
        self._type_labels: List[str] = []
        self._domain_centroids: Optional[np.ndarray] = None
        self._domain_labels: List[str] = []

        if embed_fn is not None:
            self._build_centroids()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> QueryAnalysis:
        """
        Classify *query* using embedding similarity (with keyword fallback).

        Returns a :class:`QueryAnalysis` with ``confidence`` reflecting the
        blend of keyword and embedding scores.
        """
        # Always get the keyword baseline
        kw_analysis = self._keyword_clf.classify(query)

        if self._embed_fn is None or self._type_centroids is None:
            return kw_analysis  # No embedding model available

        try:
            query_vec = self._embed_fn([query])  # (1, D)
            query_vec = self._l2_normalize(query_vec)

            # Type classification
            type_scores = query_vec @ self._type_centroids.T  # (1, C)
            best_type_idx = int(np.argmax(type_scores))
            type_cos = float(type_scores[0, best_type_idx])
            sem_type_name = self._type_labels[best_type_idx]

            # Domain classification
            domain_scores = query_vec @ self._domain_centroids.T
            best_domain_idx = int(np.argmax(domain_scores))
            domain_cos = float(domain_scores[0, best_domain_idx])
            sem_domain_name = self._domain_labels[best_domain_idx]

            # Blend: if cosine confidence is high, prefer semantic result
            if type_cos > (1.0 - self.blend_alpha):
                try:
                    query_type = QueryType(sem_type_name)
                except ValueError:
                    query_type = kw_analysis.query_type
            else:
                query_type = kw_analysis.query_type

            if domain_cos > (1.0 - self.blend_alpha):
                try:
                    domain = QueryDomain(sem_domain_name)
                except ValueError:
                    domain = kw_analysis.domain
            else:
                domain = kw_analysis.domain

            blended_confidence = (
                self.blend_alpha * (type_cos + domain_cos) / 2.0
                + (1.0 - self.blend_alpha) * kw_analysis.confidence
            )

            return QueryAnalysis(
                original_query=kw_analysis.original_query,
                query_type=query_type,
                domain=domain,
                complexity=kw_analysis.complexity,
                requires_multi_step=kw_analysis.requires_multi_step,
                requires_reasoning=kw_analysis.requires_reasoning,
                requires_external_knowledge=kw_analysis.requires_external_knowledge,
                entity_count=kw_analysis.entity_count,
                key_phrases=kw_analysis.key_phrases,
                confidence=min(1.0, blended_confidence),
            )

        except Exception as exc:
            self.logger.warning(f"Embedding routing failed, falling back to keywords: {exc}")
            return kw_analysis

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_centroids(self) -> None:
        """Pre-compute centroid embeddings for all classes."""
        try:
            # Type centroids
            type_texts, type_labels = [], []
            for label, seeds in self._TYPE_SEEDS.items():
                type_texts.extend(seeds)
                type_labels.extend([label] * len(seeds))

            type_vecs = self._l2_normalize(self._embed_fn(type_texts))
            self._type_labels = list(self._TYPE_SEEDS.keys())
            self._type_centroids = np.stack(
                [
                    type_vecs[
                        [i for i, lbl_name in enumerate(type_labels) if lbl_name == lbl]
                    ].mean(axis=0)
                    for lbl in self._type_labels
                ]
            )
            self._type_centroids = self._l2_normalize(self._type_centroids)

            # Domain centroids
            domain_texts, domain_labels = [], []
            for label, seeds in self._DOMAIN_SEEDS.items():
                domain_texts.extend(seeds)
                domain_labels.extend([label] * len(seeds))

            domain_vecs = self._l2_normalize(self._embed_fn(domain_texts))
            self._domain_labels = list(self._DOMAIN_SEEDS.keys())
            self._domain_centroids = np.stack(
                [
                    domain_vecs[
                        [i for i, lbl_name in enumerate(domain_labels) if lbl_name == lbl]
                    ].mean(axis=0)
                    for lbl in self._domain_labels
                ]
            )
            self._domain_centroids = self._l2_normalize(self._domain_centroids)

            self.logger.info(
                f"SemanticQueryRouter: built centroids for "
                f"{len(self._type_labels)} types, {len(self._domain_labels)} domains"
            )
        except Exception as exc:
            self.logger.warning(f"SemanticQueryRouter centroid build failed: {exc}")
            self._type_centroids = None
            self._domain_centroids = None

    @staticmethod
    def _l2_normalize(arr: "np.ndarray") -> "np.ndarray":
        """Row-wise L2 normalisation."""
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)


class TaskDecomposer:
    """Decomposes complex queries into subtasks."""

    def __init__(self):
        """Initialize task decomposer."""
        self.logger = logging.getLogger("TaskDecomposer")

    def decompose(self, analysis: QueryAnalysis) -> List[Task]:
        """Decompose query into tasks."""
        tasks = []

        if not analysis.needs_decomposition():
            # Single task
            tasks.append(
                Task(
                    description=f"Answer: {analysis.original_query}",
                    task_type="retrieve",
                )
            )
            return tasks

        # Multi-step decomposition
        if analysis.requires_multi_step:
            tasks.extend(self._decompose_multi_step(analysis))

        if analysis.requires_reasoning:
            tasks.extend(self._decompose_reasoning(analysis))

        # Add synthesis task
        tasks.append(
            Task(
                description="Synthesize final answer",
                task_type="synthesize",
                dependencies=[t.description for t in tasks[-2:]],
            )
        )

        return tasks

    def _decompose_multi_step(self, analysis: QueryAnalysis) -> List[Task]:
        """Decompose multi-step query."""
        tasks = []

        for i, phrase in enumerate(analysis.key_phrases[:3]):
            task = Task(
                description=f"Retrieve information about: {phrase}",
                task_type="retrieve",
                estimated_complexity=analysis.complexity / len(analysis.key_phrases),
            )
            tasks.append(task)

        return tasks

    def _decompose_reasoning(self, analysis: QueryAnalysis) -> List[Task]:
        """Decompose reasoning-heavy query."""
        tasks = []

        if analysis.query_type == QueryType.ANALYTICAL:
            tasks.extend(
                [
                    Task(
                        description="Identify causes and effects",
                        task_type="reason",
                        required_resources=["reasoning_engine"],
                    ),
                    Task(
                        description="Analyze implications",
                        task_type="reason",
                        required_resources=["reasoning_engine"],
                    ),
                ]
            )

        elif analysis.query_type == QueryType.COMPARATIVE:
            tasks.append(
                Task(
                    description="Compare entities",
                    task_type="reason",
                    required_resources=["comparison_engine"],
                )
            )

        return tasks


class Router:
    """
    Routes queries to specialized agents based on analysis.

    Implements intelligent routing with fallback mechanisms and
    support for specialized agent delegation.
    """

    def __init__(self):
        """Initialize router."""
        self.logger = logging.getLogger("Router")
        self.classifier = QueryClassifier()
        self.decomposer = TaskDecomposer()

        # Agent registry with specializations
        self.agent_registry = {
            "semantic_retriever": {
                "strengths": [QueryType.FACTUAL, QueryType.RETRIEVAL],
                "domains": [QueryDomain.GENERAL, QueryDomain.TECHNICAL],
            },
            "reasoning_engine": {
                "strengths": [QueryType.ANALYTICAL, QueryType.PROCEDURAL],
                "domains": [QueryDomain.SCIENTIFIC, QueryDomain.TECHNICAL],
            },
            "comparison_agent": {
                "strengths": [QueryType.COMPARATIVE],
                "domains": [QueryDomain.GENERAL],
            },
            "synthesis_agent": {
                "strengths": [QueryType.CREATIVE],
                "domains": [QueryDomain.GENERAL],
            },
            "domain_specialist": {
                "strengths": [QueryType.ANALYTICAL, QueryType.PROCEDURAL],
                "domains": [
                    QueryDomain.MEDICAL,
                    QueryDomain.LEGAL,
                    QueryDomain.FINANCIAL,
                ],
            },
        }

    def route(self, query: str) -> RoutingDecision:
        """Route query to appropriate agent(s)."""
        # Analyze query
        analysis = self.classifier.classify(query)

        # Select primary agent
        primary_agent = self._select_primary_agent(analysis)

        # Determine fallbacks
        fallbacks = self._get_fallback_agents(analysis, primary_agent)

        # Decompose if needed
        tasks = self.decomposer.decompose(analysis)

        decision = RoutingDecision(
            original_query=query,
            analysis=analysis,
            recommended_agent=primary_agent,
            fallback_agents=fallbacks,
            parallel_tasks=tasks,
            confidence=analysis.confidence,
            reasoning=self._explain_routing(primary_agent, analysis),
        )

        return decision

    def _select_primary_agent(self, analysis: QueryAnalysis) -> str:
        """Select best primary agent for query."""
        best_agent = "semantic_retriever"
        best_score = 0.0

        for agent_name, capabilities in self.agent_registry.items():
            score = 0.0

            # Match on query type
            if analysis.query_type in capabilities["strengths"]:
                score += 0.7

            # Match on domain
            if analysis.domain in capabilities["domains"]:
                score += 0.3

            if score > best_score:
                best_score = score
                best_agent = agent_name

        return best_agent

    def _get_fallback_agents(self, analysis: QueryAnalysis, primary: str) -> List[str]:
        """Get fallback agents."""
        fallbacks = []

        for agent_name in self.agent_registry:
            if agent_name != primary:
                # Check partial match
                if analysis.query_type in self.agent_registry[agent_name]["strengths"]:
                    fallbacks.append(agent_name)

        return fallbacks[:2]  # Limit to 2 fallbacks

    def _explain_routing(self, agent: str, analysis: QueryAnalysis) -> str:
        """Explain routing decision."""
        return (
            f"Routed to {agent} because query is "
            f"{analysis.query_type.value} type in "
            f"{analysis.domain.value} domain"
        )


# =============================================================================
# OPTIMIZATION 7: Adaptive Query Router with Learned Intent Classification
# Based on: "Learning to Route Queries in Large-Scale RAG Systems"
# and "RAFT: Adapting Language Model to Domain Specific RAG" (2024)
#
# Implements a neural query router that:
# 1. Learns intent embeddings for query classification
# 2. Adapts routing decisions based on performance feedback
# 3. Supports complex intent hierarchies for agentic workflows
# 4. Uses Thompson Sampling for exploration-exploitation tradeoff
# =============================================================================


class EmbeddingProtocol(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> List[float]:
        """Embed text into vector."""
        ...


class LLMProtocol(Protocol):
    """Protocol for LLM interactions."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response from prompt."""
        ...


@dataclass
class IntentDefinition:
    """Definition of a query intent for routing."""

    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    handler: Optional[str] = None
    priority: int = 0
    requires_retrieval: bool = True
    requires_reasoning: bool = False
    min_confidence: float = 0.5

    # Learned embedding (computed from examples)
    embedding: Optional[List[float]] = None


@dataclass
class RouteDecision:
    """A routing decision with confidence scores."""

    intent: str
    confidence: float
    handler: str
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingFeedback:
    """Feedback on a routing decision."""

    query: str
    selected_intent: str
    was_correct: bool
    actual_intent: Optional[str] = None
    response_quality: float = 0.0
    latency_ms: float = 0.0


class AdaptiveIntentRouter:
    """
    Neural adaptive query router with learned intent classification.

    Uses intent embeddings and Thompson Sampling for optimal routing
    in agentic RAG systems with multiple specialized handlers.

    Example:
        >>> router = AdaptiveIntentRouter(embedder)
        >>> router.register_intent(IntentDefinition(
        ...     name="factual_lookup",
        ...     description="Simple fact retrieval",
        ...     examples=["What is X?", "Who invented Y?"],
        ...     handler="dense_retriever"
        ... ))
        >>> decision = await router.route("What is machine learning?")
    """

    def __init__(
        self,
        embedder: Optional[EmbeddingProtocol] = None,
        llm: Optional[LLMProtocol] = None,
        similarity_threshold: float = 0.7,
        use_thompson_sampling: bool = True,
    ):
        """
        Initialize adaptive router.

        Args:
            embedder: Embedding provider for intent matching
            llm: LLM for complex intent classification
            similarity_threshold: Minimum similarity for intent match
            use_thompson_sampling: Use Thompson Sampling for exploration
        """
        self.embedder = embedder
        self.llm = llm
        self.similarity_threshold = similarity_threshold
        self.use_thompson_sampling = use_thompson_sampling

        # Intent registry
        self.intents: Dict[str, IntentDefinition] = {}
        self._intent_embeddings: Dict[str, List[float]] = {}

        # Thompson Sampling state (Beta distribution parameters)
        self._alpha: Dict[str, float] = {}  # Successes + 1
        self._beta: Dict[str, float] = {}  # Failures + 1

        # Statistics
        self._stats = {
            "total_routes": 0,
            "feedback_count": 0,
            "intent_accuracy": {},
        }

    def register_intent(self, intent: IntentDefinition) -> None:
        """Register an intent for routing."""
        self.intents[intent.name] = intent

        # Initialize Thompson Sampling parameters
        self._alpha[intent.name] = 1.0
        self._beta[intent.name] = 1.0

    async def compute_intent_embeddings(self) -> None:
        """Compute embeddings for all intent examples."""
        if not self.embedder:
            return

        for intent_name, intent in self.intents.items():
            if intent.examples:
                # Compute mean embedding of examples
                embeddings = []
                for example in intent.examples:
                    emb = await self.embedder.embed(example)
                    embeddings.append(emb)

                # Average embeddings
                if embeddings:
                    avg_embedding = [
                        sum(e[i] for e in embeddings) / len(embeddings)
                        for i in range(len(embeddings[0]))
                    ]
                    self._intent_embeddings[intent_name] = avg_embedding
                    intent.embedding = avg_embedding

    async def route(self, query: str, context: Optional[str] = None) -> RouteDecision:
        """
        Route query to appropriate handler.

        Args:
            query: User query
            context: Optional context for routing

        Returns:
            RouteDecision with handler and confidence
        """
        self._stats["total_routes"] += 1

        # Get intent scores
        scores = await self._score_intents(query)

        if not scores:
            return RouteDecision(
                intent="default",
                confidence=0.0,
                handler="semantic_retriever",
                reasoning="No intents registered",
            )

        # Apply Thompson Sampling if enabled
        if self.use_thompson_sampling:
            scores = self._apply_thompson_sampling(scores)

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        best_intent, best_score = sorted_scores[0]
        alternatives = sorted_scores[1:4]  # Top 3 alternatives

        intent_def = self.intents[best_intent]

        return RouteDecision(
            intent=best_intent,
            confidence=best_score,
            handler=intent_def.handler or "semantic_retriever",
            alternatives=alternatives,
            reasoning=f"Matched intent '{best_intent}' with confidence {best_score:.2f}",
            metadata={
                "requires_retrieval": intent_def.requires_retrieval,
                "requires_reasoning": intent_def.requires_reasoning,
            },
        )

    async def _score_intents(self, query: str) -> Dict[str, float]:
        """Score all intents for the query."""
        scores: Dict[str, float] = {}

        if self.embedder and self._intent_embeddings:
            # Use embedding similarity
            query_emb = await self.embedder.embed(query)

            for intent_name, intent_emb in self._intent_embeddings.items():
                similarity = self._cosine_similarity(query_emb, intent_emb)
                scores[intent_name] = similarity
        else:
            # Fallback to keyword matching
            for intent_name, intent in self.intents.items():
                score = self._keyword_match_score(query, intent)
                scores[intent_name] = score

        return scores

    def _apply_thompson_sampling(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply Thompson Sampling for exploration-exploitation."""
        adjusted_scores: Dict[str, float] = {}

        for intent_name, base_score in scores.items():
            # Sample from Beta distribution
            alpha = self._alpha.get(intent_name, 1.0)
            beta = self._beta.get(intent_name, 1.0)

            # Sample success probability
            sampled_prob = random.betavariate(alpha, beta)

            # Combine base score with sampled probability
            adjusted_scores[intent_name] = 0.7 * base_score + 0.3 * sampled_prob

        return adjusted_scores

    def record_feedback(self, feedback: RoutingFeedback) -> None:
        """
        Record feedback to update routing policy.

        Args:
            feedback: Routing feedback
        """
        self._stats["feedback_count"] += 1

        intent = feedback.selected_intent
        if intent not in self.intents:
            return

        # Update Thompson Sampling parameters
        if feedback.was_correct:
            self._alpha[intent] = self._alpha.get(intent, 1.0) + 1
        else:
            self._beta[intent] = self._beta.get(intent, 1.0) + 1

        # Update accuracy stats
        if intent not in self._stats["intent_accuracy"]:
            self._stats["intent_accuracy"][intent] = {"correct": 0, "total": 0}

        self._stats["intent_accuracy"][intent]["total"] += 1
        if feedback.was_correct:
            self._stats["intent_accuracy"][intent]["correct"] += 1

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _keyword_match_score(self, query: str, intent: IntentDefinition) -> float:
        """Score intent based on keyword matching."""
        query_lower = query.lower()
        score = 0.0

        # Check description keywords
        desc_words = intent.description.lower().split()
        matches = sum(1 for w in desc_words if w in query_lower)
        score += matches * 0.1

        # Check example similarity
        for example in intent.examples:
            example_words = set(example.lower().split())
            query_words = set(query_lower.split())
            overlap = len(example_words & query_words)
            score += overlap * 0.05

        return min(1.0, score)

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return dict(self._stats)


class HierarchicalIntentRouter:
    """
    Hierarchical router for complex intent taxonomies.

    Supports multi-level intent classification for agentic workflows
    with specialized sub-routers at each level.
    """

    def __init__(
        self,
        root_router: AdaptiveIntentRouter,
        sub_routers: Optional[Dict[str, AdaptiveIntentRouter]] = None,
    ):
        """
        Initialize hierarchical router.

        Args:
            root_router: Top-level router
            sub_routers: Sub-routers keyed by parent intent
        """
        self.root_router = root_router
        self.sub_routers = sub_routers or {}

    async def route(self, query: str) -> RouteDecision:
        """
        Route through hierarchy.

        Args:
            query: User query

        Returns:
            Final RouteDecision
        """
        # First-level routing
        root_decision = await self.root_router.route(query)

        # Check for sub-router
        if root_decision.intent in self.sub_routers:
            sub_router = self.sub_routers[root_decision.intent]
            sub_decision = await sub_router.route(query)

            # Combine decisions
            return RouteDecision(
                intent=f"{root_decision.intent}.{sub_decision.intent}",
                confidence=root_decision.confidence * sub_decision.confidence,
                handler=sub_decision.handler,
                alternatives=sub_decision.alternatives,
                reasoning=(
                    f"Hierarchical: {root_decision.reasoning} -> " f"{sub_decision.reasoning}"
                ),
            )

        return root_decision

    def add_sub_router(self, parent_intent: str, sub_router: AdaptiveIntentRouter) -> None:
        """Add a sub-router for a parent intent."""
        self.sub_routers[parent_intent] = sub_router


class LLMIntentRouter:
    """
    LLM-based intent router for complex queries.

    Uses LLM reasoning for queries that don't match clear intent patterns.
    """

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        available_intents: Optional[List[IntentDefinition]] = None,
    ):
        """Initialize LLM router."""
        self.llm = llm
        self.intents = available_intents or []

    async def route(self, query: str) -> RouteDecision:
        """Route using LLM reasoning."""
        if not self.llm:
            return RouteDecision(
                intent="default",
                confidence=0.0,
                handler="semantic_retriever",
            )

        # Build intent options
        intent_list = "\n".join(f"- {intent.name}: {intent.description}" for intent in self.intents)

        prompt = f"""Classify this query into one of the available intents.

Query: "{query}"

Available intents:
{intent_list}

Respond with:
INTENT: <intent_name>
CONFIDENCE: <0.0-1.0>
REASONING: <brief explanation>

Classification:"""

        response = await self.llm.generate(prompt, temperature=0.1)

        # Parse response
        intent, confidence, reasoning = self._parse_response(response)

        # Find handler
        handler = "semantic_retriever"
        for intent_def in self.intents:
            if intent_def.name == intent:
                handler = intent_def.handler or handler
                break

        return RouteDecision(
            intent=intent,
            confidence=confidence,
            handler=handler,
            reasoning=reasoning,
        )

    def _parse_response(self, response: str) -> Tuple[str, float, str]:
        """Parse LLM classification response."""
        import re

        intent = "default"
        confidence = 0.5
        reasoning = ""

        # Extract intent
        intent_match = re.search(r"INTENT:\s*(\w+)", response, re.I)
        if intent_match:
            intent = intent_match.group(1)

        # Extract confidence
        conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", response, re.I)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                pass

        # Extract reasoning
        reason_match = re.search(r"REASONING:\s*(.+)", response, re.I | re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return intent, confidence, reasoning


def create_rag_intent_router(
    embedder: Optional[EmbeddingProtocol] = None,
) -> AdaptiveIntentRouter:
    """
    Create a pre-configured RAG intent router.

    Args:
        embedder: Embedding provider

    Returns:
        Configured AdaptiveIntentRouter
    """
    router = AdaptiveIntentRouter(embedder=embedder)

    # Register common RAG intents
    router.register_intent(
        IntentDefinition(
            name="factual_lookup",
            description="Simple fact retrieval questions",
            examples=[
                "What is machine learning?",
                "Who invented the telephone?",
                "When was Python created?",
            ],
            handler="dense_retriever",
            requires_retrieval=True,
        )
    )

    router.register_intent(
        IntentDefinition(
            name="multi_hop_reasoning",
            description="Questions requiring multiple reasoning steps",
            examples=[
                "How does X relate to Y?",
                "What caused the event that led to Z?",
                "Compare the approaches of A and B",
            ],
            handler="reasoning_agent",
            requires_retrieval=True,
            requires_reasoning=True,
        )
    )

    router.register_intent(
        IntentDefinition(
            name="summarization",
            description="Summarize or synthesize information",
            examples=[
                "Summarize the key points of X",
                "What are the main ideas in this document?",
                "Give me an overview of Y",
            ],
            handler="summarization_agent",
            requires_retrieval=True,
        )
    )

    router.register_intent(
        IntentDefinition(
            name="procedural",
            description="How-to questions and procedures",
            examples=[
                "How do I configure X?",
                "What are the steps to do Y?",
                "How can I implement Z?",
            ],
            handler="procedural_retriever",
            requires_retrieval=True,
        )
    )

    return router
