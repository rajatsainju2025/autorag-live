"""
Advanced routing and task decomposition for agentic RAG.

Enables intelligent routing to specialized sub-agents based on query
characteristics and multi-step task decomposition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


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
