"""
Adaptive RAG Controller Module.

Implements dynamic pipeline selection based on query complexity analysis,
following the Adaptive-RAG pattern (Jeong et al., 2024).

Key Features:
1. Query complexity classification (simple/moderate/complex)
2. Dynamic pipeline routing (no retrieval, single, iterative)
3. Confidence-based fallback mechanisms
4. Performance-driven adaptation

Example:
    >>> controller = AdaptiveRAGController(llm, retriever)
    >>> result = await controller.run("What is machine learning?")
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from autorag_live.core.protocols import BaseLLM, Document, Message

logger = logging.getLogger(__name__)


# =============================================================================
# Complexity Classification
# =============================================================================


class QueryComplexity(str, Enum):
    """Query complexity levels for adaptive routing."""

    SIMPLE = "simple"  # Direct answer, no retrieval needed
    MODERATE = "moderate"  # Single retrieval pass
    COMPLEX = "complex"  # Multi-hop or iterative retrieval
    AMBIGUOUS = "ambiguous"  # Needs clarification


class PipelineType(str, Enum):
    """Available pipeline configurations."""

    NO_RETRIEVAL = "no_retrieval"  # LLM-only (parametric knowledge)
    SINGLE_RETRIEVAL = "single_retrieval"  # Standard RAG
    ITERATIVE_RETRIEVAL = "iterative_retrieval"  # Multi-hop RAG
    SELF_RAG = "self_rag"  # Self-reflective RAG
    HYBRID = "hybrid"  # Combination based on confidence


@dataclass
class ComplexityAnalysis:
    """
    Analysis of query complexity.

    Attributes:
        complexity: Classified complexity level
        confidence: Classification confidence (0-1)
        reasoning: Explanation for classification
        features: Extracted query features
        recommended_pipeline: Suggested pipeline type
    """

    complexity: QueryComplexity
    confidence: float = 0.8
    reasoning: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    recommended_pipeline: PipelineType = PipelineType.SINGLE_RETRIEVAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "complexity": self.complexity.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "recommended_pipeline": self.recommended_pipeline.value,
            "features": self.features,
        }


@dataclass
class AdaptiveResult:
    """
    Result from adaptive RAG execution.

    Attributes:
        answer: Generated answer
        pipeline_used: Pipeline that was executed
        complexity_analysis: Query analysis details
        retrieval_rounds: Number of retrieval passes
        latency_ms: Total execution time
        confidence: Answer confidence
        sources: Retrieved source documents
    """

    answer: str
    pipeline_used: PipelineType
    complexity_analysis: ComplexityAnalysis
    retrieval_rounds: int = 0
    latency_ms: float = 0.0
    confidence: float = 0.8
    sources: List[Document] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Complexity Classifier
# =============================================================================


class ComplexityClassifier:
    """
    Classifies query complexity for adaptive routing.

    Uses both heuristic rules and optional LLM-based classification.
    """

    # Heuristic patterns
    SIMPLE_PATTERNS = [
        r"^what\s+is\s+\w+\??$",  # "What is X?"
        r"^define\s+\w+\??$",  # "Define X"
        r"^who\s+(is|was)\s+\w+\??$",  # "Who is/was X?"
        r"^when\s+(did|was|is)\s+",  # Simple temporal
    ]

    COMPLEX_INDICATORS = [
        "compare",
        "contrast",
        "difference between",
        "how does",
        "why does",
        "explain the relationship",
        "step by step",
        "impact of",
        "consequences of",
        "advantages and disadvantages",
        "pros and cons",
    ]

    MULTI_HOP_INDICATORS = [
        "and then",
        "after that",
        "leads to",
        "causes",
        "results in",
        "therefore",
        "because of",
        "how did",
        "what happened when",
    ]

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        use_llm_classification: bool = True,
    ):
        """
        Initialize classifier.

        Args:
            llm: LLM for classification (optional)
            use_llm_classification: Use LLM for accurate classification
        """
        self.llm = llm
        self.use_llm = use_llm_classification and llm is not None

    def classify(self, query: str) -> ComplexityAnalysis:
        """
        Classify query complexity using heuristics.

        Args:
            query: User query

        Returns:
            ComplexityAnalysis with classification
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        features = self._extract_features(query)

        # Check for simple patterns
        for pattern in self.SIMPLE_PATTERNS:
            if re.match(pattern, query_lower):
                return ComplexityAnalysis(
                    complexity=QueryComplexity.SIMPLE,
                    confidence=0.85,
                    reasoning="Matches simple query pattern",
                    features=features,
                    recommended_pipeline=PipelineType.NO_RETRIEVAL,
                )

        # Check for complex indicators
        complex_matches = sum(1 for ind in self.COMPLEX_INDICATORS if ind in query_lower)
        multi_hop_matches = sum(1 for ind in self.MULTI_HOP_INDICATORS if ind in query_lower)

        # Length-based complexity
        length_score = min(len(words) / 20, 1.0)

        # Question count
        question_count = query.count("?")

        # Compute overall complexity score
        complexity_score = (
            complex_matches * 0.2
            + multi_hop_matches * 0.3
            + length_score * 0.2
            + (question_count - 1) * 0.15
        )

        # Classify based on score
        if complexity_score < 0.2:
            complexity = QueryComplexity.SIMPLE
            pipeline = PipelineType.SINGLE_RETRIEVAL
        elif complexity_score < 0.5:
            complexity = QueryComplexity.MODERATE
            pipeline = PipelineType.SINGLE_RETRIEVAL
        else:
            complexity = QueryComplexity.COMPLEX
            pipeline = PipelineType.ITERATIVE_RETRIEVAL

        # Handle multi-hop specifically
        if multi_hop_matches > 0:
            pipeline = PipelineType.ITERATIVE_RETRIEVAL

        return ComplexityAnalysis(
            complexity=complexity,
            confidence=0.7,
            reasoning=f"Score: {complexity_score:.2f}, complex_matches: {complex_matches}, multi_hop: {multi_hop_matches}",
            features=features,
            recommended_pipeline=pipeline,
        )

    async def classify_with_llm(self, query: str) -> ComplexityAnalysis:
        """
        Use LLM for more accurate classification.

        Args:
            query: User query

        Returns:
            ComplexityAnalysis with LLM classification
        """
        if not self.llm:
            return self.classify(query)

        prompt = f"""Classify the complexity of this query for a RAG system:

Query: {query}

Classify as one of:
- SIMPLE: Basic factual question that an LLM can answer from general knowledge
- MODERATE: Requires retrieving relevant documents for accurate answer
- COMPLEX: Requires multi-step reasoning or multiple retrieval passes

Also recommend a pipeline:
- NO_RETRIEVAL: LLM can answer directly
- SINGLE_RETRIEVAL: One retrieval pass is sufficient
- ITERATIVE_RETRIEVAL: Needs multiple retrieval rounds

Respond in JSON:
{{"complexity": "SIMPLE|MODERATE|COMPLEX", "pipeline": "...", "reasoning": "..."}}"""

        try:
            result = await self.llm.generate(
                [Message.user(prompt)],
                temperature=0.0,
                max_tokens=200,
            )

            import json

            data = json.loads(result.content)
            complexity = QueryComplexity(data.get("complexity", "moderate").lower())
            pipeline = PipelineType(data.get("pipeline", "single_retrieval").lower())

            return ComplexityAnalysis(
                complexity=complexity,
                confidence=0.9,
                reasoning=data.get("reasoning", ""),
                features=self._extract_features(query),
                recommended_pipeline=pipeline,
            )

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return self.classify(query)

    def _extract_features(self, query: str) -> Dict[str, Any]:
        """Extract query features for analysis."""
        query_lower = query.lower()
        words = query_lower.split()

        return {
            "length": len(words),
            "question_count": query.count("?"),
            "has_comparison": any(
                w in query_lower for w in ["compare", "vs", "versus", "difference"]
            ),
            "has_temporal": any(w in query_lower for w in ["when", "before", "after", "during"]),
            "has_causal": any(w in query_lower for w in ["why", "because", "cause", "result"]),
            "entity_count": len(re.findall(r"\b[A-Z][a-z]+\b", query)),
        }


# =============================================================================
# Pipeline Executors
# =============================================================================


class PipelineExecutor:
    """Base class for pipeline executors."""

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Optional[Callable] = None,
    ):
        """Initialize executor."""
        self.llm = llm
        self.retriever = retriever

    async def execute(
        self,
        query: str,
        **kwargs: Any,
    ) -> Tuple[str, List[Document], int]:
        """
        Execute pipeline.

        Returns:
            (answer, sources, retrieval_rounds)
        """
        raise NotImplementedError


class NoRetrievalPipeline(PipelineExecutor):
    """LLM-only pipeline for simple queries."""

    async def execute(
        self,
        query: str,
        **kwargs: Any,
    ) -> Tuple[str, List[Document], int]:
        """Execute without retrieval."""
        result = await self.llm.generate(
            [Message.user(query)],
            temperature=0.7,
        )
        return result.content, [], 0


class SingleRetrievalPipeline(PipelineExecutor):
    """Standard single-pass RAG pipeline."""

    async def execute(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> Tuple[str, List[Document], int]:
        """Execute single retrieval pass."""
        # Retrieve documents
        docs = []
        if self.retriever:
            if asyncio.iscoroutinefunction(self.retriever):
                result = await self.retriever(query, k)
            else:
                result = self.retriever(query, k)

            docs = result if isinstance(result, list) else result.documents

        # Generate with context
        context = "\n\n".join(f"[{i+1}] {d.content}" for i, d in enumerate(docs))

        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.7,
        )

        return result.content, docs, 1


class IterativeRetrievalPipeline(PipelineExecutor):
    """Multi-hop iterative retrieval pipeline."""

    MAX_ITERATIONS = 3

    async def execute(
        self,
        query: str,
        k: int = 3,
        max_iterations: int = 3,
        **kwargs: Any,
    ) -> Tuple[str, List[Document], int]:
        """Execute iterative retrieval."""
        all_docs: List[Document] = []
        current_query = query
        iterations = 0
        accumulated_context = ""

        for i in range(min(max_iterations, self.MAX_ITERATIONS)):
            iterations += 1

            # Retrieve
            if self.retriever:
                if asyncio.iscoroutinefunction(self.retriever):
                    result = await self.retriever(current_query, k)
                else:
                    result = self.retriever(current_query, k)

                new_docs = result if isinstance(result, list) else result.documents
                all_docs.extend(new_docs)

                new_context = "\n".join(d.content for d in new_docs)
                accumulated_context += f"\n--- Round {i+1} ---\n{new_context}"

            # Check if we have enough information
            check_prompt = f"""Based on the context so far, determine if you can fully answer the question.

Question: {query}

Context collected:
{accumulated_context}

Can you fully answer the question? Respond with:
- "YES" if you have sufficient information
- "NEED_MORE: <refined query>" if more retrieval is needed

Response:"""

            check_result = await self.llm.generate(
                [Message.user(check_prompt)],
                temperature=0.0,
                max_tokens=100,
            )

            response = check_result.content.strip()

            if response.startswith("YES"):
                break
            elif response.startswith("NEED_MORE:"):
                current_query = response.replace("NEED_MORE:", "").strip()
            else:
                break

        # Generate final answer
        prompt = f"""Answer the question comprehensively based on all retrieved context.

Context:
{accumulated_context}

Question: {query}

Comprehensive Answer:"""

        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.7,
        )

        # Deduplicate documents
        seen_ids = set()
        unique_docs = []
        for doc in all_docs:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                unique_docs.append(doc)

        return result.content, unique_docs, iterations


# =============================================================================
# Adaptive RAG Controller
# =============================================================================


class AdaptiveRAGController:
    """
    Adaptive RAG controller that dynamically selects pipelines.

    Implements the Adaptive-RAG pattern for query-aware pipeline routing.

    Example:
        >>> controller = AdaptiveRAGController(llm, retriever)
        >>> result = await controller.run("What is photosynthesis?")
        >>> print(f"Used: {result.pipeline_used}, Answer: {result.answer}")
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Optional[Callable] = None,
        *,
        use_llm_classification: bool = False,
        default_k: int = 5,
    ):
        """
        Initialize controller.

        Args:
            llm: Language model
            retriever: Retriever function/callable
            use_llm_classification: Use LLM for complexity classification
            default_k: Default number of documents to retrieve
        """
        self.llm = llm
        self.retriever = retriever
        self.default_k = default_k

        # Initialize classifier
        self.classifier = ComplexityClassifier(
            llm=llm if use_llm_classification else None,
            use_llm_classification=use_llm_classification,
        )

        # Initialize pipeline executors
        self.pipelines: Dict[PipelineType, PipelineExecutor] = {
            PipelineType.NO_RETRIEVAL: NoRetrievalPipeline(llm, retriever),
            PipelineType.SINGLE_RETRIEVAL: SingleRetrievalPipeline(llm, retriever),
            PipelineType.ITERATIVE_RETRIEVAL: IterativeRetrievalPipeline(llm, retriever),
        }

        # Performance tracking
        self._metrics: Dict[PipelineType, Dict[str, float]] = {
            p: {"calls": 0, "avg_latency": 0} for p in PipelineType
        }

    def register_pipeline(
        self,
        pipeline_type: PipelineType,
        executor: PipelineExecutor,
    ) -> None:
        """Register custom pipeline executor."""
        self.pipelines[pipeline_type] = executor

    async def analyze_query(self, query: str) -> ComplexityAnalysis:
        """
        Analyze query complexity.

        Args:
            query: User query

        Returns:
            ComplexityAnalysis
        """
        if self.classifier.use_llm:
            return await self.classifier.classify_with_llm(query)
        return self.classifier.classify(query)

    async def run(
        self,
        query: str,
        *,
        force_pipeline: Optional[PipelineType] = None,
        k: Optional[int] = None,
    ) -> AdaptiveResult:
        """
        Run adaptive RAG pipeline.

        Args:
            query: User query
            force_pipeline: Override automatic pipeline selection
            k: Number of documents to retrieve

        Returns:
            AdaptiveResult with answer and metadata
        """
        start_time = time.time()

        # Analyze query
        analysis = await self.analyze_query(query)

        # Select pipeline
        pipeline_type = force_pipeline or analysis.recommended_pipeline

        # Ensure pipeline exists
        if pipeline_type not in self.pipelines:
            pipeline_type = PipelineType.SINGLE_RETRIEVAL

        # Execute pipeline
        executor = self.pipelines[pipeline_type]
        answer, sources, rounds = await executor.execute(
            query,
            k=k or self.default_k,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Update metrics
        self._update_metrics(pipeline_type, latency_ms)

        return AdaptiveResult(
            answer=answer,
            pipeline_used=pipeline_type,
            complexity_analysis=analysis,
            retrieval_rounds=rounds,
            latency_ms=latency_ms,
            sources=sources,
            metadata={
                "query": query,
                "k": k or self.default_k,
            },
        )

    def _update_metrics(
        self,
        pipeline: PipelineType,
        latency_ms: float,
    ) -> None:
        """Update performance metrics."""
        metrics = self._metrics[pipeline]
        n = metrics["calls"]

        # Running average
        metrics["avg_latency"] = (metrics["avg_latency"] * n + latency_ms) / (n + 1)
        metrics["calls"] += 1

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all pipelines."""
        return {p.value: m for p, m in self._metrics.items()}


# =============================================================================
# Convenience Functions
# =============================================================================


def create_adaptive_controller(
    llm: BaseLLM,
    retriever: Optional[Callable] = None,
    *,
    smart_classification: bool = False,
) -> AdaptiveRAGController:
    """
    Create an adaptive RAG controller.

    Args:
        llm: Language model
        retriever: Retriever function
        smart_classification: Use LLM for classification

    Returns:
        AdaptiveRAGController
    """
    return AdaptiveRAGController(
        llm=llm,
        retriever=retriever,
        use_llm_classification=smart_classification,
    )
