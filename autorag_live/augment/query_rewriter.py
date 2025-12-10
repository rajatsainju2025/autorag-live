"""
Query Rewriting Engine for Agentic RAG.

Provides intelligent query transformation, expansion, and decomposition
to improve retrieval quality and handle complex multi-part questions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from autorag_live.llm.providers import LLMProvider


class RewriteStrategy(str, Enum):
    """Query rewrite strategies."""

    EXPANSION = "expansion"
    DECOMPOSITION = "decomposition"
    REFORMULATION = "reformulation"
    HYPOTHETICAL_DOCUMENT = "hypothetical_document"
    STEP_BACK = "step_back"
    MULTI_QUERY = "multi_query"


@dataclass
class RewriteResult:
    """Result of a query rewrite operation."""

    original_query: str
    rewritten_queries: list[str]
    strategy: RewriteStrategy
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def primary_query(self) -> str:
        """Get the primary rewritten query."""
        return self.rewritten_queries[0] if self.rewritten_queries else self.original_query


@dataclass
class QueryAnalysis:
    """Analysis of a query for rewriting decisions."""

    query: str
    is_complex: bool = False
    is_ambiguous: bool = False
    has_multiple_parts: bool = False
    requires_context: bool = False
    intent: str = ""
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    suggested_strategies: list[RewriteStrategy] = field(default_factory=list)


class BaseQueryRewriter(ABC):
    """Base class for query rewriters."""

    @abstractmethod
    def rewrite(self, query: str, context: Optional[str] = None) -> RewriteResult:
        """Rewrite a query."""
        pass


class QueryExpander(BaseQueryRewriter):
    """Expands queries with synonyms and related terms."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize query expander."""
        self.llm_provider = llm_provider
        self._synonym_cache: dict[str, list[str]] = {}

    def rewrite(self, query: str, context: Optional[str] = None) -> RewriteResult:
        """Expand query with synonyms and related terms."""
        if self.llm_provider:
            return self._llm_expand(query, context)
        return self._rule_based_expand(query)

    def _llm_expand(self, query: str, context: Optional[str] = None) -> RewriteResult:
        """Use LLM for query expansion."""
        prompt = f"""Expand the following query by adding relevant synonyms and related terms.
Keep the original meaning intact. Return 3-5 expanded versions.

Query: {query}
{"Context: " + context if context else ""}

Return each expanded query on a new line."""

        response = self.llm_provider.generate(prompt)  # type: ignore
        response_text = response.content if hasattr(response, "content") else str(response)
        expanded_queries = [q.strip() for q in response_text.strip().split("\n") if q.strip()]

        if not expanded_queries:
            expanded_queries = [query]

        return RewriteResult(
            original_query=query,
            rewritten_queries=expanded_queries,
            strategy=RewriteStrategy.EXPANSION,
            confidence=0.85,
        )

    def _rule_based_expand(self, query: str) -> RewriteResult:
        """Rule-based query expansion without LLM."""
        expansions = {
            "how to": ["guide for", "tutorial on", "steps to"],
            "what is": ["define", "explain", "meaning of"],
            "why": ["reason for", "cause of", "explanation of"],
            "best": ["top", "recommended", "optimal"],
        }

        expanded = [query]
        query_lower = query.lower()

        for pattern, alternatives in expansions.items():
            if pattern in query_lower:
                for alt in alternatives[:2]:
                    expanded.append(query_lower.replace(pattern, alt))

        return RewriteResult(
            original_query=query,
            rewritten_queries=expanded,
            strategy=RewriteStrategy.EXPANSION,
            confidence=0.7,
        )


class QueryDecomposer(BaseQueryRewriter):
    """Decomposes complex queries into simpler sub-queries."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize query decomposer."""
        self.llm_provider = llm_provider

    def rewrite(self, query: str, context: Optional[str] = None) -> RewriteResult:
        """Decompose a complex query into sub-queries."""
        prompt = f"""Break down this complex question into simpler sub-questions
that can be answered independently. Each sub-question should address one aspect.

Complex Question: {query}
{"Context: " + context if context else ""}

Return each sub-question on a new line, numbered 1, 2, 3, etc."""

        response = self.llm_provider.generate(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        sub_queries = []

        for line in response_text.strip().split("\n"):
            line = line.strip()
            if line:
                cleaned = line.lstrip("0123456789.)-: ")
                if cleaned:
                    sub_queries.append(cleaned)

        if not sub_queries:
            sub_queries = [query]

        return RewriteResult(
            original_query=query,
            rewritten_queries=sub_queries,
            strategy=RewriteStrategy.DECOMPOSITION,
            confidence=0.9,
            metadata={"num_sub_queries": len(sub_queries)},
        )


class HypotheticalDocumentEmbedder(BaseQueryRewriter):
    """Generates hypothetical documents for better retrieval (HyDE)."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize HyDE rewriter."""
        self.llm_provider = llm_provider

    def rewrite(self, query: str, context: Optional[str] = None) -> RewriteResult:
        """Generate a hypothetical document that would answer the query."""
        prompt = f"""Write a short paragraph that directly answers this question.
The answer should be factual and informative, as if from a reliable source.

Question: {query}
{"Context: " + context if context else ""}

Write a 2-3 sentence answer:"""

        response = self.llm_provider.generate(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        hypothetical_doc = response_text.strip()

        return RewriteResult(
            original_query=query,
            rewritten_queries=[hypothetical_doc, query],
            strategy=RewriteStrategy.HYPOTHETICAL_DOCUMENT,
            confidence=0.8,
            metadata={"type": "hyde"},
        )


class StepBackPrompting(BaseQueryRewriter):
    """Generates broader, more abstract queries for better context."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize step-back rewriter."""
        self.llm_provider = llm_provider

    def rewrite(self, query: str, context: Optional[str] = None) -> RewriteResult:
        """Generate a step-back query that captures broader context."""
        prompt = f"""Given a specific question, generate a more general step-back question
that would help provide context for answering the original question.

Specific Question: {query}

Generate a broader question that covers the general principles or background
needed to answer the specific question:"""

        response = self.llm_provider.generate(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        step_back_query = response_text.strip()

        return RewriteResult(
            original_query=query,
            rewritten_queries=[step_back_query, query],
            strategy=RewriteStrategy.STEP_BACK,
            confidence=0.85,
            metadata={"type": "step_back"},
        )


class MultiQueryGenerator(BaseQueryRewriter):
    """Generates multiple query variations for diverse retrieval."""

    def __init__(self, llm_provider: LLMProvider, num_queries: int = 3):
        """Initialize multi-query generator."""
        self.llm_provider = llm_provider
        self.num_queries = num_queries

    def rewrite(self, query: str, context: Optional[str] = None) -> RewriteResult:
        """Generate multiple query variations."""
        prompt = f"""Generate {self.num_queries} different versions of this query.
Each version should:
- Have the same meaning but different wording
- Use different keywords while preserving intent
- Help retrieve different relevant documents

Original Query: {query}
{"Context: " + context if context else ""}

Return each query variation on a new line:"""

        response = self.llm_provider.generate(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)
        variations = [q.strip() for q in response_text.strip().split("\n") if q.strip()]

        if query not in variations:
            variations.insert(0, query)

        return RewriteResult(
            original_query=query,
            rewritten_queries=variations[: self.num_queries + 1],
            strategy=RewriteStrategy.MULTI_QUERY,
            confidence=0.85,
        )


class QueryAnalyzer:
    """Analyzes queries to determine optimal rewriting strategies."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize query analyzer."""
        self.llm_provider = llm_provider
        self._complexity_indicators = [
            "and",
            "or",
            "but",
            "however",
            "also",
            "compare",
            "difference",
            "relationship",
        ]
        self._ambiguity_indicators = ["it", "this", "that", "they", "them"]

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a query for rewriting decisions."""
        query_lower = query.lower()
        words = query_lower.split()

        is_complex = len(words) > 15 or any(
            ind in query_lower for ind in self._complexity_indicators
        )

        is_ambiguous = any(word in self._ambiguity_indicators for word in words)

        question_count = query.count("?")
        has_multiple_parts = question_count > 1 or (
            is_complex and any(c in query_lower for c in ["first", "second", "then"])
        )

        suggested_strategies = []
        if has_multiple_parts:
            suggested_strategies.append(RewriteStrategy.DECOMPOSITION)
        if is_ambiguous:
            suggested_strategies.append(RewriteStrategy.REFORMULATION)
        if is_complex:
            suggested_strategies.append(RewriteStrategy.STEP_BACK)

        suggested_strategies.append(RewriteStrategy.MULTI_QUERY)

        return QueryAnalysis(
            query=query,
            is_complex=is_complex,
            is_ambiguous=is_ambiguous,
            has_multiple_parts=has_multiple_parts,
            suggested_strategies=suggested_strategies,
        )


class QueryRewriteEngine:
    """Main engine for intelligent query rewriting."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        enable_analysis: bool = True,
        default_strategies: Optional[list[RewriteStrategy]] = None,
    ):
        """Initialize query rewrite engine."""
        self.llm_provider = llm_provider
        self.enable_analysis = enable_analysis
        self.default_strategies = default_strategies or [RewriteStrategy.MULTI_QUERY]

        self.analyzer = QueryAnalyzer(llm_provider)

        self._rewriters: dict[RewriteStrategy, BaseQueryRewriter] = {}
        self._init_rewriters()

        self._rewrite_history: list[RewriteResult] = []

    def _init_rewriters(self) -> None:
        """Initialize available rewriters."""
        self._rewriters[RewriteStrategy.EXPANSION] = QueryExpander(self.llm_provider)

        if self.llm_provider:
            self._rewriters[RewriteStrategy.DECOMPOSITION] = QueryDecomposer(self.llm_provider)
            self._rewriters[RewriteStrategy.HYPOTHETICAL_DOCUMENT] = HypotheticalDocumentEmbedder(
                self.llm_provider
            )
            self._rewriters[RewriteStrategy.STEP_BACK] = StepBackPrompting(self.llm_provider)
            self._rewriters[RewriteStrategy.MULTI_QUERY] = MultiQueryGenerator(self.llm_provider)

    def rewrite(
        self,
        query: str,
        strategies: Optional[list[RewriteStrategy]] = None,
        context: Optional[str] = None,
        auto_select: bool = True,
    ) -> list[RewriteResult]:
        """Rewrite a query using specified or automatically selected strategies."""
        if auto_select and self.enable_analysis:
            analysis = self.analyzer.analyze(query)
            if analysis.suggested_strategies:
                strategies = analysis.suggested_strategies

        if not strategies:
            strategies = self.default_strategies

        results = []
        for strategy in strategies:
            rewriter = self._rewriters.get(strategy)
            if rewriter:
                result = rewriter.rewrite(query, context)
                results.append(result)
                self._rewrite_history.append(result)

        return results

    def rewrite_single(
        self,
        query: str,
        strategy: RewriteStrategy = RewriteStrategy.MULTI_QUERY,
        context: Optional[str] = None,
    ) -> RewriteResult:
        """Rewrite using a single strategy."""
        rewriter = self._rewriters.get(strategy)
        if not rewriter:
            return RewriteResult(
                original_query=query,
                rewritten_queries=[query],
                strategy=strategy,
                confidence=0.0,
                metadata={"error": f"Strategy {strategy} not available"},
            )

        result = rewriter.rewrite(query, context)
        self._rewrite_history.append(result)
        return result

    def get_all_queries(self, results: list[RewriteResult]) -> list[str]:
        """Extract all unique queries from rewrite results."""
        all_queries = set()
        for result in results:
            all_queries.update(result.rewritten_queries)
        return list(all_queries)

    def combine_results(self, results: list[RewriteResult], max_queries: int = 10) -> RewriteResult:
        """Combine multiple rewrite results into one."""
        if not results:
            return RewriteResult(
                original_query="",
                rewritten_queries=[],
                strategy=RewriteStrategy.MULTI_QUERY,
            )

        all_queries = []
        seen = set()
        for result in results:
            for q in result.rewritten_queries:
                if q not in seen:
                    all_queries.append(q)
                    seen.add(q)

        avg_confidence = sum(r.confidence for r in results) / len(results)

        return RewriteResult(
            original_query=results[0].original_query,
            rewritten_queries=all_queries[:max_queries],
            strategy=RewriteStrategy.MULTI_QUERY,
            confidence=avg_confidence,
            metadata={
                "combined_from": [r.strategy.value for r in results],
                "total_unique_queries": len(all_queries),
            },
        )

    @property
    def history(self) -> list[RewriteResult]:
        """Get rewrite history."""
        return self._rewrite_history.copy()

    def clear_history(self) -> None:
        """Clear rewrite history."""
        self._rewrite_history.clear()


def create_rewrite_engine(
    llm_provider: Optional[LLMProvider] = None,
    strategies: Optional[list[str]] = None,
) -> QueryRewriteEngine:
    """Factory function to create a query rewrite engine."""
    strategy_list = None
    if strategies:
        strategy_list = [RewriteStrategy(s) for s in strategies]

    return QueryRewriteEngine(
        llm_provider=llm_provider,
        default_strategies=strategy_list,
    )


__all__ = [
    "RewriteStrategy",
    "RewriteResult",
    "QueryAnalysis",
    "BaseQueryRewriter",
    "QueryExpander",
    "QueryDecomposer",
    "HypotheticalDocumentEmbedder",
    "StepBackPrompting",
    "MultiQueryGenerator",
    "QueryAnalyzer",
    "QueryRewriteEngine",
    "create_rewrite_engine",
]
