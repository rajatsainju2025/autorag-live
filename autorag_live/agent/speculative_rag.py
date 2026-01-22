"""
Speculative RAG Module for Parallel Speculation.

Implements speculative execution patterns for RAG to reduce latency
by generating multiple candidate answers in parallel and selecting
the best one based on verification.

Key Features:
1. Parallel candidate generation
2. Speculative retrieval with prediction
3. Early termination on high-confidence answers
4. Draft-then-verify pattern
5. Speculative decoding integration

References:
- Speculative RAG: Enhancing RAG through Drafting (Wang et al., 2024)
- Speculative Decoding (Leviathan et al., 2022)
- Medusa: Simple LLM Inference Acceleration (Cai et al., 2024)

Example:
    >>> spec_rag = SpeculativeRAG(llm, retriever)
    >>> result = await spec_rag.generate("What is quantum computing?")
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class LLMProtocol(Protocol):
    """Protocol for LLM interface."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response from prompt."""
        ...


class RetrieverProtocol(Protocol):
    """Protocol for retriever interface."""

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents for query."""
        ...


class VerifierProtocol(Protocol):
    """Protocol for answer verification."""

    async def verify(self, answer: str, context: str, query: str) -> float:
        """Verify answer against context. Returns confidence 0-1."""
        ...


# =============================================================================
# Data Structures
# =============================================================================


class SpeculationStrategy(str, Enum):
    """Strategy for speculative generation."""

    PARALLEL_DRAFTS = "parallel_drafts"  # Generate multiple drafts in parallel
    TREE_SPECULATION = "tree_speculation"  # Tree-structured speculation
    RETRIEVAL_AHEAD = "retrieval_ahead"  # Speculative retrieval
    DRAFT_VERIFY = "draft_verify"  # Draft then verify pattern
    ENSEMBLE = "ensemble"  # Ensemble of strategies


@dataclass
class SpeculativeCandidate:
    """
    A speculative candidate answer.

    Attributes:
        content: The candidate answer text
        confidence: Confidence score
        source: Source strategy/model
        latency: Generation latency in seconds
        verified: Whether verified against evidence
        verification_score: Verification confidence
        metadata: Additional metadata
    """

    content: str
    confidence: float = 0.5
    source: str = "default"
    latency: float = 0.0
    verified: bool = False
    verification_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_score(self) -> float:
        """Combined score considering confidence and verification."""
        if self.verified:
            return 0.6 * self.confidence + 0.4 * self.verification_score
        return self.confidence * 0.8  # Penalty for unverified


@dataclass
class SpeculativeResult:
    """
    Result of speculative RAG execution.

    Attributes:
        answer: Selected best answer
        candidates: All generated candidates
        selected_index: Index of selected candidate
        total_latency: Total execution time
        parallel_speedup: Speedup from parallelism
        strategy: Strategy used
    """

    answer: str
    candidates: List[SpeculativeCandidate]
    selected_index: int = 0
    total_latency: float = 0.0
    parallel_speedup: float = 1.0
    strategy: SpeculationStrategy = SpeculationStrategy.PARALLEL_DRAFTS
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_candidate(self) -> SpeculativeCandidate:
        """Get the best candidate."""
        return self.candidates[self.selected_index]


@dataclass
class RetrievalSpeculation:
    """
    Speculative retrieval result.

    Attributes:
        query: Original query
        predicted_queries: Predicted follow-up queries
        retrieved_docs: Documents per query
        hit_rate: Prediction accuracy
    """

    query: str
    predicted_queries: List[str] = field(default_factory=list)
    retrieved_docs: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    hit_rate: float = 0.0


# =============================================================================
# Speculative Generators
# =============================================================================


class DraftGenerator(ABC):
    """Base class for draft generators."""

    @abstractmethod
    async def generate_draft(
        self,
        query: str,
        context: str,
        **kwargs: Any,
    ) -> SpeculativeCandidate:
        """Generate a draft answer."""
        pass


class SimpleDraftGenerator(DraftGenerator):
    """Simple draft generator using LLM."""

    def __init__(
        self,
        llm: LLMProtocol,
        name: str = "simple",
        temperature: float = 0.7,
    ):
        """Initialize generator."""
        self.llm = llm
        self.name = name
        self.temperature = temperature

    async def generate_draft(
        self,
        query: str,
        context: str,
        **kwargs: Any,
    ) -> SpeculativeCandidate:
        """Generate draft answer."""
        start = time.time()

        prompt = f"""Answer the question based on the context provided.

Context:
{context[:3000]}

Question: {query}

Answer:"""

        try:
            answer = await self.llm.generate(
                prompt,
                temperature=self.temperature,
            )

            return SpeculativeCandidate(
                content=answer,
                confidence=0.7,
                source=self.name,
                latency=time.time() - start,
            )
        except Exception as e:
            logger.warning(f"Draft generation failed: {e}")
            return SpeculativeCandidate(
                content="",
                confidence=0.0,
                source=self.name,
                latency=time.time() - start,
                metadata={"error": str(e)},
            )


class CreativeDraftGenerator(DraftGenerator):
    """Creative draft generator with higher temperature."""

    def __init__(self, llm: LLMProtocol):
        """Initialize generator."""
        self.llm = llm

    async def generate_draft(
        self,
        query: str,
        context: str,
        **kwargs: Any,
    ) -> SpeculativeCandidate:
        """Generate creative draft."""
        start = time.time()

        prompt = f"""Provide a comprehensive and insightful answer to this question.
Think creatively and consider multiple perspectives.

Context:
{context[:3000]}

Question: {query}

Detailed Answer:"""

        try:
            answer = await self.llm.generate(prompt, temperature=0.9)

            return SpeculativeCandidate(
                content=answer,
                confidence=0.6,
                source="creative",
                latency=time.time() - start,
            )
        except Exception as e:
            logger.warning(f"Creative draft failed: {e}")
            return SpeculativeCandidate(
                content="",
                confidence=0.0,
                source="creative",
                latency=time.time() - start,
            )


class ConciseDraftGenerator(DraftGenerator):
    """Concise draft generator for brief answers."""

    def __init__(self, llm: LLMProtocol):
        """Initialize generator."""
        self.llm = llm

    async def generate_draft(
        self,
        query: str,
        context: str,
        **kwargs: Any,
    ) -> SpeculativeCandidate:
        """Generate concise draft."""
        start = time.time()

        prompt = f"""Answer briefly and precisely based on the context.

Context:
{context[:2000]}

Question: {query}

Brief Answer (1-2 sentences):"""

        try:
            answer = await self.llm.generate(prompt, temperature=0.3)

            return SpeculativeCandidate(
                content=answer,
                confidence=0.8,
                source="concise",
                latency=time.time() - start,
            )
        except Exception as e:
            logger.warning(f"Concise draft failed: {e}")
            return SpeculativeCandidate(
                content="",
                confidence=0.0,
                source="concise",
                latency=time.time() - start,
            )


# =============================================================================
# Answer Verifier
# =============================================================================


class AnswerVerifier:
    """Verifies answers against retrieved evidence."""

    def __init__(self, llm: Optional[LLMProtocol] = None):
        """Initialize verifier."""
        self.llm = llm

    async def verify(
        self,
        answer: str,
        context: str,
        query: str,
    ) -> float:
        """
        Verify answer against context.

        Args:
            answer: Candidate answer
            context: Retrieved context
            query: Original query

        Returns:
            Verification score 0-1
        """
        if not self.llm:
            return self._heuristic_verify(answer, context)

        prompt = f"""Evaluate if the answer is well-supported by the context.

Context:
{context[:2000]}

Question: {query}

Answer: {answer}

Rate the answer on a scale of 0-10 based on:
1. Factual accuracy (is it correct based on context?)
2. Relevance (does it answer the question?)
3. Completeness (is it comprehensive?)

Provide only a number 0-10:"""

        try:
            response = await self.llm.generate(prompt)
            # Extract number from response
            import re

            numbers = re.findall(r"\d+(?:\.\d+)?", response)
            if numbers:
                score = float(numbers[0])
                return min(score / 10.0, 1.0)
            return 0.5
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            return self._heuristic_verify(answer, context)

    def _heuristic_verify(self, answer: str, context: str) -> float:
        """Heuristic verification based on word overlap."""
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        if not answer_words:
            return 0.0

        overlap = len(answer_words & context_words)
        coverage = overlap / len(answer_words)

        return min(coverage, 1.0)


# =============================================================================
# Query Predictor for Speculative Retrieval
# =============================================================================


class QueryPredictor:
    """Predicts follow-up queries for speculative retrieval."""

    def __init__(self, llm: Optional[LLMProtocol] = None):
        """Initialize predictor."""
        self.llm = llm
        self._cache: Dict[str, List[str]] = {}

    async def predict(
        self,
        query: str,
        num_predictions: int = 3,
    ) -> List[str]:
        """
        Predict follow-up queries.

        Args:
            query: Original query
            num_predictions: Number of predictions

        Returns:
            List of predicted queries
        """
        cache_key = query.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key][:num_predictions]

        if not self.llm:
            return self._heuristic_predict(query, num_predictions)

        prompt = f"""Given this question, predict related follow-up questions that a user might ask.

Question: {query}

Generate {num_predictions} follow-up questions (one per line):"""

        try:
            response = await self.llm.generate(prompt)
            predictions = []

            for line in response.strip().split("\n"):
                line = line.strip()
                # Remove numbering
                import re

                line = re.sub(r"^\d+[\.\)]\s*", "", line)
                if line and "?" in line:
                    predictions.append(line)

            self._cache[cache_key] = predictions
            return predictions[:num_predictions]
        except Exception as e:
            logger.warning(f"Query prediction failed: {e}")
            return self._heuristic_predict(query, num_predictions)

    def _heuristic_predict(self, query: str, num: int) -> List[str]:
        """Simple heuristic predictions."""
        predictions = []

        # Add "why" variant
        if not query.lower().startswith("why"):
            predictions.append(f"Why {query.lower().rstrip('?')}?")

        # Add "how" variant
        if not query.lower().startswith("how"):
            predictions.append(f"How {query.lower().rstrip('?')}?")

        # Add "what are the examples" variant
        predictions.append(f"What are examples of {query.lower().rstrip('?')}?")

        return predictions[:num]


# =============================================================================
# Main Speculative RAG
# =============================================================================


class SpeculativeRAG:
    """
    Speculative RAG for reduced latency through parallel execution.

    Generates multiple candidate answers in parallel and selects
    the best one based on verification against evidence.

    Example:
        >>> spec_rag = SpeculativeRAG(llm, retriever)
        >>> result = await spec_rag.generate(
        ...     "What are transformers in ML?",
        ...     strategy=SpeculationStrategy.PARALLEL_DRAFTS
        ... )
        >>> print(f"Answer: {result.answer}")
        >>> print(f"Speedup: {result.parallel_speedup:.2f}x")
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: Optional[RetrieverProtocol] = None,
        verifier: Optional[AnswerVerifier] = None,
        num_drafts: int = 3,
        verification_threshold: float = 0.7,
    ):
        """
        Initialize Speculative RAG.

        Args:
            llm: Language model for generation
            retriever: Retriever for evidence
            verifier: Answer verifier (created if None)
            num_drafts: Number of parallel drafts
            verification_threshold: Early termination threshold
        """
        self.llm = llm
        self.retriever = retriever
        self.verifier = verifier or AnswerVerifier(llm)
        self.num_drafts = num_drafts
        self.verification_threshold = verification_threshold

        # Initialize draft generators
        self.generators: List[DraftGenerator] = [
            SimpleDraftGenerator(llm, "standard", temperature=0.5),
            CreativeDraftGenerator(llm),
            ConciseDraftGenerator(llm),
        ]

        # Query predictor for speculative retrieval
        self.query_predictor = QueryPredictor(llm)

    async def generate(
        self,
        query: str,
        *,
        strategy: SpeculationStrategy = SpeculationStrategy.PARALLEL_DRAFTS,
        top_k: int = 5,
        verify: bool = True,
    ) -> SpeculativeResult:
        """
        Generate answer with speculation.

        Args:
            query: User query
            strategy: Speculation strategy
            top_k: Documents to retrieve
            verify: Whether to verify candidates

        Returns:
            SpeculativeResult with selected answer
        """
        start_time = time.time()

        # Retrieve context
        context = ""
        if self.retriever:
            docs = await self.retriever.retrieve(query, top_k=top_k)
            context = self._format_context(docs)

        # Generate candidates based on strategy
        if strategy == SpeculationStrategy.PARALLEL_DRAFTS:
            candidates = await self._parallel_drafts(query, context)
        elif strategy == SpeculationStrategy.DRAFT_VERIFY:
            candidates = await self._draft_verify(query, context)
        elif strategy == SpeculationStrategy.RETRIEVAL_AHEAD:
            candidates = await self._retrieval_ahead(query, top_k)
        else:
            candidates = await self._parallel_drafts(query, context)

        # Verify candidates if requested
        if verify and candidates:
            await self._verify_candidates(candidates, context, query)

        # Select best candidate
        if candidates:
            candidates.sort(key=lambda c: c.final_score, reverse=True)
            selected_idx = 0
            answer = candidates[0].content
        else:
            selected_idx = 0
            answer = "Unable to generate answer."
            candidates = [
                SpeculativeCandidate(
                    content=answer,
                    confidence=0.0,
                    source="fallback",
                )
            ]

        total_latency = time.time() - start_time

        # Calculate speedup (vs sequential)
        sequential_time = sum(c.latency for c in candidates)
        parallel_speedup = sequential_time / total_latency if total_latency > 0 else 1.0

        return SpeculativeResult(
            answer=answer,
            candidates=candidates,
            selected_index=selected_idx,
            total_latency=total_latency,
            parallel_speedup=parallel_speedup,
            strategy=strategy,
        )

    async def _parallel_drafts(
        self,
        query: str,
        context: str,
    ) -> List[SpeculativeCandidate]:
        """Generate drafts in parallel."""
        tasks = [gen.generate_draft(query, context) for gen in self.generators[: self.num_drafts]]

        candidates = await asyncio.gather(*tasks, return_exceptions=True)

        return [c for c in candidates if isinstance(c, SpeculativeCandidate) and c.content]

    async def _draft_verify(
        self,
        query: str,
        context: str,
    ) -> List[SpeculativeCandidate]:
        """Draft-then-verify pattern with early termination."""
        candidates = []

        for gen in self.generators:
            candidate = await gen.generate_draft(query, context)

            if not candidate.content:
                continue

            # Verify immediately
            score = await self.verifier.verify(candidate.content, context, query)
            candidate.verified = True
            candidate.verification_score = score
            candidates.append(candidate)

            # Early termination if high confidence
            if score >= self.verification_threshold:
                logger.info(f"Early termination with score {score:.2f}")
                break

        return candidates

    async def _retrieval_ahead(
        self,
        query: str,
        top_k: int,
    ) -> List[SpeculativeCandidate]:
        """Speculative retrieval with predicted queries."""
        if not self.retriever:
            return []

        # Predict follow-up queries
        predicted = await self.query_predictor.predict(query, num_predictions=2)
        all_queries = [query] + predicted

        # Retrieve for all queries in parallel
        tasks = [self.retriever.retrieve(q, top_k=top_k) for q in all_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine contexts
        all_docs = []
        for result in results:
            if isinstance(result, list):
                all_docs.extend(result)

        # Deduplicate by content
        seen = set()
        unique_docs = []
        for doc in all_docs:
            content = doc.get("content", "")[:100]
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)

        context = self._format_context(unique_docs[: top_k * 2])

        # Generate answer with enriched context
        return await self._parallel_drafts(query, context)

    async def _verify_candidates(
        self,
        candidates: List[SpeculativeCandidate],
        context: str,
        query: str,
    ) -> None:
        """Verify candidates in parallel."""
        tasks = []
        for candidate in candidates:
            if not candidate.verified:
                tasks.append(self._verify_one(candidate, context, query))

        if tasks:
            await asyncio.gather(*tasks)

    async def _verify_one(
        self,
        candidate: SpeculativeCandidate,
        context: str,
        query: str,
    ) -> None:
        """Verify a single candidate."""
        score = await self.verifier.verify(candidate.content, context, query)
        candidate.verified = True
        candidate.verification_score = score

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Format documents as context string."""
        if not docs:
            return ""

        parts = []
        for i, doc in enumerate(docs[:5], 1):
            content = doc.get("content", doc.get("text", str(doc)))
            parts.append(f"[{i}] {content[:800]}")

        return "\n\n".join(parts)


# =============================================================================
# Speculative Retrieval Manager
# =============================================================================


class SpeculativeRetrievalManager:
    """
    Manages speculative retrieval for predictive caching.

    Pre-fetches documents for predicted queries to reduce latency
    on follow-up questions.
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        predictor: Optional[QueryPredictor] = None,
        cache_size: int = 100,
        enable_prefetch_warming: bool = True,
    ):
        """
        Initialize manager.

        Args:
            retriever: Document retriever
            predictor: Query predictor
            cache_size: Maximum cached results
            enable_prefetch_warming: Enable intelligent cache warming
        """
        self.retriever = retriever
        self.predictor = predictor or QueryPredictor()
        self.cache_size = cache_size
        self.enable_prefetch_warming = enable_prefetch_warming
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        self._pending: Dict[str, asyncio.Task] = {}
        self._query_patterns: Dict[str, int] = {}  # Track query frequency

    async def retrieve_with_speculation(
        self,
        query: str,
        top_k: int = 5,
    ) -> Tuple[List[Dict[str, Any]], RetrievalSpeculation]:
        """
        Retrieve with speculative pre-fetching.

        Args:
            query: Main query
            top_k: Documents per query

        Returns:
            Tuple of (main results, speculation details)
        """
        # Check cache first
        cache_key = self._cache_key(query)
        if cache_key in self._cache:
            docs = self._cache[cache_key]
        else:
            docs = await self.retriever.retrieve(query, top_k=top_k)
            self._cache_result(cache_key, docs)

        # Track query for pattern learning
        self._query_patterns[cache_key] = self._query_patterns.get(cache_key, 0) + 1

        # Predict and pre-fetch in background
        predicted = await self.predictor.predict(query, num_predictions=3)

        # Add pattern-based predictions if enabled
        if self.enable_prefetch_warming:
            pattern_queries = self._get_pattern_predictions(cache_key, top_n=2)
            predicted.extend(pattern_queries)

        self._start_prefetch(predicted, top_k)

        speculation = RetrievalSpeculation(
            query=query,
            predicted_queries=predicted,
        )

        return docs, speculation

    def _get_pattern_predictions(self, query_key: str, top_n: int = 2) -> List[str]:
        """Get pattern-based query predictions from historical data."""
        # Get most frequent queries that aren't already cached
        sorted_patterns = sorted(self._query_patterns.items(), key=lambda x: x[1], reverse=True)

        predictions = []
        for pattern_key, _ in sorted_patterns:
            if pattern_key != query_key and pattern_key not in self._cache:
                predictions.append(pattern_key)
                if len(predictions) >= top_n:
                    break

        return predictions

    def warm_cache(self, common_queries: List[str], top_k: int = 5) -> None:
        """
        Warm cache with common queries.

        Args:
            common_queries: List of frequent queries to pre-cache
            top_k: Documents per query
        """
        for query in common_queries:
            key = self._cache_key(query)
            if key not in self._cache:
                self._start_prefetch([query], top_k)

    def _cache_key(self, query: str) -> str:
        """Generate cache key."""
        return query.lower().strip()

    def _cache_result(self, key: str, docs: List[Dict[str, Any]]) -> None:
        """Cache retrieval result."""
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = docs

    def _start_prefetch(self, queries: List[str], top_k: int) -> None:
        """Start background pre-fetch for queries."""
        for query in queries:
            key = self._cache_key(query)
            if key not in self._cache and key not in self._pending:
                task = asyncio.create_task(self._prefetch_one(key, query, top_k))
                self._pending[key] = task

    async def _prefetch_one(self, key: str, query: str, top_k: int) -> None:
        """Pre-fetch documents for one query."""
        try:
            docs = await self.retriever.retrieve(query, top_k=top_k)
            self._cache_result(key, docs)
        except Exception as e:
            logger.warning(f"Pre-fetch failed for '{query}': {e}")
        finally:
            self._pending.pop(key, None)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_speculative_rag(
    llm: LLMProtocol,
    retriever: Optional[RetrieverProtocol] = None,
    num_drafts: int = 3,
) -> SpeculativeRAG:
    """
    Create a speculative RAG instance.

    Args:
        llm: Language model
        retriever: Document retriever
        num_drafts: Number of parallel drafts

    Returns:
        SpeculativeRAG instance
    """
    return SpeculativeRAG(llm=llm, retriever=retriever, num_drafts=num_drafts)


async def speculative_generate(
    query: str,
    llm: LLMProtocol,
    context: str = "",
) -> str:
    """
    Quick speculative generation.

    Args:
        query: User query
        llm: Language model
        context: Optional context

    Returns:
        Best answer from speculation
    """
    rag = SpeculativeRAG(llm=llm)
    result = await rag.generate(query)
    return result.answer
