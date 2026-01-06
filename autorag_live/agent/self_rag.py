"""
Self-RAG Module with Reflection Tokens.

Implements Self-RAG pattern with inline reflection tokens for
adaptive retrieval and self-assessment during generation.

Key Features:
1. Reflection tokens ([Retrieve], [ISREL], [ISSUP], [ISUSE])
2. Adaptive retrieval decisions
3. Relevance and support assessment
4. Critique-based generation refinement
5. Citation verification

References:
- Self-RAG: Learning to Retrieve, Generate, and Critique (Asai et al., 2023)
- CRAG: Corrective Retrieval Augmented Generation

Example:
    >>> self_rag = SelfRAG(llm, retriever)
    >>> result = await self_rag.generate("What is quantum computing?")
    >>> print(result.answer)
    >>> print(result.reflection_tokens)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

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


# =============================================================================
# Reflection Tokens
# =============================================================================


class ReflectionToken(str, Enum):
    """Self-RAG reflection tokens."""

    # Retrieval decision
    RETRIEVE_YES = "[Retrieve=Yes]"
    RETRIEVE_NO = "[Retrieve=No]"

    # Relevance assessment
    ISREL_RELEVANT = "[ISREL=Relevant]"
    ISREL_IRRELEVANT = "[ISREL=Irrelevant]"

    # Support assessment
    ISSUP_FULLY = "[ISSUP=Fully Supported]"
    ISSUP_PARTIALLY = "[ISSUP=Partially Supported]"
    ISSUP_NONE = "[ISSUP=No Support]"

    # Utility assessment
    ISUSE_USEFUL = "[ISUSE=Useful]"
    ISUSE_PARTIAL = "[ISUSE=Partially Useful]"
    ISUSE_NOT = "[ISUSE=Not Useful]"

    # Critique
    CRITIQUE_GOOD = "[Critique=Good]"
    CRITIQUE_NEEDS_REVISION = "[Critique=Needs Revision]"


@dataclass
class ReflectionResult:
    """
    Result of reflection token assessment.

    Attributes:
        token: The reflection token
        confidence: Confidence in the assessment
        reasoning: Reasoning for the assessment
    """

    token: ReflectionToken
    confidence: float = 0.5
    reasoning: str = ""


@dataclass
class RetrievalDecision:
    """
    Decision on whether to retrieve.

    Attributes:
        should_retrieve: Whether retrieval is needed
        confidence: Confidence in decision
        reasoning: Reasoning for decision
    """

    should_retrieve: bool
    confidence: float = 0.5
    reasoning: str = ""
    token: ReflectionToken = ReflectionToken.RETRIEVE_YES


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SelfRAGResponse:
    """
    Response from Self-RAG generation.

    Attributes:
        answer: Final generated answer
        reflection_tokens: All reflection tokens produced
        retrieved_docs: Documents retrieved (if any)
        iterations: Number of generation iterations
        retrieval_decision: Initial retrieval decision
        relevance_scores: Per-document relevance scores
        support_score: Overall support assessment
        utility_score: Final utility assessment
    """

    answer: str
    reflection_tokens: List[ReflectionResult] = field(default_factory=list)
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 1
    retrieval_decision: Optional[RetrievalDecision] = None
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    support_score: float = 0.5
    utility_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_well_supported(self) -> bool:
        """Check if answer is well supported."""
        return self.support_score >= 0.7

    def get_token_summary(self) -> str:
        """Get summary of reflection tokens."""
        return " ".join(t.token.value for t in self.reflection_tokens)


@dataclass
class CritiqueResult:
    """
    Result of answer critique.

    Attributes:
        is_satisfactory: Whether answer passes critique
        issues: List of identified issues
        suggestions: Improvement suggestions
        score: Overall quality score
    """

    is_satisfactory: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    score: float = 0.5


# =============================================================================
# Reflection Token Predictor
# =============================================================================


class ReflectionPredictor:
    """Predicts reflection tokens using LLM."""

    def __init__(self, llm: LLMProtocol):
        """Initialize predictor."""
        self.llm = llm

    async def predict_retrieval_need(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> RetrievalDecision:
        """
        Predict whether retrieval is needed.

        Args:
            query: User query
            context: Optional existing context

        Returns:
            RetrievalDecision with should_retrieve and reasoning
        """
        prompt = f"""Determine if external information retrieval is needed to answer this question.
Consider:
1. Is this a factual question requiring up-to-date information?
2. Can this be answered from general knowledge?
3. Does this require specific domain knowledge?

Question: {query}
{"Existing Context: " + context[:500] if context else "No existing context."}

Answer with:
- Decision: YES or NO
- Reasoning: Brief explanation

Decision:"""

        try:
            response = await self.llm.generate(prompt)
            should_retrieve = "yes" in response.lower().split("\n")[0].lower()

            return RetrievalDecision(
                should_retrieve=should_retrieve,
                confidence=0.8 if should_retrieve else 0.7,
                reasoning=response,
                token=(
                    ReflectionToken.RETRIEVE_YES if should_retrieve else ReflectionToken.RETRIEVE_NO
                ),
            )
        except Exception as e:
            logger.warning(f"Retrieval decision failed: {e}")
            return RetrievalDecision(
                should_retrieve=True,
                confidence=0.5,
                reasoning="Default to retrieval",
            )

    async def assess_relevance(
        self,
        query: str,
        document: Dict[str, Any],
    ) -> ReflectionResult:
        """
        Assess document relevance to query.

        Args:
            query: User query
            document: Retrieved document

        Returns:
            ReflectionResult with relevance assessment
        """
        content = document.get("content", document.get("text", ""))

        prompt = f"""Assess if this document is relevant to answering the question.

Question: {query}

Document:
{content[:1500]}

Is this document relevant? Answer RELEVANT or IRRELEVANT with brief reasoning."""

        try:
            response = await self.llm.generate(prompt)
            is_relevant = "relevant" in response.lower() and "irrelevant" not in response.lower()

            return ReflectionResult(
                token=(
                    ReflectionToken.ISREL_RELEVANT
                    if is_relevant
                    else ReflectionToken.ISREL_IRRELEVANT
                ),
                confidence=0.8 if is_relevant else 0.6,
                reasoning=response[:200],
            )
        except Exception as e:
            logger.warning(f"Relevance assessment failed: {e}")
            return ReflectionResult(
                token=ReflectionToken.ISREL_RELEVANT,
                confidence=0.5,
            )

    async def assess_support(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
    ) -> ReflectionResult:
        """
        Assess if answer is supported by documents.

        Args:
            answer: Generated answer
            documents: Retrieved documents

        Returns:
            ReflectionResult with support assessment
        """
        context = "\n\n".join(
            f"Document {i+1}: {d.get('content', d.get('text', ''))[:500]}"
            for i, d in enumerate(documents[:3])
        )

        prompt = f"""Assess how well the answer is supported by the documents.

Documents:
{context}

Answer: {answer}

Assessment options:
- FULLY SUPPORTED: All claims in answer are directly supported
- PARTIALLY SUPPORTED: Some claims are supported, others are not
- NO SUPPORT: Answer contains claims not found in documents

Assessment:"""

        try:
            response = await self.llm.generate(prompt)
            response_lower = response.lower()

            if "fully" in response_lower:
                token = ReflectionToken.ISSUP_FULLY
                confidence = 0.9
            elif "partial" in response_lower:
                token = ReflectionToken.ISSUP_PARTIALLY
                confidence = 0.6
            else:
                token = ReflectionToken.ISSUP_NONE
                confidence = 0.4

            return ReflectionResult(
                token=token,
                confidence=confidence,
                reasoning=response[:200],
            )
        except Exception as e:
            logger.warning(f"Support assessment failed: {e}")
            return ReflectionResult(
                token=ReflectionToken.ISSUP_PARTIALLY,
                confidence=0.5,
            )

    async def assess_utility(
        self,
        query: str,
        answer: str,
    ) -> ReflectionResult:
        """
        Assess answer utility for the query.

        Args:
            query: User query
            answer: Generated answer

        Returns:
            ReflectionResult with utility assessment
        """
        prompt = f"""Assess how useful this answer is for the question.

Question: {query}

Answer: {answer}

Assessment options:
- USEFUL: Answer directly and completely addresses the question
- PARTIALLY USEFUL: Answer addresses some aspects but is incomplete
- NOT USEFUL: Answer does not address the question

Assessment:"""

        try:
            response = await self.llm.generate(prompt)
            response_lower = response.lower()

            if "not useful" in response_lower:
                token = ReflectionToken.ISUSE_NOT
                confidence = 0.4
            elif "partial" in response_lower:
                token = ReflectionToken.ISUSE_PARTIAL
                confidence = 0.6
            else:
                token = ReflectionToken.ISUSE_USEFUL
                confidence = 0.8

            return ReflectionResult(
                token=token,
                confidence=confidence,
                reasoning=response[:200],
            )
        except Exception as e:
            logger.warning(f"Utility assessment failed: {e}")
            return ReflectionResult(
                token=ReflectionToken.ISUSE_PARTIAL,
                confidence=0.5,
            )


# =============================================================================
# Critique Module
# =============================================================================


class AnswerCritique:
    """Critiques generated answers for quality."""

    def __init__(self, llm: LLMProtocol):
        """Initialize critique module."""
        self.llm = llm

    async def critique(
        self,
        query: str,
        answer: str,
        documents: List[Dict[str, Any]],
    ) -> CritiqueResult:
        """
        Critique generated answer.

        Args:
            query: User query
            answer: Generated answer
            documents: Supporting documents

        Returns:
            CritiqueResult with assessment
        """
        context = "\n".join(
            f"[{i+1}] {d.get('content', d.get('text', ''))[:400]}"
            for i, d in enumerate(documents[:3])
        )

        prompt = f"""Critique this answer based on the question and available evidence.

Question: {query}

Evidence:
{context}

Answer: {answer}

Evaluate:
1. Factual accuracy (is it correct based on evidence?)
2. Completeness (does it fully address the question?)
3. Clarity (is it well-written and clear?)
4. Grounding (are claims supported by evidence?)

Provide:
- Overall assessment: GOOD or NEEDS REVISION
- Issues (if any): List specific problems
- Suggestions: How to improve

Assessment:"""

        try:
            response = await self.llm.generate(prompt)
            is_good = "good" in response.lower() and "needs revision" not in response.lower()

            # Extract issues
            issues = []
            for line in response.split("\n"):
                line_lower = line.lower()
                if "issue" in line_lower or "problem" in line_lower:
                    issues.append(line.strip())

            # Extract suggestions
            suggestions = []
            for line in response.split("\n"):
                line_lower = line.lower()
                if "suggest" in line_lower or "improve" in line_lower:
                    suggestions.append(line.strip())

            return CritiqueResult(
                is_satisfactory=is_good,
                issues=issues[:3],
                suggestions=suggestions[:3],
                score=0.8 if is_good else 0.4,
            )
        except Exception as e:
            logger.warning(f"Critique failed: {e}")
            return CritiqueResult(
                is_satisfactory=True,
                score=0.5,
            )


# =============================================================================
# Main Self-RAG
# =============================================================================


class SelfRAG:
    """
    Self-RAG with reflection tokens.

    Implements adaptive retrieval and self-assessment during
    generation using reflection tokens.

    Example:
        >>> self_rag = SelfRAG(llm, retriever)
        >>> result = await self_rag.generate(
        ...     "What causes climate change?",
        ...     max_iterations=3
        ... )
        >>> print(f"Answer: {result.answer}")
        >>> print(f"Support: {result.support_score:.2f}")
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: Optional[RetrieverProtocol] = None,
        max_iterations: int = 3,
        support_threshold: float = 0.6,
    ):
        """
        Initialize Self-RAG.

        Args:
            llm: Language model
            retriever: Document retriever
            max_iterations: Maximum refinement iterations
            support_threshold: Minimum support score to accept
        """
        self.llm = llm
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.support_threshold = support_threshold

        self.reflection_predictor = ReflectionPredictor(llm)
        self.critique = AnswerCritique(llm)

    async def generate(
        self,
        query: str,
        *,
        top_k: int = 5,
        force_retrieval: bool = False,
        include_critique: bool = True,
    ) -> SelfRAGResponse:
        """
        Generate answer with self-reflection.

        Args:
            query: User query
            top_k: Documents to retrieve
            force_retrieval: Force retrieval regardless of prediction
            include_critique: Include critique step

        Returns:
            SelfRAGResponse with answer and reflection tokens
        """
        reflection_tokens = []

        # Step 1: Decide if retrieval is needed
        retrieval_decision = await self.reflection_predictor.predict_retrieval_need(query)
        reflection_tokens.append(
            ReflectionResult(
                token=retrieval_decision.token,
                confidence=retrieval_decision.confidence,
                reasoning=retrieval_decision.reasoning,
            )
        )

        # Step 2: Retrieve if needed
        documents = []
        relevance_scores = {}

        if force_retrieval or retrieval_decision.should_retrieve:
            if self.retriever:
                documents = await self.retriever.retrieve(query, top_k=top_k)

                # Assess relevance of each document
                for i, doc in enumerate(documents):
                    relevance = await self.reflection_predictor.assess_relevance(query, doc)
                    reflection_tokens.append(relevance)
                    relevance_scores[f"doc_{i}"] = relevance.confidence

        # Step 3: Generate answer
        answer = await self._generate_answer(query, documents)

        # Step 4: Assess support
        if documents:
            support_result = await self.reflection_predictor.assess_support(answer, documents)
            reflection_tokens.append(support_result)
            support_score = support_result.confidence
        else:
            support_score = 0.5

        # Step 5: Assess utility
        utility_result = await self.reflection_predictor.assess_utility(query, answer)
        reflection_tokens.append(utility_result)
        utility_score = utility_result.confidence

        # Step 6: Critique and refine if needed
        iterations = 1
        if include_critique and utility_score < 0.7:
            for i in range(self.max_iterations - 1):
                critique_result = await self.critique.critique(query, answer, documents)

                if critique_result.is_satisfactory:
                    break

                # Refine answer based on critique
                answer = await self._refine_answer(query, answer, documents, critique_result)
                iterations += 1

                # Re-assess
                utility_result = await self.reflection_predictor.assess_utility(query, answer)
                utility_score = utility_result.confidence

                if utility_score >= 0.7:
                    break

        return SelfRAGResponse(
            answer=answer,
            reflection_tokens=reflection_tokens,
            retrieved_docs=documents,
            iterations=iterations,
            retrieval_decision=retrieval_decision,
            relevance_scores=relevance_scores,
            support_score=support_score,
            utility_score=utility_score,
        )

    async def _generate_answer(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> str:
        """Generate initial answer."""
        if documents:
            context = "\n\n".join(
                f"[{i+1}] {d.get('content', d.get('text', ''))[:800]}"
                for i, d in enumerate(documents[:5])
            )

            prompt = f"""Answer the question based on the provided context.
Only use information from the context. If the context doesn't contain
enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""Answer the following question based on your knowledge.
Be accurate and acknowledge any uncertainty.

Question: {query}

Answer:"""

        return await self.llm.generate(prompt)

    async def _refine_answer(
        self,
        query: str,
        answer: str,
        documents: List[Dict[str, Any]],
        critique: CritiqueResult,
    ) -> str:
        """Refine answer based on critique."""
        context = "\n\n".join(
            f"[{i+1}] {d.get('content', d.get('text', ''))[:600]}"
            for i, d in enumerate(documents[:3])
        )

        issues = "\n".join(f"- {issue}" for issue in critique.issues)
        suggestions = "\n".join(f"- {s}" for s in critique.suggestions)

        prompt = f"""Improve this answer based on the critique.

Question: {query}

Current Answer: {answer}

Context:
{context}

Issues identified:
{issues or "None specified"}

Suggestions:
{suggestions or "None specified"}

Improved Answer:"""

        return await self.llm.generate(prompt)


# =============================================================================
# Self-RAG with Inline Tokens
# =============================================================================


class InlineTokenSelfRAG:
    """
    Self-RAG that generates inline reflection tokens.

    Produces output with embedded reflection tokens for
    training or interpretability.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: Optional[RetrieverProtocol] = None,
    ):
        """Initialize inline token Self-RAG."""
        self.llm = llm
        self.retriever = retriever
        self.self_rag = SelfRAG(llm, retriever)

    async def generate_with_inline_tokens(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Generate answer with inline reflection tokens.

        Args:
            query: User query
            top_k: Documents to retrieve

        Returns:
            Answer string with inline reflection tokens
        """
        result = await self.self_rag.generate(query, top_k=top_k)

        # Build output with inline tokens
        parts = []

        # Add retrieval decision
        parts.append(result.retrieval_decision.token.value)

        # Add per-document relevance
        for i, doc in enumerate(result.retrieved_docs[:3]):
            doc_id = f"doc_{i}"
            score = result.relevance_scores.get(doc_id, 0.5)
            if score > 0.6:
                parts.append(ReflectionToken.ISREL_RELEVANT.value)
            else:
                parts.append(ReflectionToken.ISREL_IRRELEVANT.value)

        # Add answer
        parts.append(result.answer)

        # Add support assessment
        if result.support_score > 0.7:
            parts.append(ReflectionToken.ISSUP_FULLY.value)
        elif result.support_score > 0.4:
            parts.append(ReflectionToken.ISSUP_PARTIALLY.value)
        else:
            parts.append(ReflectionToken.ISSUP_NONE.value)

        # Add utility assessment
        if result.utility_score > 0.7:
            parts.append(ReflectionToken.ISUSE_USEFUL.value)
        elif result.utility_score > 0.4:
            parts.append(ReflectionToken.ISUSE_PARTIAL.value)
        else:
            parts.append(ReflectionToken.ISUSE_NOT.value)

        return " ".join(parts)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_self_rag(
    llm: LLMProtocol,
    retriever: Optional[RetrieverProtocol] = None,
) -> SelfRAG:
    """
    Create a Self-RAG instance.

    Args:
        llm: Language model
        retriever: Document retriever

    Returns:
        SelfRAG instance
    """
    return SelfRAG(llm=llm, retriever=retriever)


async def self_rag_generate(
    query: str,
    llm: LLMProtocol,
    retriever: Optional[RetrieverProtocol] = None,
) -> str:
    """
    Quick Self-RAG generation.

    Args:
        query: User query
        llm: Language model
        retriever: Optional retriever

    Returns:
        Generated answer
    """
    self_rag = SelfRAG(llm=llm, retriever=retriever)
    result = await self_rag.generate(query)
    return result.answer
