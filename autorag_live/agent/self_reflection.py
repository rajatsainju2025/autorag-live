"""
Self-Reflection and Corrective RAG (CRAG) Module.

Implements retrieval quality assessment, self-correction, and query refinement
based on the CRAG pattern and Self-RAG concepts.

Based on:
- "Corrective Retrieval Augmented Generation" (Yan et al., 2024)
- "Self-RAG: Learning to Retrieve, Generate, and Critique" (Asai et al., 2023)

Key Features:
1. Retrieval quality grading (Correct/Incorrect/Ambiguous)
2. Query rewriting for improved retrieval
3. Web search fallback for knowledge gaps
4. Answer verification and self-critique
5. Hallucination detection

Example:
    >>> reflector = SelfReflector(llm=my_llm)
    >>> assessment = await reflector.grade_retrieval(query, documents)
    >>> if assessment.needs_correction:
    ...     new_query = await reflector.rewrite_query(query, assessment)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from autorag_live.core.protocols import BaseLLM, Document, Message, RetrievalResult

logger = logging.getLogger(__name__)


# =============================================================================
# Assessment Types
# =============================================================================


class RetrievalGrade(str, Enum):
    """Grades for retrieval quality."""

    CORRECT = "correct"  # Documents are relevant
    INCORRECT = "incorrect"  # Documents are not relevant
    AMBIGUOUS = "ambiguous"  # Unclear if relevant


class KnowledgeType(str, Enum):
    """Types of knowledge required."""

    IN_CONTEXT = "in_context"  # Can answer from retrieved docs
    REQUIRES_SEARCH = "requires_search"  # Needs external search
    REQUIRES_REASONING = "requires_reasoning"  # Needs multi-hop reasoning
    UNANSWERABLE = "unanswerable"  # Cannot be answered


class CritiqueType(str, Enum):
    """Types of answer critiques."""

    SUPPORTED = "supported"  # Answer supported by sources
    PARTIALLY_SUPPORTED = "partially_supported"  # Some claims not supported
    NOT_SUPPORTED = "not_supported"  # Answer contradicts or lacks support
    HALLUCINATED = "hallucinated"  # Contains fabricated information


@dataclass
class RetrievalAssessment:
    """
    Assessment of retrieval quality.

    Attributes:
        grade: Overall retrieval grade
        relevance_scores: Per-document relevance scores
        reasoning: Explanation for the assessment
        needs_correction: Whether correction is recommended
        suggested_action: Recommended corrective action
        confidence: Confidence in the assessment
    """

    grade: RetrievalGrade
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    needs_correction: bool = False
    suggested_action: str = ""
    confidence: float = 0.5

    @property
    def is_correct(self) -> bool:
        """Check if retrieval is assessed as correct."""
        return self.grade == RetrievalGrade.CORRECT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "grade": self.grade.value,
            "relevance_scores": self.relevance_scores,
            "reasoning": self.reasoning,
            "needs_correction": self.needs_correction,
            "suggested_action": self.suggested_action,
            "confidence": self.confidence,
        }


@dataclass
class KnowledgeAssessment:
    """
    Assessment of knowledge requirements.

    Attributes:
        knowledge_type: Type of knowledge needed
        has_sufficient_context: Whether context is sufficient
        missing_information: What information is missing
        reasoning: Explanation for the assessment
    """

    knowledge_type: KnowledgeType
    has_sufficient_context: bool = False
    missing_information: List[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "knowledge_type": self.knowledge_type.value,
            "has_sufficient_context": self.has_sufficient_context,
            "missing_information": self.missing_information,
            "reasoning": self.reasoning,
        }


@dataclass
class AnswerCritique:
    """
    Critique of generated answer.

    Attributes:
        critique_type: Type of critique
        is_faithful: Whether answer is faithful to sources
        unsupported_claims: Claims not supported by sources
        reasoning: Explanation for the critique
        confidence: Confidence in the critique
        suggested_revision: Suggested improved answer
    """

    critique_type: CritiqueType
    is_faithful: bool = True
    unsupported_claims: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.5
    suggested_revision: Optional[str] = None

    @property
    def needs_revision(self) -> bool:
        """Check if answer needs revision."""
        return self.critique_type not in (
            CritiqueType.SUPPORTED,
            CritiqueType.PARTIALLY_SUPPORTED,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "critique_type": self.critique_type.value,
            "is_faithful": self.is_faithful,
            "unsupported_claims": self.unsupported_claims,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "suggested_revision": self.suggested_revision,
        }


@dataclass
class QueryRewrite:
    """
    Rewritten query for improved retrieval.

    Attributes:
        original_query: Original user query
        rewritten_query: Improved query
        reasoning: Why the rewrite was made
        decomposed_queries: Sub-queries if decomposed
    """

    original_query: str
    rewritten_query: str
    reasoning: str = ""
    decomposed_queries: List[str] = field(default_factory=list)

    def get_queries(self) -> List[str]:
        """Get all queries to execute (rewritten + decomposed)."""
        queries = [self.rewritten_query]
        queries.extend(self.decomposed_queries)
        return queries


# =============================================================================
# Prompts
# =============================================================================

GRADE_RETRIEVAL_PROMPT = """You are a retrieval quality assessor. Evaluate whether the retrieved documents are relevant to answering the query.

Query: {query}

Retrieved Documents:
{documents}

Assess the relevance of each document and provide an overall grade.

Output format (JSON):
{{
    "grade": "correct" | "incorrect" | "ambiguous",
    "relevance_scores": {{"doc_id": score}},  // 0.0 to 1.0
    "reasoning": "explanation of assessment",
    "needs_correction": true | false,
    "suggested_action": "what to do if correction needed",
    "confidence": 0.0-1.0
}}

Respond with ONLY the JSON object."""

ASSESS_KNOWLEDGE_PROMPT = """Analyze whether the retrieved context contains sufficient information to answer the query.

Query: {query}

Context:
{context}

Determine:
1. What type of knowledge is needed
2. Whether the context is sufficient
3. What information is missing (if any)

Output format (JSON):
{{
    "knowledge_type": "in_context" | "requires_search" | "requires_reasoning" | "unanswerable",
    "has_sufficient_context": true | false,
    "missing_information": ["list", "of", "missing", "info"],
    "reasoning": "explanation"
}}

Respond with ONLY the JSON object."""

CRITIQUE_ANSWER_PROMPT = """You are a fact-checker. Evaluate whether the answer is faithful to the provided sources.

Query: {query}

Sources:
{sources}

Answer: {answer}

Check:
1. Is every claim in the answer supported by the sources?
2. Are there any unsupported or contradicted claims?
3. Is there any hallucinated information?

Output format (JSON):
{{
    "critique_type": "supported" | "partially_supported" | "not_supported" | "hallucinated",
    "is_faithful": true | false,
    "unsupported_claims": ["list of unsupported claims"],
    "reasoning": "explanation",
    "confidence": 0.0-1.0,
    "suggested_revision": "improved answer if needed, null otherwise"
}}

Respond with ONLY the JSON object."""

REWRITE_QUERY_PROMPT = """You are a query optimization expert. Improve the query for better retrieval results.

Original Query: {query}

Assessment: {assessment}

Task: Rewrite the query to be:
- More specific and searchable
- Decomposed into sub-queries if complex
- Focused on the missing information

Output format (JSON):
{{
    "rewritten_query": "improved query",
    "reasoning": "why this rewrite helps",
    "decomposed_queries": ["sub-query 1", "sub-query 2"]  // optional, for complex queries
}}

Respond with ONLY the JSON object."""


# =============================================================================
# Self-Reflector
# =============================================================================


class SelfReflector:
    """
    Self-reflection engine for CRAG-style assessment and correction.

    Provides:
    - Retrieval quality grading
    - Knowledge gap assessment
    - Answer faithfulness critique
    - Query rewriting for correction

    Example:
        >>> reflector = SelfReflector(llm)
        >>> assessment = await reflector.grade_retrieval(query, docs)
        >>> if not assessment.is_correct:
        ...     rewrite = await reflector.rewrite_query(query, assessment)
    """

    def __init__(
        self,
        llm: BaseLLM,
        *,
        relevance_threshold: float = 0.5,
        confidence_threshold: float = 0.7,
        verbose: bool = False,
    ):
        """
        Initialize self-reflector.

        Args:
            llm: Language model for assessments
            relevance_threshold: Minimum relevance score
            confidence_threshold: Minimum confidence for decisions
            verbose: Enable verbose logging
        """
        self.llm = llm
        self.relevance_threshold = relevance_threshold
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose

    def _format_documents(self, documents: List[Document]) -> str:
        """Format documents for prompt."""
        formatted = []
        for i, doc in enumerate(documents):
            doc_id = doc.id or f"doc_{i}"
            formatted.append(f"[{doc_id}] {doc.content}")
        return "\n\n".join(formatted)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        import json

        # Try to find JSON in response
        try:
            # First try direct parsing
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON block
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Return empty dict if parsing fails
        logger.warning(f"Failed to parse JSON from response: {response[:200]}...")
        return {}

    async def grade_retrieval(
        self,
        query: str,
        documents: List[Document],
    ) -> RetrievalAssessment:
        """
        Grade the quality of retrieved documents.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            RetrievalAssessment with grade and details
        """
        if not documents:
            return RetrievalAssessment(
                grade=RetrievalGrade.INCORRECT,
                reasoning="No documents retrieved",
                needs_correction=True,
                suggested_action="retry_retrieval",
                confidence=1.0,
            )

        # Build prompt
        prompt = GRADE_RETRIEVAL_PROMPT.format(
            query=query,
            documents=self._format_documents(documents),
        )

        # Get assessment from LLM
        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.0,  # Deterministic for evaluation
        )

        # Parse response
        data = self._parse_json_response(result.content)

        # Build assessment
        grade = RetrievalGrade(data.get("grade", "ambiguous"))
        relevance_scores = data.get("relevance_scores", {})

        # Determine if correction needed
        avg_relevance = (
            sum(relevance_scores.values()) / len(relevance_scores) if relevance_scores else 0.0
        )
        needs_correction = (
            grade != RetrievalGrade.CORRECT or avg_relevance < self.relevance_threshold
        )

        assessment = RetrievalAssessment(
            grade=grade,
            relevance_scores=relevance_scores,
            reasoning=data.get("reasoning", ""),
            needs_correction=needs_correction,
            suggested_action=data.get("suggested_action", ""),
            confidence=data.get("confidence", 0.5),
        )

        if self.verbose:
            logger.info(f"Retrieval grade: {grade.value}, confidence: {assessment.confidence}")

        return assessment

    async def assess_knowledge(
        self,
        query: str,
        context: str,
    ) -> KnowledgeAssessment:
        """
        Assess whether context has sufficient knowledge.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            KnowledgeAssessment with type and gaps
        """
        prompt = ASSESS_KNOWLEDGE_PROMPT.format(
            query=query,
            context=context,
        )

        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.0,
        )

        data = self._parse_json_response(result.content)

        assessment = KnowledgeAssessment(
            knowledge_type=KnowledgeType(data.get("knowledge_type", "in_context")),
            has_sufficient_context=data.get("has_sufficient_context", False),
            missing_information=data.get("missing_information", []),
            reasoning=data.get("reasoning", ""),
        )

        if self.verbose:
            logger.info(
                f"Knowledge type: {assessment.knowledge_type.value}, "
                f"sufficient: {assessment.has_sufficient_context}"
            )

        return assessment

    async def critique_answer(
        self,
        query: str,
        answer: str,
        sources: List[Document],
    ) -> AnswerCritique:
        """
        Critique an answer for faithfulness to sources.

        Args:
            query: User query
            answer: Generated answer
            sources: Source documents

        Returns:
            AnswerCritique with assessment
        """
        prompt = CRITIQUE_ANSWER_PROMPT.format(
            query=query,
            answer=answer,
            sources=self._format_documents(sources),
        )

        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.0,
        )

        data = self._parse_json_response(result.content)

        critique = AnswerCritique(
            critique_type=CritiqueType(data.get("critique_type", "supported")),
            is_faithful=data.get("is_faithful", True),
            unsupported_claims=data.get("unsupported_claims", []),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.5),
            suggested_revision=data.get("suggested_revision"),
        )

        if self.verbose:
            logger.info(
                f"Answer critique: {critique.critique_type.value}, "
                f"faithful: {critique.is_faithful}"
            )

        return critique

    async def rewrite_query(
        self,
        query: str,
        assessment: RetrievalAssessment,
    ) -> QueryRewrite:
        """
        Rewrite query for improved retrieval.

        Args:
            query: Original query
            assessment: Retrieval assessment

        Returns:
            QueryRewrite with improved query
        """
        prompt = REWRITE_QUERY_PROMPT.format(
            query=query,
            assessment=assessment.reasoning,
        )

        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.3,  # Slight creativity for rewrites
        )

        data = self._parse_json_response(result.content)

        rewrite = QueryRewrite(
            original_query=query,
            rewritten_query=data.get("rewritten_query", query),
            reasoning=data.get("reasoning", ""),
            decomposed_queries=data.get("decomposed_queries", []),
        )

        if self.verbose:
            logger.info(f"Query rewritten: {rewrite.rewritten_query}")
            if rewrite.decomposed_queries:
                logger.info(f"Decomposed into: {rewrite.decomposed_queries}")

        return rewrite


# =============================================================================
# CRAG Pipeline
# =============================================================================


@dataclass
class CRAGResult:
    """
    Result from CRAG pipeline.

    Attributes:
        query: Original query
        answer: Final answer
        sources: Source documents used
        retrieval_assessment: Retrieval quality assessment
        knowledge_assessment: Knowledge gap assessment
        answer_critique: Answer faithfulness critique
        corrections_made: Number of corrections applied
        is_verified: Whether answer passed verification
    """

    query: str
    answer: str
    sources: List[Document] = field(default_factory=list)
    retrieval_assessment: Optional[RetrievalAssessment] = None
    knowledge_assessment: Optional[KnowledgeAssessment] = None
    answer_critique: Optional[AnswerCritique] = None
    corrections_made: int = 0
    is_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "retrieval_assessment": (
                self.retrieval_assessment.to_dict() if self.retrieval_assessment else None
            ),
            "knowledge_assessment": (
                self.knowledge_assessment.to_dict() if self.knowledge_assessment else None
            ),
            "answer_critique": (self.answer_critique.to_dict() if self.answer_critique else None),
            "corrections_made": self.corrections_made,
            "is_verified": self.is_verified,
        }


class CRAGPipeline:
    """
    Corrective RAG Pipeline with self-reflection.

    Implements the full CRAG pattern:
    1. Retrieve documents
    2. Grade retrieval quality
    3. Correct if needed (rewrite query, web search)
    4. Generate answer
    5. Critique and verify answer

    Example:
        >>> pipeline = CRAGPipeline(llm, retriever)
        >>> result = await pipeline.run("What is machine learning?")
        >>> print(result.answer)
        >>> print(f"Verified: {result.is_verified}")
    """

    def __init__(
        self,
        llm: BaseLLM,
        retriever: Any,  # Should implement retrieve(query, k) -> List[Document]
        *,
        max_corrections: int = 2,
        enable_web_search: bool = False,
        web_search_fn: Optional[Any] = None,
        verbose: bool = False,
    ):
        """
        Initialize CRAG pipeline.

        Args:
            llm: Language model
            retriever: Document retriever
            max_corrections: Maximum correction attempts
            enable_web_search: Enable web search fallback
            web_search_fn: Web search function (query) -> List[Document]
            verbose: Enable verbose logging
        """
        self.llm = llm
        self.retriever = retriever
        self.max_corrections = max_corrections
        self.enable_web_search = enable_web_search
        self.web_search_fn = web_search_fn
        self.verbose = verbose

        self.reflector = SelfReflector(llm, verbose=verbose)

    async def _retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Execute retrieval."""
        import asyncio

        if asyncio.iscoroutinefunction(self.retriever.retrieve):
            result = await self.retriever.retrieve(query, k=k)
        else:
            result = self.retriever.retrieve(query, k=k)

        if isinstance(result, RetrievalResult):
            return result.documents
        return result

    async def _generate_answer(
        self,
        query: str,
        context: str,
    ) -> str:
        """Generate answer from context."""
        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Provide a clear, accurate answer based only on the information in the context.
If the context doesn't contain enough information, say so.

Answer:"""

        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.3,
        )

        return result.content.strip()

    async def run(
        self,
        query: str,
        k: int = 5,
    ) -> CRAGResult:
        """
        Run the full CRAG pipeline.

        Args:
            query: User query
            k: Number of documents to retrieve

        Returns:
            CRAGResult with answer and assessments
        """
        corrections_made = 0
        current_query = query

        # Step 1: Initial retrieval
        documents = await self._retrieve(current_query, k)

        # Step 2: Grade retrieval
        retrieval_assessment = await self.reflector.grade_retrieval(current_query, documents)

        # Step 3: Correct if needed
        while retrieval_assessment.needs_correction and corrections_made < self.max_corrections:
            if self.verbose:
                logger.info(
                    f"Correction {corrections_made + 1}: {retrieval_assessment.suggested_action}"
                )

            # Try query rewriting
            rewrite = await self.reflector.rewrite_query(current_query, retrieval_assessment)
            current_query = rewrite.rewritten_query

            # Re-retrieve with new query
            new_docs = await self._retrieve(current_query, k)

            # Also try decomposed queries
            for sub_query in rewrite.decomposed_queries:
                sub_docs = await self._retrieve(sub_query, k // 2)
                new_docs.extend(sub_docs)

            # Deduplicate
            seen_ids = set()
            unique_docs = []
            for doc in new_docs:
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    unique_docs.append(doc)
            documents = unique_docs

            # Re-grade
            retrieval_assessment = await self.reflector.grade_retrieval(query, documents)
            corrections_made += 1

        # Step 4: Assess knowledge
        context = "\n\n".join(doc.content for doc in documents)
        knowledge_assessment = await self.reflector.assess_knowledge(query, context)

        # Step 5: Handle knowledge gaps
        if (
            self.enable_web_search
            and self.web_search_fn
            and knowledge_assessment.knowledge_type == KnowledgeType.REQUIRES_SEARCH
        ):
            if self.verbose:
                logger.info("Falling back to web search")

            web_docs = await self.web_search_fn(query)
            documents.extend(web_docs)
            context = "\n\n".join(doc.content for doc in documents)

        # Step 6: Generate answer
        answer = await self._generate_answer(query, context)

        # Step 7: Critique answer
        answer_critique = await self.reflector.critique_answer(query, answer, documents)

        # Step 8: Revise if needed
        if answer_critique.needs_revision and answer_critique.suggested_revision:
            if self.verbose:
                logger.info("Applying suggested revision")
            answer = answer_critique.suggested_revision

        # Build result
        return CRAGResult(
            query=query,
            answer=answer,
            sources=documents,
            retrieval_assessment=retrieval_assessment,
            knowledge_assessment=knowledge_assessment,
            answer_critique=answer_critique,
            corrections_made=corrections_made,
            is_verified=answer_critique.is_faithful,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


async def grade_retrieval(
    llm: BaseLLM,
    query: str,
    documents: List[Document],
) -> RetrievalAssessment:
    """
    Quick function to grade retrieval quality.

    Args:
        llm: Language model
        query: User query
        documents: Retrieved documents

    Returns:
        RetrievalAssessment
    """
    reflector = SelfReflector(llm)
    return await reflector.grade_retrieval(query, documents)


async def verify_answer(
    llm: BaseLLM,
    query: str,
    answer: str,
    sources: List[Document],
) -> bool:
    """
    Quick function to verify answer faithfulness.

    Args:
        llm: Language model
        query: User query
        answer: Generated answer
        sources: Source documents

    Returns:
        True if answer is faithful to sources
    """
    reflector = SelfReflector(llm)
    critique = await reflector.critique_answer(query, answer, sources)
    return critique.is_faithful
