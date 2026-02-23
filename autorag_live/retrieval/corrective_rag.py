"""
Corrective RAG (CRAG).

Implements the CRAG pattern from "Corrective Retrieval Augmented Generation"
(Yan et al., 2024 - https://arxiv.org/abs/2401.15884).

Core idea:
  1. Retrieve documents for the query.
  2. Grade each document with a lightweight evaluator (correct / ambiguous / incorrect).
  3. Based on the aggregate grade:
     - CORRECT  → use retrieved docs as-is.
     - AMBIGUOUS → combine retrieved docs with web-search results.
     - INCORRECT → discard retrieved docs, fall back to web search only.
  4. Refine / strip retrieved knowledge before passing to the generator.

This corrects for retrieval failures that silently produce hallucinations.

References:
    https://arxiv.org/abs/2401.15884
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
LLMFn = Callable[[str], Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------


class DocumentGrade(str, Enum):
    """Relevance verdict for a retrieved document."""

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class OverallGrade(str, Enum):
    """Aggregate decision for retrieval quality."""

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class GradedDocument:
    """A retrieved document paired with its relevance grade."""

    content: str
    grade: DocumentGrade
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CRAGResult:
    """Full CRAG pipeline result."""

    answer: str
    overall_grade: OverallGrade
    graded_docs: List[GradedDocument]
    used_web_search: bool
    refined_knowledge: str


# ---------------------------------------------------------------------------
# CRAG Implementation
# ---------------------------------------------------------------------------


class CorrectiveRAG:
    """
    Corrective Retrieval Augmented Generation (CRAG).

    Usage::

        async def my_llm(prompt: str) -> str: ...
        async def web_search(query: str) -> list[str]: ...
        async def retrieve(query: str) -> list[str]: ...

        crag = CorrectiveRAG(llm_fn=my_llm, retriever_fn=retrieve, web_search_fn=web_search)
        result = await crag.run("What are the effects of climate change on coral reefs?")
        print(result.answer)
    """

    _GRADE_PROMPT = """\
You are grading the relevance of a retrieved document to a user question.

Question: {question}

Retrieved Document:
{document}

Is this document relevant to answering the question?
- Respond "correct" if it directly and clearly helps answer the question.
- Respond "ambiguous" if it is partially relevant but may be incomplete or off-topic.
- Respond "incorrect" if it is clearly irrelevant or misleading.

Respond with exactly one word: correct, ambiguous, or incorrect.
"""

    _REFINE_PROMPT = """\
You are refining retrieved knowledge to extract only the most relevant facts.

Question: {question}

Retrieved Documents:
{documents}

Extract and present only the key facts that directly answer the question.
Remove any irrelevant, redundant, or off-topic content.
Present as concise bullet points.
"""

    _GENERATE_PROMPT = """\
Answer the following question using the provided knowledge.

Question: {question}

Knowledge:
{knowledge}

Provide a comprehensive, accurate answer:"""

    def __init__(
        self,
        llm_fn: LLMFn,
        retriever_fn: Callable[[str], Coroutine[Any, Any, List[str]]],
        web_search_fn: Optional[Callable[[str], Coroutine[Any, Any, List[str]]]] = None,
        correct_threshold: float = 0.5,
    ) -> None:
        """
        Initialize CRAG.

        Args:
            llm_fn: Async function to call the LLM with a prompt string.
            retriever_fn: Async function that returns a list of document strings.
            web_search_fn: Optional async web-search fallback.
            correct_threshold: Fraction of docs graded "correct" to consider overall CORRECT.
        """
        self._llm = llm_fn
        self._retriever = retriever_fn
        self._web_search = web_search_fn
        self.correct_threshold = correct_threshold

    async def _grade_document(self, question: str, document: str) -> DocumentGrade:
        """Grade a single document for relevance."""
        prompt = self._GRADE_PROMPT.format(question=question, document=document[:2000])
        response = (await self._llm(prompt)).strip().lower()

        if "correct" in response:
            return DocumentGrade.CORRECT
        if "ambiguous" in response:
            return DocumentGrade.AMBIGUOUS
        return DocumentGrade.INCORRECT

    async def _grade_all(self, question: str, docs: Sequence[str]) -> List[GradedDocument]:
        """Grade all documents concurrently."""
        grades = await asyncio.gather(*[self._grade_document(question, d) for d in docs])
        return [GradedDocument(content=doc, grade=grade) for doc, grade in zip(docs, grades)]

    def _aggregate_grade(self, graded: List[GradedDocument]) -> OverallGrade:
        """Determine overall retrieval quality from individual grades."""
        if not graded:
            return OverallGrade.INCORRECT

        n = len(graded)
        correct_count = sum(1 for g in graded if g.grade == DocumentGrade.CORRECT)
        incorrect_count = sum(1 for g in graded if g.grade == DocumentGrade.INCORRECT)

        if correct_count / n >= self.correct_threshold:
            return OverallGrade.CORRECT
        if incorrect_count / n >= self.correct_threshold:
            return OverallGrade.INCORRECT
        return OverallGrade.AMBIGUOUS

    async def _refine_knowledge(self, question: str, docs: List[GradedDocument]) -> str:
        """Strip noise; extract key facts from relevant documents."""
        relevant = [d.content for d in docs if d.grade != DocumentGrade.INCORRECT]
        if not relevant:
            return ""
        combined = "\n\n---\n\n".join(f"Doc {i + 1}:\n{d}" for i, d in enumerate(relevant))
        prompt = self._REFINE_PROMPT.format(question=question, documents=combined[:4000])
        return await self._llm(prompt)

    async def run(self, question: str) -> CRAGResult:
        """
        Execute the full CRAG pipeline.

        Args:
            question: The user query to answer.

        Returns:
            CRAGResult with final answer and grading metadata.
        """
        # Step 1: Retrieve
        raw_docs = await self._retriever(question)
        logger.info("CRAG: retrieved %d documents", len(raw_docs))

        # Step 2: Grade
        graded_docs = await self._grade_all(question, raw_docs)
        overall_grade = self._aggregate_grade(graded_docs)
        logger.info("CRAG: overall grade = %s", overall_grade.value)

        used_web = False
        web_docs: List[str] = []

        # Step 3: Decide knowledge source
        if overall_grade == OverallGrade.INCORRECT:
            # Discard retrieval; rely entirely on web search
            if self._web_search:
                web_docs = await self._web_search(question)
                graded_docs = [
                    GradedDocument(content=d, grade=DocumentGrade.CORRECT) for d in web_docs
                ]
                used_web = True
                logger.info("CRAG: using %d web results (retrieval discarded)", len(web_docs))
            else:
                logger.warning("CRAG: retrieval incorrect and no web search fallback available")

        elif overall_grade == OverallGrade.AMBIGUOUS:
            # Augment with web search
            if self._web_search:
                web_docs = await self._web_search(question)
                extra = [GradedDocument(content=d, grade=DocumentGrade.CORRECT) for d in web_docs]
                graded_docs.extend(extra)
                used_web = True
                logger.info(
                    "CRAG: augmented with %d web results (ambiguous retrieval)", len(web_docs)
                )

        # Step 4: Refine knowledge
        refined = await self._refine_knowledge(question, graded_docs)

        # Step 5: Generate answer
        knowledge = refined if refined else "\n\n".join(d.content for d in graded_docs)
        prompt = self._GENERATE_PROMPT.format(question=question, knowledge=knowledge[:6000])
        answer = await self._llm(prompt)

        return CRAGResult(
            answer=answer.strip(),
            overall_grade=overall_grade,
            graded_docs=graded_docs,
            used_web_search=used_web,
            refined_knowledge=refined,
        )
