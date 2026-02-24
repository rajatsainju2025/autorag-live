"""
Constitutional Critic Agent for RAG Output Refinement.

Inspired by Constitutional AI (Bai et al., 2022 — https://arxiv.org/abs/2212.08073),
this agent applies a set of *constitutional principles* to evaluate and iteratively
revise RAG-generated answers.

Each principle is a natural-language rule (e.g. "The answer must be factually
grounded in the retrieved context. Do not introduce information not present in
the context."). The critic checks the answer against every principle and, if
a violation is detected, generates a revision.

Workflow per revision round:
  1. Critique: For each principle, ask the LLM whether the answer violates it.
  2. Revision: For each violated principle, produce a revised answer.
  3. Select: Accumulate revisions; the final output is the last clean answer.

This is a zero-shot, training-free approach to enforce factuality, coherence,
conciseness, and safety constraints on RAG outputs.

Default principles target the core RAG failure modes:
  - Hallucination / confabulation
  - Answer-context faithfulness
  - Answer completeness
  - Conciseness
  - Hedging when uncertain

Usage::

    critic = ConstitutionalCriticAgent(llm_fn=my_llm)
    result = await critic.critique_and_revise(
        question="What causes aurora borealis?",
        answer="It is caused by solar wind particles ...",
        context="[retrieved docs]",
    )
    print(result.final_answer)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List, Optional

logger = logging.getLogger(__name__)

LLMFn = Callable[[str], Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# Default RAG constitutional principles
# ---------------------------------------------------------------------------

DEFAULT_PRINCIPLES: List[str] = [
    (
        "Faithfulness: The answer must only use information present in the "
        "retrieved context. It must not introduce facts, names, or figures "
        "that are not grounded in the provided documents."
    ),
    (
        "Completeness: The answer must address all aspects of the question "
        "that can be resolved from the retrieved context. Do not omit key "
        "relevant information that appears in the context."
    ),
    (
        "Conciseness: The answer must be as concise as possible while remaining "
        "complete and accurate. Remove any redundant or off-topic sentences."
    ),
    (
        "Uncertainty hedging: When the retrieved context does not contain "
        "sufficient information to answer part of the question with confidence, "
        "the answer must explicitly acknowledge this limitation (e.g., "
        "'Based on the available information...' or 'The context does not "
        "provide details about...')."
    ),
    (
        "No hallucination: The answer must not contain any claims that contradict "
        "the retrieved context or that are presented with false certainty about "
        "topics not covered by the context."
    ),
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PrincipleEvaluation:
    """Result of evaluating a single constitutional principle."""

    principle: str
    violated: bool
    critique: str  # LLM's reasoning for the verdict
    revision: Optional[str] = None  # Revised answer if violated


@dataclass
class ConstitutionalCritiqueResult:
    """Full result of the constitutional critique-and-revise loop."""

    original_answer: str
    final_answer: str
    evaluations: List[PrincipleEvaluation]
    n_violations: int
    n_rounds: int
    principles_applied: List[str]


# ---------------------------------------------------------------------------
# Constitutional Critic Agent
# ---------------------------------------------------------------------------


class ConstitutionalCriticAgent:
    """
    Constitutional Critic for RAG answer quality control.

    Applies a configurable set of natural-language principles to iteratively
    critique and revise RAG-generated answers.

    Args:
        llm_fn: Async callable ``(prompt: str) -> str``.
        principles: Constitutional principles to enforce. Defaults to ``DEFAULT_PRINCIPLES``.
        max_rounds: Maximum critique-revision cycles.
        parallel_critique: If True, evaluate all principles concurrently.

    Example::

        critic = ConstitutionalCriticAgent(llm_fn=my_llm)
        result = await critic.critique_and_revise(
            question="...", answer="...", context="..."
        )
        print(result.final_answer)
    """

    _CRITIQUE_PROMPT = """\
You are a strict constitutional evaluator for AI-generated answers.

Constitutional Principle:
{principle}

---
Question: {question}

Retrieved Context:
{context}

Answer Being Evaluated:
{answer}
---

Does the above answer VIOLATE the constitutional principle?
First, briefly explain your reasoning (1-2 sentences).
Then output exactly one of: VIOLATED or SATISFIED

Reasoning:"""

    _REVISION_PROMPT = """\
You are revising an AI-generated answer to comply with a constitutional principle.

Constitutional Principle:
{principle}

---
Question: {question}

Retrieved Context:
{context}

Original Answer:
{answer}

Critique of the violation:
{critique}
---

Write a revised answer that corrects the violation while keeping all correct,
useful content from the original. The revision must still be grounded in the
retrieved context.

Revised Answer:"""

    def __init__(
        self,
        llm_fn: LLMFn,
        principles: Optional[List[str]] = None,
        max_rounds: int = 2,
        parallel_critique: bool = True,
    ) -> None:
        self._llm = llm_fn
        self.principles = principles if principles is not None else DEFAULT_PRINCIPLES
        self.max_rounds = max_rounds
        self.parallel_critique = parallel_critique

    async def critique_and_revise(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> ConstitutionalCritiqueResult:
        """
        Run the constitutional critique-and-revise pipeline.

        Args:
            question: The original user question.
            answer: The RAG-generated answer to evaluate.
            context: The retrieved context used to generate the answer.

        Returns:
            ConstitutionalCritiqueResult with the final revised answer and evaluation trace.
        """
        original_answer = answer
        current_answer = answer
        all_evaluations: List[PrincipleEvaluation] = []
        total_violations = 0

        for round_num in range(1, self.max_rounds + 1):
            logger.info("Constitutional critique round %d/%d", round_num, self.max_rounds)

            # Evaluate all principles (parallel or sequential)
            if self.parallel_critique:
                evaluations = await asyncio.gather(
                    *[
                        self._evaluate_principle(p, question, current_answer, context)
                        for p in self.principles
                    ]
                )
            else:
                evaluations = []
                for p in self.principles:
                    ev = await self._evaluate_principle(p, question, current_answer, context)
                    evaluations.append(ev)

            round_violations = [ev for ev in evaluations if ev.violated]
            all_evaluations.extend(evaluations)
            total_violations += len(round_violations)

            if not round_violations:
                logger.info("No violations in round %d; stopping early.", round_num)
                break

            # Apply revisions sequentially (each revision feeds the next)
            for ev in round_violations:
                revision = await self._revise(
                    ev.principle, question, current_answer, context, ev.critique
                )
                ev.revision = revision
                current_answer = revision
                logger.debug("Revised for principle '%s...'", ev.principle[:50])

        return ConstitutionalCritiqueResult(
            original_answer=original_answer,
            final_answer=current_answer,
            evaluations=all_evaluations,
            n_violations=total_violations,
            n_rounds=min(round_num, self.max_rounds),
            principles_applied=self.principles,
        )

    async def _evaluate_principle(
        self, principle: str, question: str, answer: str, context: str
    ) -> PrincipleEvaluation:
        """Check a single principle against the current answer."""
        prompt = self._CRITIQUE_PROMPT.format(
            principle=principle,
            question=question,
            context=context[:3000],
            answer=answer,
        )
        response = (await self._llm(prompt)).strip()

        violated = "VIOLATED" in response.upper()
        # Isolate the reasoning from the verdict line
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        critique_lines = [
            ln for ln in lines if "VIOLATED" not in ln.upper() and "SATISFIED" not in ln.upper()
        ]
        critique = " ".join(critique_lines) or response

        return PrincipleEvaluation(
            principle=principle,
            violated=violated,
            critique=critique,
        )

    async def _revise(
        self,
        principle: str,
        question: str,
        answer: str,
        context: str,
        critique: str,
    ) -> str:
        """Generate a revised answer that addresses a specific violation."""
        prompt = self._REVISION_PROMPT.format(
            principle=principle,
            question=question,
            context=context[:3000],
            answer=answer,
            critique=critique,
        )
        return (await self._llm(prompt)).strip()
