"""
Constitutional AI Critique-Revision Loop.

Implements the Constitutional AI (CAI) self-critique and revision process from
Anthropic (Bai et al., 2022), adapted for agentic RAG answer generation.

Overview
--------
After an initial answer is generated, the model is asked to:
1. **Critique** the answer against each constitutional principle.
2. **Revise** the answer to address violations found.
3. Repeat up to ``max_revisions`` times, stopping early when all principles pass.

Why CAI for RAG?
----------------
RAG answers can violate safety, accuracy, or helpfulness principles even when
grounded in retrieved documents — the LLM may overstate confidence, include
speculative claims, or generate harmful adjacent content.  CAI provides a
systematic post-generation safety net without requiring a separate reward model.

Principles
----------
The default ``HHH`` principles (Harmless, Helpful, Honest) mirror Anthropic's
original Constitutional AI paper.  Custom principles can be added via
``ConstitutionalAI.add_principle()``.

Each principle provides:
- ``critique_prompt``  — asks the model to detect violations
- ``revision_prompt``  — asks the model to fix detected violations
- ``critique_request`` — optional follow-up to extract a structured verdict

References
----------
- Bai et al. 2022. "Constitutional AI: Harmlessness from AI Feedback."
  https://arxiv.org/abs/2212.08073
- Perez et al. 2022. "Red Teaming Language Models with Language Models."

Features
--------
- Pure async, semaphore-bounded LLM calls
- Pluggable ``LLMFn`` — no hard provider dependency
- Per-revision history stored for downstream audit
- ``violation_detected()`` heuristic parsing — no structured JSON needed
- ``ConstitutionalAI.revise()`` returns :class:`RevisionResult` with full trace
- ``create_cai()`` factory with preset principle bundles
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

LLMFn = Callable[[str], Coroutine[Any, Any, str]]
"""Async callable: prompt → response text."""


# ---------------------------------------------------------------------------
# Principle model
# ---------------------------------------------------------------------------


@dataclass
class ConstitutionalPrinciple:
    """
    A single principle in a constitutional AI revision loop.

    Attributes
    ----------
    name:
        Short identifier (e.g. ``"harmlessness"``).
    critique_prompt:
        Prompt appended to the current answer asking the model to identify
        whether this principle is violated.
    revision_prompt:
        Prompt asking the model to rewrite the answer fixing the violation.
    description:
        Human-readable description of the principle.
    required:
        If True, the loop will not exit early while this principle is violated.
    """

    name: str
    critique_prompt: str
    revision_prompt: str
    description: str = ""
    required: bool = True


# ---------------------------------------------------------------------------
# Default HHH principles
# ---------------------------------------------------------------------------

_HARMLESSNESS = ConstitutionalPrinciple(
    name="harmlessness",
    description=(
        "The answer must not encourage harm, violence, discrimination, " "or illegal activity."
    ),
    critique_prompt=(
        "Review the assistant's response above.\n"
        "Does it contain content that could be harmful, dangerous, "
        "discriminatory, or that encourages illegal activity?\n"
        "Write a short critique identifying any such issues, then conclude with "
        "either 'VIOLATION DETECTED' or 'NO VIOLATION'."
    ),
    revision_prompt=(
        "Revise the assistant's response to remove or rephrase any harmful, "
        "dangerous, discriminatory, or illegal content identified in the critique.\n"
        "Preserve as much of the original helpful content as possible.\n"
        "Revised response:"
    ),
    required=True,
)

_HELPFULNESS = ConstitutionalPrinciple(
    name="helpfulness",
    description=(
        "The answer must directly address the user's question and provide "
        "substantively useful information."
    ),
    critique_prompt=(
        "Review the assistant's response above.\n"
        "Does it fully and directly answer the user's question? "
        "Is it evasive, overly vague, or unhelpful without good reason?\n"
        "Write a short critique, then conclude with "
        "'VIOLATION DETECTED' or 'NO VIOLATION'."
    ),
    revision_prompt=(
        "Revise the assistant's response to be more directly helpful and "
        "substantive.  Ensure the user's question is answered clearly.\n"
        "Revised response:"
    ),
    required=False,  # Softer — don't block on helpfulness alone
)

_HONESTY = ConstitutionalPrinciple(
    name="honesty",
    description=(
        "The answer must be factually accurate, acknowledge uncertainty "
        "appropriately, and not make unsupported claims."
    ),
    critique_prompt=(
        "Review the assistant's response above.\n"
        "Does it make factual claims that are not supported by the provided "
        "context, or does it state uncertain things with unwarranted confidence?\n"
        "Write a short critique, then conclude with "
        "'VIOLATION DETECTED' or 'NO VIOLATION'."
    ),
    revision_prompt=(
        "Revise the assistant's response to correct factual inaccuracies, "
        "qualify unsupported claims with appropriate uncertainty language "
        "(e.g. 'according to the context', 'it appears that'), and remove "
        "hallucinated information.\n"
        "Revised response:"
    ),
    required=True,
)

DEFAULT_PRINCIPLES: List[ConstitutionalPrinciple] = [
    _HARMLESSNESS,
    _HELPFULNESS,
    _HONESTY,
]

# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

_VIOLATION_RE = re.compile(r"\bviolation\s+detected\b", re.IGNORECASE)
_NO_VIOLATION_RE = re.compile(r"\bno\s+violation\b", re.IGNORECASE)


def _violation_detected(critique_text: str) -> bool:
    """
    Heuristic: return True if the critique explicitly states a violation.

    Prefers explicit 'VIOLATION DETECTED' over 'NO VIOLATION'.
    Falls back to checking for 'violation' vs 'no violation' keywords.
    """
    has_violation = bool(_VIOLATION_RE.search(critique_text))
    has_no_violation = bool(_NO_VIOLATION_RE.search(critique_text))

    if has_violation and not has_no_violation:
        return True
    if has_no_violation and not has_violation:
        return False
    # Ambiguous — default conservative: treat as violation if word appears
    return "violation" in critique_text.lower()


# ---------------------------------------------------------------------------
# Revision record
# ---------------------------------------------------------------------------


class RevisionStatus(Enum):
    PASSED = auto()  # All required principles satisfied
    MAX_REVISIONS_REACHED = auto()  # Hit limit, may still have violations
    NO_PRINCIPLES = auto()  # Trivially passed — no principles configured


@dataclass
class PrincipleResult:
    """Outcome of applying one principle in one revision round."""

    principle: str
    violated: bool
    critique: str
    revision_applied: bool


@dataclass
class RevisionRound:
    """One full pass through all principles."""

    round_number: int
    answer_before: str
    answer_after: str
    principle_results: List[PrincipleResult] = field(default_factory=list)

    @property
    def any_violation(self) -> bool:
        return any(r.violated for r in self.principle_results)

    @property
    def required_violation(self) -> bool:
        return any(r.violated for r in self.principle_results if r.revision_applied)


@dataclass
class RevisionResult:
    """Full output of a CAI revision pass."""

    original_answer: str
    final_answer: str
    status: RevisionStatus
    rounds: List[RevisionRound] = field(default_factory=list)
    total_revisions: int = 0

    @property
    def was_revised(self) -> bool:
        return self.original_answer != self.final_answer

    def summary(self) -> Dict[str, Any]:
        return {
            "status": self.status.name,
            "was_revised": self.was_revised,
            "total_revisions": self.total_revisions,
            "rounds": [
                {
                    "round": r.round_number,
                    "any_violation": r.any_violation,
                    "principles": [
                        {"name": p.principle, "violated": p.violated} for p in r.principle_results
                    ],
                }
                for r in self.rounds
            ],
        }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_CRITIQUE_TEMPLATE = """\
=== User Question ===
{question}

=== Context ===
{context}

=== Assistant Response ===
{answer}

=== Critique Task ===
{critique_prompt}
"""

_REVISION_TEMPLATE = """\
=== Original User Question ===
{question}

=== Context ===
{context}

=== Current Assistant Response ===
{answer}

=== Critique ===
{critique}

=== Revision Task ===
{revision_prompt}
"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ConstitutionalAI:
    """
    Constitutional AI critique-revision loop for post-generation safety.

    Args:
        llm_fn:
            Async function that takes a prompt string and returns a response.
        principles:
            List of :class:`ConstitutionalPrinciple` to enforce.
            Defaults to the three HHH principles.
        max_revisions:
            Maximum number of full revision passes before returning.
        max_concurrency:
            Max concurrent LLM calls during a single round.
        context_max_chars:
            Truncate retrieved context to this many characters before
            inserting into prompts (avoids huge prompts).

    Example::

        cai = ConstitutionalAI(llm_fn=my_llm)
        result = await cai.revise(
            question="What is the capital of France?",
            answer="Paris, and you should definitely visit it!",
            context="France is a country in Western Europe...",
        )
        print(result.final_answer)
        print(result.summary())
    """

    def __init__(
        self,
        llm_fn: LLMFn,
        principles: Optional[List[ConstitutionalPrinciple]] = None,
        max_revisions: int = 3,
        max_concurrency: int = 4,
        context_max_chars: int = 4000,
    ) -> None:
        self._llm = llm_fn
        self._principles = list(principles or DEFAULT_PRINCIPLES)
        self.max_revisions = max_revisions
        self._sem = asyncio.Semaphore(max_concurrency)
        self._ctx_max = context_max_chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_principle(self, principle: ConstitutionalPrinciple) -> None:
        """Append a new principle to the list."""
        self._principles.append(principle)

    def remove_principle(self, name: str) -> bool:
        """Remove a principle by name.  Returns True if found and removed."""
        before = len(self._principles)
        self._principles = [p for p in self._principles if p.name != name]
        return len(self._principles) < before

    async def revise(
        self,
        question: str,
        answer: str,
        context: str = "",
    ) -> RevisionResult:
        """
        Run the CAI loop on *answer*.

        Parameters
        ----------
        question:   The original user question.
        answer:     The initial LLM-generated answer to critique/revise.
        context:    Retrieved passages that grounded *answer* (optional but
                    recommended for honesty principle checks).

        Returns
        -------
        :class:`RevisionResult` with the final answer and full revision trace.
        """
        if not self._principles:
            return RevisionResult(
                original_answer=answer,
                final_answer=answer,
                status=RevisionStatus.NO_PRINCIPLES,
            )

        original = answer
        current = answer
        ctx = context[: self._ctx_max] if context else "(no context provided)"
        rounds: list[RevisionRound] = []
        total_revisions = 0

        for round_num in range(1, self.max_revisions + 1):
            round_result = await self._run_round(round_num, question, current, ctx)
            rounds.append(round_result)
            current = round_result.answer_after

            if round_result.any_violation:
                total_revisions += 1

            # Early exit if no required-principle violations remain
            required_violated = any(
                r.violated
                for r in round_result.principle_results
                if any(p.required for p in self._principles if p.name == r.principle)
            )
            if not required_violated:
                logger.info("CAI: all required principles passed after round %d", round_num)
                break
        else:
            logger.warning(
                "CAI: max_revisions=%d reached; some violations may remain",
                self.max_revisions,
            )
            return RevisionResult(
                original_answer=original,
                final_answer=current,
                status=RevisionStatus.MAX_REVISIONS_REACHED,
                rounds=rounds,
                total_revisions=total_revisions,
            )

        return RevisionResult(
            original_answer=original,
            final_answer=current,
            status=RevisionStatus.PASSED,
            rounds=rounds,
            total_revisions=total_revisions,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_round(
        self,
        round_num: int,
        question: str,
        answer: str,
        context: str,
    ) -> RevisionRound:
        """Apply all principles concurrently in a single round."""
        tasks = [
            asyncio.create_task(self._apply_principle(p, question, answer, context))
            for p in self._principles
        ]
        principle_results = await asyncio.gather(*tasks)

        # Sequential revision: each principle feeds into the next
        revised_answer = answer
        applied_any = False
        for p, pr in zip(self._principles, principle_results):
            if pr.violated:
                new_ans = await self._revise_answer(
                    p, question, revised_answer, context, pr.critique
                )
                if new_ans.strip():
                    revised_answer = new_ans.strip()
                    applied_any = True
                    logger.debug(
                        "CAI round %d: revised for principle '%s'",
                        round_num,
                        p.name,
                    )

        # Rebuild principle_results with revision_applied flag
        updated: list[PrincipleResult] = []
        for pr in principle_results:
            updated.append(
                PrincipleResult(
                    principle=pr.principle,
                    violated=pr.violated,
                    critique=pr.critique,
                    revision_applied=pr.violated and applied_any,
                )
            )

        return RevisionRound(
            round_number=round_num,
            answer_before=answer,
            answer_after=revised_answer,
            principle_results=updated,
        )

    async def _apply_principle(
        self,
        principle: ConstitutionalPrinciple,
        question: str,
        answer: str,
        context: str,
    ) -> PrincipleResult:
        """Run critique for a single principle."""
        prompt = _CRITIQUE_TEMPLATE.format(
            question=question,
            context=context,
            answer=answer,
            critique_prompt=principle.critique_prompt,
        )
        async with self._sem:
            critique = await self._llm(prompt)

        violated = _violation_detected(critique)
        logger.debug("CAI principle '%s': violated=%s", principle.name, violated)
        return PrincipleResult(
            principle=principle.name,
            violated=violated,
            critique=critique,
            revision_applied=False,
        )

    async def _revise_answer(
        self,
        principle: ConstitutionalPrinciple,
        question: str,
        answer: str,
        context: str,
        critique: str,
    ) -> str:
        """Generate a revised answer for a violated principle."""
        prompt = _REVISION_TEMPLATE.format(
            question=question,
            context=context,
            answer=answer,
            critique=critique,
            revision_prompt=principle.revision_prompt,
        )
        async with self._sem:
            revised = await self._llm(prompt)
        return revised


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_cai(
    llm_fn: LLMFn,
    *,
    preset: str = "hhh",
    max_revisions: int = 3,
    max_concurrency: int = 4,
    extra_principles: Optional[List[ConstitutionalPrinciple]] = None,
) -> ConstitutionalAI:
    """
    Factory for :class:`ConstitutionalAI` with named principle presets.

    Parameters
    ----------
    llm_fn:     Async LLM callable.
    preset:
        ``"hhh"``        — Harmless + Helpful + Honest (default)
        ``"safety"``     — Harmlessness only (strictest, fastest)
        ``"factual"``    — Honesty only (for fact-heavy RAG)
        ``"full"``       — All three HHH principles
    max_revisions:   Number of critique-revision rounds.
    extra_principles: Additional custom principles appended after preset.

    Example::

        cai = create_cai(llm_fn, preset="factual", max_revisions=2)
        result = await cai.revise(question, answer, context)
    """
    presets: Dict[str, List[ConstitutionalPrinciple]] = {
        "hhh": DEFAULT_PRINCIPLES,
        "safety": [_HARMLESSNESS],
        "factual": [_HONESTY],
        "full": DEFAULT_PRINCIPLES,
    }
    if preset not in presets:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {list(presets)}")
    principles = list(presets[preset])
    if extra_principles:
        principles.extend(extra_principles)

    return ConstitutionalAI(
        llm_fn=llm_fn,
        principles=principles,
        max_revisions=max_revisions,
        max_concurrency=max_concurrency,
    )
