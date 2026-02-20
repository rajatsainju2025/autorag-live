"""
Chain-of-Verification (CoVe) for Claim-Level Hallucination Reduction.

Implements the CoVe framework from Dhuliawala et al. (2023), adapted for
agentic RAG pipelines.  CoVe decomposes an initial answer into atomic claims,
generates independent verification questions for each claim, retrieves evidence
and verifies each question independently (to avoid follow-on hallucination),
and then produces a revised answer that corrects any failed verifications.

Why CoVe for RAG?
-----------------
Even when an LLM answer is grounded in retrieved passages, the synthesis step
can introduce factual distortions: misquoted numbers, swapped entities, or
over-generalised statements.  CoVe treats each atomic claim as an independent
verification sub-task, preventing the hallucination cascade where one false
claim makes subsequent claims more likely to be wrong.

Pipeline Stages
---------------
1. **Baseline generation** — Produce an initial answer (or accept one as input)
2. **Claim decomposition** — Extract N atomic, independently verifiable claims
3. **Question generation** — Generate one verification question per claim
4. **Independent verification** — Execute each question with no shared context
   (uses async concurrent retrieval + LLM verification)
5. **Revision** — Rewrite the answer incorporating only verified facts;
   flag or remove unverified/false claims

References
----------
- Dhuliawala et al. 2023. "Chain-of-Verification Reduces Hallucination in
  Large Language Models." https://arxiv.org/abs/2309.11495

Features
--------
- Fully async, semaphore-bounded throughout
- Pluggable ``LLMFn`` and optional ``RetrievalFn`` for evidence lookup
- Configurable maximum claims to verify (cost control)
- Returns :class:`CoVeResult` with per-claim verification trace
- ``create_cove()`` factory with preset configurations
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
"""Async function: prompt → LLM response text."""

RetrievalFn = Callable[[str], Coroutine[Any, Any, List[Dict[str, Any]]]]
"""Async function: query → list of retrieved doc dicts with 'text' key."""


# ---------------------------------------------------------------------------
# Claim verification states
# ---------------------------------------------------------------------------


class VerificationStatus(Enum):
    VERIFIED = auto()  # Claim confirmed by evidence
    REFUTED = auto()  # Claim contradicted by evidence
    UNVERIFIABLE = auto()  # Evidence insufficient to decide
    SKIPPED = auto()  # Claim could not be parsed/extracted


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class AtomicClaim:
    """A single atomic, independently verifiable claim extracted from an answer."""

    index: int
    text: str
    verification_question: str = ""


@dataclass
class ClaimVerification:
    """Verification result for one atomic claim."""

    claim: AtomicClaim
    evidence_passages: List[str] = field(default_factory=list)
    verification_response: str = ""
    status: VerificationStatus = VerificationStatus.UNVERIFIABLE
    confidence: float = 0.5  # 0.0–1.0

    @property
    def verified(self) -> bool:
        return self.status == VerificationStatus.VERIFIED

    @property
    def refuted(self) -> bool:
        return self.status == VerificationStatus.REFUTED


@dataclass
class CoVeResult:
    """Full output of a Chain-of-Verification pass."""

    original_question: str
    baseline_answer: str
    final_answer: str
    claims: List[AtomicClaim] = field(default_factory=list)
    verifications: List[ClaimVerification] = field(default_factory=list)
    was_revised: bool = False

    @property
    def verified_count(self) -> int:
        return sum(1 for v in self.verifications if v.verified)

    @property
    def refuted_count(self) -> int:
        return sum(1 for v in self.verifications if v.refuted)

    @property
    def unverifiable_count(self) -> int:
        return sum(1 for v in self.verifications if v.status == VerificationStatus.UNVERIFIABLE)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_claims": len(self.claims),
            "verified": self.verified_count,
            "refuted": self.refuted_count,
            "unverifiable": self.unverifiable_count,
            "was_revised": self.was_revised,
            "claims": [
                {
                    "index": v.claim.index,
                    "claim": v.claim.text,
                    "question": v.claim.verification_question,
                    "status": v.status.name,
                    "confidence": v.confidence,
                }
                for v in self.verifications
            ],
        }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """\
You are given a question and an answer.  Extract all atomic, independently
verifiable factual claims from the answer.  A claim is atomic if it asserts
exactly one fact that can be checked independently.

Format each claim on a new line starting with "CLAIM N:" where N is the
claim number (starting from 1).

=== Question ===
{question}

=== Answer ===
{answer}

=== Atomic Claims ===
"""

_QUESTION_GEN_PROMPT = """\
Convert the following factual claim into a single, clear, self-contained
verification question that can be answered independently without seeing the
original answer.  The question should target the specific fact asserted.

Claim: {claim}

Verification question:"""

_VERIFY_PROMPT = """\
Answer the following question using ONLY the evidence passages below.
If the evidence supports the claim, write "VERIFIED: <explanation>".
If the evidence contradicts the claim, write "REFUTED: <explanation>".
If the evidence is insufficient, write "UNVERIFIABLE: <explanation>".

=== Question ===
{question}

=== Evidence ===
{evidence}

=== Verdict ===
"""

_REVISE_PROMPT = """\
You are revising an answer based on claim verification results.

=== Original Question ===
{question}

=== Original Answer ===
{answer}

=== Verification Results ===
{verification_summary}

Instructions:
- Keep all VERIFIED claims exactly as stated.
- Remove or qualify any REFUTED claims (prefix with "However, it should be
  noted that...").
- Mark UNVERIFIABLE claims with appropriate hedging ("It is unclear whether...").
- Do NOT introduce any new factual claims not present in the original.
- The revised answer must remain fluent and coherent.

=== Revised Answer ===
"""

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_CLAIM_RE = re.compile(r"CLAIM\s+(\d+)\s*:\s*(.+)", re.IGNORECASE)
_VERIFIED_RE = re.compile(r"^\s*VERIFIED\s*:", re.IGNORECASE | re.MULTILINE)
_REFUTED_RE = re.compile(r"^\s*REFUTED\s*:", re.IGNORECASE | re.MULTILINE)
_CONFIDENCE_RE = re.compile(r"confidence\s*[:=]\s*([0-9.]+)", re.IGNORECASE)


def _parse_claims(llm_output: str) -> List[str]:
    """Extract claim texts from LLM decomposition output."""
    claims: list[str] = []
    for m in _CLAIM_RE.finditer(llm_output):
        text = m.group(2).strip()
        if text:
            claims.append(text)
    # Fallback: numbered list lines
    if not claims:
        for line in llm_output.splitlines():
            line = line.strip()
            m2 = re.match(r"^\d+[\.\)]\s+(.+)", line)
            if m2:
                claims.append(m2.group(1))
    return claims


def _parse_verification_status(text: str) -> VerificationStatus:
    if _VERIFIED_RE.search(text):
        return VerificationStatus.VERIFIED
    if _REFUTED_RE.search(text):
        return VerificationStatus.REFUTED
    return VerificationStatus.UNVERIFIABLE


def _parse_confidence(text: str, status: VerificationStatus) -> float:
    m = _CONFIDENCE_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Default confidence by status
    defaults = {
        VerificationStatus.VERIFIED: 0.85,
        VerificationStatus.REFUTED: 0.80,
        VerificationStatus.UNVERIFIABLE: 0.50,
    }
    return defaults.get(status, 0.5)


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------


class CoVeAgent:
    """
    Chain-of-Verification agent for claim-level hallucination reduction.

    Args:
        llm_fn:
            Async callable used for all LLM calls
            (decomposition, question generation, verification, revision).
        retrieval_fn:
            Optional async callable that retrieves evidence for a verification
            question.  If ``None``, the agent uses the original context only.
        max_claims:
            Maximum number of claims to verify per answer (cost control).
        max_concurrency:
            Semaphore bound on concurrent LLM/retrieval calls.
        context_max_chars:
            Maximum chars of original context to include in evidence.
        revise_threshold:
            Fraction of claims that must be REFUTED to trigger revision.
            Default 0.0 means revise if *any* claim is refuted.

    Example::

        cove = CoVeAgent(llm_fn=my_llm, retrieval_fn=my_retriever)
        result = await cove.verify(
            question="Who founded OpenAI?",
            answer="OpenAI was founded in 2015 by Elon Musk and Sam Altman.",
            context="...retrieved passages...",
        )
        print(result.final_answer)
        print(result.summary())
    """

    def __init__(
        self,
        llm_fn: LLMFn,
        retrieval_fn: Optional[RetrievalFn] = None,
        max_claims: int = 10,
        max_concurrency: int = 6,
        context_max_chars: int = 3000,
        revise_threshold: float = 0.0,
    ) -> None:
        self._llm = llm_fn
        self._retrieval = retrieval_fn
        self.max_claims = max_claims
        self._sem = asyncio.Semaphore(max_concurrency)
        self._ctx_max = context_max_chars
        self.revise_threshold = revise_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify(
        self,
        question: str,
        answer: str,
        context: str = "",
    ) -> CoVeResult:
        """
        Run CoVe on *answer*.

        Parameters
        ----------
        question:   The original user question.
        answer:     The LLM-generated answer to verify.
        context:    Retrieved passages used to generate the answer (used as
                    fallback evidence when no retrieval_fn is provided).

        Returns
        -------
        :class:`CoVeResult` with final (possibly revised) answer and full trace.
        """
        # Stage 1: Decompose into atomic claims
        claims = await self._decompose(question, answer)
        if not claims:
            logger.warning("CoVe: no claims extracted — returning original answer")
            return CoVeResult(
                original_question=question,
                baseline_answer=answer,
                final_answer=answer,
            )

        # Stage 2: Generate verification questions concurrently
        claims = await self._generate_questions(claims)

        # Stage 3: Verify each claim independently
        ctx_truncated = context[: self._ctx_max] if context else ""
        verifications = await self._verify_all(claims, ctx_truncated)

        # Stage 4: Decide whether to revise
        n_refuted = sum(1 for v in verifications if v.refuted)
        refuted_fraction = n_refuted / len(verifications) if verifications else 0.0
        needs_revision = refuted_fraction > self.revise_threshold

        final_answer = answer
        if needs_revision:
            logger.info(
                "CoVe: %d/%d claims refuted — triggering revision",
                n_refuted,
                len(verifications),
            )
            final_answer = await self._revise(question, answer, verifications)

        return CoVeResult(
            original_question=question,
            baseline_answer=answer,
            final_answer=final_answer,
            claims=claims,
            verifications=verifications,
            was_revised=needs_revision,
        )

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    async def _decompose(self, question: str, answer: str) -> List[AtomicClaim]:
        """Stage 1: Extract atomic claims from the answer."""
        prompt = _DECOMPOSE_PROMPT.format(question=question, answer=answer)
        async with self._sem:
            output = await self._llm(prompt)

        raw_claims = _parse_claims(output)
        raw_claims = raw_claims[: self.max_claims]

        atomic: list[AtomicClaim] = []
        for i, text in enumerate(raw_claims, start=1):
            atomic.append(AtomicClaim(index=i, text=text))

        logger.debug("CoVe: extracted %d claims", len(atomic))
        return atomic

    async def _generate_questions(self, claims: List[AtomicClaim]) -> List[AtomicClaim]:
        """Stage 2: Generate verification questions for all claims concurrently."""
        tasks = [asyncio.create_task(self._gen_question(claim)) for claim in claims]
        updated = await asyncio.gather(*tasks)
        return list(updated)

    async def _gen_question(self, claim: AtomicClaim) -> AtomicClaim:
        prompt = _QUESTION_GEN_PROMPT.format(claim=claim.text)
        async with self._sem:
            question = await self._llm(prompt)
        claim.verification_question = question.strip()
        return claim

    async def _verify_all(
        self,
        claims: List[AtomicClaim],
        fallback_context: str,
    ) -> List[ClaimVerification]:
        """Stage 3: Verify all claims independently and concurrently."""
        tasks = [
            asyncio.create_task(self._verify_claim(claim, fallback_context)) for claim in claims
        ]
        return list(await asyncio.gather(*tasks))

    async def _verify_claim(
        self,
        claim: AtomicClaim,
        fallback_context: str,
    ) -> ClaimVerification:
        """Verify a single claim: retrieve evidence, then ask LLM."""
        # Retrieve evidence
        evidence_passages: list[str] = []
        if self._retrieval and claim.verification_question:
            try:
                async with self._sem:
                    docs = await self._retrieval(claim.verification_question)
                evidence_passages = [d.get("text", "") for d in docs if d.get("text")]
            except Exception as exc:  # noqa: BLE001
                logger.warning("CoVe retrieval failed for claim %d: %s", claim.index, exc)

        if not evidence_passages and fallback_context:
            evidence_passages = [fallback_context]

        evidence_str = (
            "\n\n---\n\n".join(evidence_passages[:5])
            if evidence_passages
            else "(no evidence available)"
        )

        # Verify against evidence
        prompt = _VERIFY_PROMPT.format(
            question=claim.verification_question or claim.text,
            evidence=evidence_str,
        )
        async with self._sem:
            response = await self._llm(prompt)

        status = _parse_verification_status(response)
        confidence = _parse_confidence(response, status)

        return ClaimVerification(
            claim=claim,
            evidence_passages=evidence_passages,
            verification_response=response,
            status=status,
            confidence=confidence,
        )

    async def _revise(
        self,
        question: str,
        answer: str,
        verifications: List[ClaimVerification],
    ) -> str:
        """Stage 4: Revise the answer based on verification results."""
        lines: list[str] = []
        for v in verifications:
            lines.append(
                f"CLAIM {v.claim.index}: {v.claim.text}\n"
                f"  STATUS: {v.status.name} (confidence={v.confidence:.2f})\n"
                f"  VERDICT: {v.verification_response[:200]}"
            )
        verification_summary = "\n\n".join(lines)

        prompt = _REVISE_PROMPT.format(
            question=question,
            answer=answer,
            verification_summary=verification_summary,
        )
        async with self._sem:
            revised = await self._llm(prompt)
        return revised.strip() or answer  # fallback to original if empty


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_cove(
    llm_fn: LLMFn,
    *,
    retrieval_fn: Optional[RetrievalFn] = None,
    max_claims: int = 10,
    max_concurrency: int = 6,
    revise_threshold: float = 0.0,
) -> CoVeAgent:
    """
    Factory for :class:`CoVeAgent` with keyword-only configuration.

    Args:
        llm_fn:           Async LLM callable.
        retrieval_fn:     Optional async retrieval callable for evidence.
        max_claims:       Cap on number of claims to verify (cost control).
        max_concurrency:  Concurrent LLM / retrieval calls.
        revise_threshold: Fraction of refuted claims needed to trigger revision.
                          ``0.0`` = revise on any single refuted claim.

    Example::

        cove = create_cove(llm_fn=my_llm, retrieval_fn=my_retriever, max_claims=8)
        result = await cove.verify(question, answer, context)
        if result.was_revised:
            print("Answer was corrected:", result.final_answer)
    """
    return CoVeAgent(
        llm_fn=llm_fn,
        retrieval_fn=retrieval_fn,
        max_claims=max_claims,
        max_concurrency=max_concurrency,
        revise_threshold=revise_threshold,
    )
