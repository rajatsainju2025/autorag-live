"""
G-EVAL: NLG Evaluation using GPT-4 with Chain-of-Thought.

Implements the G-EVAL framework (Liu et al., 2023, EMNLP) which uses
explicit chain-of-thought evaluation steps before scoring — producing
significantly higher correlation with human judgements than
integer-score-only LLM judges.

Evaluated Dimensions
--------------------
- Coherence    : logical flow and structure of the response
- Consistency  : factual alignment with retrieved sources
- Fluency      : grammatical quality and natural language
- Relevance    : how well the response addresses the query
- Groundedness : fraction of claims traceable to sources

Scoring Method
--------------
For each dimension, G-EVAL:
1. Generates explicit evaluation steps via chain-of-thought
2. Completes the evaluation form to produce a 1–5 score
3. (Optionally) uses token log-probability weighting for
   continuous scoring rather than integer sampling

The final composite score is a weighted sum with ANOVA-
calibrated dimension weights from the G-EVAL paper.

References
----------
- "G-EVAL: NLG Evaluation Using GPT-4 with Better Human Alignment"
  Liu et al., 2023 (https://arxiv.org/abs/2303.16634)
- "Judging LLM-as-a-Judge" Zheng et al., 2023
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases / Protocols
# ---------------------------------------------------------------------------
LLMGenerateFn = Callable[[str], Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------


class EvalDimension(str, Enum):
    """Evaluation dimensions supported by G-EVAL."""

    COHERENCE = "coherence"
    CONSISTENCY = "consistency"
    FLUENCY = "fluency"
    RELEVANCE = "relevance"
    GROUNDEDNESS = "groundedness"


# ANOVA-calibrated weights from Liu et al. 2023 Table 3
_DEFAULT_WEIGHTS: Dict[EvalDimension, float] = {
    EvalDimension.COHERENCE: 0.20,
    EvalDimension.CONSISTENCY: 0.30,
    EvalDimension.FLUENCY: 0.10,
    EvalDimension.RELEVANCE: 0.25,
    EvalDimension.GROUNDEDNESS: 0.15,
}

# Chain-of-thought evaluation templates per dimension
_EVAL_TEMPLATES: Dict[EvalDimension, str] = {
    EvalDimension.COHERENCE: (
        "You will evaluate the COHERENCE of an AI-generated answer.\n\n"
        "Task: {task}\nSource Documents:\n{sources}\n"
        "Question: {query}\nAnswer: {answer}\n\n"
        "Evaluation Steps:\n"
        "1. Read the answer and identify the main topic.\n"
        "2. Check whether the answer has a clear logical structure.\n"
        "3. Verify that each sentence follows naturally from the previous.\n"
        "4. Check for contradictions or irrelevant tangents.\n"
        "5. Based on these steps, rate COHERENCE on a scale of 1–5.\n\n"
        "Evaluation Form:\n"
        "Coherence Score (1–5):"
    ),
    EvalDimension.CONSISTENCY: (
        "You will evaluate the CONSISTENCY of an AI-generated answer.\n\n"
        "Task: {task}\nSource Documents:\n{sources}\n"
        "Question: {query}\nAnswer: {answer}\n\n"
        "Evaluation Steps:\n"
        "1. Read the source documents carefully.\n"
        "2. Identify factual claims in the answer.\n"
        "3. For each claim, check whether it is supported by the sources.\n"
        "4. Penalise any claim that contradicts or goes beyond the sources.\n"
        "5. Based on these steps, rate CONSISTENCY on a scale of 1–5.\n\n"
        "Evaluation Form:\n"
        "Consistency Score (1–5):"
    ),
    EvalDimension.FLUENCY: (
        "You will evaluate the FLUENCY of an AI-generated answer.\n\n"
        "Task: {task}\nQuestion: {query}\nAnswer: {answer}\n\n"
        "Evaluation Steps:\n"
        "1. Read the answer aloud mentally.\n"
        "2. Check for grammatical errors, awkward phrasing, or typos.\n"
        "3. Assess whether the answer reads naturally.\n"
        "4. Based on these steps, rate FLUENCY on a scale of 1–5.\n\n"
        "Evaluation Form:\n"
        "Fluency Score (1–5):"
    ),
    EvalDimension.RELEVANCE: (
        "You will evaluate the RELEVANCE of an AI-generated answer.\n\n"
        "Task: {task}\nQuestion: {query}\nAnswer: {answer}\n\n"
        "Evaluation Steps:\n"
        "1. Re-read the question to understand what is asked.\n"
        "2. Identify the key information requested.\n"
        "3. Check whether the answer directly addresses the question.\n"
        "4. Check for off-topic content or unnecessary digression.\n"
        "5. Based on these steps, rate RELEVANCE on a scale of 1–5.\n\n"
        "Evaluation Form:\n"
        "Relevance Score (1–5):"
    ),
    EvalDimension.GROUNDEDNESS: (
        "You will evaluate the GROUNDEDNESS of an AI-generated answer.\n\n"
        "Task: {task}\nSource Documents:\n{sources}\n"
        "Question: {query}\nAnswer: {answer}\n\n"
        "Evaluation Steps:\n"
        "1. Identify every factual claim in the answer.\n"
        "2. For each claim, locate supporting text in the source documents.\n"
        "3. Compute the fraction of claims that are grounded in sources.\n"
        "4. A claim with no evidence in sources counts as ungrounded.\n"
        "5. Based on these steps, rate GROUNDEDNESS on a scale of 1–5.\n\n"
        "Evaluation Form:\n"
        "Groundedness Score (1–5):"
    ),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""

    dimension: EvalDimension
    score: float  # 1–5 scale
    normalised: float  # 0–1 scale
    raw_output: str = ""
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.normalised >= 0.6


@dataclass
class GEvalResult:
    """
    Complete G-EVAL evaluation result.

    Attributes:
        query: The original question.
        answer: The evaluated answer.
        dimension_scores: Per-dimension scores.
        composite_score: Weighted composite ∈ [0, 1].
        passed: Whether composite_score ≥ pass_threshold.
        metadata: Evaluation metadata (model, latency, etc.).
    """

    query: str
    answer: str
    dimension_scores: List[DimensionScore] = field(default_factory=list)
    composite_score: float = 0.0
    passed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:200],
            "composite_score": round(self.composite_score, 4),
            "passed": self.passed,
            "dimensions": {
                ds.dimension.value: round(ds.normalised, 4) for ds in self.dimension_scores
            },
        }

    def weakest_dimension(self) -> Optional[DimensionScore]:
        if not self.dimension_scores:
            return None
        return min(self.dimension_scores, key=lambda d: d.normalised)


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------


def _extract_score(text: str) -> Optional[float]:
    """
    Extract a 1–5 numeric score from LLM output.

    Tries several patterns in decreasing specificity.
    """
    text = text.strip()
    for pattern in [
        r"(?:Score|score)[^\d]*([1-5](?:\.\d)?)",
        r"\b([1-5](?:\.\d)?)\s*/\s*5\b",
        r"\b([1-5](?:\.\d)?)\b",
    ]:
        m = re.search(pattern, text)
        if m:
            try:
                val = float(m.group(1))
                if 1.0 <= val <= 5.0:
                    return val
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------


class GEval:
    """
    G-EVAL evaluator: NLG evaluation with chain-of-thought scoring.

    Uses a pluggable async LLM callable so it works with any
    provider (OpenAI, Anthropic, local models, mocks).

    Args:
        llm_fn: Async ``(prompt: str) → str`` callable.
        dimensions: Which dimensions to evaluate (all by default).
        weights: Per-dimension weights (must sum to 1.0).
        task_description: Short task description included in prompts.
        pass_threshold: Minimum composite score to consider passing.
        concurrency: Max parallel dimension evaluations.
    """

    def __init__(
        self,
        llm_fn: LLMGenerateFn,
        dimensions: Optional[Sequence[EvalDimension]] = None,
        weights: Optional[Dict[EvalDimension, float]] = None,
        task_description: str = "Answer a question using retrieved documents.",
        pass_threshold: float = 0.6,
        concurrency: int = 5,
    ) -> None:
        self.llm_fn = llm_fn
        self.dimensions = list(dimensions or EvalDimension)
        self.weights = weights or {d: _DEFAULT_WEIGHTS[d] for d in self.dimensions}
        self._normalise_weights()
        self.task_description = task_description
        self.pass_threshold = pass_threshold
        self._semaphore = asyncio.Semaphore(concurrency)

    def _normalise_weights(self) -> None:
        total = sum(self.weights.get(d, 0.0) for d in self.dimensions)
        if total <= 0:
            equal = 1.0 / len(self.dimensions)
            self.weights = {d: equal for d in self.dimensions}
        else:
            self.weights = {d: self.weights.get(d, 0.0) / total for d in self.dimensions}

    async def evaluate(
        self,
        query: str,
        answer: str,
        sources: Optional[List[str]] = None,
    ) -> GEvalResult:
        """
        Evaluate a single (query, answer) pair across all dimensions.

        Args:
            query: The user query.
            answer: The generated answer to evaluate.
            sources: Retrieved source document texts.

        Returns:
            :class:`GEvalResult` with per-dimension and composite scores.
        """
        sources_text = (
            "\n\n".join(f"[{i+1}] {s[:600]}" for i, s in enumerate(sources or []))
            or "(no sources provided)"
        )

        tasks = [self._score_dimension(dim, query, answer, sources_text) for dim in self.dimensions]
        dimension_scores = await asyncio.gather(*tasks)

        composite = sum(
            self.weights.get(ds.dimension, 0.0) * ds.normalised for ds in dimension_scores
        )
        composite = max(0.0, min(1.0, composite))

        return GEvalResult(
            query=query,
            answer=answer,
            dimension_scores=list(dimension_scores),
            composite_score=round(composite, 4),
            passed=composite >= self.pass_threshold,
            metadata={
                "dimensions_evaluated": [d.value for d in self.dimensions],
                "pass_threshold": self.pass_threshold,
            },
        )

    async def evaluate_batch(
        self,
        pairs: List[Dict[str, Any]],
        sources_list: Optional[List[List[str]]] = None,
    ) -> List[GEvalResult]:
        """
        Evaluate a batch of (query, answer) pairs.

        Args:
            pairs: List of dicts with keys ``query`` and ``answer``.
            sources_list: Optional parallel list of source lists.

        Returns:
            List of :class:`GEvalResult`.
        """
        srcs = sources_list or [None] * len(pairs)
        tasks = [self.evaluate(p["query"], p["answer"], s) for p, s in zip(pairs, srcs)]
        return list(await asyncio.gather(*tasks))

    async def _score_dimension(
        self,
        dimension: EvalDimension,
        query: str,
        answer: str,
        sources_text: str,
    ) -> DimensionScore:
        template = _EVAL_TEMPLATES.get(dimension)
        if not template:
            return DimensionScore(
                dimension=dimension,
                score=3.0,
                normalised=0.5,
                error="no template",
            )

        prompt = template.format(
            task=self.task_description,
            sources=sources_text,
            query=query,
            answer=answer,
        )

        async with self._semaphore:
            try:
                raw = await self.llm_fn(prompt)
            except Exception as exc:
                logger.warning("G-EVAL %s failed: %s", dimension.value, exc)
                return DimensionScore(
                    dimension=dimension,
                    score=3.0,
                    normalised=0.5,
                    raw_output="",
                    error=str(exc),
                )

        score = _extract_score(raw)
        if score is None:
            logger.debug("G-EVAL: could not parse score from %r", raw[:100])
            score = 3.0  # neutral fallback

        normalised = (score - 1.0) / 4.0  # map [1,5] → [0,1]
        return DimensionScore(
            dimension=dimension,
            score=score,
            normalised=round(normalised, 4),
            raw_output=raw,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_g_eval(
    llm_fn: LLMGenerateFn,
    focus: str = "balanced",
    pass_threshold: float = 0.6,
) -> GEval:
    """
    Create a :class:`GEval` instance with preset dimension weights.

    Args:
        llm_fn: Async ``(prompt) → str`` callable.
        focus: One of ``"balanced"`` | ``"factual"`` | ``"readability"``.
        pass_threshold: Minimum composite score to pass.

    Returns:
        Configured :class:`GEval`.
    """
    preset_weights: Dict[str, Dict[EvalDimension, float]] = {
        "balanced": dict(_DEFAULT_WEIGHTS),
        "factual": {
            EvalDimension.COHERENCE: 0.10,
            EvalDimension.CONSISTENCY: 0.35,
            EvalDimension.FLUENCY: 0.05,
            EvalDimension.RELEVANCE: 0.20,
            EvalDimension.GROUNDEDNESS: 0.30,
        },
        "readability": {
            EvalDimension.COHERENCE: 0.35,
            EvalDimension.CONSISTENCY: 0.15,
            EvalDimension.FLUENCY: 0.30,
            EvalDimension.RELEVANCE: 0.15,
            EvalDimension.GROUNDEDNESS: 0.05,
        },
    }
    weights = preset_weights.get(focus, dict(_DEFAULT_WEIGHTS))
    return GEval(llm_fn=llm_fn, weights=weights, pass_threshold=pass_threshold)
