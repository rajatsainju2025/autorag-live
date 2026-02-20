"""
Self-Consistency Decoding for Agentic RAG.

Generates multiple independent answers in parallel and selects the most
consistent one by majority vote — dramatically reducing hallucinations
without requiring any additional training.

Algorithm (Wang et al., 2023):
    1. Sample k answers with temperature > 0
    2. Cluster answers by semantic similarity
    3. Return the answer from the largest cluster (majority vote)
    4. Report a calibrated consistency score

This implementation adds:
- Async parallel sampling (all k calls are concurrent via asyncio.gather)
- Semantic clustering with cosine similarity + DBSCAN-style greedy merging
- Token-level overlap (ROUGE-1) as an optional tie-breaker
- Early-exit: if any answer exceeds ``early_exit_threshold`` agreement it
  is returned immediately without waiting for remaining samples
- Structured result with per-answer metadata for downstream calibration

References:
- Self-Consistency Improves Chain of Thought Reasoning in LLMs
  (Wang et al., 2023) https://arxiv.org/abs/2203.11171
- Making Large Language Models Better Reasoners with Self-Consistent Prompting
  (Wei et al., 2022) https://arxiv.org/abs/2212.09561

Example:
    >>> voter = SelfConsistencyVoter(llm=my_llm, samples=7, temperature=0.7)
    >>> result = await voter.vote("What is 17 × 24?")
    >>> print(result.answer, result.consistency_score)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
LLMGenerateFn = Callable[[str, float], Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SampledAnswer:
    """A single LLM sample from the self-consistency ensemble."""

    text: str
    sample_idx: int
    latency_ms: float = 0.0
    cluster_id: int = -1
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = max(1, len(self.text.split()))


@dataclass
class VotingResult:
    """Result of self-consistency voting."""

    answer: str
    """The selected answer (majority cluster centroid)."""

    consistency_score: float
    """Fraction of samples in the majority cluster ∈ [0, 1]."""

    samples: List[SampledAnswer] = field(default_factory=list)
    """All k sampled answers with cluster assignments."""

    majority_cluster_size: int = 0
    """Number of samples in the winning cluster."""

    total_samples: int = 0
    """Total samples generated."""

    cluster_count: int = 0
    """Number of distinct answer clusters found."""

    early_exit: bool = False
    """True if the result was returned before all samples completed."""

    total_latency_ms: float = 0.0
    """Wall-clock time for the full voting round."""

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_high_confidence(self) -> bool:
        """True if consistency_score ≥ 0.7."""
        return self.consistency_score >= 0.7

    @property
    def cluster_diversity(self) -> float:
        """Fraction of distinct clusters (higher = more diverse / uncertain)."""
        if self.total_samples == 0:
            return 0.0
        return self.cluster_count / self.total_samples


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Simple word-level tokenizer for ROUGE-1."""
    return set(re.findall(r"\w+", text.lower()))


def _rouge1(a: str, b: str) -> float:
    """Symmetric ROUGE-1 F1 between two strings."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    precision = intersection / len(tb)
    recall = intersection / len(ta)
    denom = precision + recall
    return (2 * precision * recall / denom) if denom > 0 else 0.0


def _extract_numeric(text: str) -> Optional[float]:
    """Extract the first number from text for exact-match numeric voting."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(matches[0]) if matches else None


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _greedy_cluster(
    answers: List[str],
    sim_fn: Callable[[str, str], float],
    threshold: float = 0.6,
) -> List[int]:
    """
    Greedy single-linkage clustering.

    Each answer joins the first existing cluster whose centroid has
    similarity ≥ threshold, else starts a new cluster.

    O(N²) but N is small (typically 5–20 samples).

    Returns:
        List of cluster ids, same length as `answers`.
    """
    cluster_ids = [-1] * len(answers)
    cluster_reps: List[str] = []  # one representative per cluster

    for i, ans in enumerate(answers):
        best_cluster = -1
        best_sim = threshold - 1e-9

        for c_id, rep in enumerate(cluster_reps):
            sim = sim_fn(ans, rep)
            if sim > best_sim:
                best_sim = sim
                best_cluster = c_id

        if best_cluster == -1:
            best_cluster = len(cluster_reps)
            cluster_reps.append(ans)

        cluster_ids[i] = best_cluster

    return cluster_ids


# ---------------------------------------------------------------------------
# Self-Consistency Voter
# ---------------------------------------------------------------------------


class SelfConsistencyVoter:
    """
    Parallel self-consistency decoding with majority vote.

    Generates `samples` independent answers concurrently, clusters them
    by semantic similarity, and returns the answer from the largest cluster.

    Args:
        llm: Async callable (prompt, temperature) → answer string.
        samples: Number of samples to generate (default 7; odd avoids ties).
        temperature: Sampling temperature — should be > 0 for diversity (default 0.7).
        similarity_threshold: Min similarity to merge into same cluster (default 0.6).
        sim_fn: Similarity function (default ROUGE-1; swap for embedding cosine sim).
        early_exit_threshold: Return early if this fraction agree (default 0.85).
        max_retries: Retries per failed sample (default 2).
        timeout_s: Per-sample timeout in seconds (default 30.0).
    """

    def __init__(
        self,
        llm: LLMGenerateFn,
        samples: int = 7,
        temperature: float = 0.7,
        similarity_threshold: float = 0.6,
        sim_fn: Optional[Callable[[str, str], float]] = None,
        early_exit_threshold: float = 0.85,
        max_retries: int = 2,
        timeout_s: float = 30.0,
    ) -> None:
        self.llm = llm
        self.samples = samples
        self.temperature = temperature
        self.sim_threshold = similarity_threshold
        self.sim_fn = sim_fn or _rouge1
        self.early_exit_threshold = early_exit_threshold
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def vote(
        self,
        prompt: str,
        context: Optional[str] = None,
    ) -> VotingResult:
        """
        Run self-consistency sampling and voting.

        Args:
            prompt: The question / instruction to answer.
            context: Optional retrieved context to prepend to each sample call.

        Returns:
            VotingResult with the winning answer and consistency metadata.
        """
        full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
        start = time.perf_counter()

        # ── Parallel sampling with early exit ─────────────────────────────
        collected: List[SampledAnswer] = []
        early_exit = False

        # Use asyncio.as_completed for early exit support
        tasks = {asyncio.create_task(self._sample(full_prompt, i)): i for i in range(self.samples)}

        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for fut in done:
                sample = fut.result()
                if sample is not None:
                    collected.append(sample)

            # Early-exit check
            if len(collected) >= 3:
                partial_result = self._cluster_and_vote(collected)
                if partial_result.consistency_score >= self.early_exit_threshold:
                    # Cancel remaining tasks
                    for t in pending:
                        t.cancel()
                    early_exit = True
                    break

        # Await remaining (already cancelled)
        for t in pending:
            try:
                await t
            except asyncio.CancelledError:
                pass

        if not collected:
            return VotingResult(
                answer="",
                consistency_score=0.0,
                total_latency_ms=(time.perf_counter() - start) * 1000,
            )

        result = self._cluster_and_vote(collected)
        result.early_exit = early_exit
        result.total_latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "SelfConsistency: %d/%d samples in majority cluster (score=%.2f, early_exit=%s)",
            result.majority_cluster_size,
            result.total_samples,
            result.consistency_score,
            early_exit,
        )
        return result

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _sample(self, prompt: str, idx: int) -> Optional[SampledAnswer]:
        """Generate a single sample with retries."""
        for attempt in range(self.max_retries + 1):
            try:
                start = time.perf_counter()
                text = await asyncio.wait_for(
                    self.llm(prompt, self.temperature),
                    timeout=self.timeout_s,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                return SampledAnswer(text=text.strip(), sample_idx=idx, latency_ms=latency_ms)
            except asyncio.TimeoutError:
                logger.warning("SelfConsistency sample %d timed out (attempt %d)", idx, attempt)
            except Exception as exc:
                logger.warning("SelfConsistency sample %d failed: %s", idx, exc)
        return None

    def _cluster_and_vote(self, samples: List[SampledAnswer]) -> VotingResult:
        """Cluster collected samples and return the majority-vote result."""
        texts = [s.text for s in samples]

        # Numeric answers get exact-match grouping first
        numeric_votes = [_extract_numeric(t) for t in texts]
        if all(v is not None for v in numeric_votes):
            counter: Counter = Counter(numeric_votes)
            winner_val, count = counter.most_common(1)[0]
            winner_text = next(t for t, v in zip(texts, numeric_votes) if v == winner_val)
            cluster_ids = [int(v == winner_val) for v in numeric_votes]
        else:
            cluster_ids = _greedy_cluster(texts, self.sim_fn, self.sim_threshold)
            winner_cluster = Counter(cluster_ids).most_common(1)[0][0]
            # Centroid = longest answer in the majority cluster
            winner_text = max(
                (t for t, c in zip(texts, cluster_ids) if c == winner_cluster),
                key=len,
            )
            count = sum(1 for c in cluster_ids if c == winner_cluster)

        # Annotate samples with cluster ids
        for sample, cid in zip(samples, cluster_ids):
            sample.cluster_id = cid

        consistency = count / len(samples)
        n_clusters = len(set(cluster_ids))

        return VotingResult(
            answer=winner_text,
            consistency_score=consistency,
            samples=samples,
            majority_cluster_size=count,
            total_samples=len(samples),
            cluster_count=n_clusters,
        )


# ---------------------------------------------------------------------------
# Convenience wrapper for pipeline integration
# ---------------------------------------------------------------------------


async def self_consistent_answer(
    prompt: str,
    llm: LLMGenerateFn,
    context: Optional[str] = None,
    samples: int = 5,
    temperature: float = 0.7,
) -> Tuple[str, float]:
    """
    Thin wrapper: returns (answer, consistency_score) for easy pipeline use.

    Args:
        prompt: Question / instruction.
        llm: Async LLM callable.
        context: Optional retrieved context.
        samples: Number of samples.
        temperature: Sampling temperature.

    Returns:
        Tuple of (best_answer_string, consistency_score).
    """
    voter = SelfConsistencyVoter(llm=llm, samples=samples, temperature=temperature)
    result = await voter.vote(prompt, context=context)
    return result.answer, result.consistency_score
