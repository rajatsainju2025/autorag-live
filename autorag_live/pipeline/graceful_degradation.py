"""
Graceful Degradation Pipeline for Agentic RAG.

Production RAG systems must remain functional even when components fail.
This module implements a tiered fallback pipeline with the following properties:

1. **Tiers**: Each tier is a complete RAG strategy (e.g. full agentic → simple
   retrieval + generate → cached response → static fallback message).
2. **Health checks**: Each tier declares whether it is currently healthy.
3. **Automatic promotion/demotion**: If a tier fails, the pipeline silently
   moves to the next. On success it records the winning tier.
4. **Partial degradation metrics**: Tracks how often each tier is used so
   engineers can monitor SLA degradation in production dashboards.
5. **Timeout enforcement**: Each tier has an independent timeout to prevent
   slow upstreams from cascading.

Usage::

    from autorag_live.pipeline.graceful_degradation import (
        GracefulDegradationPipeline,
        PipelineTier,
    )

    async def full_agentic_rag(query: str) -> str: ...
    async def simple_rag(query: str) -> str: ...
    async def cached_response(query: str) -> str: ...

    pipeline = GracefulDegradationPipeline(
        tiers=[
            PipelineTier("full_agentic", full_agentic_rag, timeout=30.0),
            PipelineTier("simple_rag",   simple_rag,        timeout=10.0),
            PipelineTier("cached",        cached_response,   timeout=2.0),
        ],
        static_fallback="I'm currently unable to answer this question. Please try again later.",
    )

    answer = await pipeline.run("What is retrieval augmented generation?")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

RAGFn = Callable[[str], Awaitable[str]]
HealthFn = Callable[[], Awaitable[bool]]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TierMetrics:
    """Runtime statistics for a single pipeline tier."""

    name: str
    invocations: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.invocations if self.invocations > 0 else 0.0

    @property
    def mean_latency_ms(self) -> float:
        return self.total_latency_ms / self.successes if self.successes > 0 else 0.0


@dataclass
class PipelineTier:
    """
    A single tier in the graceful degradation pipeline.

    Args:
        name: Human-readable tier identifier (used in logs and metrics).
        fn: Async function ``(query: str) -> str`` implementing this tier.
        timeout: Maximum seconds to wait before considering this tier failed.
        health_fn: Optional async callable returning True if this tier is healthy.
                   Unhealthy tiers are skipped entirely.
        enabled: Set to False to permanently disable this tier.
    """

    name: str
    fn: RAGFn
    timeout: float = 30.0
    health_fn: Optional[HealthFn] = None
    enabled: bool = True
    _metrics: TierMetrics = field(init=False)

    def __post_init__(self) -> None:
        self._metrics = TierMetrics(name=self.name)

    @property
    def metrics(self) -> TierMetrics:
        return self._metrics


@dataclass
class DegradationResult:
    """Result of a graceful degradation pipeline run."""

    answer: str
    winning_tier: str  # Name of the tier that produced the answer
    tiers_attempted: List[str]
    fully_degraded: bool  # True if static_fallback was used
    total_latency_ms: float


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class GracefulDegradationPipeline:
    """
    Multi-tier RAG pipeline with automatic graceful degradation.

    Tries each tier in order, falling back to the next on failure or timeout.
    If all tiers fail, returns a static fallback message.

    Args:
        tiers: Ordered list of PipelineTier objects (highest quality first).
        static_fallback: Message returned when all tiers fail.
        record_metrics: Whether to accumulate per-tier metrics.

    Example::

        pipeline = GracefulDegradationPipeline(tiers=[tier1, tier2], static_fallback="Sorry.")
        result = await pipeline.run("What is RAG?")
        print(result.answer, result.winning_tier)
    """

    def __init__(
        self,
        tiers: List[PipelineTier],
        static_fallback: str = "I'm unable to provide an answer at this time.",
        record_metrics: bool = True,
    ) -> None:
        if not tiers:
            raise ValueError("At least one tier is required.")
        self.tiers = tiers
        self.static_fallback = static_fallback
        self.record_metrics = record_metrics

    async def run(self, query: str) -> DegradationResult:
        """
        Execute the pipeline, falling back through tiers as needed.

        Args:
            query: The user question.

        Returns:
            DegradationResult with the answer and metadata.
        """
        start = time.monotonic()
        attempted: List[str] = []

        for tier in self.tiers:
            if not tier.enabled:
                logger.debug("Tier '%s' is disabled; skipping.", tier.name)
                continue

            # Health check (skip unhealthy tiers)
            if tier.health_fn is not None:
                try:
                    healthy = await asyncio.wait_for(tier.health_fn(), timeout=2.0)
                    if not healthy:
                        logger.warning("Tier '%s' failed health check; skipping.", tier.name)
                        continue
                except (asyncio.TimeoutError, Exception) as exc:
                    logger.warning("Tier '%s' health check error (%s); skipping.", tier.name, exc)
                    continue

            attempted.append(tier.name)
            tier_start = time.monotonic()

            if self.record_metrics:
                tier._metrics.invocations += 1

            try:
                answer = await asyncio.wait_for(tier.fn(query), timeout=tier.timeout)
                latency_ms = (time.monotonic() - tier_start) * 1000.0

                if self.record_metrics:
                    tier._metrics.successes += 1
                    tier._metrics.total_latency_ms += latency_ms

                logger.info("Query answered by tier '%s' in %.1fms.", tier.name, latency_ms)
                return DegradationResult(
                    answer=answer,
                    winning_tier=tier.name,
                    tiers_attempted=attempted,
                    fully_degraded=False,
                    total_latency_ms=(time.monotonic() - start) * 1000.0,
                )

            except asyncio.TimeoutError:
                latency_ms = (time.monotonic() - tier_start) * 1000.0
                logger.warning(
                    "Tier '%s' timed out after %.1fs; trying next tier.",
                    tier.name,
                    tier.timeout,
                )
                if self.record_metrics:
                    tier._metrics.failures += 1
                    tier._metrics.timeouts += 1
                    tier._metrics.total_latency_ms += latency_ms

            except Exception as exc:
                latency_ms = (time.monotonic() - tier_start) * 1000.0
                logger.warning(
                    "Tier '%s' raised %s: %s; trying next tier.",
                    tier.name,
                    type(exc).__name__,
                    exc,
                )
                if self.record_metrics:
                    tier._metrics.failures += 1
                    tier._metrics.total_latency_ms += latency_ms

        # All tiers failed — return static fallback
        logger.error(
            "All %d tier(s) failed for query '%s...'; returning static fallback.",
            len(attempted),
            query[:60],
        )
        return DegradationResult(
            answer=self.static_fallback,
            winning_tier="static_fallback",
            tiers_attempted=attempted,
            fully_degraded=True,
            total_latency_ms=(time.monotonic() - start) * 1000.0,
        )

    def metrics_report(self) -> Dict[str, Any]:
        """Return a metrics snapshot for all tiers."""
        report: Dict[str, Any] = {}
        for tier in self.tiers:
            m = tier._metrics
            report[tier.name] = {
                "invocations": m.invocations,
                "successes": m.successes,
                "failures": m.failures,
                "timeouts": m.timeouts,
                "success_rate": round(m.success_rate, 4),
                "mean_latency_ms": round(m.mean_latency_ms, 1),
            }
        return report

    def disable_tier(self, name: str) -> None:
        """Dynamically disable a tier by name (e.g. during an incident)."""
        for tier in self.tiers:
            if tier.name == name:
                tier.enabled = False
                logger.warning("Tier '%s' has been disabled.", name)
                return
        logger.warning("disable_tier: tier '%s' not found.", name)

    def enable_tier(self, name: str) -> None:
        """Re-enable a previously disabled tier."""
        for tier in self.tiers:
            if tier.name == name:
                tier.enabled = True
                logger.info("Tier '%s' has been re-enabled.", name)
                return
        logger.warning("enable_tier: tier '%s' not found.", name)
