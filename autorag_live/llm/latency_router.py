"""Latency-aware model router.

Routes LLM requests to the best candidate model under latency SLO, cost cap,
and quality floor using rolling performance telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class ModelTelemetry:
    model: str
    p50_latency_ms: float
    p95_latency_ms: float
    cost_per_1k_tokens: float
    quality_score: float  # [0,1] from offline/online eval


@dataclass
class RoutingDecision:
    model: str
    expected_latency_ms: float
    expected_cost_per_1k_tokens: float
    expected_quality: float
    reason: str


class LatencyAwareModelRouter:
    """Multi-objective model selector for agentic RAG generation."""

    def __init__(
        self,
        latency_weight: float = 0.45,
        cost_weight: float = 0.25,
        quality_weight: float = 0.30,
    ) -> None:
        total = latency_weight + cost_weight + quality_weight
        if total <= 0:
            raise ValueError("weights must sum to > 0")
        self.w_latency = latency_weight / total
        self.w_cost = cost_weight / total
        self.w_quality = quality_weight / total

    def route(
        self,
        telemetry: List[ModelTelemetry],
        slo_ms: float,
        max_cost_per_1k_tokens: float,
        quality_floor: float = 0.0,
    ) -> RoutingDecision:
        if not telemetry:
            raise ValueError("telemetry list is empty")

        feasible = [
            t
            for t in telemetry
            if t.p95_latency_ms <= slo_ms
            and t.cost_per_1k_tokens <= max_cost_per_1k_tokens
            and t.quality_score >= quality_floor
        ]
        candidates = feasible if feasible else telemetry

        lat = np.asarray([c.p95_latency_ms for c in candidates], dtype=np.float32)
        cst = np.asarray([c.cost_per_1k_tokens for c in candidates], dtype=np.float32)
        qlt = np.asarray([c.quality_score for c in candidates], dtype=np.float32)

        def _norm(x: np.ndarray, invert: bool = False) -> np.ndarray:
            if np.allclose(x.max(), x.min()):
                out = np.ones_like(x)
            else:
                out = (x - x.min()) / (x.max() - x.min())
            return 1.0 - out if invert else out

        # lower latency/cost is better; higher quality is better
        lat_s = _norm(lat, invert=True)
        cst_s = _norm(cst, invert=True)
        qlt_s = _norm(qlt, invert=False)

        score = self.w_latency * lat_s + self.w_cost * cst_s + self.w_quality * qlt_s
        idx = int(np.argmax(score))
        chosen = candidates[idx]

        reason = (
            f"score={score[idx]:.3f}, p95={chosen.p95_latency_ms:.0f}ms, "
            f"cost={chosen.cost_per_1k_tokens:.4f}, quality={chosen.quality_score:.3f}"
        )
        return RoutingDecision(
            model=chosen.model,
            expected_latency_ms=chosen.p95_latency_ms,
            expected_cost_per_1k_tokens=chosen.cost_per_1k_tokens,
            expected_quality=chosen.quality_score,
            reason=reason,
        )

    def route_many(
        self,
        telemetry_by_tier: Dict[str, List[ModelTelemetry]],
        slo_ms: float,
        max_cost_per_1k_tokens: float,
        quality_floor: float = 0.0,
    ) -> Dict[str, RoutingDecision]:
        """Tier-aware routing (e.g., short_answer, long_form, reasoning)."""
        out: Dict[str, RoutingDecision] = {}
        for tier, telemetry in telemetry_by_tier.items():
            out[tier] = self.route(
                telemetry=telemetry,
                slo_ms=slo_ms,
                max_cost_per_1k_tokens=max_cost_per_1k_tokens,
                quality_floor=quality_floor,
            )
        return out


def create_latency_aware_model_router() -> LatencyAwareModelRouter:
    """Factory for `LatencyAwareModelRouter`."""
    return LatencyAwareModelRouter()
