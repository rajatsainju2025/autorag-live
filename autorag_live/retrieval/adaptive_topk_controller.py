"""Adaptive Top-K controller for retrieval.

Chooses k dynamically from query uncertainty and score tail shape instead of
using a static top-k. This reduces wasted reranking tokens on easy queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class TopKDecision:
    k: int
    confidence: float
    reason: str


class AdaptiveTopKController:
    """Heuristic controller mapping uncertainty to retrieval depth."""

    def __init__(
        self,
        min_k: int = 4,
        max_k: int = 40,
        margin_threshold: float = 0.08,
    ) -> None:
        self.min_k = min_k
        self.max_k = max_k
        self.margin_threshold = margin_threshold

    def choose_k(
        self,
        scores: Iterable[float],
        query_uncertainty: float = 0.5,
    ) -> TopKDecision:
        """Pick k using score concentration and uncertainty.

        `query_uncertainty` is expected in [0,1], where higher means broader
        retrieval should be attempted.
        """
        arr = np.asarray(list(scores), dtype=np.float32)
        if arr.size == 0:
            return TopKDecision(k=self.min_k, confidence=0.0, reason="no scores")

        arr = np.sort(arr)[::-1]
        best = float(arr[0])
        second = float(arr[1]) if arr.size > 1 else best
        margin = best - second

        # Tail entropy proxy: flatter tails imply ambiguity -> increase k.
        probs = arr - arr.min() + 1e-6
        probs = probs / probs.sum()
        entropy = (
            float(-(probs * np.log(probs)).sum() / np.log(len(probs))) if len(probs) > 1 else 0.0
        )

        ambiguity = (
            0.45 * query_uncertainty + 0.35 * entropy + 0.20 * (margin < self.margin_threshold)
        )
        ambiguity = float(np.clip(ambiguity, 0.0, 1.0))

        k = int(round(self.min_k + ambiguity * (self.max_k - self.min_k)))
        k = max(self.min_k, min(self.max_k, k))

        confidence = float(np.clip(1.0 - ambiguity, 0.0, 1.0))
        reason = (
            f"uncertainty={query_uncertainty:.2f}, margin={margin:.3f}, " f"entropy={entropy:.2f}"
        )
        return TopKDecision(k=k, confidence=confidence, reason=reason)


def create_adaptive_topk_controller(
    min_k: int = 4,
    max_k: int = 40,
) -> AdaptiveTopKController:
    """Factory for `AdaptiveTopKController`."""
    return AdaptiveTopKController(min_k=min_k, max_k=max_k)
