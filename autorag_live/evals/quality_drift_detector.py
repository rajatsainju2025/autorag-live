"""Online quality drift detector.

Tracks rolling quality signals (faithfulness, answer relevance, user rating)
and flags statistically meaningful degradation to trigger safe fallbacks or
auto-tuning.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable

import numpy as np


@dataclass
class DriftStatus:
    drift_detected: bool
    z_score: float
    baseline_mean: float
    recent_mean: float
    threshold: float


class QualityDriftDetector:
    """Windowed z-score drift detector for online RAG quality monitoring."""

    def __init__(
        self,
        baseline_window: int = 200,
        recent_window: int = 40,
        z_threshold: float = -2.0,
    ) -> None:
        if baseline_window <= recent_window:
            raise ValueError("baseline_window must be greater than recent_window")
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.z_threshold = z_threshold
        self._scores: Deque[float] = deque(maxlen=baseline_window)

    def update(self, quality_score: float) -> None:
        self._scores.append(float(np.clip(quality_score, 0.0, 1.0)))

    def update_many(self, scores: Iterable[float]) -> None:
        for s in scores:
            self.update(s)

    def check(self) -> DriftStatus:
        if len(self._scores) < self.recent_window * 2:
            return DriftStatus(
                drift_detected=False,
                z_score=0.0,
                baseline_mean=float(np.mean(self._scores)) if self._scores else 0.0,
                recent_mean=float(np.mean(self._scores)) if self._scores else 0.0,
                threshold=self.z_threshold,
            )

        arr = np.asarray(self._scores, dtype=np.float32)
        recent = arr[-self.recent_window :]
        baseline = arr[: -self.recent_window]

        baseline_mean = float(np.mean(baseline))
        baseline_std = float(np.std(baseline) + 1e-8)
        recent_mean = float(np.mean(recent))
        z = (recent_mean - baseline_mean) / baseline_std

        # negative z means quality degraded
        drift = z <= self.z_threshold
        return DriftStatus(
            drift_detected=drift,
            z_score=float(z),
            baseline_mean=baseline_mean,
            recent_mean=recent_mean,
            threshold=self.z_threshold,
        )

    def snapshot(self) -> Dict[str, float]:
        status = self.check()
        return {
            "count": float(len(self._scores)),
            "baseline_mean": round(status.baseline_mean, 4),
            "recent_mean": round(status.recent_mean, 4),
            "z_score": round(status.z_score, 4),
            "threshold": status.threshold,
            "drift_detected": 1.0 if status.drift_detected else 0.0,
        }


def create_quality_drift_detector(
    baseline_window: int = 200,
    recent_window: int = 40,
    z_threshold: float = -2.0,
) -> QualityDriftDetector:
    """Factory for `QualityDriftDetector`."""
    return QualityDriftDetector(
        baseline_window=baseline_window,
        recent_window=recent_window,
        z_threshold=z_threshold,
    )
