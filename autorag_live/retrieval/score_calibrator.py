"""Cross-retriever score calibration.

Normalizes heterogeneous score ranges (BM25, dense, reranker logits) into a
comparable [0, 1] signal for stable downstream fusion and routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class ScoredItem:
    item_id: str
    score: float
    source: str


class ScoreCalibrator:
    """Per-source robust z-score calibration mapped to [0,1]."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon

    def calibrate(self, items: Iterable[ScoredItem]) -> List[ScoredItem]:
        items_list = list(items)
        if not items_list:
            return []

        by_source: Dict[str, List[ScoredItem]] = {}
        for item in items_list:
            by_source.setdefault(item.source, []).append(item)

        out: list[ScoredItem] = []
        for source, group in by_source.items():
            scores = np.asarray([x.score for x in group], dtype=np.float32)
            med = float(np.median(scores))
            mad = float(np.median(np.abs(scores - med))) + self.epsilon
            z = (scores - med) / (1.4826 * mad)
            # Logistic squash to [0,1]
            cal = 1.0 / (1.0 + np.exp(-z))

            for item, val in zip(group, cal):
                out.append(
                    ScoredItem(
                        item_id=item.item_id,
                        score=float(val),
                        source=source,
                    )
                )
        return out

    def blend(
        self,
        items: Iterable[ScoredItem],
        source_weights: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """Blend calibrated scores by item across retrieval sources."""
        calibrated = self.calibrate(items)
        weights = source_weights or {}

        agg: Dict[str, float] = {}
        norm: Dict[str, float] = {}
        for item in calibrated:
            w = float(weights.get(item.source, 1.0))
            agg[item.item_id] = agg.get(item.item_id, 0.0) + item.score * w
            norm[item.item_id] = norm.get(item.item_id, 0.0) + w

        return {k: (agg[k] / norm[k] if norm[k] else 0.0) for k in agg}


def create_score_calibrator() -> ScoreCalibrator:
    """Factory for `ScoreCalibrator`."""
    return ScoreCalibrator()
