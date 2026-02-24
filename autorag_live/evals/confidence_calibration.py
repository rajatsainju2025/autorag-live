"""
Confidence Calibration Metrics for RAG Systems.

Well-calibrated models produce confidence scores that match empirical accuracy.
If a model says "I'm 80% confident" across 100 predictions, ~80 should be correct.

This module provides:
  - **Expected Calibration Error (ECE)** — primary calibration metric.
  - **Maximum Calibration Error (MCE)** — worst-case bin error.
  - **Reliability diagram data** — for visualisation.
  - **Temperature scaling** — post-hoc calibration with a single scalar T.
  - **Adaptive binning (equal-mass bins)** — robust to sparse probability ranges.

References:
    - Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
    - Minderer et al., "Revisiting the Calibration of Modern Neural Networks", NeurIPS 2021.
    - Zhao et al., "Calibrating Sequence-Level Retrieval", ACL Findings 2021.

Usage::

    from autorag_live.evals.confidence_calibration import CalibrationEvaluator

    evaluator = CalibrationEvaluator(n_bins=15, strategy="equal_mass")
    confidences = [0.9, 0.7, 0.6, 0.4, 0.85, ...]  # model confidence scores
    labels      = [1,   1,   0,   0,   1, ...]       # 1=correct, 0=incorrect
    report = evaluator.evaluate(confidences, labels)
    print(report)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CalibrationBin:
    """Statistics for a single confidence bin."""

    lower: float
    upper: float
    n_samples: int
    mean_confidence: float
    mean_accuracy: float
    calibration_error: float  # |mean_confidence - mean_accuracy|

    @property
    def weight(self) -> float:
        """Fractional weight relative to total (set externally)."""
        return self._weight

    @weight.setter
    def weight(self, v: float) -> None:
        self._weight = v

    _weight: float = field(default=0.0, repr=False, compare=False)


@dataclass
class CalibrationReport:
    """Full calibration evaluation result."""

    ece: float  # Expected Calibration Error (weighted mean |conf - acc|)
    mce: float  # Maximum Calibration Error
    ace: float  # Average Calibration Error (unweighted mean)
    overconfidence: float  # Mean(conf - acc) when conf > acc
    underconfidence: float  # Mean(acc - conf) when acc > conf
    n_samples: int
    n_bins: int
    bins: List[CalibrationBin]
    strategy: str

    def __str__(self) -> str:
        lines = [
            "=== Calibration Report ===",
            f"  ECE  : {self.ece:.4f}  (lower is better; <0.05 is well-calibrated)",
            f"  MCE  : {self.mce:.4f}",
            f"  ACE  : {self.ace:.4f}",
            f"  Over-confidence  : {self.overconfidence:.4f}",
            f"  Under-confidence : {self.underconfidence:.4f}",
            f"  Samples: {self.n_samples}  |  Bins: {self.n_bins}  |  Strategy: {self.strategy}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class CalibrationEvaluator:
    """
    Compute calibration metrics for RAG confidence scores.

    Args:
        n_bins: Number of equal-width or equal-mass bins.
        strategy: ``"equal_width"`` or ``"equal_mass"`` binning.

    Example::

        ev = CalibrationEvaluator(n_bins=10, strategy="equal_width")
        report = ev.evaluate(confidences=[0.9, 0.4, 0.7], labels=[1, 0, 1])
        print(report.ece)
    """

    def __init__(
        self,
        n_bins: int = 10,
        strategy: Literal["equal_width", "equal_mass"] = "equal_width",
    ) -> None:
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        self.n_bins = n_bins
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def evaluate(
        self,
        confidences: Sequence[float],
        labels: Sequence[int],
    ) -> CalibrationReport:
        """
        Compute calibration metrics.

        Args:
            confidences: Model confidence scores in [0, 1].
            labels: Binary correctness labels (1 = correct, 0 = incorrect).

        Returns:
            CalibrationReport with ECE, MCE, ACE, and per-bin statistics.
        """
        if len(confidences) != len(labels):
            raise ValueError(
                f"confidences and labels must have the same length ({len(confidences)} vs {len(labels)})"
            )
        if len(confidences) == 0:
            raise ValueError("confidences must not be empty")

        confs = [float(c) for c in confidences]
        labs = [int(b) for b in labels]
        n = len(confs)

        bins = self._bin_samples(confs, labs)

        if not bins:
            logger.warning("No bins could be constructed; returning zero-error report.")
            return CalibrationReport(
                ece=0.0,
                mce=0.0,
                ace=0.0,
                overconfidence=0.0,
                underconfidence=0.0,
                n_samples=n,
                n_bins=0,
                bins=[],
                strategy=self.strategy,
            )

        # Set per-bin weight = n_bin / n_total
        for b in bins:
            b.weight = b.n_samples / n

        ece = sum(b.weight * b.calibration_error for b in bins)
        mce = max(b.calibration_error for b in bins)
        ace = sum(b.calibration_error for b in bins) / len(bins)

        over_errors = [
            b.mean_confidence - b.mean_accuracy for b in bins if b.mean_confidence > b.mean_accuracy
        ]
        under_errors = [
            b.mean_accuracy - b.mean_confidence for b in bins if b.mean_accuracy > b.mean_confidence
        ]

        overconfidence = sum(over_errors) / len(over_errors) if over_errors else 0.0
        underconfidence = sum(under_errors) / len(under_errors) if under_errors else 0.0

        return CalibrationReport(
            ece=round(ece, 6),
            mce=round(mce, 6),
            ace=round(ace, 6),
            overconfidence=round(overconfidence, 6),
            underconfidence=round(underconfidence, 6),
            n_samples=n,
            n_bins=len(bins),
            bins=bins,
            strategy=self.strategy,
        )

    # ------------------------------------------------------------------
    # Temperature scaling
    # ------------------------------------------------------------------

    @staticmethod
    def temperature_scale(confidences: Sequence[float], temperature: float) -> List[float]:
        """
        Apply temperature scaling to logit-derived confidence scores.

        Args:
            confidences: Confidence probabilities in (0, 1).
            temperature: T > 1 softens (reduces overconfidence);
                         T < 1 sharpens (reduces underconfidence).

        Returns:
            Rescaled confidence values.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        scaled = []
        for p in confidences:
            p = max(1e-9, min(1 - 1e-9, float(p)))
            logit = math.log(p / (1 - p))
            rescaled_logit = logit / temperature
            new_p = 1.0 / (1.0 + math.exp(-rescaled_logit))
            scaled.append(new_p)
        return scaled

    @staticmethod
    def find_optimal_temperature(
        confidences: Sequence[float],
        labels: Sequence[int],
        temperatures: Optional[Sequence[float]] = None,
    ) -> float:
        """
        Grid-search for the temperature that minimises ECE.

        Args:
            confidences: Model confidence scores.
            labels: Binary correctness labels.
            temperatures: Candidate temperatures to search (default: 0.1 to 3.0).

        Returns:
            Temperature value with lowest ECE.
        """
        if temperatures is None:
            temperatures = [round(0.1 * i, 1) for i in range(1, 31)]  # 0.1 … 3.0

        evaluator = CalibrationEvaluator(n_bins=10, strategy="equal_width")
        best_t, best_ece = 1.0, float("inf")

        for t in temperatures:
            scaled = CalibrationEvaluator.temperature_scale(confidences, t)
            report = evaluator.evaluate(scaled, labels)
            if report.ece < best_ece:
                best_ece, best_t = report.ece, float(t)

        logger.info("Optimal temperature=%.2f (ECE=%.4f)", best_t, best_ece)
        return best_t

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bin_samples(self, confs: List[float], labs: List[int]) -> List[CalibrationBin]:
        """Create calibration bins according to the configured strategy."""
        pairs = sorted(zip(confs, labs), key=lambda x: x[0])

        if self.strategy == "equal_mass":
            return self._equal_mass_bins(pairs)
        return self._equal_width_bins(pairs)

    def _equal_width_bins(self, pairs: List[tuple[float, int]]) -> List[CalibrationBin]:
        bins: List[CalibrationBin] = []
        bin_size = 1.0 / self.n_bins

        for i in range(self.n_bins):
            lo = i * bin_size
            hi = (i + 1) * bin_size
            bucket = [
                (c, lbl) for c, lbl in pairs if lo <= c < hi or (i == self.n_bins - 1 and c == 1.0)
            ]
            if not bucket:
                continue
            mean_conf = sum(c for c, _ in bucket) / len(bucket)
            mean_acc = sum(lbl for _, lbl in bucket) / len(bucket)
            bins.append(
                CalibrationBin(
                    lower=lo,
                    upper=hi,
                    n_samples=len(bucket),
                    mean_confidence=round(mean_conf, 6),
                    mean_accuracy=round(mean_acc, 6),
                    calibration_error=round(abs(mean_conf - mean_acc), 6),
                )
            )
        return bins

    def _equal_mass_bins(self, pairs: List[tuple[float, int]]) -> List[CalibrationBin]:
        n = len(pairs)
        bin_size = max(1, n // self.n_bins)
        bins: List[CalibrationBin] = []

        for i in range(0, n, bin_size):
            bucket = pairs[i : i + bin_size]
            if not bucket:
                continue
            lo = bucket[0][0]
            hi = bucket[-1][0]
            mean_conf = sum(c for c, _ in bucket) / len(bucket)
            mean_acc = sum(lbl for _, lbl in bucket) / len(bucket)
            bins.append(
                CalibrationBin(
                    lower=lo,
                    upper=hi,
                    n_samples=len(bucket),
                    mean_confidence=round(mean_conf, 6),
                    mean_accuracy=round(mean_acc, 6),
                    calibration_error=round(abs(mean_conf - mean_acc), 6),
                )
            )
        return bins
