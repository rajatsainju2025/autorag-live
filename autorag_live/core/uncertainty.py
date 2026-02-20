"""
Calibrated uncertainty estimation for RAG pipelines.

Replaces ad-hoc heuristic confidence scores with principled calibration:
- TemperatureScaler  : Platt/temperature scaling on retrieval score logits
- EnsembleUncertainty: Variance-based epistemic uncertainty from score spread
- AnswerLengthSignal : Proxy for model hedging (verbose ≈ uncertain)
- UncertaintyEstimator: Unified interface returning a calibrated [0, 1] score

Calibration methodology
-----------------------
Temperature scaling (Guo et al., 2017) learns a single scalar T that maps
raw logit s → σ(s / T) to minimise Expected Calibration Error (ECE).
Without labelled data we default to T=1.5 (empirically over-confident
baseline for dense retrievers) and allow online refinement via
``update()`` whenever binary relevance feedback is available.

Based on:
- "On Calibration of Modern Neural Networks" (Guo et al., 2017, ICML)
- "RAG Uncertainty Quantification" (Jiang et al., 2023, EMNLP)
- "Uncertainty-Aware RAG" (Geng et al., 2024)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level calibration helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _expected_calibration_error(
    confidences: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    """
    Compute ECE: weighted average |confidence − accuracy| across bins.

    Args:
        confidences: Predicted probabilities ∈ [0, 1].
        labels: Binary ground-truth relevance labels (0 or 1).
        n_bins: Number of equal-width bins.

    Returns:
        ECE ∈ [0, 1]; lower is better.
    """
    confs = np.asarray(confidences, dtype=float)
    labs = np.asarray(labels, dtype=float)
    n = len(confs)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs >= lo) & (confs < hi)
        if not mask.any():
            continue
        bin_conf = confs[mask].mean()
        bin_acc = labs[mask].mean()
        ece += mask.sum() / n * abs(bin_conf - bin_acc)
    return float(ece)


# ---------------------------------------------------------------------------
# Temperature scaler (single-parameter calibration)
# ---------------------------------------------------------------------------


class TemperatureScaler:
    """
    Single-parameter temperature scaling for retrieval score calibration.

    Maps a raw relevance score s ∈ ℝ → σ(s / T) ∈ [0, 1].

    The temperature T is initialised empirically (default T=1.5, which
    compensates for the over-confidence common in dense-retrieval cosine
    scores) and can be refined online via ``update()`` using relevance
    feedback.

    Args:
        init_temperature: Starting temperature (> 0; higher → less confident).
        lr: Gradient-descent learning rate for online updates.
        min_temp: Minimum allowed temperature (prevents collapse to 0).
        max_temp: Maximum allowed temperature.
    """

    def __init__(
        self,
        init_temperature: float = 1.5,
        lr: float = 0.05,
        min_temp: float = 0.1,
        max_temp: float = 10.0,
    ) -> None:
        if init_temperature <= 0:
            raise ValueError("init_temperature must be > 0")
        self.T = float(init_temperature)
        self.lr = lr
        self.min_temp = min_temp
        self.max_temp = max_temp
        self._update_count = 0

    def calibrate(self, score: float) -> float:
        """Return calibrated probability for a single raw score."""
        return _sigmoid(score / self.T)

    def calibrate_batch(self, scores: Sequence[float]) -> List[float]:
        """Return calibrated probabilities for a batch of raw scores."""
        return [self.calibrate(s) for s in scores]

    def update(self, score: float, label: int) -> None:
        """
        One-step online gradient update using binary cross-entropy loss.

        dL/dT = (σ(s/T) − y) · (−s / T²)

        Args:
            score: Raw retrieval score used in the prediction.
            label: Binary relevance label (1 = relevant, 0 = not relevant).
        """
        p = self.calibrate(score)
        grad_T = (p - label) * (-score / (self.T**2))
        self.T -= self.lr * grad_T
        self.T = max(self.min_temp, min(self.max_temp, self.T))
        self._update_count += 1

    def ece(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        n_bins: int = 10,
    ) -> float:
        """Return Expected Calibration Error on a held-out set."""
        probs = self.calibrate_batch(scores)
        return _expected_calibration_error(probs, labels, n_bins)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "T": self.T,
            "lr": self.lr,
            "min_temp": self.min_temp,
            "max_temp": self.max_temp,
            "update_count": self._update_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.T = float(state["T"])
        self.lr = float(state.get("lr", self.lr))
        self._update_count = int(state.get("update_count", 0))


# ---------------------------------------------------------------------------
# Ensemble (variance-based) uncertainty
# ---------------------------------------------------------------------------


class EnsembleUncertainty:
    """
    Epistemic uncertainty from the spread of retrieval scores.

    A high variance in the top-k document scores signals that the retriever
    is unsure about the correct answer — the confidence is deflated
    proportionally.

    Confidence = μ_score / (1 + α · σ_score)

    where μ and σ are the mean and standard deviation of the calibrated
    scores for the top-k documents, and α is a sensitivity parameter.

    Args:
        alpha: Spread sensitivity (default 2.0 — moderate penalisation).
        top_k: Number of top documents to consider for spread estimation.
    """

    def __init__(self, alpha: float = 2.0, top_k: int = 5) -> None:
        self.alpha = alpha
        self.top_k = top_k

    def estimate(self, calibrated_scores: Sequence[float]) -> float:
        """
        Return confidence ∈ [0, 1] from calibrated score distribution.

        Args:
            calibrated_scores: Calibrated probabilities from TemperatureScaler.

        Returns:
            Confidence value ∈ [0, 1].
        """
        scores = list(calibrated_scores[: self.top_k])
        if not scores:
            return 0.0
        mu = float(np.mean(scores))
        sigma = float(np.std(scores)) if len(scores) > 1 else 0.0
        return float(np.clip(mu / (1.0 + self.alpha * sigma), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Answer-length signal (proxy for model hedging)
# ---------------------------------------------------------------------------


class AnswerLengthSignal:
    """
    Proxy uncertainty from answer verbosity.

    Dense retrievers tend to produce verbose hedged answers ("It depends…",
    "There are several factors…") when uncertain.  Abnormally long answers
    (> long_threshold tokens) therefore indicate lower confidence.

    Args:
        short_threshold: Answers shorter than this are confident (≤ 50 tokens).
        long_threshold: Answers longer than this are hedged (≥ 300 tokens).
    """

    def __init__(
        self,
        short_threshold: int = 50,
        long_threshold: int = 300,
    ) -> None:
        self.short_threshold = short_threshold
        self.long_threshold = long_threshold

    def penalty(self, answer: str) -> float:
        """
        Return a length-based confidence penalty ∈ [0, 0.2].

        Short answers → penalty ≈ 0; very long answers → penalty ≈ 0.2.
        """
        n_tokens = len(answer.split())
        if n_tokens <= self.short_threshold:
            return 0.0
        if n_tokens >= self.long_threshold:
            return 0.20
        t = (n_tokens - self.short_threshold) / (self.long_threshold - self.short_threshold)
        return round(0.20 * t, 4)


# ---------------------------------------------------------------------------
# UncertaintyResult
# ---------------------------------------------------------------------------


@dataclass
class UncertaintyResult:
    """
    Full uncertainty breakdown for a RAG response.

    Attributes:
        confidence: Final calibrated confidence ∈ [0, 1].
        raw_score: Mean raw retrieval score before calibration.
        calibrated_score: Temperature-scaled mean probability.
        ensemble_confidence: Spread-adjusted confidence.
        length_penalty: Deduction from answer verbosity.
        temperature: Scaler temperature used at inference time.
        diagnostics: Auxiliary debugging information.
    """

    confidence: float
    raw_score: float
    calibrated_score: float
    ensemble_confidence: float
    length_penalty: float
    temperature: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        return self.confidence >= threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": round(self.confidence, 4),
            "raw_score": round(self.raw_score, 4),
            "calibrated_score": round(self.calibrated_score, 4),
            "ensemble_confidence": round(self.ensemble_confidence, 4),
            "length_penalty": round(self.length_penalty, 4),
            "temperature": round(self.temperature, 4),
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# Unified estimator
# ---------------------------------------------------------------------------


class UncertaintyEstimator:
    """
    Unified calibrated uncertainty estimator for RAG pipelines.

    Combines three signals with a configurable weighted blend:
    1. Temperature-scaled mean retrieval probability  (weight w_cal)
    2. Score-spread ensemble penalty                 (weight w_ens)
    3. Answer-length hedging signal                  (subtractive)

    The final confidence is::

        c = w_cal * calibrated_mean + w_ens * ensemble_conf - length_penalty

    clamped to [0, 1].

    Args:
        temperature_scaler: Pre-configured :class:`TemperatureScaler` instance.
        w_cal: Weight for calibrated probability signal (default 0.55).
        w_ens: Weight for ensemble spread signal (default 0.45).
        short_threshold: Answer-length lower bound for hedging detection.
        long_threshold: Answer-length upper bound for hedging detection.
    """

    def __init__(
        self,
        temperature_scaler: Optional[TemperatureScaler] = None,
        w_cal: float = 0.55,
        w_ens: float = 0.45,
        short_threshold: int = 50,
        long_threshold: int = 300,
    ) -> None:
        if not math.isclose(w_cal + w_ens, 1.0, abs_tol=1e-6):
            raise ValueError("w_cal + w_ens must equal 1.0")
        self.scaler = temperature_scaler or TemperatureScaler()
        self.ensemble = EnsembleUncertainty()
        self.length_signal = AnswerLengthSignal(short_threshold, long_threshold)
        self.w_cal = w_cal
        self.w_ens = w_ens

    def estimate(
        self,
        documents: List[Dict[str, Any]],
        answer: str = "",
        score_key: str = "score",
    ) -> UncertaintyResult:
        """
        Compute calibrated uncertainty for a retrieval + generation result.

        Args:
            documents: Retrieved document list; each must have ``score_key``.
            answer: Generated answer text (used for length signal).
            score_key: Dict key for retrieval relevance score (default "score").

        Returns:
            :class:`UncertaintyResult` with full breakdown.
        """
        raw_scores: List[float] = [
            float(d.get(score_key, 0.0)) for d in documents if d.get(score_key) is not None
        ]
        if not raw_scores:
            return UncertaintyResult(
                confidence=0.0,
                raw_score=0.0,
                calibrated_score=0.0,
                ensemble_confidence=0.0,
                length_penalty=0.0,
                temperature=self.scaler.T,
                diagnostics={"reason": "no_scores"},
            )

        raw_mean = float(np.mean(raw_scores))
        calibrated = self.scaler.calibrate_batch(raw_scores)
        cal_mean = float(np.mean(calibrated))
        ens_conf = self.ensemble.estimate(calibrated)
        l_penalty = self.length_signal.penalty(answer)

        confidence = float(
            np.clip(self.w_cal * cal_mean + self.w_ens * ens_conf - l_penalty, 0.0, 1.0)
        )
        return UncertaintyResult(
            confidence=round(confidence, 4),
            raw_score=round(raw_mean, 4),
            calibrated_score=round(cal_mean, 4),
            ensemble_confidence=round(ens_conf, 4),
            length_penalty=round(l_penalty, 4),
            temperature=round(self.scaler.T, 4),
            diagnostics={
                "n_docs": len(raw_scores),
                "score_std": round(float(np.std(raw_scores)), 4),
                "answer_tokens": len(answer.split()),
            },
        )

    def update(self, score: float, label: int) -> None:
        """Pass relevance feedback to the temperature scaler for online calibration."""
        self.scaler.update(score, label)

    def ece(
        self,
        documents_batch: List[List[Dict[str, Any]]],
        labels_batch: List[List[int]],
        score_key: str = "score",
    ) -> float:
        """
        Compute ECE on a batch of (document_list, label_list) pairs.

        Useful for offline evaluation of calibration quality.
        """
        all_scores: List[float] = []
        all_labels: List[int] = []
        for docs, labs in zip(documents_batch, labels_batch):
            scores = [float(d.get(score_key, 0.0)) for d in docs]
            probs = self.scaler.calibrate_batch(scores)
            all_scores.extend(probs)
            all_labels.extend(labs)
        return _expected_calibration_error(all_scores, all_labels)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "scaler": self.scaler.state_dict(),
            "w_cal": self.w_cal,
            "w_ens": self.w_ens,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.scaler.load_state_dict(state["scaler"])
        self.w_cal = float(state.get("w_cal", self.w_cal))
        self.w_ens = float(state.get("w_ens", self.w_ens))
