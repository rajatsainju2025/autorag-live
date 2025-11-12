"""Performance regression detection with baseline tracking."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""

    metric_name: str
    baseline_value: float
    threshold_percent: float = 5.0
    measurements: List[float] = field(default_factory=list)

    def check_regression(self, current_value: float) -> bool:
        """Check if current value indicates regression."""
        if not self.baseline_value:
            return False

        percent_change = abs(current_value - self.baseline_value) / self.baseline_value * 100
        is_regression = percent_change > self.threshold_percent
        self.measurements.append(current_value)
        return is_regression

    def get_stats(self) -> Dict:
        """Get statistics."""
        if not self.measurements:
            return {}
        return {
            "baseline": self.baseline_value,
            "current": self.measurements[-1] if self.measurements else None,
            "threshold_percent": self.threshold_percent,
            "measurements": len(self.measurements),
        }


class RegressionDetector:
    """Detects performance regressions."""

    def __init__(self):
        """Initialize regression detector."""
        self._baselines: Dict[str, PerformanceBaseline] = {}

    def register_metric(
        self,
        metric_name: str,
        baseline_value: float,
        threshold_percent: float = 5.0,
    ):
        """Register a metric for regression detection."""
        self._baselines[metric_name] = PerformanceBaseline(
            metric_name=metric_name,
            baseline_value=baseline_value,
            threshold_percent=threshold_percent,
        )

    def check_all(self, measurements: Dict[str, float]) -> Dict[str, bool]:
        """Check all metrics for regressions."""
        regressions = {}
        for metric_name, value in measurements.items():
            if metric_name in self._baselines:
                regressions[metric_name] = self._baselines[metric_name].check_regression(value)
        return regressions

    def get_report(self) -> Dict:
        """Get regression report."""
        report = {}
        for name, baseline in self._baselines.items():
            report[name] = baseline.get_stats()
        return report
