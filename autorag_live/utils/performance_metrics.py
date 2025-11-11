"""Performance metrics collection and reporting.

This module provides lightweight performance monitoring with:
- Execution time tracking
- Memory usage monitoring
- Throughput metrics
- Batch operation statistics

Example:
    >>> metrics = PerformanceMetrics()
    >>> with metrics.timer("query_execution"):
    ...     results = retriever.retrieve("query", k=10)
    >>> print(metrics.summary())
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: float = field(default_factory=time.time)
    value: float = 0.0
    unit: str = ""


@dataclass
class MetricStats:
    """Statistics for a metric."""

    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = 0.0
    unit: str = ""

    @property
    def mean(self) -> float:
        """Calculate mean."""
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def summary(self) -> str:
        """Get summary string."""
        if self.count == 0:
            return "No data"
        unit_str = f" {self.unit}" if self.unit else ""
        return (
            f"count={self.count}, "
            f"mean={self.mean:.3f}{unit_str}, "
            f"min={self.min:.3f}{unit_str}, "
            f"max={self.max:.3f}{unit_str}"
        )


class PerformanceMetrics:
    """Thread-safe performance metrics collector."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.stats: Dict[str, MetricStats] = {}
        self.lock = threading.RLock()

    def record(self, metric_name: str, value: float, unit: str = "") -> None:
        """Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement (optional)
        """
        with self.lock:
            point = MetricPoint(value=value, unit=unit)
            self.metrics[metric_name].append(point)

            # Update stats
            if metric_name not in self.stats:
                self.stats[metric_name] = MetricStats(unit=unit)
            stats = self.stats[metric_name]
            stats.count += 1
            stats.total += value
            stats.min = min(stats.min, value)
            stats.max = max(stats.max, value)

    def timer(self, metric_name: str, unit: str = "ms"):
        """Context manager for timing operations.

        Args:
            metric_name: Name of the metric
            unit: Unit of measurement

        Returns:
            Timer context manager
        """
        return _TimerContext(self, metric_name, unit)

    def get_stats(self, metric_name: str) -> Optional[MetricStats]:
        """Get statistics for a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            MetricStats or None if not found
        """
        with self.lock:
            return self.stats.get(metric_name)

    def summary(self) -> str:
        """Get summary of all metrics.

        Returns:
            Formatted summary string
        """
        with self.lock:
            if not self.stats:
                return "No metrics recorded"

            lines = ["Performance Metrics Summary:"]
            for name, stats in sorted(self.stats.items()):
                lines.append(f"  {name}: {stats.summary}")
            return "\n".join(lines)

    def clear(self) -> None:
        """Clear all metrics."""
        with self.lock:
            self.metrics.clear()
            self.stats.clear()


class _TimerContext:
    """Context manager for timing operations."""

    def __init__(
        self,
        metrics: PerformanceMetrics,
        metric_name: str,
        unit: str = "ms",
    ):
        """Initialize timer context.

        Args:
            metrics: PerformanceMetrics instance
            metric_name: Name of the metric
            unit: Unit of measurement
        """
        self.metrics = metrics
        self.metric_name = metric_name
        self.unit = unit
        self.start_time: Optional[float] = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and record metric."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time

            # Convert to requested unit
            if self.unit == "ms":
                value = elapsed * 1000
            elif self.unit == "s":
                value = elapsed
            else:
                value = elapsed

            self.metrics.record(self.metric_name, value, self.unit)


class MemoryMonitor:
    """Monitor memory usage of processes."""

    @staticmethod
    def get_memory_usage() -> Optional[float]:
        """Get current memory usage in MB.

        Returns:
            Memory usage in MB or None if psutil unavailable
        """
        if not PSUTIL_AVAILABLE or psutil is None:
            return None

        try:
            process = psutil.Process()  # type: ignore
            return process.memory_info().rss / (1024 * 1024)  # type: ignore
        except Exception:
            return None

    @staticmethod
    def get_memory_peak() -> Optional[float]:
        """Get peak memory usage in MB.

        Returns:
            Peak memory in MB or None if psutil unavailable
        """
        if not PSUTIL_AVAILABLE or psutil is None:
            return None

        try:
            # Note: peak memory tracking requires resource module on Unix
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return None
