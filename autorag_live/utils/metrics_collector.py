"""
Lightweight metrics collection for AutoRAG-Live performance tracking.

This module provides simple metrics collection without heavy dependencies,
tracking retrieval performance, cache efficiency, and system health.

Features:
    - Counter metrics (requests, cache hits/misses)
    - Gauge metrics (current values, resource usage)
    - Histogram metrics (latencies, response times)
    - Time-series tracking
    - In-memory storage with optional persistence

Example:
    >>> from autorag_live.utils.metrics_collector import MetricsCollector, get_metrics
    >>>
    >>> metrics = get_metrics()
    >>> metrics.increment("retrieval.requests")
    >>> metrics.record_latency("retrieval.dense", 0.123)
    >>> metrics.set_gauge("cache.size", 1000)
    >>>
    >>> # Get current metrics
    >>> summary = metrics.summary()
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Deque, Dict, Optional

import numpy as np


@dataclass
class MetricPoint:
    """Single metric data point."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""

    name: str
    count: int
    sum: float
    min: float
    max: float
    mean: float
    std: float
    p50: float
    p95: float
    p99: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "count": self.count,
            "sum": round(self.sum, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "p50": round(self.p50, 4),
            "p95": round(self.p95, 4),
            "p99": round(self.p99, 4),
        }


class MetricsCollector:
    """
    Lightweight metrics collector for performance tracking.

    Tracks:
    - Counters: Monotonically increasing values (requests, errors)
    - Gauges: Point-in-time values (memory usage, queue size)
    - Histograms: Distribution tracking (latencies, sizes)
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.

        Args:
            max_history: Maximum number of historical points per metric
        """
        self._lock = Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=max_history))
        self._start_time = time.time()

    def increment(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name (e.g., "retrieval.requests")
            value: Increment amount
            tags: Optional tags for filtering

        Example:
            >>> metrics.increment("cache.hits")
            >>> metrics.increment("errors.validation", tags={"type": "query"})
        """
        with self._lock:
            self._counters[name] += value

    def decrement(self, name: str, value: int = 1) -> None:
        """
        Decrement a counter metric.

        Args:
            name: Metric name
            value: Decrement amount
        """
        with self._lock:
            self._counters[name] -= value

    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric to a specific value.

        Args:
            name: Metric name (e.g., "memory.usage_mb")
            value: Current value
            tags: Optional tags

        Example:
            >>> metrics.set_gauge("cache.size", 1500)
        """
        with self._lock:
            self._gauges[name] = value

    def record_value(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a value in a histogram.

        Args:
            name: Metric name (e.g., "retrieval.latency_ms")
            value: Value to record
            tags: Optional tags

        Example:
            >>> metrics.record_value("query.length", 45)
        """
        with self._lock:
            self._histograms[name].append(value)

    def record_latency(self, name: str, duration_seconds: float) -> None:
        """
        Record latency in milliseconds.

        Args:
            name: Operation name
            duration_seconds: Duration in seconds

        Example:
            >>> start = time.time()
            >>> # ... do work ...
            >>> metrics.record_latency("dense.retrieve", time.time() - start)
        """
        self.record_value(f"{name}.latency_ms", duration_seconds * 1000)

    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        with self._lock:
            return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value."""
        with self._lock:
            return self._gauges.get(name)

    def get_histogram_summary(self, name: str) -> Optional[MetricSummary]:
        """
        Get summary statistics for a histogram.

        Args:
            name: Metric name

        Returns:
            MetricSummary with statistics or None if no data
        """
        with self._lock:
            if name not in self._histograms or not self._histograms[name]:
                return None

            values = np.array(list(self._histograms[name]))
            return MetricSummary(
                name=name,
                count=len(values),
                sum=float(np.sum(values)),
                min=float(np.min(values)),
                max=float(np.max(values)),
                mean=float(np.mean(values)),
                std=float(np.std(values)),
                p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)),
            )

    def summary(self) -> Dict[str, Any]:
        """
        Get complete metrics summary.

        Returns:
            Dictionary with all metrics

        Example:
            >>> summary = metrics.summary()
            >>> print(summary["counters"])
        """
        with self._lock:
            histogram_summaries = {}
            for name in self._histograms:
                summary = self.get_histogram_summary(name)
                if summary:
                    histogram_summaries[name] = summary.to_dict()

            return {
                "uptime_seconds": round(time.time() - self._start_time, 2),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": histogram_summaries,
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._start_time = time.time()

    def calculate_rate(self, counter_name: str, window_seconds: float = 60.0) -> float:
        """
        Calculate rate for a counter (per second).

        Args:
            counter_name: Counter name
            window_seconds: Time window for rate calculation

        Returns:
            Rate per second

        Example:
            >>> rate = metrics.calculate_rate("retrieval.requests")
            >>> print(f"Requests/sec: {rate:.2f}")
        """
        count = self.get_counter(counter_name)
        uptime = time.time() - self._start_time
        effective_window = min(uptime, window_seconds)

        if effective_window > 0:
            return count / effective_window
        return 0.0


class TimedMetric:
    """
    Context manager for timing operations.

    Example:
        >>> metrics = MetricsCollector()
        >>> with TimedMetric(metrics, "database.query"):
        ...     # do work
        ...     pass
    """

    def __init__(self, collector: MetricsCollector, name: str):
        """
        Initialize timed metric.

        Args:
            collector: Metrics collector instance
            name: Operation name
        """
        self.collector = collector
        self.name = name
        self.start_time: Optional[float] = None

    def __enter__(self) -> "TimedMetric":
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Record timing."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_latency(self.name, duration)

            # Track errors separately
            if exc_type is not None:
                self.collector.increment(f"{self.name}.errors")
            else:
                self.collector.increment(f"{self.name}.success")


# Global metrics collector instance
_global_metrics: Optional[MetricsCollector] = None
_metrics_lock = Lock()


def get_metrics() -> MetricsCollector:
    """
    Get or create global metrics collector.

    Returns:
        Global MetricsCollector instance

    Example:
        >>> metrics = get_metrics()
        >>> metrics.increment("requests")
    """
    global _global_metrics

    if _global_metrics is None:
        with _metrics_lock:
            if _global_metrics is None:
                _global_metrics = MetricsCollector()

    return _global_metrics


def reset_metrics() -> None:
    """Reset global metrics collector."""
    metrics = get_metrics()
    metrics.reset()


# Convenience functions for common patterns
def track_cache_hit() -> None:
    """Track a cache hit."""
    get_metrics().increment("cache.hits")


def track_cache_miss() -> None:
    """Track a cache miss."""
    get_metrics().increment("cache.misses")


def get_cache_hit_rate() -> float:
    """
    Calculate cache hit rate.

    Returns:
        Hit rate as percentage (0-100)
    """
    metrics = get_metrics()
    hits = metrics.get_counter("cache.hits")
    misses = metrics.get_counter("cache.misses")

    total = hits + misses
    if total == 0:
        return 0.0

    return (hits / total) * 100


def print_metrics_report() -> None:
    """Print formatted metrics report to console."""
    metrics = get_metrics()
    summary = metrics.summary()

    print("\n=== AutoRAG-Live Metrics ===")
    print(f"Uptime: {summary['uptime_seconds']:.1f}s")

    print("\nCounters:")
    for name, value in summary["counters"].items():
        print(f"  {name}: {value}")

    print("\nGauges:")
    for name, value in summary["gauges"].items():
        print(f"  {name}: {value}")

    print("\nHistograms:")
    for name, stats in summary["histograms"].items():
        print(f"  {name}:")
        print(f"    count: {stats['count']}, mean: {stats['mean']:.2f}ms")
        print(
            f"    p50: {stats['p50']:.2f}ms, p95: {stats['p95']:.2f}ms, p99: {stats['p99']:.2f}ms"
        )

    # Calculate cache hit rate if available
    if "cache.hits" in summary["counters"] or "cache.misses" in summary["counters"]:
        hit_rate = get_cache_hit_rate()
        print(f"\nCache Hit Rate: {hit_rate:.1f}%")
