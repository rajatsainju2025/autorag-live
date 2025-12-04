"""
Performance monitoring and profiling utilities for AutoRAG-Live.

This module provides tools for monitoring system performance, collecting
metrics, and profiling expensive operations with minimal overhead.
"""

import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

import psutil

from autorag_live.utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Cache process object to avoid repeated lookups
_PROCESS = psutil.Process()
_PROCESS_LOCK = threading.Lock()


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation."""

    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    cpu_percent: Optional[float] = None
    memory_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, skip_system_metrics: bool = False) -> None:
        """Mark operation as complete and calculate metrics.

        Args:
            skip_system_metrics: Skip expensive system metric collection
        """
        self.end_time = time.perf_counter()  # Use perf_counter for better precision
        self.duration = self.end_time - self.start_time

        if not skip_system_metrics:
            # Get system metrics with cached process
            with _PROCESS_LOCK:
                try:
                    self.cpu_percent = _PROCESS.cpu_percent()
                    memory_info = _PROCESS.memory_info()
                    self.memory_mb = (
                        memory_info.rss / 1048576
                    )  # Division is faster than / 1024 / 1024

                    # Peak memory (approximate)
                    self.memory_peak_mb = self.memory_mb
                except Exception:
                    # Silently fail on metrics collection errors
                    pass


@dataclass
class AggregatedMetrics:
    """Aggregated performance metrics."""

    operation_name: str
    call_count: int = 0
    total_duration: float = 0.0
    avg_duration: float = 0.0
    min_duration: float = float("inf")
    max_duration: float = 0.0
    durations: List[float] = field(default_factory=list)
    total_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    memory_peaks: List[float] = field(default_factory=list)

    def add_metrics(self, metrics: PerformanceMetrics) -> None:
        """Add metrics to aggregation."""
        if metrics.duration is None:
            return

        self.call_count += 1
        self.total_duration += metrics.duration
        self.avg_duration = self.total_duration / self.call_count
        self.min_duration = min(self.min_duration, metrics.duration)
        self.max_duration = max(self.max_duration, metrics.duration)
        self.durations.append(metrics.duration)

        if metrics.memory_mb:
            self.total_memory_mb += metrics.memory_mb
            self.avg_memory_mb = self.total_memory_mb / self.call_count

        if metrics.memory_peak_mb:
            self.memory_peaks.append(metrics.memory_peak_mb)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "operation": self.operation_name,
            "calls": self.call_count,
            "total_duration": round(self.total_duration, 3),
            "avg_duration": round(self.avg_duration, 3),
            "min_duration": round(self.min_duration, 3) if self.min_duration != float("inf") else 0,
            "max_duration": round(self.max_duration, 3),
            "p50_duration": round(statistics.median(self.durations), 3) if self.durations else 0,
            "p95_duration": round(statistics.quantiles(self.durations, n=20)[18], 3)
            if len(self.durations) >= 20
            else round(self.max_duration, 3),
            "avg_memory_mb": round(self.avg_memory_mb, 1),
            "peak_memory_mb": round(max(self.memory_peaks), 1) if self.memory_peaks else 0,
        }


class PerformanceMonitor:
    """Central performance monitoring system."""

    def __init__(self) -> None:
        self._metrics: Dict[str, AggregatedMetrics] = {}
        self._lock = threading.Lock()
        self._enabled = True

    def enable(self) -> None:
        """Enable performance monitoring."""
        self._enabled = True

    def disable(self) -> None:
        """Disable performance monitoring."""
        self._enabled = False

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        if not self._enabled:
            return

        with self._lock:
            if metrics.operation_name not in self._metrics:
                self._metrics[metrics.operation_name] = AggregatedMetrics(metrics.operation_name)
            agg = self._metrics[metrics.operation_name]
            agg.add_metrics(metrics)

    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated metrics."""
        with self._lock:
            if operation_name:
                agg = self._metrics.get(operation_name)
                return agg.get_summary() if agg else {}
            else:
                return {name: agg.get_summary() for name, agg in self._metrics.items()}

    def reset(self, operation_name: Optional[str] = None) -> None:
        """Reset metrics."""
        with self._lock:
            if operation_name:
                self._metrics.pop(operation_name, None)
            else:
                self._metrics.clear()

    def log_summary(self) -> None:
        """Log performance summary."""
        if not self._enabled:
            return

        metrics = self.get_metrics()
        if not metrics:
            logger.info("No performance metrics recorded")
            return

        logger.info("=== Performance Summary ===")
        for op_name, op_metrics in metrics.items():
            logger.info(f"Operation: {op_name}")
            logger.info(f"  Calls: {op_metrics['calls']}")
            logger.info(f"  Total Duration: {op_metrics['total_duration']}s")
            logger.info(f"  Avg Duration: {op_metrics['avg_duration']}s")
            logger.info(f"  P50 Duration: {op_metrics['p50_duration']}s")
            logger.info(f"  P95 Duration: {op_metrics['p95_duration']}s")
            logger.info(f"  Peak Memory: {op_metrics['peak_memory_mb']}MB")
        logger.info("==========================")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


@contextmanager
def monitor_performance(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for monitoring performance metrics.

    Automatically tracks execution time, memory usage, and custom metadata
    for the code block. Metrics are recorded to the global performance monitor.

    Args:
        operation_name: Descriptive name for the operation being monitored.
        metadata: Optional dictionary of additional context to record.

    Yields:
        PerformanceMetrics: Metrics object that can be updated during execution.

    Example:
        >>> with monitor_performance("dense_retrieval", {"corpus_size": 1000}):
        ...     results = dense_retrieve(query, corpus)
        >>>
        >>> # Access metrics during execution
        >>> with monitor_performance("complex_operation") as metrics:
        ...     # Do work
        ...     metrics.add_metadata("items_processed", 100)
    """
    if not performance_monitor._enabled:
        yield
        return

    metrics = PerformanceMetrics(
        operation_name=operation_name, start_time=time.time(), metadata=metadata or {}
    )

    try:
        yield metrics
    finally:
        metrics.complete()
        performance_monitor.record_metrics(metrics)


def profile_function(operation_name: Optional[str] = None):
    """Decorator for profiling function performance.

    Wraps a function to automatically track its execution metrics including
    time, memory usage, and call frequency. If operation_name is not provided,
    uses the function's module and name.

    Args:
        operation_name: Optional custom name for the operation.
            If None, uses "module.function_name".

    Returns:
        Callable: Decorator function that wraps the target function.

    Example:
        >>> @profile_function("custom_retrieve")
        ... def retrieve_documents(query: str, k: int = 5):
        ...     # Function implementation
        ...     pass
        >>>
        >>> # Auto-named profiling
        >>> @profile_function()
        ... def complex_calculation(data):
        ...     # Function implementation
        ...     pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"

        def wrapper(*args, **kwargs) -> T:
            with monitor_performance(op_name):
                result = func(*args, **kwargs)
                return result

        return wrapper

    return decorator


class SystemMonitor:
    """System resource monitoring."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._metrics_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def start_monitoring(self) -> None:
        """Start background system monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Started system monitoring")

    def stop_monitoring(self) -> None:
        """Stop background system monitoring."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Stopped system monitoring")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._get_system_metrics()
                with self._lock:
                    self._metrics_history.append(metrics)
                    # Keep only last 1000 entries
                    if len(self._metrics_history) > 1000:
                        self._metrics_history = self._metrics_history[-1000:]
            except Exception as e:
                logger.warning(f"Error in system monitoring: {e}")

            time.sleep(self.interval)

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": psutil.virtual_memory().used / 1024 / 1024,
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "process_count": len(psutil.pids()),
        }

    def get_recent_metrics(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent system metrics."""
        with self._lock:
            return self._metrics_history[-last_n:] if self._metrics_history else []

    def get_average_metrics(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get average system metrics."""
        with self._lock:
            metrics = self._metrics_history[-last_n:] if last_n else self._metrics_history

        if not metrics:
            return {}

        # Calculate averages
        avg_metrics = {}
        keys = [
            "cpu_percent",
            "memory_percent",
            "memory_used_mb",
            "disk_usage_percent",
            "process_count",
        ]

        for key in keys:
            values = [m[key] for m in metrics if key in m]
            if values:
                avg_metrics[f"avg_{key}"] = round(statistics.mean(values), 2)
                avg_metrics[f"max_{key}"] = round(max(values), 2)
                avg_metrics[f"min_{key}"] = round(min(values), 2)

        return avg_metrics


# Global system monitor instance
system_monitor = SystemMonitor()


def start_system_monitoring(interval: float = 1.0) -> None:
    """Start system monitoring."""
    global system_monitor
    system_monitor = SystemMonitor(interval)
    system_monitor.start_monitoring()


def stop_system_monitoring() -> None:
    """Stop system monitoring."""
    system_monitor.stop_monitoring()


def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics."""
    return system_monitor.get_average_metrics(last_n=10)


# Optimization: perf(pool): add executor pooling for concurrent retrieval
