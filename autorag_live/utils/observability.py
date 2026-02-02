"""
Real-time metrics and observability for production agentic RAG.

Provides comprehensive monitoring, tracing, and metrics collection
for debugging and optimization.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

from autorag_live.utils import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Point-in-time value
    HISTOGRAM = "histogram"  # Distribution
    TIMER = "timer"  # Timing measurements


@dataclass
class MetricValue:
    """Single metric value."""

    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSnapshot:
    """Snapshot of metric state."""

    name: str
    metric_type: MetricType
    value: float
    count: int
    min_value: float
    max_value: float
    avg_value: float
    p50: float
    p95: float
    p99: float
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "count": self.count,
            "min": self.min_value,
            "max": self.max_value,
            "avg": self.avg_value,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "tags": self.tags,
        }


class Metric:
    """Base metric class."""

    def __init__(self, name: str, metric_type: MetricType, max_history: int = 1000):
        self.name = name
        self.metric_type = metric_type
        self.max_history = max_history
        self.values: Deque[MetricValue] = deque(maxlen=max_history)
        self.tags: Dict[str, str] = {}

    def record(self, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        metric_value = MetricValue(
            timestamp=time.time(),
            value=value,
            tags=tags or {},
        )
        self.values.append(metric_value)

    def get_snapshot(self) -> MetricSnapshot:
        """Get current metric snapshot."""
        if not self.values:
            return MetricSnapshot(
                name=self.name,
                metric_type=self.metric_type,
                value=0.0,
                count=0,
                min_value=0.0,
                max_value=0.0,
                avg_value=0.0,
                p50=0.0,
                p95=0.0,
                p99=0.0,
            )

        values_list = [v.value for v in self.values]
        values_sorted = sorted(values_list)

        return MetricSnapshot(
            name=self.name,
            metric_type=self.metric_type,
            value=values_list[-1],
            count=len(values_list),
            min_value=min(values_list),
            max_value=max(values_list),
            avg_value=sum(values_list) / len(values_list),
            p50=self._percentile(values_sorted, 50),
            p95=self._percentile(values_sorted, 95),
            p99=self._percentile(values_sorted, 99),
            tags=self.tags,
        )

    def _percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not sorted_values:
            return 0.0

        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = f + 1

        if c >= len(sorted_values):
            return sorted_values[-1]

        d0 = sorted_values[f]
        d1 = sorted_values[c]

        return d0 + (d1 - d0) * (k - f)


class MetricsCollector:
    """
    Central metrics collector for agentic RAG.

    Tracks:
    - Query latency
    - Retrieval performance
    - Cache hit rates
    - Token usage
    - Error rates
    """

    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.start_time = time.time()

        # Initialize standard metrics
        self._initialize_standard_metrics()

    def _initialize_standard_metrics(self) -> None:
        """Initialize standard metrics."""
        # Latency metrics
        self.register_metric("query_latency_ms", MetricType.TIMER)
        self.register_metric("retrieval_latency_ms", MetricType.TIMER)
        self.register_metric("generation_latency_ms", MetricType.TIMER)

        # Performance metrics
        self.register_metric("cache_hit_rate", MetricType.GAUGE)
        self.register_metric("retrieval_score", MetricType.HISTOGRAM)
        self.register_metric("answer_confidence", MetricType.HISTOGRAM)

        # Resource metrics
        self.register_metric("token_count", MetricType.COUNTER)
        self.register_metric("document_count", MetricType.COUNTER)
        self.register_metric("active_queries", MetricType.GAUGE)

        # Error metrics
        self.register_metric("error_count", MetricType.COUNTER)
        self.register_metric("timeout_count", MetricType.COUNTER)

    def register_metric(self, name: str, metric_type: MetricType, max_history: int = 1000) -> None:
        """Register a new metric."""
        if name in self.metrics:
            logger.warning(f"Metric {name} already registered")
            return

        self.metrics[name] = Metric(name, metric_type, max_history)
        logger.debug(f"Registered metric: {name}")

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record metric value."""
        if name not in self.metrics:
            logger.warning(f"Unknown metric: {name}")
            return

        self.metrics[name].record(value, tags)

    def increment(self, name: str, value: float = 1.0) -> None:
        """Increment counter metric."""
        if name not in self.metrics:
            logger.warning(f"Unknown metric: {name}")
            return

        current = self.get_latest_value(name)
        self.record(name, current + value)

    def get_latest_value(self, name: str) -> float:
        """Get latest value for metric."""
        if name not in self.metrics:
            return 0.0

        metric = self.metrics[name]
        if not metric.values:
            return 0.0

        return metric.values[-1].value

    def get_snapshot(self, name: str) -> Optional[MetricSnapshot]:
        """Get snapshot for specific metric."""
        if name not in self.metrics:
            return None

        return self.metrics[name].get_snapshot()

    def get_all_snapshots(self) -> Dict[str, MetricSnapshot]:
        """Get snapshots for all metrics."""
        return {name: metric.get_snapshot() for name, metric in self.metrics.items()}

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        snapshots = self.get_all_snapshots()

        return {
            "uptime_seconds": time.time() - self.start_time,
            "metrics": {name: snap.to_dict() for name, snap in snapshots.items()},
        }


class Timer:
    """Context manager for timing operations."""

    def __init__(
        self,
        collector: MetricsCollector,
        metric_name: str,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.collector = collector
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time: Optional[float] = None

    def __enter__(self) -> Timer:
        """Start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timer and record."""
        if self.start_time is not None:
            elapsed_ms = (time.time() - self.start_time) * 1000
            self.collector.record(self.metric_name, elapsed_ms, self.tags)


class DistributedTracer:
    """
    Distributed tracing for agentic RAG operations.

    Tracks execution flow across components.
    """

    def __init__(self):
        self.traces: Dict[str, Trace] = {}
        self.active_traces: Dict[str, str] = {}  # operation_id -> trace_id

    def start_trace(self, trace_id: str, operation: str) -> Trace:
        """Start a new trace."""
        trace = Trace(trace_id, operation)
        self.traces[trace_id] = trace
        return trace

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID."""
        return self.traces.get(trace_id)

    def end_trace(self, trace_id: str) -> None:
        """End a trace."""
        if trace_id in self.traces:
            self.traces[trace_id].end()


@dataclass
class Span:
    """Single span in trace."""

    span_id: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    parent_span_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Calculate span duration."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "span_id": self.span_id,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "parent_span_id": self.parent_span_id,
            "tags": self.tags,
            "logs": self.logs,
        }


class Trace:
    """Trace containing multiple spans."""

    def __init__(self, trace_id: str, operation: str):
        self.trace_id = trace_id
        self.operation = operation
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.spans: Dict[str, Span] = {}
        self.root_span_id: Optional[str] = None

    def start_span(
        self,
        span_id: str,
        operation: str,
        parent_span_id: Optional[str] = None,
    ) -> Span:
        """Start a new span."""
        span = Span(
            span_id=span_id,
            operation=operation,
            start_time=time.time(),
            parent_span_id=parent_span_id,
        )

        self.spans[span_id] = span

        if self.root_span_id is None:
            self.root_span_id = span_id

        return span

    def end_span(self, span_id: str) -> None:
        """End a span."""
        if span_id in self.spans:
            self.spans[span_id].end_time = time.time()

    def end(self) -> None:
        """End trace."""
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        """Calculate trace duration."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "trace_id": self.trace_id,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "spans": {sid: span.to_dict() for sid, span in self.spans.items()},
        }


class ObservabilityHub:
    """
    Central hub for observability.

    Combines metrics and tracing.
    """

    def __init__(self):
        self.metrics = MetricsCollector()
        self.tracer = DistributedTracer()

    def record_query(
        self,
        query: str,
        latency_ms: float,
        success: bool,
        confidence: float = 0.0,
    ) -> None:
        """Record query execution."""
        self.metrics.record("query_latency_ms", latency_ms)
        self.metrics.increment("active_queries", -1 if success else 0)

        if success:
            self.metrics.record("answer_confidence", confidence)
        else:
            self.metrics.increment("error_count")

    def record_retrieval(
        self,
        latency_ms: float,
        doc_count: int,
        avg_score: float,
    ) -> None:
        """Record retrieval operation."""
        self.metrics.record("retrieval_latency_ms", latency_ms)
        self.metrics.increment("document_count", doc_count)
        self.metrics.record("retrieval_score", avg_score)

    def record_cache_access(self, hit: bool) -> None:
        """Record cache access."""
        # Update hit rate
        current_rate = self.metrics.get_latest_value("cache_hit_rate")
        new_rate = current_rate * 0.9 + (1.0 if hit else 0.0) * 0.1  # EMA
        self.metrics.record("cache_hit_rate", new_rate)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display."""
        return {
            "summary": self.metrics.get_summary(),
            "key_metrics": {
                "avg_query_latency_ms": self.metrics.get_snapshot("query_latency_ms").avg_value,
                "cache_hit_rate": self.metrics.get_latest_value("cache_hit_rate"),
                "error_rate": self._calculate_error_rate(),
                "throughput_qps": self._calculate_throughput(),
            },
        }

    def _calculate_error_rate(self) -> float:
        """Calculate error rate."""
        errors = self.metrics.get_latest_value("error_count")
        total = errors + self.metrics.get_snapshot("query_latency_ms").count
        return errors / total if total > 0 else 0.0

    def _calculate_throughput(self) -> float:
        """Calculate queries per second."""
        uptime = time.time() - self.metrics.start_time
        total_queries = self.metrics.get_snapshot("query_latency_ms").count
        return total_queries / uptime if uptime > 0 else 0.0


# Global observability hub
_observability_hub: Optional[ObservabilityHub] = None


def get_observability_hub() -> ObservabilityHub:
    """Get global observability hub."""
    global _observability_hub
    if _observability_hub is None:
        _observability_hub = ObservabilityHub()
    return _observability_hub


# Example usage
def example_observability():
    """Example of observability usage."""
    hub = get_observability_hub()

    # Simulate queries
    for i in range(10):
        # Start trace
        trace = hub.tracer.start_trace(f"trace_{i}", "query_processing")

        # Simulate retrieval
        _ = trace.start_span("retrieval", "retrieve_documents")
        time.sleep(0.05)
        trace.end_span("retrieval")
        hub.record_retrieval(50.0, 5, 0.8)

        # Simulate generation
        _ = trace.start_span("generation", "generate_answer", "retrieval")
        time.sleep(0.1)
        trace.end_span("generation")

        # End trace
        hub.tracer.end_trace(f"trace_{i}")

        # Record query
        hub.record_query(f"Query {i}", 150.0, True, 0.9)

        # Cache access
        hub.record_cache_access(i % 3 == 0)

    # Get dashboard data
    dashboard = hub.get_dashboard_data()

    print("Dashboard Data:")
    print(f"  Avg Latency: {dashboard['key_metrics']['avg_query_latency_ms']:.1f}ms")
    print(f"  Cache Hit Rate: {dashboard['key_metrics']['cache_hit_rate']:.2%}")
    print(f"  Error Rate: {dashboard['key_metrics']['error_rate']:.2%}")
    print(f"  Throughput: {dashboard['key_metrics']['throughput_qps']:.2f} QPS")


if __name__ == "__main__":
    example_observability()
