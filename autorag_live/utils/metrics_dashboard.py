"""
Metrics Dashboard for Agentic RAG Pipeline.

Comprehensive metrics collection, aggregation, and visualization
for monitoring all components of the RAG pipeline.
"""

import json
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class ComponentType(str, Enum):
    """Pipeline component types."""

    LLM = "llm"
    RETRIEVAL = "retrieval"
    ROUTER = "router"
    AGENT = "agent"
    RERANKER = "reranker"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    SAFETY = "safety"
    CACHE = "cache"
    PIPELINE = "pipeline"
    EVALUATION = "evaluation"


@dataclass
class MetricPoint:
    """A single metric data point."""

    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricDefinition:
    """Definition of a metric."""

    name: str
    metric_type: MetricType
    component: ComponentType
    description: str = ""
    unit: str = ""
    labels: list[str] = field(default_factory=list)


class Metric:
    """A single metric with its data points."""

    def __init__(self, definition: MetricDefinition, max_points: int = 10000):
        """Initialize metric."""
        self.definition = definition
        self.max_points = max_points
        self._points: list[MetricPoint] = []
        self._lock = threading.Lock()

        self._counter_value = 0.0
        self._gauge_value = 0.0
        self._histogram_values: list[float] = []

    def record(self, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Record a metric value."""
        with self._lock:
            point = MetricPoint(value=value, labels=labels or {})
            self._points.append(point)

            if len(self._points) > self.max_points:
                self._points = self._points[-self.max_points :]

            if self.definition.metric_type == MetricType.COUNTER:
                self._counter_value += value
            elif self.definition.metric_type == MetricType.GAUGE:
                self._gauge_value = value
            elif self.definition.metric_type in (MetricType.HISTOGRAM, MetricType.TIMER):
                self._histogram_values.append(value)
                if len(self._histogram_values) > self.max_points:
                    self._histogram_values = self._histogram_values[-self.max_points :]

    def increment(self, value: float = 1.0) -> None:
        """Increment counter metric."""
        self.record(value)

    def set_gauge(self, value: float) -> None:
        """Set gauge metric value."""
        with self._lock:
            self._gauge_value = value
            self._points.append(MetricPoint(value=value))

    def get_current(self) -> float:
        """Get current metric value."""
        with self._lock:
            if self.definition.metric_type == MetricType.COUNTER:
                return self._counter_value
            elif self.definition.metric_type == MetricType.GAUGE:
                return self._gauge_value
            elif self._points:
                return self._points[-1].value
            return 0.0

    def get_stats(self) -> dict[str, float]:
        """Get statistical summary."""
        with self._lock:
            if self.definition.metric_type in (MetricType.HISTOGRAM, MetricType.TIMER):
                values = self._histogram_values
            else:
                values = [p.value for p in self._points]

            if not values:
                return {"count": 0, "sum": 0, "mean": 0, "min": 0, "max": 0}

            return {
                "count": len(values),
                "sum": sum(values),
                "mean": statistics.mean(values),
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values) if values else 0,
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            }

    def get_percentile(self, percentile: float) -> float:
        """Get percentile value."""
        with self._lock:
            if self.definition.metric_type in (MetricType.HISTOGRAM, MetricType.TIMER):
                values = sorted(self._histogram_values)
            else:
                values = sorted(p.value for p in self._points)

            if not values:
                return 0.0

            index = int(len(values) * percentile / 100)
            return values[min(index, len(values) - 1)]

    def get_points(self, since: Optional[datetime] = None, limit: int = 1000) -> list[MetricPoint]:
        """Get metric data points."""
        with self._lock:
            points = self._points
            if since:
                points = [p for p in points if p.timestamp >= since]
            return points[-limit:]


class Timer:
    """Context manager for timing operations."""

    def __init__(self, metric: Metric, labels: Optional[dict[str, str]] = None):
        """Initialize timer."""
        self.metric = metric
        self.labels = labels or {}
        self.start_time = 0.0

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and record."""
        elapsed_ms = (time.time() - self.start_time) * 1000
        self.metric.record(elapsed_ms, self.labels)


class MetricsRegistry:
    """Registry for all metrics."""

    def __init__(self):
        """Initialize registry."""
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()

    def register(self, definition: MetricDefinition) -> Metric:
        """Register a new metric."""
        with self._lock:
            if definition.name in self._metrics:
                return self._metrics[definition.name]

            metric = Metric(definition)
            self._metrics[definition.name] = metric
            return metric

    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self._metrics.get(name)

    def get_or_create(
        self,
        name: str,
        metric_type: MetricType,
        component: ComponentType,
        description: str = "",
    ) -> Metric:
        """Get existing metric or create new one."""
        if name in self._metrics:
            return self._metrics[name]

        definition = MetricDefinition(
            name=name,
            metric_type=metric_type,
            component=component,
            description=description,
        )
        return self.register(definition)

    def list_metrics(self, component: Optional[ComponentType] = None) -> list[MetricDefinition]:
        """List all registered metrics."""
        metrics = list(self._metrics.values())
        if component:
            metrics = [m for m in metrics if m.definition.component == component]
        return [m.definition for m in metrics]

    def get_all_current_values(self) -> dict[str, float]:
        """Get current values for all metrics."""
        return {name: m.get_current() for name, m in self._metrics.items()}


_default_registry = MetricsRegistry()


def get_default_registry() -> MetricsRegistry:
    """Get the default metrics registry."""
    return _default_registry


class MetricsDashboard:
    """Dashboard for viewing and analyzing metrics."""

    def __init__(self, registry: Optional[MetricsRegistry] = None):
        """Initialize dashboard."""
        self.registry = registry or get_default_registry()
        self._alerts: list[dict[str, Any]] = []
        self._alert_callbacks: list[Callable[[str, float, float], None]] = []

    def get_component_summary(self, component: ComponentType) -> dict[str, Any]:
        """Get summary metrics for a component."""
        metrics = [
            m for m in self.registry._metrics.values() if m.definition.component == component
        ]

        summary = {
            "component": component.value,
            "metrics_count": len(metrics),
            "metrics": {},
        }

        for metric in metrics:
            summary["metrics"][metric.definition.name] = {
                "type": metric.definition.metric_type.value,
                "current": metric.get_current(),
                "stats": metric.get_stats(),
            }

        return summary

    def get_pipeline_overview(self) -> dict[str, Any]:
        """Get overview of entire pipeline metrics."""
        overview = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "totals": {
                "total_requests": 0,
                "total_errors": 0,
                "avg_latency_ms": 0,
            },
        }

        latencies = []
        for component in ComponentType:
            summary = self.get_component_summary(component)
            if summary["metrics"]:
                overview["components"][component.value] = summary

                for name, data in summary["metrics"].items():
                    if "latency" in name or "time" in name:
                        if data["stats"]["count"] > 0:
                            latencies.append(data["stats"]["mean"])
                    if "request" in name and "counter" in data["type"]:
                        overview["totals"]["total_requests"] += data["current"]
                    if "error" in name:
                        overview["totals"]["total_errors"] += data["current"]

        if latencies:
            overview["totals"]["avg_latency_ms"] = statistics.mean(latencies)

        return overview

    def get_time_series(
        self,
        metric_name: str,
        duration_minutes: int = 60,
        bucket_seconds: int = 60,
    ) -> list[dict[str, Any]]:
        """Get time series data for a metric."""
        metric = self.registry.get(metric_name)
        if not metric:
            return []

        since = datetime.now() - timedelta(minutes=duration_minutes)
        points = metric.get_points(since=since)

        buckets: dict[int, list[float]] = defaultdict(list)
        for point in points:
            bucket_key = int(point.timestamp.timestamp()) // bucket_seconds
            buckets[bucket_key].append(point.value)

        series = []
        for bucket_key in sorted(buckets.keys()):
            values = buckets[bucket_key]
            ts = datetime.fromtimestamp(bucket_key * bucket_seconds)
            series.append(
                {
                    "timestamp": ts.isoformat(),
                    "value": statistics.mean(values),
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                }
            )

        return series

    def add_alert(
        self,
        metric_name: str,
        threshold: float,
        comparison: str = "gt",
        callback: Optional[Callable[[str, float, float], None]] = None,
    ) -> None:
        """Add an alert for a metric."""
        alert = {
            "metric_name": metric_name,
            "threshold": threshold,
            "comparison": comparison,
        }
        self._alerts.append(alert)
        if callback:
            self._alert_callbacks.append(callback)

    def check_alerts(self) -> list[dict[str, Any]]:
        """Check all alerts and return triggered ones."""
        triggered = []

        for alert in self._alerts:
            metric = self.registry.get(alert["metric_name"])
            if not metric:
                continue

            current = metric.get_current()
            threshold = alert["threshold"]
            comparison = alert["comparison"]

            is_triggered = False
            if comparison == "gt" and current > threshold:
                is_triggered = True
            elif comparison == "lt" and current < threshold:
                is_triggered = True
            elif comparison == "eq" and current == threshold:
                is_triggered = True
            elif comparison == "gte" and current >= threshold:
                is_triggered = True
            elif comparison == "lte" and current <= threshold:
                is_triggered = True

            if is_triggered:
                triggered.append(
                    {
                        "metric": alert["metric_name"],
                        "current_value": current,
                        "threshold": threshold,
                        "comparison": comparison,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                for callback in self._alert_callbacks:
                    try:
                        callback(alert["metric_name"], current, threshold)
                    except Exception:
                        pass

        return triggered

    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "metrics": {},
        }

        for name, metric in self.registry._metrics.items():
            data["metrics"][name] = {
                "definition": {
                    "name": metric.definition.name,
                    "type": metric.definition.metric_type.value,
                    "component": metric.definition.component.value,
                    "description": metric.definition.description,
                },
                "current_value": metric.get_current(),
                "stats": metric.get_stats(),
            }

        if format == "json":
            return json.dumps(data, indent=2)
        elif format == "prometheus":
            return self._to_prometheus_format(data)
        else:
            return json.dumps(data, indent=2)

    def _to_prometheus_format(self, data: dict[str, Any]) -> str:
        """Convert metrics to Prometheus format."""
        lines = []
        for name, metric_data in data["metrics"].items():
            clean_name = name.replace(".", "_").replace("-", "_")
            metric_type = metric_data["definition"]["type"]

            lines.append(f"# HELP {clean_name} {metric_data['definition']['description']}")
            lines.append(f"# TYPE {clean_name} {metric_type}")

            if metric_type in ("counter", "gauge"):
                lines.append(f"{clean_name} {metric_data['current_value']}")
            elif metric_type in ("histogram", "timer"):
                stats = metric_data["stats"]
                lines.append(f"{clean_name}_count {stats['count']}")
                lines.append(f"{clean_name}_sum {stats['sum']}")
                lines.append(f"{clean_name}_mean {stats['mean']}")

        return "\n".join(lines)


def create_llm_metrics(registry: Optional[MetricsRegistry] = None) -> dict[str, Metric]:
    """Create standard LLM metrics."""
    reg = registry or get_default_registry()
    return {
        "requests": reg.get_or_create(
            "llm.requests.total",
            MetricType.COUNTER,
            ComponentType.LLM,
            "Total LLM requests",
        ),
        "latency": reg.get_or_create(
            "llm.latency.ms",
            MetricType.TIMER,
            ComponentType.LLM,
            "LLM request latency in milliseconds",
        ),
        "tokens_input": reg.get_or_create(
            "llm.tokens.input",
            MetricType.COUNTER,
            ComponentType.LLM,
            "Total input tokens",
        ),
        "tokens_output": reg.get_or_create(
            "llm.tokens.output",
            MetricType.COUNTER,
            ComponentType.LLM,
            "Total output tokens",
        ),
        "errors": reg.get_or_create(
            "llm.errors.total",
            MetricType.COUNTER,
            ComponentType.LLM,
            "Total LLM errors",
        ),
    }


def create_retrieval_metrics(
    registry: Optional[MetricsRegistry] = None,
) -> dict[str, Metric]:
    """Create standard retrieval metrics."""
    reg = registry or get_default_registry()
    return {
        "queries": reg.get_or_create(
            "retrieval.queries.total",
            MetricType.COUNTER,
            ComponentType.RETRIEVAL,
            "Total retrieval queries",
        ),
        "latency": reg.get_or_create(
            "retrieval.latency.ms",
            MetricType.TIMER,
            ComponentType.RETRIEVAL,
            "Retrieval latency in milliseconds",
        ),
        "results_count": reg.get_or_create(
            "retrieval.results.count",
            MetricType.HISTOGRAM,
            ComponentType.RETRIEVAL,
            "Number of results per query",
        ),
        "cache_hits": reg.get_or_create(
            "retrieval.cache.hits",
            MetricType.COUNTER,
            ComponentType.RETRIEVAL,
            "Retrieval cache hits",
        ),
    }


def create_pipeline_metrics(
    registry: Optional[MetricsRegistry] = None,
) -> dict[str, Metric]:
    """Create standard pipeline metrics."""
    reg = registry or get_default_registry()
    return {
        "requests": reg.get_or_create(
            "pipeline.requests.total",
            MetricType.COUNTER,
            ComponentType.PIPELINE,
            "Total pipeline requests",
        ),
        "latency": reg.get_or_create(
            "pipeline.latency.ms",
            MetricType.TIMER,
            ComponentType.PIPELINE,
            "End-to-end pipeline latency",
        ),
        "success_rate": reg.get_or_create(
            "pipeline.success.rate",
            MetricType.GAUGE,
            ComponentType.PIPELINE,
            "Pipeline success rate",
        ),
        "active_requests": reg.get_or_create(
            "pipeline.requests.active",
            MetricType.GAUGE,
            ComponentType.PIPELINE,
            "Currently active requests",
        ),
    }


__all__ = [
    "MetricType",
    "ComponentType",
    "MetricPoint",
    "MetricDefinition",
    "Metric",
    "Timer",
    "MetricsRegistry",
    "MetricsDashboard",
    "get_default_registry",
    "create_llm_metrics",
    "create_retrieval_metrics",
    "create_pipeline_metrics",
]
