"""Observability and metrics export for AutoRAG-Live."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LatencyMetric:
    """Latency metric for operations."""

    operation: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class CacheMetric:
    """Cache hit/miss metrics."""

    cache_name: str
    hits: int = 0
    misses: int = 0
    size: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


class MetricsExporter:
    """Export metrics for observability (OpenTelemetry, Prometheus compatible)."""

    def __init__(self, service_name: str = "autorag-live"):
        """Initialize metrics exporter.

        Args:
            service_name: Name of the service for labeling metrics
        """
        self.service_name = service_name
        self.latency_metrics: list[LatencyMetric] = []
        self.cache_metrics: Dict[str, CacheMetric] = {}

    def record_latency(
        self, operation: str, duration_ms: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record operation latency.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            tags: Optional tags (e.g., {"model": "gpt-4", "status": "success"})
        """
        metric = LatencyMetric(operation, duration_ms, tags=tags or {})
        self.latency_metrics.append(metric)

    def record_cache_hit(self, cache_name: str) -> None:
        """Record cache hit.

        Args:
            cache_name: Name of the cache
        """
        if cache_name not in self.cache_metrics:
            self.cache_metrics[cache_name] = CacheMetric(cache_name)
        self.cache_metrics[cache_name].hits += 1

    def record_cache_miss(self, cache_name: str) -> None:
        """Record cache miss.

        Args:
            cache_name: Name of the cache
        """
        if cache_name not in self.cache_metrics:
            self.cache_metrics[cache_name] = CacheMetric(cache_name)
        self.cache_metrics[cache_name].misses += 1

    def get_cache_hit_rate(self, cache_name: str) -> float:
        """Get cache hit rate.

        Args:
            cache_name: Name of the cache

        Returns:
            Hit rate (0.0 to 1.0)
        """
        if cache_name not in self.cache_metrics:
            return 0.0
        return self.cache_metrics[cache_name].hit_rate

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # Latency metrics
        for metric in self.latency_metrics:
            tags_str = ",".join([f'{k}="{v}"' for k, v in metric.tags.items()])
            if tags_str:
                tags_str = f"{{{tags_str}}}"
            lines.append(f"autorag_operation_latency_ms{tags_str} {metric.duration_ms}")

        # Cache metrics
        for cache_name, cache_metric in self.cache_metrics.items():
            lines.append(f'autorag_cache_hits{{cache="{cache_name}"}} {cache_metric.hits}')
            lines.append(f'autorag_cache_misses{{cache="{cache_name}"}} {cache_metric.misses}')
            lines.append(f'autorag_cache_hit_rate{{cache="{cache_name}"}} {cache_metric.hit_rate}')

        return "\n".join(lines)

    def export_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary.

        Returns:
            Dictionary with metrics
        """
        return {
            "service_name": self.service_name,
            "latency_metrics": [
                {
                    "operation": m.operation,
                    "duration_ms": m.duration_ms,
                    "timestamp": m.timestamp,
                    "tags": m.tags,
                }
                for m in self.latency_metrics
            ],
            "cache_metrics": {
                name: {
                    "cache_name": metric.cache_name,
                    "hits": metric.hits,
                    "misses": metric.misses,
                    "hit_rate": metric.hit_rate,
                    "size": metric.size,
                }
                for name, metric in self.cache_metrics.items()
            },
        }

    def clear(self) -> None:
        """Clear all recorded metrics."""
        self.latency_metrics.clear()
        self.cache_metrics.clear()
