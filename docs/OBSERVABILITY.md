# Observability Guide for AutoRAG-Live

This guide explains how to monitor, trace, and debug AutoRAG-Live systems using built-in observability tools.

## Table of Contents

1. [Observability Overview](#observability-overview)
2. [Metrics Collection](#metrics-collection)
3. [Latency Tracking](#latency-tracking)
4. [Cache Metrics](#cache-metrics)
5. [Custom Metrics](#custom-metrics)
6. [Exporting Metrics](#exporting-metrics)
7. [Dashboards and Visualization](#dashboards-and-visualization)
8. [Performance Profiling](#performance-profiling)
9. [Best Practices](#best-practices)

## Observability Overview

Observability in AutoRAG-Live provides visibility into system behavior through three pillars:

1. **Metrics**: Quantitative measurements (latency, throughput, hit rate)
2. **Logs**: Timestamped events with context
3. **Traces**: Request flows across system components

### Why Observability Matters

For RAG systems:
- **Performance Optimization**: Identify bottlenecks in retrieval/reranking
- **Reliability**: Detect failures before users notice
- **Cost Management**: Monitor resource utilization
- **User Experience**: Understand query latency and success rates
- **System Health**: Track cache hit rates, indexing lag, etc.

### Architecture Overview

```
Application
    ↓
MetricsExporter (Core)
    ├→ Latency Metrics
    ├→ Cache Metrics
    ├→ Custom Metrics
    ↓
Exporters
    ├→ Prometheus Format
    ├→ JSON Export
    ├→ Custom Backends
    ↓
Monitoring Stack
    ├→ Prometheus/Grafana
    ├→ Datadog/New Relic
    ├→ ELK Stack
    └→ Custom Alerting
```

## Metrics Collection

### MetricsExporter

Core metrics collection component:

```python
from autorag_live.core.metrics_export import MetricsExporter

# Create metrics exporter
exporter = MetricsExporter()

# Record latency for a retrieval operation
import time

start = time.time()
results = retriever.retrieve(query)
duration = (time.time() - start) * 1000  # Convert to ms

exporter.record_latency(
    operation="retrieval",
    duration_ms=duration,
    tags={
        "retriever_type": "semantic",
        "query_length": str(len(query)),
    }
)

# Record cache metrics
exporter.record_cache_hit(
    operation="retrieval_cache",
    hit=True,
    tags={"cache_backend": "redis"}
)

# Get metrics
metrics = exporter.export_metrics()
print(f"Avg latency: {metrics['avg_latency']:.2f}ms")
print(f"Cache hit rate: {metrics['hit_rate']:.1%}")
```

### Latency Metrics

Track operation performance:

```python
from autorag_live.core.metrics_export import LatencyMetric
import time

class LatencyTracker:
    """Track and analyze latency metrics."""

    def __init__(self, exporter: MetricsExporter):
        self.exporter = exporter

    def measure(self, operation: str, func, *args, **kwargs):
        """Measure function execution time."""
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = (time.perf_counter() - start) * 1000
            self.exporter.record_latency(
                operation=operation,
                duration_ms=duration,
                tags=kwargs.get("tags", {})
            )

# Usage
tracker = LatencyTracker(exporter)

def retrieve_and_rerank(query):
    # Measure individual operations
    results = tracker.measure(
        "retrieval",
        retriever.retrieve,
        query,
        tags={"query_type": "semantic"}
    )

    reranked = tracker.measure(
        "reranking",
        reranker.rerank,
        results,
        query,
        tags={"reranker_type": "mmr"}
    )

    return reranked
```

### Percentile Analysis

Analyze latency distributions:

```python
class LatencyAnalyzer:
    """Analyze latency metrics with percentiles."""

    def __init__(self, exporter: MetricsExporter):
        self.exporter = exporter

    def get_percentiles(self, operation: str) -> dict:
        """Get latency percentiles for operation."""
        metrics = self.exporter.get_latency_metrics(operation)

        if not metrics:
            return {}

        durations = sorted([m.duration_ms for m in metrics])
        n = len(durations)

        return {
            "min": durations[0],
            "p50": durations[n // 2],
            "p95": durations[int(n * 0.95)],
            "p99": durations[int(n * 0.99)],
            "max": durations[-1],
            "avg": sum(durations) / n,
        }

    def identify_slowdowns(
        self,
        operation: str,
        baseline_ms: float = 100
    ) -> List[LatencyMetric]:
        """Identify metrics above baseline."""
        metrics = self.exporter.get_latency_metrics(operation)
        return [m for m in metrics if m.duration_ms > baseline_ms]

    def get_trend(self, operation: str, window: int = 100) -> dict:
        """Get latency trend over recent operations."""
        metrics = self.exporter.get_latency_metrics(operation)

        if len(metrics) < window:
            recent = metrics
        else:
            recent = metrics[-window:]

        early = sum(m.duration_ms for m in recent[:window//2]) / (window//2)
        late = sum(m.duration_ms for m in recent[window//2:]) / (window//2)

        return {
            "early_avg": early,
            "late_avg": late,
            "trend": "improving" if late < early else "degrading",
            "change_percent": ((late - early) / early) * 100,
        }

# Usage
analyzer = LatencyAnalyzer(exporter)

# Get performance distribution
percentiles = analyzer.get_percentiles("retrieval")
print(f"Retrieval latency P95: {percentiles['p95']:.0f}ms")

# Find slow operations
slowdowns = analyzer.identify_slowdowns("retrieval", baseline_ms=150)
if slowdowns:
    logger.warning(f"Found {len(slowdowns)} slow retrievals")

# Check for degradation
trend = analyzer.get_trend("retrieval")
if trend["trend"] == "degrading":
    logger.alert(f"Latency degrading: {trend['change_percent']:.1f}%")
```

## Cache Metrics

### Hit Rate Analysis

Track cache effectiveness:

```python
from autorag_live.core.metrics_export import CacheMetric

class CacheAnalyzer:
    """Analyze cache performance."""

    def __init__(self, exporter: MetricsExporter):
        self.exporter = exporter

    def get_hit_rate(self, operation: str = None) -> float:
        """Get cache hit rate."""
        metrics = self.exporter.get_cache_metrics(operation)

        if not metrics:
            return 0.0

        hits = sum(1 for m in metrics if m.hit)
        return hits / len(metrics)

    def get_hit_rate_by_backend(self) -> dict:
        """Get hit rate by cache backend."""
        all_metrics = self.exporter.get_cache_metrics()

        backend_metrics = {}
        for metric in all_metrics:
            backend = metric.tags.get("cache_backend", "unknown")
            if backend not in backend_metrics:
                backend_metrics[backend] = {"hits": 0, "total": 0}

            backend_metrics[backend]["total"] += 1
            if metric.hit:
                backend_metrics[backend]["hits"] += 1

        return {
            backend: metrics["hits"] / metrics["total"]
            for backend, metrics in backend_metrics.items()
        }

    def analyze_cache_distribution(self) -> dict:
        """Analyze cache usage patterns."""
        metrics = self.exporter.get_cache_metrics()

        if not metrics:
            return {}

        # Group by operation
        by_operation = {}
        for metric in metrics:
            op = metric.tags.get("operation", "unknown")
            if op not in by_operation:
                by_operation[op] = {"hits": 0, "total": 0}

            by_operation[op]["total"] += 1
            if metric.hit:
                by_operation[op]["hits"] += 1

        return {
            op: {
                "hit_rate": stats["hits"] / stats["total"],
                "total_queries": stats["total"],
            }
            for op, stats in by_operation.items()
        }

# Usage
cache_analyzer = CacheAnalyzer(exporter)

# Monitor overall hit rate
hit_rate = cache_analyzer.get_hit_rate()
print(f"Cache hit rate: {hit_rate:.1%}")

# Check backend performance
backend_rates = cache_analyzer.get_hit_rate_by_backend()
print(f"Redis hit rate: {backend_rates['redis']:.1%}")
print(f"Memory hit rate: {backend_rates['memory']:.1%}")

# Find cache issues
distribution = cache_analyzer.analyze_cache_distribution()
for op, stats in distribution.items():
    if stats["hit_rate"] < 0.5:
        logger.warning(f"Low cache hit rate for {op}: {stats['hit_rate']:.1%}")
```

## Custom Metrics

### Define Custom Metrics

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
import time

@dataclass
class CustomMetric:
    """Custom application metric."""
    name: str
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str]

    def to_prometheus(self) -> str:
        """Export to Prometheus format."""
        labels = ",".join(f'{k}="{v}"' for k, v in self.tags.items())
        return f"{self.name}{{{labels}}} {self.value}"

class CustomMetricsCollector:
    """Collect custom application metrics."""

    def __init__(self, exporter: MetricsExporter):
        self.exporter = exporter
        self.metrics = []

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "count",
        tags: Dict[str, str] = None
    ):
        """Record custom metric."""
        metric = CustomMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            tags=tags or {}
        )
        self.metrics.append(metric)

    def record_query_quality(self, score: float, query: str):
        """Record query result quality."""
        self.record_metric(
            "query_quality_score",
            value=score,
            unit="score",
            tags={"query_length": str(len(query))}
        )

    def record_retrieval_count(self, count: int, retriever_type: str):
        """Record number of documents retrieved."""
        self.record_metric(
            "documents_retrieved",
            value=count,
            unit="count",
            tags={"retriever_type": retriever_type}
        )

    def record_model_latency(self, model_name: str, latency_ms: float):
        """Record LLM inference latency."""
        self.record_metric(
            "llm_latency",
            value=latency_ms,
            unit="ms",
            tags={"model": model_name}
        )

    def get_summary(self) -> dict:
        """Get metrics summary."""
        if not self.metrics:
            return {}

        by_name = {}
        for metric in self.metrics:
            if metric.name not in by_name:
                by_name[metric.name] = []
            by_name[metric.name].append(metric.value)

        return {
            name: {
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
            for name, values in by_name.items()
        }

# Usage
collector = CustomMetricsCollector(exporter)

def rag_pipeline(query: str) -> str:
    """RAG with custom metrics."""

    # Record input
    start = time.time()

    # Retrieval
    results = retriever.retrieve(query)
    collector.record_retrieval_count(len(results), "semantic")

    # Reranking
    reranked = reranker.rerank(results, query)

    # LLM inference
    llm_start = time.time()
    answer = llm.generate(query, reranked)
    llm_latency = (time.time() - llm_start) * 1000
    collector.record_model_latency("gpt-4", llm_latency)

    # Quality assessment
    quality = evaluate_answer(answer, query)
    collector.record_query_quality(quality, query)

    total_latency = time.time() - start
    print(f"Total latency: {total_latency:.2f}s")

    return answer

# Get metrics summary
summary = collector.get_summary()
print(f"Avg LLM latency: {summary['llm_latency']['avg']:.0f}ms")
```

## Exporting Metrics

### Prometheus Format Export

```python
class PrometheusExporter:
    """Export metrics in Prometheus text format."""

    def __init__(self, exporter: MetricsExporter):
        self.exporter = exporter

    def export_text_format(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = []
        timestamp = int(time.time() * 1000)

        # Latency metrics
        latency_metrics = self.exporter.get_latency_metrics()
        for metric in latency_metrics:
            lines.append(
                f'latency_ms{{operation="{metric.operation}",'
                f'tags="{metric.tags}"}} {metric.duration_ms} {timestamp}'
            )

        # Cache metrics
        cache_metrics = self.exporter.get_cache_metrics()
        for metric in cache_metrics:
            hit = 1 if metric.hit else 0
            lines.append(
                f'cache_hit{{operation="{metric.tags.get("operation", "")}"}} '
                f'{hit} {timestamp}'
            )

        return "\n".join(lines) + "\n"

    def export_to_file(self, filepath: str):
        """Export metrics to file."""
        with open(filepath, "w") as f:
            f.write(self.export_text_format())

# Usage
prometheus_exporter = PrometheusExporter(exporter)

# Export metrics
metrics_text = prometheus_exporter.export_text_format()
print(metrics_text)

# Save to file for Prometheus scraping
prometheus_exporter.export_to_file("/tmp/rag_metrics.txt")
```

### JSON Export

```python
import json

class JSONExporter:
    """Export metrics as JSON."""

    def __init__(self, exporter: MetricsExporter):
        self.exporter = exporter

    def export_json(self) -> str:
        """Export all metrics as JSON."""
        data = {
            "timestamp": time.time(),
            "latency": self._export_latency(),
            "cache": self._export_cache(),
            "summary": self._export_summary(),
        }
        return json.dumps(data, indent=2)

    def _export_latency(self) -> list:
        """Export latency metrics."""
        metrics = self.exporter.get_latency_metrics()
        return [
            {
                "operation": m.operation,
                "duration_ms": m.duration_ms,
                "timestamp": m.timestamp,
                "tags": m.tags,
            }
            for m in metrics
        ]

    def _export_cache(self) -> list:
        """Export cache metrics."""
        metrics = self.exporter.get_cache_metrics()
        return [
            {
                "operation": m.tags.get("operation"),
                "hit": m.hit,
                "timestamp": m.timestamp,
                "tags": m.tags,
            }
            for m in metrics
        ]

    def _export_summary(self) -> dict:
        """Export summary statistics."""
        latency = self.exporter.get_latency_metrics()
        cache = self.exporter.get_cache_metrics()

        if not latency:
            avg_latency = 0
        else:
            avg_latency = sum(m.duration_ms for m in latency) / len(latency)

        if not cache:
            hit_rate = 0
        else:
            hits = sum(1 for m in cache if m.hit)
            hit_rate = hits / len(cache)

        return {
            "avg_latency_ms": avg_latency,
            "cache_hit_rate": hit_rate,
            "total_latency_samples": len(latency),
            "total_cache_samples": len(cache),
        }

# Usage
json_exporter = JSONExporter(exporter)

metrics_json = json_exporter.export_json()
print(metrics_json)

# Save to file
with open("metrics.json", "w") as f:
    f.write(metrics_json)
```

## Dashboards and Visualization

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "AutoRAG-Live Performance",
    "panels": [
      {
        "title": "Retrieval Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, latency_ms{operation='retrieval'})"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "cache_hit_rate"
          }
        ]
      },
      {
        "title": "Documents Retrieved",
        "targets": [
          {
            "expr": "rate(documents_retrieved[5m])"
          }
        ]
      },
      {
        "title": "LLM Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, llm_latency)"
          }
        ]
      }
    ]
  }
}
```

### Custom Dashboard

```python
class MetricsDashboard:
    """Generate HTML dashboard from metrics."""

    def __init__(self, exporter: MetricsExporter):
        self.exporter = exporter

    def generate_html(self) -> str:
        """Generate HTML dashboard."""
        html = """
        <html>
        <head>
            <title>AutoRAG-Live Metrics</title>
            <style>
                body { font-family: sans-serif; margin: 20px; }
                .metric {
                    display: inline-block;
                    margin: 10px;
                    padding: 10px;
                    background: #f0f0f0;
                    border-radius: 5px;
                }
                .value { font-size: 24px; font-weight: bold; }
                .label { font-size: 12px; color: #666; }
            </style>
        </head>
        <body>
            <h1>AutoRAG-Live Metrics Dashboard</h1>
            <div id="metrics">
        """

        # Add latency metrics
        latency_metrics = self.exporter.get_latency_metrics()
        if latency_metrics:
            avg = sum(m.duration_ms for m in latency_metrics) / len(latency_metrics)
            html += f"""
                <div class="metric">
                    <div class="value">{avg:.1f}ms</div>
                    <div class="label">Avg Latency</div>
                </div>
            """

        # Add cache metrics
        cache_metrics = self.exporter.get_cache_metrics()
        if cache_metrics:
            hit_rate = sum(1 for m in cache_metrics if m.hit) / len(cache_metrics)
            html += f"""
                <div class="metric">
                    <div class="value">{hit_rate:.1%}</div>
                    <div class="label">Cache Hit Rate</div>
                </div>
            """

        html += """
            </div>
        </body>
        </html>
        """

        return html

    def save_dashboard(self, filepath: str):
        """Save dashboard to file."""
        with open(filepath, "w") as f:
            f.write(self.generate_html())

# Usage
dashboard = MetricsDashboard(exporter)
dashboard.save_dashboard("dashboard.html")
```

## Performance Profiling

### Operation Profiler

```python
import cProfile
import pstats
from io import StringIO

class OperationProfiler:
    """Profile operation performance."""

    def __init__(self):
        self.profilers = {}

    def profile_operation(self, operation_name: str, func, *args, **kwargs):
        """Profile a function."""
        if operation_name not in self.profilers:
            self.profilers[operation_name] = cProfile.Profile()

        profiler = self.profilers[operation_name]
        profiler.enable()

        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()

        return result

    def get_stats(self, operation_name: str, top_n: int = 10) -> str:
        """Get profiling stats for operation."""
        if operation_name not in self.profilers:
            return ""

        profiler = self.profilers[operation_name]
        stats = pstats.Stats(profiler)

        output = StringIO()
        stats.sort_stats("cumulative")
        stats.print_stats(top_n, file=output)

        return output.getvalue()

    def print_profile_report(self, operation_name: str):
        """Print profiling report."""
        print(f"Profile for {operation_name}:")
        print(self.get_stats(operation_name))

# Usage
profiler = OperationProfiler()

def retrieval_operation():
    results = retriever.retrieve("query")
    return results

# Profile the operation
result = profiler.profile_operation(
    "retrieval",
    retrieval_operation
)

# Print report
profiler.print_profile_report("retrieval")
```

## Best Practices

### 1. Contextualize Metrics

```python
from contextvars import ContextVar

# Store request context
request_id = ContextVar("request_id")
user_id = ContextVar("user_id")

def record_with_context(exporter, operation, duration_ms):
    """Record metrics with request context."""
    exporter.record_latency(
        operation=operation,
        duration_ms=duration_ms,
        tags={
            "request_id": request_id.get(None),
            "user_id": user_id.get(None),
        }
    )
```

### 2. Set Performance Budgets

```python
class PerformanceBudget:
    """Track metrics against performance budgets."""

    def __init__(self):
        self.budgets = {
            "retrieval_p95_ms": 200,
            "cache_hit_rate": 0.7,
            "llm_latency_p99_ms": 5000,
        }

    def check_budget(self, metrics: dict) -> dict:
        """Check if metrics exceed budgets."""
        violations = {}

        for key, limit in self.budgets.items():
            if key in metrics and metrics[key] > limit:
                violations[key] = {
                    "limit": limit,
                    "actual": metrics[key],
                    "exceeded_by": metrics[key] - limit,
                }

        return violations
```

### 3. Sample Metrics for High-Volume Systems

```python
import random

class SampledMetricsExporter(MetricsExporter):
    """Sample metrics to reduce overhead."""

    def __init__(self, sample_rate: float = 0.1):
        super().__init__()
        self.sample_rate = sample_rate

    def record_latency(self, operation, duration_ms, tags=None):
        """Record latency with sampling."""
        if random.random() < self.sample_rate:
            super().record_latency(operation, duration_ms, tags)
```

### 4. Aggregate Metrics Periodically

```python
class MetricsAggregator:
    """Periodically aggregate and clear metrics."""

    def __init__(self, exporter: MetricsExporter, interval_seconds: int = 60):
        self.exporter = exporter
        self.interval = interval_seconds
        self.aggregates = []

    def aggregate(self):
        """Aggregate current metrics."""
        aggregate = {
            "timestamp": time.time(),
            "latency_avg": self._avg_latency(),
            "cache_hit_rate": self._cache_hit_rate(),
        }

        self.aggregates.append(aggregate)
        self.exporter.clear()  # Clear to save memory

    def _avg_latency(self) -> float:
        metrics = self.exporter.get_latency_metrics()
        if not metrics:
            return 0.0
        return sum(m.duration_ms for m in metrics) / len(metrics)

    def _cache_hit_rate(self) -> float:
        metrics = self.exporter.get_cache_metrics()
        if not metrics:
            return 0.0
        hits = sum(1 for m in metrics if m.hit)
        return hits / len(metrics)
```

## Conclusion

Observability in AutoRAG-Live provides:

- **Metrics**: Track latency, cache performance, and custom application metrics
- **Analysis**: Identify bottlenecks and performance trends
- **Exporters**: Integrate with Prometheus, Grafana, JSON, and custom backends
- **Dashboards**: Visualize system health and performance

Use MetricsExporter for built-in metrics, create custom metrics for application-specific insights, and export to your monitoring stack for comprehensive observability.
