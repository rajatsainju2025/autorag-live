"""
Pipeline monitoring and observability module for AutoRAG-Live.

Provides comprehensive monitoring capabilities for RAG pipelines
with metrics collection, tracing, and alerting.

Features:
- Metrics collection (latency, throughput, errors)
- Distributed tracing with spans
- Health checks and status monitoring
- Alerting with configurable thresholds
- Dashboard-ready metrics export
- Request tracking and correlation
- Resource usage monitoring
- Performance profiling

Example usage:
    >>> monitor = PipelineMonitor()
    >>> 
    >>> with monitor.trace("retrieval") as span:
    ...     results = retriever.search(query)
    ...     span.set_attribute("result_count", len(results))
    >>> 
    >>> metrics = monitor.get_metrics()
"""

from __future__ import annotations

import asyncio
import functools
import logging
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import uuid

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MetricType(Enum):
    """Types of metrics."""
    
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()
    SUMMARY = auto()


class HealthStatus(Enum):
    """Health status levels."""
    
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    UNKNOWN = auto()


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class SpanStatus(Enum):
    """Span status."""
    
    OK = auto()
    ERROR = auto()
    CANCELLED = auto()


@dataclass
class MetricValue:
    """A metric value."""
    
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """A metric definition."""
    
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    
    # Values
    values: List[MetricValue] = field(default_factory=list)
    
    # For counters
    _counter: float = 0.0
    
    # For gauges
    _gauge: float = 0.0
    
    # For histograms
    _histogram: List[float] = field(default_factory=list)
    buckets: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    
    def inc(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter."""
        if self.metric_type != MetricType.COUNTER:
            raise ValueError("inc() only valid for counters")
        self._counter += value
        self.values.append(MetricValue(value=self._counter, labels=labels or {}))
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value."""
        if self.metric_type != MetricType.GAUGE:
            raise ValueError("set() only valid for gauges")
        self._gauge = value
        self.values.append(MetricValue(value=value, labels=labels or {}))
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record histogram observation."""
        if self.metric_type not in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
            raise ValueError("observe() only valid for histograms/summaries")
        self._histogram.append(value)
        self.values.append(MetricValue(value=value, labels=labels or {}))
    
    def get_value(self) -> float:
        """Get current value."""
        if self.metric_type == MetricType.COUNTER:
            return self._counter
        elif self.metric_type == MetricType.GAUGE:
            return self._gauge
        elif self.metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
            if self._histogram:
                return statistics.mean(self._histogram)
            return 0.0
        return 0.0
    
    def get_percentile(self, p: float) -> float:
        """Get percentile (for histograms)."""
        if not self._histogram:
            return 0.0
        sorted_values = sorted(self._histogram)
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]


@dataclass
class Span:
    """A tracing span."""
    
    span_id: str
    name: str
    
    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    
    # Hierarchy
    trace_id: str = ""
    parent_span_id: Optional[str] = None
    
    # Status
    status: SpanStatus = SpanStatus.OK
    error: Optional[str] = None
    
    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Events
    events: List[Tuple[float, str, Dict[str, Any]]] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000
    
    @property
    def is_finished(self) -> bool:
        """Check if span is finished."""
        return self.end_time > 0
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add span event."""
        self.events.append((time.time(), name, attributes or {}))
    
    def finish(self, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None) -> None:
        """Finish span."""
        self.end_time = time.time()
        self.status = status
        if error:
            self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'span_id': self.span_id,
            'trace_id': self.trace_id,
            'name': self.name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms,
            'status': self.status.name,
            'error': self.error,
            'attributes': self.attributes,
            'parent_span_id': self.parent_span_id,
        }


@dataclass
class Trace:
    """A distributed trace."""
    
    trace_id: str
    name: str
    
    # Spans
    spans: List[Span] = field(default_factory=list)
    
    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    
    # Metadata
    service_name: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Get trace duration."""
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000
    
    @property
    def span_count(self) -> int:
        return len(self.spans)
    
    def add_span(self, span: Span) -> None:
        """Add span to trace."""
        span.trace_id = self.trace_id
        self.spans.append(span)
    
    def get_root_span(self) -> Optional[Span]:
        """Get root span."""
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return None
    
    def finish(self) -> None:
        """Finish trace."""
        self.end_time = time.time()


@dataclass
class Alert:
    """An alert notification."""
    
    alert_id: str
    name: str
    severity: AlertSeverity
    
    # Message
    message: str
    description: str = ""
    
    # Source
    metric_name: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    
    # Timing
    triggered_at: float = field(default_factory=time.time)
    resolved_at: float = 0.0
    
    # State
    is_active: bool = True
    
    labels: Dict[str, str] = field(default_factory=dict)
    
    def resolve(self) -> None:
        """Resolve alert."""
        self.resolved_at = time.time()
        self.is_active = False


@dataclass
class AlertRule:
    """An alert rule definition."""
    
    rule_id: str
    name: str
    metric_name: str
    
    # Condition
    condition: str  # "gt", "lt", "eq", "gte", "lte"
    threshold: float
    
    # Severity
    severity: AlertSeverity = AlertSeverity.WARNING
    
    # Message
    message: str = ""
    
    # Options
    enabled: bool = True
    cooldown_seconds: float = 300.0
    
    # State
    last_triggered: float = 0.0
    
    def check(self, value: float) -> bool:
        """Check if condition is met."""
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "eq":
            return value == self.threshold
        elif self.condition == "gte":
            return value >= self.threshold
        elif self.condition == "lte":
            return value <= self.threshold
        return False


@dataclass
class HealthCheck:
    """A health check definition."""
    
    name: str
    check_fn: Callable[[], bool]
    
    # Options
    timeout: float = 5.0
    critical: bool = True
    
    # State
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: float = 0.0
    last_error: Optional[str] = None
    
    async def run(self) -> HealthStatus:
        """Run health check."""
        try:
            start = time.time()
            
            if asyncio.iscoroutinefunction(self.check_fn):
                result = await asyncio.wait_for(
                    self.check_fn(),
                    timeout=self.timeout,
                )
            else:
                result = self.check_fn()
            
            self.last_check = time.time()
            self.status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            self.last_error = None
            
        except asyncio.TimeoutError:
            self.status = HealthStatus.UNHEALTHY
            self.last_error = "Health check timeout"
        except Exception as e:
            self.status = HealthStatus.UNHEALTHY
            self.last_error = str(e)
        
        return self.status


class MetricsRegistry:
    """
    Registry for metrics.
    
    Example:
        >>> registry = MetricsRegistry()
        >>> counter = registry.counter("requests_total", "Total requests")
        >>> counter.inc()
    """
    
    def __init__(self):
        """Initialize registry."""
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()
    
    def counter(
        self,
        name: str,
        description: str = "",
        unit: str = "",
    ) -> Metric:
        """Create or get counter metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Metric(
                    name=name,
                    metric_type=MetricType.COUNTER,
                    description=description,
                    unit=unit,
                )
            return self._metrics[name]
    
    def gauge(
        self,
        name: str,
        description: str = "",
        unit: str = "",
    ) -> Metric:
        """Create or get gauge metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Metric(
                    name=name,
                    metric_type=MetricType.GAUGE,
                    description=description,
                    unit=unit,
                )
            return self._metrics[name]
    
    def histogram(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        buckets: Optional[List[float]] = None,
    ) -> Metric:
        """Create or get histogram metric."""
        with self._lock:
            if name not in self._metrics:
                metric = Metric(
                    name=name,
                    metric_type=MetricType.HISTOGRAM,
                    description=description,
                    unit=unit,
                )
                if buckets:
                    metric.buckets = buckets
                self._metrics[name] = metric
            return self._metrics[name]
    
    def get(self, name: str) -> Optional[Metric]:
        """Get metric by name."""
        return self._metrics.get(name)
    
    def all(self) -> Dict[str, Metric]:
        """Get all metrics."""
        return self._metrics.copy()


class Tracer:
    """
    Distributed tracing.
    
    Example:
        >>> tracer = Tracer(service_name="rag-pipeline")
        >>> 
        >>> with tracer.start_span("retrieval") as span:
        ...     span.set_attribute("query", query)
        ...     results = retriever.search(query)
    """
    
    def __init__(
        self,
        service_name: str = "autorag",
        max_traces: int = 1000,
    ):
        """
        Initialize tracer.
        
        Args:
            service_name: Service name
            max_traces: Maximum traces to keep
        """
        self.service_name = service_name
        self.max_traces = max_traces
        
        self._traces: Deque[Trace] = deque(maxlen=max_traces)
        self._current_trace: Optional[Trace] = None
        self._current_span: Optional[Span] = None
        self._lock = threading.Lock()
    
    def start_trace(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Trace:
        """Start a new trace."""
        trace = Trace(
            trace_id=str(uuid.uuid4()),
            name=name,
            service_name=self.service_name,
            attributes=attributes or {},
        )
        
        with self._lock:
            self._traces.append(trace)
            self._current_trace = trace
        
        return trace
    
    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """
        Start a span context.
        
        Args:
            name: Span name
            attributes: Span attributes
            
        Yields:
            Span
        """
        # Create span
        span = Span(
            span_id=str(uuid.uuid4()),
            name=name,
            trace_id=self._current_trace.trace_id if self._current_trace else str(uuid.uuid4()),
            parent_span_id=self._current_span.span_id if self._current_span else None,
            attributes=attributes or {},
        )
        
        # Set as current
        previous_span = self._current_span
        self._current_span = span
        
        # Add to trace
        if self._current_trace:
            self._current_trace.add_span(span)
        
        try:
            yield span
            span.finish(SpanStatus.OK)
        except Exception as e:
            span.finish(SpanStatus.ERROR, str(e))
            raise
        finally:
            self._current_span = previous_span
    
    def get_current_span(self) -> Optional[Span]:
        """Get current span."""
        return self._current_span
    
    def get_traces(
        self,
        limit: int = 100,
    ) -> List[Trace]:
        """Get recent traces."""
        return list(self._traces)[-limit:]


class AlertManager:
    """
    Alert management.
    
    Example:
        >>> manager = AlertManager()
        >>> manager.add_rule(AlertRule(
        ...     rule_id="high_latency",
        ...     name="High Latency",
        ...     metric_name="request_latency_p99",
        ...     condition="gt",
        ...     threshold=5.0,
        ... ))
    """
    
    def __init__(
        self,
        on_alert: Optional[Callable[[Alert], None]] = None,
    ):
        """
        Initialize alert manager.
        
        Args:
            on_alert: Callback when alert is triggered
        """
        self.on_alert = on_alert
        
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: List[Alert] = []
        self._lock = threading.Lock()
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self._rules[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove alert rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False
    
    def check(
        self,
        metric_name: str,
        value: float,
    ) -> List[Alert]:
        """
        Check metric against rules.
        
        Args:
            metric_name: Metric name
            value: Current value
            
        Returns:
            List of triggered alerts
        """
        triggered = []
        
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            
            if rule.metric_name != metric_name:
                continue
            
            # Check cooldown
            if time.time() - rule.last_triggered < rule.cooldown_seconds:
                continue
            
            # Check condition
            if rule.check(value):
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    name=rule.name,
                    severity=rule.severity,
                    message=rule.message or f"{metric_name} {rule.condition} {rule.threshold}",
                    metric_name=metric_name,
                    current_value=value,
                    threshold=rule.threshold,
                )
                
                with self._lock:
                    self._alerts.append(alert)
                    rule.last_triggered = time.time()
                
                triggered.append(alert)
                
                if self.on_alert:
                    self.on_alert(alert)
        
        return triggered
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return [a for a in self._alerts if a.is_active]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolve()
                return True
        return False


class PipelineMonitor:
    """
    Main pipeline monitoring interface.
    
    Example:
        >>> monitor = PipelineMonitor(service_name="rag-service")
        >>> 
        >>> # Track requests
        >>> monitor.track_request("retrieval", duration_ms=45.2)
        >>> 
        >>> # Use tracing
        >>> with monitor.trace("generation") as span:
        ...     span.set_attribute("model", "gpt-4")
        ...     response = generate(query, context)
        >>> 
        >>> # Get metrics
        >>> metrics = monitor.get_metrics_summary()
    """
    
    def __init__(
        self,
        service_name: str = "autorag",
    ):
        """
        Initialize monitor.
        
        Args:
            service_name: Service name
        """
        self.service_name = service_name
        
        # Components
        self._metrics = MetricsRegistry()
        self._tracer = Tracer(service_name=service_name)
        self._alerts = AlertManager()
        
        # Health checks
        self._health_checks: Dict[str, HealthCheck] = {}
        
        # Initialize default metrics
        self._init_metrics()
    
    def _init_metrics(self) -> None:
        """Initialize default metrics."""
        # Request metrics
        self._metrics.counter("requests_total", "Total requests")
        self._metrics.counter("requests_failed", "Failed requests")
        self._metrics.histogram("request_latency_ms", "Request latency", "ms")
        
        # Component metrics
        self._metrics.histogram("retrieval_latency_ms", "Retrieval latency", "ms")
        self._metrics.histogram("generation_latency_ms", "Generation latency", "ms")
        self._metrics.histogram("reranking_latency_ms", "Reranking latency", "ms")
        
        # Throughput
        self._metrics.gauge("requests_per_second", "Current RPS")
        self._metrics.gauge("active_requests", "Active requests")
        
        # Resource metrics
        self._metrics.gauge("memory_usage_mb", "Memory usage", "MB")
        self._metrics.gauge("cache_hit_rate", "Cache hit rate")
    
    def track_request(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Track a request.
        
        Args:
            operation: Operation name
            duration_ms: Duration in ms
            success: Was request successful
            labels: Additional labels
        """
        # Update counters
        self._metrics.get("requests_total").inc(labels=labels)
        
        if not success:
            self._metrics.get("requests_failed").inc(labels=labels)
        
        # Update latency
        self._metrics.get("request_latency_ms").observe(duration_ms, labels=labels)
        
        # Operation-specific metrics
        latency_metric = f"{operation}_latency_ms"
        if self._metrics.get(latency_metric):
            self._metrics.get(latency_metric).observe(duration_ms, labels=labels)
        
        # Check alerts
        self._alerts.check("request_latency_ms", duration_ms)
    
    @contextmanager
    def trace(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """
        Create tracing span context.
        
        Args:
            name: Span name
            attributes: Span attributes
            
        Yields:
            Span
        """
        with self._tracer.start_span(name, attributes) as span:
            yield span
            
            # Track metrics from span
            self.track_request(
                operation=name,
                duration_ms=span.duration_ms,
                success=span.status == SpanStatus.OK,
            )
    
    def timed(
        self,
        operation: str,
    ) -> Callable:
        """
        Decorator to time function execution.
        
        Args:
            operation: Operation name
            
        Returns:
            Decorator
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                success = True
                
                try:
                    return func(*args, **kwargs)
                except Exception:
                    success = False
                    raise
                finally:
                    duration_ms = (time.time() - start) * 1000
                    self.track_request(operation, duration_ms, success)
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                success = True
                
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    success = False
                    raise
                finally:
                    duration_ms = (time.time() - start) * 1000
                    self.track_request(operation, duration_ms, success)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return wrapper
        
        return decorator
    
    def add_health_check(
        self,
        name: str,
        check_fn: Callable[[], bool],
        critical: bool = True,
    ) -> None:
        """
        Add health check.
        
        Args:
            name: Check name
            check_fn: Check function
            critical: Is check critical
        """
        self._health_checks[name] = HealthCheck(
            name=name,
            check_fn=check_fn,
            critical=critical,
        )
    
    async def run_health_checks(self) -> Dict[str, HealthStatus]:
        """Run all health checks."""
        results = {}
        
        for name, check in self._health_checks.items():
            results[name] = await check.run()
        
        return results
    
    def get_health_status(self) -> HealthStatus:
        """Get overall health status."""
        if not self._health_checks:
            return HealthStatus.UNKNOWN
        
        has_unhealthy_critical = False
        has_unhealthy = False
        
        for check in self._health_checks.values():
            if check.status == HealthStatus.UNHEALTHY:
                if check.critical:
                    has_unhealthy_critical = True
                else:
                    has_unhealthy = True
        
        if has_unhealthy_critical:
            return HealthStatus.UNHEALTHY
        if has_unhealthy:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add alert rule."""
        self._alerts.add_rule(rule)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return self._alerts.get_active_alerts()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary.
        
        Returns:
            Dictionary with metric values
        """
        metrics = self._metrics.all()
        summary = {}
        
        for name, metric in metrics.items():
            if metric.metric_type == MetricType.COUNTER:
                summary[name] = metric.get_value()
            elif metric.metric_type == MetricType.GAUGE:
                summary[name] = metric.get_value()
            elif metric.metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                summary[name] = {
                    'mean': metric.get_value(),
                    'p50': metric.get_percentile(50),
                    'p95': metric.get_percentile(95),
                    'p99': metric.get_percentile(99),
                }
        
        return summary
    
    def get_traces(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get recent traces."""
        traces = self._tracer.get_traces(limit)
        return [
            {
                'trace_id': t.trace_id,
                'name': t.name,
                'duration_ms': t.duration_ms,
                'span_count': t.span_count,
                'spans': [s.to_dict() for s in t.spans],
            }
            for t in traces
        ]
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics
        """
        lines = []
        metrics = self._metrics.all()
        
        for name, metric in metrics.items():
            # Add help
            if metric.description:
                lines.append(f"# HELP {name} {metric.description}")
            
            # Add type
            type_name = metric.metric_type.name.lower()
            lines.append(f"# TYPE {name} {type_name}")
            
            # Add value
            if metric.metric_type == MetricType.COUNTER:
                lines.append(f"{name} {metric.get_value()}")
            elif metric.metric_type == MetricType.GAUGE:
                lines.append(f"{name} {metric.get_value()}")
            elif metric.metric_type == MetricType.HISTOGRAM:
                # Add bucket counts
                for bucket in metric.buckets:
                    count = sum(1 for v in metric._histogram if v <= bucket)
                    lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')
                lines.append(f'{name}_bucket{{le="+Inf"}} {len(metric._histogram)}')
                lines.append(f"{name}_sum {sum(metric._histogram)}")
                lines.append(f"{name}_count {len(metric._histogram)}")
        
        return "\n".join(lines)


# Convenience functions

def create_monitor(
    service_name: str = "autorag",
) -> PipelineMonitor:
    """
    Create pipeline monitor.
    
    Args:
        service_name: Service name
        
    Returns:
        PipelineMonitor instance
    """
    return PipelineMonitor(service_name=service_name)


def timed(
    operation: str,
    monitor: Optional[PipelineMonitor] = None,
) -> Callable:
    """
    Timing decorator.
    
    Args:
        operation: Operation name
        monitor: Monitor instance
        
    Returns:
        Decorator
    """
    if monitor is None:
        monitor = PipelineMonitor()
    
    return monitor.timed(operation)
