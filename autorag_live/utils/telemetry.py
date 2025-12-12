"""
Telemetry and observability for AutoRAG-Live.

Provides comprehensive telemetry collection for monitoring,
debugging, and performance analysis.

Features:
- Request/response tracing
- Performance metrics
- Error tracking
- Usage analytics
- Custom events
- Export to multiple backends

Example usage:
    >>> telemetry = Telemetry()
    >>> with telemetry.trace("retrieval") as span:
    ...     results = retrieve(query)
    ...     span.set_attribute("result_count", len(results))
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)


class SpanStatus(str, Enum):
    """Span status codes."""
    
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class MetricType(str, Enum):
    """Types of metrics."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class EventSeverity(str, Enum):
    """Event severity levels."""
    
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Span:
    """Represents a trace span."""
    
    name: str
    trace_id: str
    span_id: str
    
    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Hierarchy
    parent_id: Optional[str] = None
    
    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    status: SpanStatus = SpanStatus.UNSET
    error_message: Optional[str] = None
    
    # Events within span
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute."""
        self.attributes[key] = value
    
    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        if message:
            self.error_message = message
    
    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })
    
    def end(self, status: Optional[SpanStatus] = None) -> None:
        """End the span."""
        self.end_time = time.time()
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "error_message": self.error_message,
            "attributes": self.attributes,
            "events": self.events,
        }


@dataclass
class Metric:
    """Represents a metric data point."""
    
    name: str
    metric_type: MetricType
    value: float
    timestamp: float = field(default_factory=time.time)
    
    # Labels
    labels: Dict[str, str] = field(default_factory=dict)
    
    # For histograms/summaries
    bucket: Optional[str] = None
    quantile: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels,
            "bucket": self.bucket,
            "quantile": self.quantile,
        }


@dataclass
class Event:
    """Represents a telemetry event."""
    
    name: str
    severity: EventSeverity = EventSeverity.INFO
    timestamp: float = field(default_factory=time.time)
    
    # Event data
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "attributes": self.attributes,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }


class MetricCollector:
    """Collect and aggregate metrics."""
    
    def __init__(self):
        """Initialize collector."""
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value
    
    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value
    
    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
    
    def get_metrics(self) -> List[Metric]:
        """Get all collected metrics."""
        metrics = []
        
        with self._lock:
            # Counters
            for key, value in self._counters.items():
                name, labels = self._parse_key(key)
                metrics.append(Metric(
                    name=name,
                    metric_type=MetricType.COUNTER,
                    value=value,
                    labels=labels,
                ))
            
            # Gauges
            for key, value in self._gauges.items():
                name, labels = self._parse_key(key)
                metrics.append(Metric(
                    name=name,
                    metric_type=MetricType.GAUGE,
                    value=value,
                    labels=labels,
                ))
            
            # Histograms (emit percentiles)
            for key, values in self._histograms.items():
                name, labels = self._parse_key(key)
                if values:
                    sorted_values = sorted(values)
                    
                    # P50, P90, P99
                    for p in [0.5, 0.9, 0.99]:
                        idx = int(len(sorted_values) * p)
                        metrics.append(Metric(
                            name=name,
                            metric_type=MetricType.HISTOGRAM,
                            value=sorted_values[min(idx, len(sorted_values)-1)],
                            labels=labels,
                            quantile=p,
                        ))
        
        return metrics
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create metric key from name and labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _parse_key(self, key: str) -> tuple[str, Dict[str, str]]:
        """Parse metric key into name and labels."""
        if "{" not in key:
            return key, {}
        
        name, label_str = key.split("{", 1)
        label_str = label_str.rstrip("}")
        
        labels = {}
        if label_str:
            for part in label_str.split(","):
                k, v = part.split("=", 1)
                labels[k] = v
        
        return name, labels


class TelemetryExporter(ABC):
    """Base class for telemetry exporters."""
    
    @abstractmethod
    def export_spans(self, spans: List[Span]) -> None:
        """Export spans."""
        pass
    
    @abstractmethod
    def export_metrics(self, metrics: List[Metric]) -> None:
        """Export metrics."""
        pass
    
    @abstractmethod
    def export_events(self, events: List[Event]) -> None:
        """Export events."""
        pass


class ConsoleExporter(TelemetryExporter):
    """Export telemetry to console."""
    
    def __init__(self, pretty: bool = True):
        """Initialize exporter."""
        self.pretty = pretty
    
    def export_spans(self, spans: List[Span]) -> None:
        """Export spans to console."""
        for span in spans:
            if self.pretty:
                print(f"[SPAN] {span.name} ({span.duration_ms:.2f}ms) - {span.status.value}")
            else:
                print(json.dumps(span.to_dict()))
    
    def export_metrics(self, metrics: List[Metric]) -> None:
        """Export metrics to console."""
        for metric in metrics:
            if self.pretty:
                label_str = ",".join(f"{k}={v}" for k, v in metric.labels.items())
                print(f"[METRIC] {metric.name}{{{label_str}}} = {metric.value}")
            else:
                print(json.dumps(metric.to_dict()))
    
    def export_events(self, events: List[Event]) -> None:
        """Export events to console."""
        for event in events:
            if self.pretty:
                print(f"[{event.severity.value.upper()}] {event.name}: {event.attributes}")
            else:
                print(json.dumps(event.to_dict()))


class FileExporter(TelemetryExporter):
    """Export telemetry to files."""
    
    def __init__(
        self,
        directory: str = "telemetry",
        rotate_size_mb: float = 10.0,
    ):
        """
        Initialize exporter.
        
        Args:
            directory: Output directory
            rotate_size_mb: Rotate file when size exceeds this
        """
        self.directory = directory
        self.rotate_size_mb = rotate_size_mb
        
        os.makedirs(directory, exist_ok=True)
    
    def export_spans(self, spans: List[Span]) -> None:
        """Export spans to file."""
        self._append_to_file("spans.jsonl", [s.to_dict() for s in spans])
    
    def export_metrics(self, metrics: List[Metric]) -> None:
        """Export metrics to file."""
        self._append_to_file("metrics.jsonl", [m.to_dict() for m in metrics])
    
    def export_events(self, events: List[Event]) -> None:
        """Export events to file."""
        self._append_to_file("events.jsonl", [e.to_dict() for e in events])
    
    def _append_to_file(self, filename: str, items: List[Dict[str, Any]]) -> None:
        """Append items to file."""
        filepath = os.path.join(self.directory, filename)
        
        # Check for rotation
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb >= self.rotate_size_mb:
                self._rotate_file(filepath)
        
        with open(filepath, "a") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
    
    def _rotate_file(self, filepath: str) -> None:
        """Rotate a file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated = f"{filepath}.{timestamp}"
        os.rename(filepath, rotated)


class InMemoryExporter(TelemetryExporter):
    """Export telemetry to memory for testing."""
    
    def __init__(self, max_items: int = 10000):
        """Initialize exporter."""
        self.max_items = max_items
        self.spans: List[Span] = []
        self.metrics: List[Metric] = []
        self.events: List[Event] = []
        self._lock = threading.Lock()
    
    def export_spans(self, spans: List[Span]) -> None:
        """Store spans in memory."""
        with self._lock:
            self.spans.extend(spans)
            if len(self.spans) > self.max_items:
                self.spans = self.spans[-self.max_items:]
    
    def export_metrics(self, metrics: List[Metric]) -> None:
        """Store metrics in memory."""
        with self._lock:
            self.metrics.extend(metrics)
            if len(self.metrics) > self.max_items:
                self.metrics = self.metrics[-self.max_items:]
    
    def export_events(self, events: List[Event]) -> None:
        """Store events in memory."""
        with self._lock:
            self.events.extend(events)
            if len(self.events) > self.max_items:
                self.events = self.events[-self.max_items:]
    
    def clear(self) -> None:
        """Clear all stored data."""
        with self._lock:
            self.spans.clear()
            self.metrics.clear()
            self.events.clear()


class BatchProcessor:
    """Process telemetry data in batches."""
    
    def __init__(
        self,
        exporter: TelemetryExporter,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        """
        Initialize batch processor.
        
        Args:
            exporter: Telemetry exporter
            batch_size: Maximum batch size
            flush_interval: Seconds between flushes
        """
        self.exporter = exporter
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self._span_queue: queue.Queue[Span] = queue.Queue()
        self._metric_queue: queue.Queue[Metric] = queue.Queue()
        self._event_queue: queue.Queue[Event] = queue.Queue()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the batch processor."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        
        # Register shutdown hook
        atexit.register(self.stop)
    
    def stop(self) -> None:
        """Stop the batch processor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        
        # Final flush
        self._flush()
    
    def add_span(self, span: Span) -> None:
        """Add a span to be processed."""
        self._span_queue.put(span)
    
    def add_metric(self, metric: Metric) -> None:
        """Add a metric to be processed."""
        self._metric_queue.put(metric)
    
    def add_event(self, event: Event) -> None:
        """Add an event to be processed."""
        self._event_queue.put(event)
    
    def _process_loop(self) -> None:
        """Main processing loop."""
        last_flush = time.time()
        
        while self._running:
            time.sleep(0.1)
            
            # Check if we should flush
            should_flush = (
                time.time() - last_flush >= self.flush_interval or
                self._span_queue.qsize() >= self.batch_size or
                self._metric_queue.qsize() >= self.batch_size or
                self._event_queue.qsize() >= self.batch_size
            )
            
            if should_flush:
                self._flush()
                last_flush = time.time()
    
    def _flush(self) -> None:
        """Flush all queued data."""
        # Flush spans
        spans = []
        while not self._span_queue.empty() and len(spans) < self.batch_size:
            try:
                spans.append(self._span_queue.get_nowait())
            except queue.Empty:
                break
        
        if spans:
            try:
                self.exporter.export_spans(spans)
            except Exception as e:
                logger.error(f"Failed to export spans: {e}")
        
        # Flush metrics
        metrics = []
        while not self._metric_queue.empty() and len(metrics) < self.batch_size:
            try:
                metrics.append(self._metric_queue.get_nowait())
            except queue.Empty:
                break
        
        if metrics:
            try:
                self.exporter.export_metrics(metrics)
            except Exception as e:
                logger.error(f"Failed to export metrics: {e}")
        
        # Flush events
        events = []
        while not self._event_queue.empty() and len(events) < self.batch_size:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                break
        
        if events:
            try:
                self.exporter.export_events(events)
            except Exception as e:
                logger.error(f"Failed to export events: {e}")


class SpanContext:
    """Context manager for spans."""
    
    def __init__(
        self,
        span: Span,
        telemetry: "Telemetry",
    ):
        """Initialize context."""
        self.span = span
        self.telemetry = telemetry
    
    def __enter__(self) -> Span:
        """Enter context."""
        return self.span
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context."""
        if exc_type is not None:
            self.span.set_status(SpanStatus.ERROR, str(exc_val))
        
        self.span.end()
        self.telemetry._end_span(self.span)


class Telemetry:
    """
    Main telemetry interface.
    
    Example:
        >>> telemetry = Telemetry()
        >>> 
        >>> # Trace an operation
        >>> with telemetry.trace("retrieval", attributes={"query": "test"}) as span:
        ...     results = retrieve(query)
        ...     span.set_attribute("result_count", len(results))
        >>> 
        >>> # Record metrics
        >>> telemetry.counter("requests_total", labels={"endpoint": "/search"})
        >>> telemetry.histogram("response_time_ms", 150.5)
        >>> 
        >>> # Log events
        >>> telemetry.event("cache_hit", severity=EventSeverity.INFO)
    """
    
    def __init__(
        self,
        service_name: str = "autorag",
        exporter: Optional[TelemetryExporter] = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        """
        Initialize telemetry.
        
        Args:
            service_name: Service name for traces
            exporter: Telemetry exporter
            batch_size: Batch size for export
            flush_interval: Seconds between flushes
        """
        self.service_name = service_name
        self.exporter = exporter or InMemoryExporter()
        
        self._metrics = MetricCollector()
        self._processor = BatchProcessor(
            self.exporter,
            batch_size=batch_size,
            flush_interval=flush_interval,
        )
        
        # Current trace context
        self._current_trace_id: Optional[str] = None
        self._current_span_id: Optional[str] = None
        self._span_stack: List[Span] = []
        
        self._started = False
    
    def start(self) -> None:
        """Start telemetry collection."""
        if self._started:
            return
        
        self._processor.start()
        self._started = True
    
    def stop(self) -> None:
        """Stop telemetry collection."""
        if not self._started:
            return
        
        self._processor.stop()
        self._started = False
    
    @contextmanager
    def trace(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Span]:
        """
        Create a trace span.
        
        Args:
            name: Span name
            attributes: Initial attributes
            
        Yields:
            Span
        """
        span = self._create_span(name, attributes)
        
        # Push onto stack
        self._span_stack.append(span)
        self._current_span_id = span.span_id
        
        try:
            yield span
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            raise
        finally:
            span.end()
            self._end_span(span)
            
            # Pop from stack
            self._span_stack.pop()
            if self._span_stack:
                self._current_span_id = self._span_stack[-1].span_id
            else:
                self._current_span_id = None
    
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SpanContext:
        """
        Start a span (returns context manager).
        
        Args:
            name: Span name
            attributes: Initial attributes
            
        Returns:
            SpanContext
        """
        span = self._create_span(name, attributes)
        self._span_stack.append(span)
        self._current_span_id = span.span_id
        
        return SpanContext(span, self)
    
    def _create_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Create a new span."""
        # Use existing trace or create new
        if not self._current_trace_id:
            self._current_trace_id = str(uuid.uuid4())
        
        span = Span(
            name=name,
            trace_id=self._current_trace_id,
            span_id=str(uuid.uuid4()),
            parent_id=self._current_span_id,
            attributes=attributes or {},
        )
        
        # Add service name
        span.attributes["service.name"] = self.service_name
        
        return span
    
    def _end_span(self, span: Span) -> None:
        """End and export a span."""
        self._processor.add_span(span)
    
    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Increment value
            labels: Metric labels
        """
        self._metrics.increment(name, value, labels)
    
    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set a gauge metric.
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Metric labels
        """
        self._metrics.gauge(name, value, labels)
    
    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a histogram observation.
        
        Args:
            name: Metric name
            value: Observation value
            labels: Metric labels
        """
        self._metrics.histogram(name, value, labels)
    
    def event(
        self,
        name: str,
        severity: EventSeverity = EventSeverity.INFO,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an event.
        
        Args:
            name: Event name
            severity: Event severity
            attributes: Event attributes
        """
        event = Event(
            name=name,
            severity=severity,
            attributes=attributes or {},
            trace_id=self._current_trace_id,
            span_id=self._current_span_id,
        )
        
        self._processor.add_event(event)
    
    def flush_metrics(self) -> None:
        """Flush collected metrics."""
        metrics = self._metrics.get_metrics()
        if metrics:
            try:
                self.exporter.export_metrics(metrics)
            except Exception as e:
                logger.error(f"Failed to export metrics: {e}")
    
    def get_metrics(self) -> List[Metric]:
        """Get collected metrics."""
        return self._metrics.get_metrics()
    
    def timed(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Callable:
        """
        Decorator to time function execution.
        
        Args:
            name: Metric name
            labels: Metric labels
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration_ms = (time.time() - start) * 1000
                    self.histogram(name, duration_ms, labels)
            
            return wrapper
        return decorator
    
    def counted(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Callable:
        """
        Decorator to count function calls.
        
        Args:
            name: Metric name
            labels: Metric labels
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                self.counter(name, 1.0, labels)
                return func(*args, **kwargs)
            
            return wrapper
        return decorator


class RequestTelemetry:
    """Telemetry helper for request tracking."""
    
    def __init__(self, telemetry: Telemetry):
        """Initialize request telemetry."""
        self.telemetry = telemetry
    
    @contextmanager
    def track_request(
        self,
        endpoint: str,
        method: str = "POST",
        **kwargs: Any,
    ) -> Iterator[Span]:
        """
        Track an API request.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            **kwargs: Additional attributes
            
        Yields:
            Request span
        """
        attributes = {
            "http.endpoint": endpoint,
            "http.method": method,
            **kwargs,
        }
        
        with self.telemetry.trace(f"request_{endpoint}", attributes) as span:
            self.telemetry.counter(
                "http_requests_total",
                labels={"endpoint": endpoint, "method": method},
            )
            yield span
    
    def record_response(
        self,
        span: Span,
        status_code: int,
        response_size: int = 0,
    ) -> None:
        """
        Record response metrics.
        
        Args:
            span: Request span
            status_code: HTTP status code
            response_size: Response size in bytes
        """
        span.set_attribute("http.status_code", status_code)
        span.set_attribute("http.response_size", response_size)
        
        # Status based on code
        if 200 <= status_code < 400:
            span.set_status(SpanStatus.OK)
        else:
            span.set_status(SpanStatus.ERROR, f"HTTP {status_code}")


# Global telemetry instance
_default_telemetry: Optional[Telemetry] = None


def get_telemetry(
    service_name: str = "autorag",
    auto_start: bool = True,
) -> Telemetry:
    """Get or create the default telemetry instance."""
    global _default_telemetry
    if _default_telemetry is None:
        _default_telemetry = Telemetry(service_name=service_name)
        if auto_start:
            _default_telemetry.start()
    return _default_telemetry


def trace(name: str, **kwargs: Any) -> Iterator[Span]:
    """Convenience function to create a trace span."""
    return get_telemetry().trace(name, kwargs)


def counter(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
    """Convenience function to increment a counter."""
    get_telemetry().counter(name, value, labels)


def gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Convenience function to set a gauge."""
    get_telemetry().gauge(name, value, labels)


def histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Convenience function to record a histogram observation."""
    get_telemetry().histogram(name, value, labels)


def event(
    name: str,
    severity: EventSeverity = EventSeverity.INFO,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """Convenience function to log an event."""
    get_telemetry().event(name, severity, attributes)
