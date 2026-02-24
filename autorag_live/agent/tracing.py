"""
Agent Observability and Tracing Module.

Implements OpenTelemetry-compatible distributed tracing for RAG agent operations,
enabling end-to-end visibility into query processing, retrieval, and generation.

Key Features:
1. Span-based tracing for all operations
2. Automatic context propagation
3. Metrics collection and export
4. Error tracking and alerting
5. Performance profiling

Example:
    >>> tracer = AgentTracer("rag-service")
    >>> with tracer.span("query_processing") as span:
    ...     span.set_attribute("query", query)
    ...     result = await process(query)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Enums and Constants
# =============================================================================


class SpanStatus(str, Enum):
    """Status of a span."""

    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


class SpanKind(str, Enum):
    """Kind of span."""

    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class OperationType(str, Enum):
    """Type of RAG operation."""

    QUERY = "query"
    RETRIEVAL = "retrieval"
    RERANK = "rerank"
    AUGMENTATION = "augmentation"
    GENERATION = "generation"
    EMBEDDING = "embedding"
    CACHE_LOOKUP = "cache_lookup"
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    VERIFICATION = "verification"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SpanContext:
    """
    Context for distributed tracing.

    Attributes:
        trace_id: Unique trace identifier
        span_id: Unique span identifier
        parent_span_id: Parent span ID (if any)
        baggage: Key-value metadata
    """

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)

    def with_new_span(self) -> "SpanContext":
        """Create child context with new span ID."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=self.span_id,
            baggage=dict(self.baggage),
        )


@dataclass
class SpanEvent:
    """
    Event within a span.

    Attributes:
        name: Event name
        timestamp: When event occurred
        attributes: Event attributes
    """

    name: str
    timestamp: datetime = field(default_factory=datetime.now)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """
    A trace span representing a unit of work.

    Attributes:
        name: Span name
        context: Span context
        kind: Span kind
        operation_type: RAG operation type
        start_time: When span started
        end_time: When span ended
        status: Span status
        attributes: Span attributes
        events: Span events
        links: Links to other spans
    """

    name: str
    context: SpanContext
    kind: SpanKind = SpanKind.INTERNAL
    operation_type: Optional[OperationType] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanContext] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    @property
    def is_active(self) -> bool:
        """Check if span is still active."""
        return self.end_time is None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """Set multiple attributes."""
        self.attributes.update(attributes)

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an event to the span."""
        self.events.append(SpanEvent(name=name, attributes=attributes or {}))

    def set_status(
        self,
        status: SpanStatus,
        message: str = "",
    ) -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def record_exception(
        self,
        exception: Exception,
        escaped: bool = True,
    ) -> None:
        """Record an exception."""
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                "exception.escaped": escaped,
            },
        )
        self.set_status(SpanStatus.ERROR, str(exception))

    def end(self) -> None:
        """End the span."""
        if self.end_time is None:
            self.end_time = datetime.now()
            if self.status == SpanStatus.UNSET:
                self.status = SpanStatus.OK

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "kind": self.kind.value,
            "operation_type": self.operation_type.value if self.operation_type else None,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
        }


@dataclass
class Trace:
    """
    A complete trace with all spans.

    Attributes:
        trace_id: Unique trace identifier
        spans: All spans in the trace
        metadata: Trace metadata
    """

    trace_id: str
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def root_span(self) -> Optional[Span]:
        """Get root span."""
        for span in self.spans:
            if span.context.parent_span_id is None:
                return span
        return None

    @property
    def duration_ms(self) -> float:
        """Get total trace duration."""
        root = self.root_span
        return root.duration_ms if root else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms,
            "span_count": len(self.spans),
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans],
        }


# =============================================================================
# Span Exporter Interface
# =============================================================================


class SpanExporter(ABC):
    """Abstract base class for span exporters."""

    @abstractmethod
    def export(self, spans: List[Span]) -> None:
        """Export spans."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass


class ConsoleExporter(SpanExporter):
    """Exports spans to console."""

    def __init__(self, verbose: bool = False):
        """Initialize exporter."""
        self.verbose = verbose

    def export(self, spans: List[Span]) -> None:
        """Export spans to console."""
        for span in spans:
            status_emoji = "✓" if span.status == SpanStatus.OK else "✗"
            print(
                f"[{status_emoji}] {span.name} "
                f"({span.duration_ms:.2f}ms) "
                f"[{span.context.trace_id[:8]}]"
            )
            if self.verbose:
                for key, value in span.attributes.items():
                    print(f"    {key}: {value}")

    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass


class InMemoryExporter(SpanExporter):
    """Exports spans to memory for testing."""

    def __init__(self, max_spans: int = 10000):
        """Initialize exporter."""
        self.spans: List[Span] = []
        self.max_spans = max_spans

    def export(self, spans: List[Span]) -> None:
        """Export spans to memory."""
        self.spans.extend(spans)
        # Limit size
        if len(self.spans) > self.max_spans:
            self.spans = self.spans[-self.max_spans :]

    def get_spans(self) -> List[Span]:
        """Get all exported spans."""
        return list(self.spans)

    def clear(self) -> None:
        """Clear exported spans."""
        self.spans.clear()

    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass


class JSONFileExporter(SpanExporter):
    """Exports spans to JSON file."""

    def __init__(self, filepath: str):
        """Initialize exporter."""
        import json

        self.filepath = filepath
        self._json = json

    def export(self, spans: List[Span]) -> None:
        """Export spans to file."""
        with open(self.filepath, "a") as f:
            for span in spans:
                f.write(self._json.dumps(span.to_dict()) + "\n")

    def shutdown(self) -> None:
        """Shutdown exporter."""
        pass


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class Metric:
    """A metric measurement."""

    name: str
    value: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates metrics.

    Tracks counters, gauges, and histograms.
    """

    def __init__(self):
        """Initialize collector."""
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._labels: Dict[str, Dict[str, str]] = {}

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value
        if labels:
            self._labels[key] = labels

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        if labels:
            self._labels[key] = labels

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram value."""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        if labels:
            self._labels[key] = labels

    def get_counter(self, name: str) -> float:
        """Get counter value."""
        return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        """Get gauge value."""
        return self._gauges.get(name, 0)

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        values = self._histograms.get(name, [])
        if not values:
            return {"count": 0, "sum": 0, "min": 0, "max": 0, "avg": 0}

        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {k: self.get_histogram_stats(k) for k in self._histograms},
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()

    def _make_key(
        self,
        name: str,
        labels: Optional[Dict[str, str]],
    ) -> str:
        """Make metric key with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# =============================================================================
# Adaptive Head-Based Sampler
# =============================================================================


class AdaptiveSampler:
    """
    Head-based probabilistic sampler with adaptive rate control.

    At high QPS, recording every span burns 5-15 % of CPU on telemetry.
    This sampler maintains a *target* sample rate and adjusts it dynamically
    so that the exporter is never overwhelmed.

    Algorithm
    ---------
    * Every incoming trace is assigned a ``[0, 1)`` random number.
    * A trace is sampled iff its random number is below ``current_rate``.
    * Every ``window_seconds`` the sampler checks the observed throughput
      and raises/lowers ``current_rate`` using a simple AIMD controller:

      - If throughput > ``max_spans_per_sec``: halve the rate (× 0.5).
      - Otherwise: increase rate additively by ``rate_step``.

    The result is a well-understood, low-overhead controller that converges
    to the highest sustainable sample rate without dropping the exporter.

    Usage::

        sampler = AdaptiveSampler(initial_rate=1.0, max_spans_per_sec=200)
        tracer  = AgentTracer("svc", sampler=sampler)

    Args:
        initial_rate:      Starting sample probability (0–1].
        max_spans_per_sec: Exporter throughput ceiling.
        min_rate:          Never drop below this rate (always sample errors).
        window_seconds:    How often (seconds) to re-evaluate the rate.
        rate_step:         Additive increase per window when under capacity.
    """

    def __init__(
        self,
        initial_rate: float = 1.0,
        max_spans_per_sec: float = 500.0,
        min_rate: float = 0.01,
        window_seconds: float = 10.0,
        rate_step: float = 0.05,
    ) -> None:
        import random
        import threading

        self._rng = random.Random()  # Instance-local RNG for thread safety
        self.current_rate = max(min_rate, min(1.0, initial_rate))
        self.max_spans_per_sec = max_spans_per_sec
        self.min_rate = min_rate
        self.window_seconds = window_seconds
        self.rate_step = rate_step

        self._lock = threading.Lock()
        self._span_count: int = 0
        self._window_start: float = time.monotonic()

        # Stats
        self.total_considered: int = 0
        self.total_sampled: int = 0

    def should_sample(self, force_sample: bool = False) -> bool:
        """
        Decide whether to sample the current trace.

        Args:
            force_sample: Always sample (e.g. error spans).

        Returns:
            ``True`` if this trace should be recorded.
        """
        with self._lock:
            self.total_considered += 1
            self._maybe_update_rate()

            if force_sample or self._rng.random() < self.current_rate:
                self._span_count += 1
                self.total_sampled += 1
                return True
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return current sampler statistics."""
        with self._lock:
            return {
                "current_rate": round(self.current_rate, 4),
                "total_considered": self.total_considered,
                "total_sampled": self.total_sampled,
                "effective_rate": (
                    round(self.total_sampled / self.total_considered, 4)
                    if self.total_considered
                    else 0.0
                ),
            }

    # ------------------------------------------------------------------

    def _maybe_update_rate(self) -> None:
        """AIMD rate controller — called under lock."""
        now = time.monotonic()
        elapsed = now - self._window_start

        if elapsed < self.window_seconds:
            return

        observed_qps = self._span_count / max(elapsed, 1e-9)

        if observed_qps > self.max_spans_per_sec:
            # Multiplicative decrease
            self.current_rate = max(self.min_rate, self.current_rate * 0.5)
            logger.debug(
                f"AdaptiveSampler: throughput={observed_qps:.0f}/s > limit="
                f"{self.max_spans_per_sec:.0f}/s — rate ↓ {self.current_rate:.3f}"
            )
        else:
            # Additive increase
            self.current_rate = min(1.0, self.current_rate + self.rate_step)

        self._span_count = 0
        self._window_start = now


# =============================================================================
# Tracer
# =============================================================================


class AgentTracer:
    """
    Main tracer for RAG agent operations.

    Provides span management, context propagation, and metrics.

    Example:
        >>> tracer = AgentTracer("rag-service")
        >>> async with tracer.span("query") as span:
        ...     span.set_attribute("query.text", query)
        ...     result = await process(query)
    """

    # Context variable for current span
    _current_context: Optional[SpanContext] = None
    _current_span: Optional[Span] = None

    def __init__(
        self,
        service_name: str,
        exporter: Optional[SpanExporter] = None,
        auto_export: bool = True,
        sampler: Optional["AdaptiveSampler"] = None,
    ):
        """
        Initialize tracer.

        Args:
            service_name: Name of the service
            exporter: Span exporter
            auto_export: Whether to auto-export on span end
            sampler: Optional :class:`AdaptiveSampler`.  When *None* every
                     span is recorded (rate = 1.0).  Pass an
                     ``AdaptiveSampler`` to shed load at high QPS.
        """
        self.service_name = service_name
        self.exporter = exporter or InMemoryExporter()
        self.auto_export = auto_export
        self.sampler = sampler  # None ⇒ always sample
        self.metrics = MetricsCollector()

        self._traces: Dict[str, Trace] = {}
        self._active_spans: List[Span] = []

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        operation_type: Optional[OperationType] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a span context manager.

        Args:
            name: Span name
            kind: Span kind
            operation_type: RAG operation type
            attributes: Initial attributes

        Yields:
            Span object
        """
        span = self._create_span(name, kind, operation_type, attributes)
        self._active_spans.append(span)

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            self._active_spans.remove(span)
            self._record_span(span)

    @asynccontextmanager
    async def async_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        operation_type: Optional[OperationType] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Create an async span context manager.

        Args:
            name: Span name
            kind: Span kind
            operation_type: RAG operation type
            attributes: Initial attributes

        Yields:
            Span object
        """
        span = self._create_span(name, kind, operation_type, attributes)
        self._active_spans.append(span)

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            self._active_spans.remove(span)
            self._record_span(span)

    def trace(
        self,
        name: Optional[str] = None,
        operation_type: Optional[OperationType] = None,
    ) -> Callable[[F], F]:
        """
        Decorator for tracing functions.

        Args:
            name: Span name (defaults to function name)
            operation_type: Operation type

        Returns:
            Decorated function
        """

        def decorator(func: F) -> F:
            span_name = name or func.__name__

            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    async with self.async_span(
                        span_name,
                        operation_type=operation_type,
                    ) as span:
                        span.set_attribute("function", func.__name__)
                        return await func(*args, **kwargs)

                return async_wrapper  # type: ignore
            else:

                @functools.wraps(func)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    with self.span(
                        span_name,
                        operation_type=operation_type,
                    ) as span:
                        span.set_attribute("function", func.__name__)
                        return func(*args, **kwargs)

                return sync_wrapper  # type: ignore

        return decorator

    def get_current_span(self) -> Optional[Span]:
        """Get current active span."""
        return self._active_spans[-1] if self._active_spans else None

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID."""
        return self._traces.get(trace_id)

    def get_all_traces(self) -> List[Trace]:
        """Get all traces."""
        return list(self._traces.values())

    def export_pending(self) -> None:
        """Export all pending spans."""
        spans = []
        for trace in self._traces.values():
            spans.extend(trace.spans)
        if spans:
            self.exporter.export(spans)

    def shutdown(self) -> None:
        """Shutdown tracer."""
        self.export_pending()
        self.exporter.shutdown()

    def _create_span(
        self,
        name: str,
        kind: SpanKind,
        operation_type: Optional[OperationType],
        attributes: Optional[Dict[str, Any]],
    ) -> Span:
        """Create a new span."""
        # Get or create context
        if self._active_spans:
            parent = self._active_spans[-1]
            context = parent.context.with_new_span()
        else:
            context = SpanContext()

        span = Span(
            name=name,
            context=context,
            kind=kind,
            operation_type=operation_type,
            attributes={
                "service.name": self.service_name,
                **(attributes or {}),
            },
        )

        return span

    def _record_span(self, span: Span) -> None:
        """Record completed span (subject to sampler decision)."""
        # Error spans are always recorded regardless of sampling rate.
        force = span.status == SpanStatus.ERROR
        if self.sampler is not None and not self.sampler.should_sample(force_sample=force):
            return  # Dropped — do not export or store

        trace_id = span.context.trace_id

        if trace_id not in self._traces:
            self._traces[trace_id] = Trace(trace_id=trace_id)

        self._traces[trace_id].spans.append(span)

        # Record metrics
        self.metrics.increment(
            "spans_total",
            labels={"operation": span.operation_type.value if span.operation_type else "unknown"},
        )
        self.metrics.record_histogram(
            "span_duration_ms",
            span.duration_ms,
            labels={"name": span.name},
        )

        if span.status == SpanStatus.ERROR:
            self.metrics.increment("span_errors_total")

        # Auto-export if enabled
        if self.auto_export:
            self.exporter.export([span])


# =============================================================================
# RAG-Specific Tracing
# =============================================================================


class RAGTracer(AgentTracer):
    """
    Specialized tracer for RAG operations.

    Provides convenience methods for common RAG operations.
    """

    @asynccontextmanager
    async def trace_query(
        self,
        query: str,
        user_id: Optional[str] = None,
    ):
        """Trace a query operation."""
        async with self.async_span(
            "query",
            operation_type=OperationType.QUERY,
            attributes={
                "query.text": query[:200],
                "query.length": len(query),
                "user.id": user_id or "anonymous",
            },
        ) as span:
            yield span

    @asynccontextmanager
    async def trace_retrieval(
        self,
        query: str,
        k: int = 5,
        retriever_type: str = "dense",
    ):
        """Trace a retrieval operation."""
        async with self.async_span(
            "retrieval",
            operation_type=OperationType.RETRIEVAL,
            attributes={
                "retrieval.query": query[:200],
                "retrieval.k": k,
                "retrieval.type": retriever_type,
            },
        ) as span:
            yield span

    @asynccontextmanager
    async def trace_generation(
        self,
        model: str,
        prompt_tokens: int = 0,
    ):
        """Trace a generation operation."""
        async with self.async_span(
            "generation",
            operation_type=OperationType.GENERATION,
            attributes={
                "llm.model": model,
                "llm.prompt_tokens": prompt_tokens,
            },
        ) as span:
            yield span

    @asynccontextmanager
    async def trace_embedding(
        self,
        model: str,
        text_count: int = 1,
    ):
        """Trace an embedding operation."""
        async with self.async_span(
            "embedding",
            operation_type=OperationType.EMBEDDING,
            attributes={
                "embedding.model": model,
                "embedding.text_count": text_count,
            },
        ) as span:
            yield span

    @asynccontextmanager
    async def trace_rerank(
        self,
        model: str,
        doc_count: int = 0,
    ):
        """Trace a reranking operation."""
        async with self.async_span(
            "rerank",
            operation_type=OperationType.RERANK,
            attributes={
                "rerank.model": model,
                "rerank.doc_count": doc_count,
            },
        ) as span:
            yield span


# =============================================================================
# Convenience Functions
# =============================================================================


_global_tracer: Optional[AgentTracer] = None


def get_tracer(service_name: str = "rag-service") -> AgentTracer:
    """
    Get or create global tracer.

    Args:
        service_name: Service name for new tracer

    Returns:
        AgentTracer instance
    """
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = AgentTracer(service_name)
    return _global_tracer


def get_rag_tracer(service_name: str = "rag-service") -> RAGTracer:
    """
    Get a RAG-specific tracer.

    Args:
        service_name: Service name

    Returns:
        RAGTracer instance
    """
    return RAGTracer(service_name)


def trace_function(
    name: Optional[str] = None,
    operation_type: Optional[OperationType] = None,
) -> Callable[[F], F]:
    """
    Decorator for tracing functions with global tracer.

    Args:
        name: Span name
        operation_type: Operation type

    Returns:
        Decorator function
    """
    tracer = get_tracer()
    return tracer.trace(name, operation_type)
