"""
Observability bridge for agentic RAG pipelines.

Provides structured tracing, metrics, and logging without hard-coupling
to any vendor SDK.  Ships a lightweight ``Tracer`` protocol and two
concrete backends:

1. **InMemoryTracer** — zero-dependency; stores spans in a list for
   testing and local debugging.
2. **OpenTelemetryTracer** — delegates to the OpenTelemetry SDK when
   available; gracefully degrades to ``InMemoryTracer`` if the
   ``opentelemetry-api`` package is not installed.

Key Features:
    * ``@traced`` decorator — auto-creates a span around any sync/async
      function with arguments captured as span attributes.
    * ``trace_graph_execution()`` — context manager that wraps a full
      ``CompiledGraph.invoke()`` in a root span and attaches child spans
      for each graph node via the callback hooks.
    * Minimal overhead: InMemoryTracer adds < 5 µs per span.

Example:
    >>> from autorag_live.core.observability import InMemoryTracer, traced
    >>>
    >>> tracer = InMemoryTracer()
    >>>
    >>> @traced(tracer, "retrieve")
    ... async def retrieve(query: str) -> list[str]:
    ...     return ["doc1", "doc2"]
    ...
    >>> import asyncio; asyncio.run(retrieve("What is RLHF?"))
    >>> print(tracer.spans[0].name)  # "retrieve"
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class SpanStatus(str, Enum):
    """Outcome status for a completed span."""

    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


@dataclass(frozen=True)
class Span:
    """Immutable record of a single traced operation."""

    span_id: str
    name: str
    parent_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Wall-clock duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000.0


# ---------------------------------------------------------------------------
# Tracer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Tracer(Protocol):
    """Minimal tracing interface — any backend must implement these."""

    def start_span(
        self,
        name: str,
        *,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Begin a span and return its unique id."""
        ...

    def end_span(
        self,
        span_id: str,
        *,
        status: SpanStatus = SpanStatus.OK,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Finish a span, recording its status and optional extra attrs."""
        ...

    def add_event(
        self,
        span_id: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Attach a timestamped event to an open span."""
        ...


# ---------------------------------------------------------------------------
# InMemoryTracer
# ---------------------------------------------------------------------------


class InMemoryTracer:
    """Zero-dependency tracer that stores spans in a plain list.

    Ideal for unit tests, local debugging, and pipeline introspection.
    """

    def __init__(self) -> None:
        self._open: Dict[str, dict] = {}
        self.spans: List[Span] = []

    # -- Tracer protocol ----------------------------------------------------

    def start_span(
        self,
        name: str,
        *,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        span_id = uuid.uuid4().hex[:16]
        self._open[span_id] = {
            "span_id": span_id,
            "name": name,
            "parent_id": parent_id,
            "start_time": time.monotonic(),
            "attributes": dict(attributes or {}),
            "events": [],
        }
        return span_id

    def end_span(
        self,
        span_id: str,
        *,
        status: SpanStatus = SpanStatus.OK,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = self._open.pop(span_id, None)
        if record is None:
            logger.warning("end_span called for unknown span %s", span_id)
            return
        if attributes:
            record["attributes"].update(attributes)
        record["end_time"] = time.monotonic()
        record["status"] = status
        self.spans.append(Span(**record))

    def add_event(
        self,
        span_id: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = self._open.get(span_id)
        if record is None:
            return
        record["events"].append(
            {"name": name, "timestamp": time.monotonic(), "attributes": attributes or {}}
        )

    # -- Convenience --------------------------------------------------------

    def clear(self) -> None:
        """Discard all recorded spans."""
        self._open.clear()
        self.spans.clear()

    def find_spans(self, name: str) -> List[Span]:
        """Return all completed spans matching *name*."""
        return [s for s in self.spans if s.name == name]

    @property
    def root_spans(self) -> List[Span]:
        """Return spans that have no parent."""
        return [s for s in self.spans if s.parent_id is None]


# ---------------------------------------------------------------------------
# OpenTelemetryTracer (graceful degradation)
# ---------------------------------------------------------------------------


class OpenTelemetryTracer:
    """Delegates to the OpenTelemetry SDK when ``opentelemetry-api`` is
    installed.  Falls back to :class:`InMemoryTracer` otherwise.

    Parameters
    ----------
    service_name:
        Logical name of the service (appears in Jaeger / Zipkin).
    """

    def __init__(self, service_name: str = "autorag-live") -> None:
        self._service_name = service_name
        self._otel_available = False
        self._fallback: Optional[InMemoryTracer] = None
        self._otel_spans: Dict[str, Any] = {}

        try:
            from opentelemetry import trace  # type: ignore[import-untyped]

            self._otel_tracer = trace.get_tracer(service_name)
            self._otel_available = True
        except ImportError:
            logger.info("opentelemetry-api not installed — falling back to InMemoryTracer")
            self._fallback = InMemoryTracer()

    # -- Tracer protocol ----------------------------------------------------

    def start_span(
        self,
        name: str,
        *,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self._otel_available:
            assert self._fallback is not None
            return self._fallback.start_span(name, parent_id=parent_id, attributes=attributes)

        span = self._otel_tracer.start_span(name, attributes=attributes or {})
        span_id = uuid.uuid4().hex[:16]
        self._otel_spans[span_id] = span
        return span_id

    def end_span(
        self,
        span_id: str,
        *,
        status: SpanStatus = SpanStatus.OK,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._otel_available:
            assert self._fallback is not None
            self._fallback.end_span(span_id, status=status, attributes=attributes)
            return

        from opentelemetry import trace  # type: ignore[import-untyped]

        span = self._otel_spans.pop(span_id, None)
        if span is None:
            return
        if attributes:
            span.set_attributes(attributes)
        if status == SpanStatus.ERROR:
            span.set_status(trace.StatusCode.ERROR)
        else:
            span.set_status(trace.StatusCode.OK)
        span.end()

    def add_event(
        self,
        span_id: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._otel_available:
            assert self._fallback is not None
            self._fallback.add_event(span_id, name, attributes)
            return

        span = self._otel_spans.get(span_id)
        if span is not None:
            span.add_event(name, attributes=attributes or {})

    # -- Convenience --------------------------------------------------------

    @property
    def spans(self) -> List[Span]:
        """Access spans (only available with fallback backend)."""
        if self._fallback:
            return self._fallback.spans
        return []


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------


def traced(
    tracer: Tracer,
    span_name: Optional[str] = None,
    *,
    capture_args: bool = True,
) -> Callable:
    """Decorator that wraps a sync or async function in a tracer span.

    Parameters
    ----------
    tracer:
        A :class:`Tracer` instance (InMemoryTracer, OpenTelemetryTracer, …).
    span_name:
        Override for the span name.  Defaults to the function's ``__name__``.
    capture_args:
        If ``True``, scalar positional/keyword arguments are stored as
        span attributes for debugging.
    """

    def decorator(fn: Callable) -> Callable:
        name = span_name or fn.__name__

        import asyncio as _asyncio

        if _asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                attrs = _capture_attrs(args, kwargs) if capture_args else {}
                sid = tracer.start_span(name, attributes=attrs)
                try:
                    result = await fn(*args, **kwargs)
                    tracer.end_span(sid, status=SpanStatus.OK)
                    return result
                except Exception as exc:
                    tracer.end_span(
                        sid,
                        status=SpanStatus.ERROR,
                        attributes={"error": str(exc)},
                    )
                    raise

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                attrs = _capture_attrs(args, kwargs) if capture_args else {}
                sid = tracer.start_span(name, attributes=attrs)
                try:
                    result = fn(*args, **kwargs)
                    tracer.end_span(sid, status=SpanStatus.OK)
                    return result
                except Exception as exc:
                    tracer.end_span(
                        sid,
                        status=SpanStatus.ERROR,
                        attributes={"error": str(exc)},
                    )
                    raise

            return sync_wrapper

    return decorator


def _capture_attrs(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract JSON-safe scalar args/kwargs for span attributes."""
    attrs: Dict[str, Any] = {}
    for i, v in enumerate(args):
        if isinstance(v, (str, int, float, bool)):
            attrs[f"arg_{i}"] = v
    for k, v in kwargs.items():
        if isinstance(v, (str, int, float, bool)):
            attrs[k] = v
    return attrs


# ---------------------------------------------------------------------------
# Graph execution tracing helper
# ---------------------------------------------------------------------------


@contextmanager
def trace_graph_execution(
    tracer: Tracer,
    graph_name: str,
    *,
    context_id: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager that creates a root span for an entire graph run
    and provides callback functions for node-level child spans.

    Usage::

        with trace_graph_execution(tracer, "rag_pipeline") as hooks:
            result = await compiled.invoke(ctx,
                on_node_start=hooks["on_node_start"],
                on_node_end=hooks["on_node_end"])

    The resulting ``tracer.spans`` will contain one root span and one
    child span per executed graph node.
    """
    root_attrs: Dict[str, Any] = {"graph.name": graph_name}
    if context_id:
        root_attrs["context.id"] = context_id

    root_id = tracer.start_span(f"graph:{graph_name}", attributes=root_attrs)
    child_spans: Dict[str, str] = {}

    def on_node_start(node_name: str) -> None:
        sid = tracer.start_span(
            f"node:{node_name}",
            parent_id=root_id,
            attributes={"node.name": node_name},
        )
        child_spans[node_name] = sid

    def on_node_end(node_name: str) -> None:
        sid = child_spans.pop(node_name, None)
        if sid:
            tracer.end_span(sid, status=SpanStatus.OK)

    hooks: Dict[str, Any] = {
        "on_node_start": on_node_start,
        "on_node_end": on_node_end,
        "root_span_id": root_id,
    }

    try:
        yield hooks
        tracer.end_span(root_id, status=SpanStatus.OK)
    except Exception as exc:
        tracer.end_span(
            root_id,
            status=SpanStatus.ERROR,
            attributes={"error": str(exc)},
        )
        raise
