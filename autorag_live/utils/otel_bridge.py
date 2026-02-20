"""OpenTelemetry bridge + RAG pipeline profiler.

Bridges the custom :class:`~autorag_live.utils.telemetry.TelemetryExporter`
hierarchy to the OpenTelemetry SDK so that every span, metric, and event
produced by the existing ``Telemetry`` class is *also* exported via OTLP
(gRPC or HTTP) to any compatible backend (Jaeger, Grafana Tempo, Datadog,
Honeycomb, …).

On top of the bridge, :class:`PipelineProfiler` provides **stage-level**
instrumentation for a complete RAG pipeline:

    retrieval → rerank → augmentation → generation → evaluation

Each stage gets a child span with standardised attributes
(``rag.stage``, ``rag.doc_count``, ``rag.latency_ms``, …) and the
profiler accumulates latency histograms so callers can inspect P50/P90/P99
per-stage without any external collector.

Usage::

    from autorag_live.utils.otel_bridge import (
        OTLPBridgeExporter,
        PipelineProfiler,
        setup_otel_pipeline_profiling,
    )

    # Quick one-liner
    profiler = setup_otel_pipeline_profiling(
        service_name="autorag",
        otlp_endpoint="http://localhost:4317",
    )

    # Instrument a pipeline run
    async with profiler.trace_pipeline(query="what is RAG?") as ctx:
        with ctx.stage("retrieval", doc_count=10):
            docs = await retriever.retrieve(query)
        with ctx.stage("rerank", doc_count=5):
            docs = reranker.rerank(query, docs)
        with ctx.stage("generation"):
            answer = await llm.generate(query, docs)

    # Inspect local histograms
    print(profiler.stage_latencies)
"""

from __future__ import annotations

import bisect
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional

from .telemetry import (
    Event,
    EventSeverity,
    Metric,
    MetricType,
    Span,
    SpanStatus,
    Telemetry,
    TelemetryExporter,
    get_telemetry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenTelemetry bridge exporter
# ---------------------------------------------------------------------------

# Lazy OTel SDK imports — the SDK is an *optional* dependency.  When absent
# the bridge gracefully degrades to a no-op so that the rest of the library
# still works.

_otel_available: Optional[bool] = None


def _check_otel() -> bool:
    """Return True if ``opentelemetry-sdk`` is importable."""
    global _otel_available
    if _otel_available is None:
        try:
            import opentelemetry.sdk.trace  # noqa: F401

            _otel_available = True
        except ImportError:
            _otel_available = False
    return _otel_available


class OTLPBridgeExporter(TelemetryExporter):
    """Bridge from AutoRAG custom telemetry to OpenTelemetry OTLP.

    When ``opentelemetry-sdk`` is installed the exporter converts every
    :class:`Span`, :class:`Metric`, and :class:`Event` into native OTel
    objects and pushes them to the configured OTLP endpoint (gRPC by
    default, HTTP/protobuf with ``use_http=True``).

    When the SDK is **not** installed the exporter silently drops data and
    logs a one-time warning.

    Args:
        service_name: ``service.name`` resource attribute.
        otlp_endpoint: OTLP collector endpoint (e.g. ``http://localhost:4317``).
        use_http: Use ``OTLPSpanExporter`` over HTTP instead of gRPC.
        headers: Extra headers for the exporter (e.g. auth tokens).
        insecure: Allow unencrypted gRPC (default ``True`` for local dev).
    """

    def __init__(
        self,
        service_name: str = "autorag",
        otlp_endpoint: str = "http://localhost:4317",
        use_http: bool = False,
        headers: Optional[Dict[str, str]] = None,
        insecure: bool = True,
    ) -> None:
        self._service_name = service_name
        self._endpoint = otlp_endpoint
        self._use_http = use_http
        self._headers = headers or {}
        self._insecure = insecure

        self._tracer: Any = None  # opentelemetry.trace.Tracer | None
        self._meter: Any = None  # opentelemetry.metrics.Meter | None
        self._warned = False

        if _check_otel():
            self._init_otel()
        else:
            logger.warning(
                "opentelemetry-sdk not installed — "
                "OTLPBridgeExporter will operate as a no-op.  "
                "Install with: pip install opentelemetry-sdk "
                "opentelemetry-exporter-otlp"
            )

    # -- OTel SDK initialisation --------------------------------------------

    def _init_otel(self) -> None:
        """Wire up OTel TracerProvider + MeterProvider with OTLP exporters."""
        from opentelemetry import metrics as otel_metrics
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": self._service_name})

        # --- Traces ---
        if self._use_http:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            span_exporter = OTLPSpanExporter(
                endpoint=f"{self._endpoint}/v1/traces",
                headers=self._headers,
            )
        else:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            span_exporter = OTLPSpanExporter(
                endpoint=self._endpoint,
                insecure=self._insecure,
                headers=self._headers or None,
            )

        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        otel_trace.set_tracer_provider(tracer_provider)
        self._tracer = otel_trace.get_tracer("autorag.pipeline")

        # --- Metrics ---
        try:
            if self._use_http:
                from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                    OTLPMetricExporter,
                )

                metric_exporter = OTLPMetricExporter(
                    endpoint=f"{self._endpoint}/v1/metrics",
                    headers=self._headers,
                )
            else:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                    OTLPMetricExporter,
                )

                metric_exporter = OTLPMetricExporter(
                    endpoint=self._endpoint,
                    insecure=self._insecure,
                    headers=self._headers or None,
                )

            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

            reader = PeriodicExportingMetricReader(metric_exporter)
            meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            otel_metrics.set_meter_provider(meter_provider)
            self._meter = otel_metrics.get_meter("autorag.pipeline")
        except Exception as exc:
            logger.warning(f"Could not initialise OTel metrics exporter: {exc}")
            self._meter = None

    # -- TelemetryExporter interface ----------------------------------------

    def export_spans(self, spans: List[Span]) -> None:
        """Convert AutoRAG spans → OTel spans and export."""
        if self._tracer is None:
            return

        from opentelemetry import trace as otel_trace

        for span in spans:
            otel_span = self._tracer.start_span(
                name=span.name,
                attributes=_safe_attributes(span.attributes),
            )
            # Map status
            if span.status == SpanStatus.ERROR:
                otel_span.set_status(
                    otel_trace.StatusCode.ERROR,
                    span.error_message or "unknown error",
                )
            elif span.status == SpanStatus.OK:
                otel_span.set_status(otel_trace.StatusCode.OK)

            # Forward events
            for evt in span.events:
                otel_span.add_event(
                    evt.get("name", "event"),
                    attributes=_safe_attributes(evt.get("attributes", {})),
                )

            otel_span.end()

    def export_metrics(self, metrics: List[Metric]) -> None:
        """Convert AutoRAG metrics → OTel metric instruments."""
        if self._meter is None:
            return

        for m in metrics:
            try:
                if m.metric_type == MetricType.COUNTER:
                    counter = self._meter.create_counter(
                        m.name,
                        description=f"AutoRAG counter: {m.name}",
                    )
                    counter.add(m.value, attributes=m.labels)

                elif m.metric_type == MetricType.GAUGE:
                    gauge = self._meter.create_up_down_counter(
                        m.name,
                        description=f"AutoRAG gauge: {m.name}",
                    )
                    gauge.add(m.value, attributes=m.labels)

                elif m.metric_type in (MetricType.HISTOGRAM, MetricType.SUMMARY):
                    hist = self._meter.create_histogram(
                        m.name,
                        description=f"AutoRAG histogram: {m.name}",
                    )
                    hist.record(m.value, attributes=m.labels)
            except Exception as exc:
                logger.debug(f"OTel metric export failed for {m.name}: {exc}")

    def export_events(self, events: List[Event]) -> None:
        """Convert AutoRAG events → OTel span events on a synthetic span."""
        if self._tracer is None:
            return

        from opentelemetry import trace as otel_trace

        # Group events into a single synthetic span so they appear in the
        # trace timeline.
        if not events:
            return

        severity_map = {
            EventSeverity.DEBUG: "DEBUG",
            EventSeverity.INFO: "INFO",
            EventSeverity.WARNING: "WARNING",
            EventSeverity.ERROR: "ERROR",
            EventSeverity.CRITICAL: "CRITICAL",
        }

        span = self._tracer.start_span("autorag.events")
        for evt in events:
            attrs = {
                "severity": severity_map.get(evt.severity, "INFO"),
                **_safe_attributes(evt.attributes),
            }
            span.add_event(evt.name, attributes=attrs)
        span.set_status(otel_trace.StatusCode.OK)
        span.end()


def _safe_attributes(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """OTel only accepts str | bool | int | float | Sequence[…] values."""
    safe: Dict[str, Any] = {}
    for k, v in attrs.items():
        if isinstance(v, (str, bool, int, float)):
            safe[k] = v
        elif isinstance(v, (list, tuple)):
            safe[k] = [str(x) for x in v]
        else:
            safe[k] = str(v)
    return safe


# ---------------------------------------------------------------------------
# Local latency histogram (no external dependency)
# ---------------------------------------------------------------------------


class _LatencyHistogram:
    """Thread-safe latency histogram with O(log n) insertion.

    Stores raw observations so callers can compute arbitrary percentiles.
    Caps at ``max_samples`` (FIFO eviction) to bound memory.
    """

    __slots__ = ("_values", "_max", "_lock")

    def __init__(self, max_samples: int = 10_000) -> None:
        self._values: List[float] = []
        self._max = max_samples
        self._lock = threading.Lock()

    def record(self, value_ms: float) -> None:
        with self._lock:
            bisect.insort(self._values, value_ms)
            if len(self._values) > self._max:
                self._values.pop(0)

    def percentile(self, p: float) -> float:
        """Return *p*-th percentile (0–100).  Returns 0 if empty."""
        with self._lock:
            n = len(self._values)
            if n == 0:
                return 0.0
            idx = min(int(n * p / 100.0), n - 1)
            return self._values[idx]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._values)

    @property
    def mean(self) -> float:
        with self._lock:
            if not self._values:
                return 0.0
            return sum(self._values) / len(self._values)

    def summary(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean, 2),
            "p50_ms": round(self.percentile(50), 2),
            "p90_ms": round(self.percentile(90), 2),
            "p99_ms": round(self.percentile(99), 2),
        }


# ---------------------------------------------------------------------------
# Pipeline profiler
# ---------------------------------------------------------------------------

# Standard RAG pipeline stages
RAG_STAGES = (
    "retrieval",
    "rerank",
    "augmentation",
    "generation",
    "evaluation",
    "cache_lookup",
    "cache_store",
    "routing",
    "safety_check",
)


@dataclass
class _StageRecord:
    """Accumulated profiling data for one pipeline stage."""

    name: str
    histogram: _LatencyHistogram = field(default_factory=_LatencyHistogram)
    call_count: int = 0
    error_count: int = 0


class PipelineProfileContext:
    """Context returned by :meth:`PipelineProfiler.trace_pipeline`.

    Use :meth:`stage` to create child spans for each RAG stage.
    """

    def __init__(
        self,
        profiler: "PipelineProfiler",
        telemetry: Telemetry,
        parent_span: Span,
        query: str,
    ) -> None:
        self._profiler = profiler
        self._telemetry = telemetry
        self._parent = parent_span
        self._query = query

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        doc_count: Optional[int] = None,
        extra_attrs: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Span]:
        """Instrument a single pipeline stage.

        Args:
            name: Stage name (e.g. ``"retrieval"``, ``"rerank"``).
            doc_count: Number of documents entering/exiting this stage.
            extra_attrs: Additional span attributes.

        Yields:
            The child :class:`Span` for the stage.
        """
        attrs: Dict[str, Any] = {
            "rag.stage": name,
            "rag.query": self._query[:200],
            **(extra_attrs or {}),
        }
        if doc_count is not None:
            attrs["rag.doc_count"] = doc_count

        t0 = time.monotonic()
        record = self._profiler._ensure_stage(name)
        record.call_count += 1

        with self._telemetry.trace(f"rag.{name}", attributes=attrs) as span:
            try:
                yield span
            except Exception:
                record.error_count += 1
                raise
            finally:
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                span.set_attribute("rag.latency_ms", round(elapsed_ms, 2))
                record.histogram.record(elapsed_ms)

                # Also push to the telemetry histogram
                self._telemetry.histogram(
                    "rag.stage.latency_ms",
                    elapsed_ms,
                    labels={"stage": name},
                )


class PipelineProfiler:
    """RAG pipeline profiler with per-stage latency histograms.

    Accumulates timing data across many pipeline runs so you can query
    P50/P90/P99 per-stage latencies *without* any external metrics backend.

    Args:
        telemetry: :class:`Telemetry` instance to use.  Defaults to the
            global singleton from :func:`get_telemetry`.
        service_name: ``service.name`` for traces.
    """

    def __init__(
        self,
        telemetry: Optional[Telemetry] = None,
        service_name: str = "autorag",
    ) -> None:
        self._telemetry = telemetry or get_telemetry(service_name)
        self._stages: Dict[str, _StageRecord] = {}
        self._lock = threading.Lock()
        self._pipeline_count = 0

    # -- public API ---------------------------------------------------------

    @contextmanager
    def trace_pipeline(
        self,
        query: str,
        *,
        pipeline_id: Optional[str] = None,
        extra_attrs: Optional[Dict[str, Any]] = None,
    ) -> Iterator[PipelineProfileContext]:
        """Trace an entire pipeline run.

        Creates a root span ``rag.pipeline`` and returns a context with a
        ``.stage()`` helper for child spans.

        Args:
            query: The user query driving this pipeline execution.
            pipeline_id: Optional unique identifier for the run.
            extra_attrs: Extra root-span attributes.

        Yields:
            :class:`PipelineProfileContext` with a ``.stage()`` helper.
        """
        pid = pipeline_id or str(uuid.uuid4())
        attrs: Dict[str, Any] = {
            "rag.pipeline_id": pid,
            "rag.query": query[:200],
            **(extra_attrs or {}),
        }

        t0 = time.monotonic()
        with self._telemetry.trace("rag.pipeline", attributes=attrs) as root:
            ctx = PipelineProfileContext(self, self._telemetry, root, query)
            try:
                yield ctx
            finally:
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                root.set_attribute("rag.total_latency_ms", round(elapsed_ms, 2))
                self._pipeline_count += 1
                self._telemetry.histogram(
                    "rag.pipeline.latency_ms",
                    elapsed_ms,
                )

    @property
    def pipeline_count(self) -> int:
        """Number of completed pipeline traces."""
        return self._pipeline_count

    @property
    def stage_latencies(self) -> Dict[str, Dict[str, float]]:
        """Per-stage latency summaries (count, mean, P50, P90, P99)."""
        with self._lock:
            return {
                name: {
                    **rec.histogram.summary(),
                    "error_count": rec.error_count,
                    "call_count": rec.call_count,
                }
                for name, rec in self._stages.items()
            }

    def reset(self) -> None:
        """Clear all accumulated profiling data."""
        with self._lock:
            self._stages.clear()
            self._pipeline_count = 0

    # -- internal -----------------------------------------------------------

    def _ensure_stage(self, name: str) -> _StageRecord:
        with self._lock:
            if name not in self._stages:
                self._stages[name] = _StageRecord(name=name)
            return self._stages[name]


# ---------------------------------------------------------------------------
# Convenience setup
# ---------------------------------------------------------------------------


def setup_otel_pipeline_profiling(
    service_name: str = "autorag",
    otlp_endpoint: str = "http://localhost:4317",
    use_http: bool = False,
    headers: Optional[Dict[str, str]] = None,
) -> PipelineProfiler:
    """One-liner to wire up OTel export + pipeline profiling.

    Creates an :class:`OTLPBridgeExporter`, plugs it into a fresh
    :class:`Telemetry` instance, and returns a :class:`PipelineProfiler`
    ready to instrument pipeline runs.

    Args:
        service_name: OpenTelemetry ``service.name``.
        otlp_endpoint: OTLP collector endpoint.
        use_http: Use HTTP/protobuf instead of gRPC.
        headers: Extra exporter headers.

    Returns:
        A fully wired :class:`PipelineProfiler`.
    """
    exporter = OTLPBridgeExporter(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        use_http=use_http,
        headers=headers,
    )
    telemetry = Telemetry(
        service_name=service_name,
        exporter=exporter,
    )
    telemetry.start()

    profiler = PipelineProfiler(telemetry=telemetry, service_name=service_name)
    logger.info(f"OTel pipeline profiling enabled → {otlp_endpoint} " f"(service={service_name})")
    return profiler


__all__ = [
    "OTLPBridgeExporter",
    "PipelineProfiler",
    "PipelineProfileContext",
    "RAG_STAGES",
    "setup_otel_pipeline_profiling",
    # Async GenAI semantic conventions
    "ATTR_GEN_AI_REQUEST_MODEL",
    "ATTR_GEN_AI_REQUEST_TEMPERATURE",
    "ATTR_GEN_AI_USAGE_INPUT_TOKENS",
    "ATTR_GEN_AI_USAGE_OUTPUT_TOKENS",
    "ATTR_RAG_QUERY",
    "ATTR_RAG_CACHE_HIT",
    "ATTR_RAG_CONSISTENCY_SCORE",
    "ATTR_RAG_HYDE_N_HYPOTHESES",
    "ATTR_RAG_RAPTOR_DEPTH",
    "async_rag_span",
    "async_pipeline_span",
]


# =============================================================================
# Async-native span context managers with GenAI semantic conventions
# (OpenTelemetry GenAI SemConv: https://opentelemetry.io/docs/specs/semconv/gen-ai/)
# =============================================================================

# ── GenAI Semantic Convention attribute names ─────────────────────────────
ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ATTR_GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
ATTR_GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
ATTR_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# ── Custom RAG attributes ─────────────────────────────────────────────────
ATTR_RAG_QUERY = "rag.query"
ATTR_RAG_QUERY_INTENT = "rag.query.intent"
ATTR_RAG_QUERY_COMPLEXITY = "rag.query.complexity"
ATTR_RAG_DOCS_RETRIEVED = "rag.retrieval.docs_retrieved"
ATTR_RAG_CACHE_HIT = "rag.cache.hit"
ATTR_RAG_CACHE_SIMILARITY = "rag.cache.similarity"
ATTR_RAG_CONFIDENCE = "rag.answer.confidence"
ATTR_RAG_CONSISTENCY_SCORE = "rag.answer.consistency_score"
ATTR_RAG_HYDE_N_HYPOTHESES = "rag.hyde.n_hypotheses"
ATTR_RAG_HYDE_ALPHA = "rag.hyde.alpha"
ATTR_RAG_RAPTOR_DEPTH = "rag.raptor.tree_depth"
ATTR_RAG_RAPTOR_NODES = "rag.raptor.nodes_searched"
ATTR_RAG_BUDGET_UTILISATION = "rag.token_budget.utilisation"
ATTR_RAG_BUDGET_REMAINING = "rag.token_budget.remaining_tokens"


class _AsyncNoOpSpan:
    """No-op async span — used when OTel SDK is not installed."""

    def set_attribute(self, key: str, value: Any) -> "_AsyncNoOpSpan":
        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "_AsyncNoOpSpan":
        return self

    def record_exception(self, exc: Exception) -> "_AsyncNoOpSpan":
        return self

    def set_status(self, *args: Any, **kwargs: Any) -> "_AsyncNoOpSpan":
        return self


@asynccontextmanager
async def async_rag_span(
    span_name: str,
    profiler: Optional["PipelineProfiler"] = None,
    query: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Any, None]:
    """
    Async context manager that creates an OTel-compatible RAG span.

    Works whether or not ``opentelemetry-sdk`` is installed:
    - If a ``PipelineProfiler`` is provided, stages are recorded in its
      latency histograms *and* forwarded to the OTLP bridge.
    - If no profiler is available, a no-op span is yielded.

    Args:
        span_name: OTel span name (e.g. ``"rag.retrieval"``).
        profiler: Optional PipelineProfiler for metric recording.
        query: Sets ``rag.query`` attribute automatically.
        attributes: Additional span attributes to set on entry.

    Yields:
        Span-like object (real OTel span or _AsyncNoOpSpan).

    Example:
        >>> async with async_rag_span("rag.retrieval", profiler=p, query=q) as span:
        ...     span.set_attribute(ATTR_RAG_DOCS_RETRIEVED, 10)
        ...     docs = await retriever.retrieve(q)
    """
    start_ns = time.perf_counter_ns()

    if profiler is None:
        span: Any = _AsyncNoOpSpan()
        if query:
            span.set_attribute(ATTR_RAG_QUERY, query[:500])
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        try:
            yield span
        finally:
            pass
        return

    # Extract stage name from span_name (e.g. "rag.retrieval" → "retrieval")
    stage = span_name.split(".")[-1]

    with profiler.trace_stage(stage) as ctx_span:
        if query:
            ctx_span.set_attribute(ATTR_RAG_QUERY, query[:500])
        if attributes:
            for k, v in attributes.items():
                ctx_span.set_attribute(k, v)
        try:
            yield ctx_span
        except Exception as exc:
            ctx_span.record_exception(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            ctx_span.set_attribute("duration_ms", round(elapsed_ms, 2))


@asynccontextmanager
async def async_pipeline_span(
    query: str,
    profiler: Optional["PipelineProfiler"] = None,
) -> AsyncGenerator[Any, None]:
    """
    Root span for an entire RAG pipeline call.

    Creates a ``rag.pipeline`` parent span and yields a span object.
    Nested stage spans should be opened inside this context.

    Args:
        query: The user query (recorded as ``rag.query``).
        profiler: Optional PipelineProfiler.

    Yields:
        Span-like object for the pipeline root span.
    """
    async with async_rag_span("rag.pipeline", profiler=profiler, query=query) as span:
        yield span
