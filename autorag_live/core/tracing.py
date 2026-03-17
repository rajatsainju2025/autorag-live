"""OpenTelemetry tracing integration for AutoRAG-Live."""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("autorag_live.tracing")

# Lazy import to avoid hard dependency
_tracer = None


def get_tracer(name: str = "autorag_live"):
    """Get or initialize tracer."""
    global _tracer
    if _tracer is None:
        try:
            from opentelemetry import trace

            _tracer = trace.get_tracer(name)
        except ImportError:
            logger.warning(
                "OpenTelemetry not installed. Tracing disabled. "
                "Install with: pip install opentelemetry-api"
            )
            _tracer = NoOpTracer()
    return _tracer


class NoOpTracer:
    """No-op tracer when OpenTelemetry is not available."""

    def start_as_current_span(self, name: str, **kwargs):
        """Return a no-op context manager."""
        return NoOpSpan()


class NoOpSpan:
    """No-op span context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key: str, value: Any):
        pass

    def add_event(self, name: str, attributes: dict = None):  # type: ignore
        pass


def traced(span_name: Optional[str] = None):
    """Decorator to create a tracing span for an async function.

    Usage:
        @traced("retrieve_documents")
        async def retrieve(query: str):
            ...
    """

    def _decorator(func: Callable[..., Any]):
        name = span_name or func.__name__

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def _async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.start_as_current_span(name) as span:
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("status", "success")
                        return result
                    except Exception as e:
                        span.set_attribute("status", "error")
                        span.set_attribute("error.type", type(e).__name__)
                        span.add_event("exception", {"message": str(e)})
                        raise

            return _async_wrapper
        else:

            @functools.wraps(func)
            def _sync_wrapper(*args, **kwargs):
                tracer = get_tracer()
                with tracer.start_as_current_span(name) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("status", "success")
                        return result
                    except Exception as e:
                        span.set_attribute("status", "error")
                        span.set_attribute("error.type", type(e).__name__)
                        span.add_event("exception", {"message": str(e)})
                        raise

            return _sync_wrapper

    return _decorator


def setup_tracing(
    exporter_type: str = "console",
    service_name: str = "autorag-live",
) -> None:
    """Setup OpenTelemetry tracing.

    Args:
        exporter_type: "console", "jaeger", "otlp"
        service_name: service name for traces
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider = TracerProvider()

        if exporter_type == "console":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif exporter_type == "jaeger":
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter

                jaeger_exporter = JaegerExporter(
                    agent_host_name=service_name,
                )
                provider.add_span_processor(SimpleSpanProcessor(jaeger_exporter))
            except ImportError:
                logger.warning(
                    "Jaeger exporter not installed. "
                    "Install with: pip install opentelemetry-exporter-jaeger"
                )
        elif exporter_type == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

                otlp_exporter = OTLPSpanExporter()
                provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))
            except ImportError:
                logger.warning(
                    "OTLP exporter not installed. "
                    "Install with: pip install opentelemetry-exporter-otlp"
                )

        trace.set_tracer_provider(provider)
        logger.info(
            "OpenTelemetry tracing initialized with %s exporter",
            exporter_type,
        )
    except ImportError:
        logger.warning(
            "OpenTelemetry not installed. Tracing disabled. "
            "Install with: pip install opentelemetry-api"
        )
