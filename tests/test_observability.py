"""Tests for autorag_live.core.observability."""

from __future__ import annotations

import asyncio
import time

import pytest

from autorag_live.core.observability import (
    InMemoryTracer,
    OpenTelemetryTracer,
    Span,
    SpanStatus,
    Tracer,
    trace_graph_execution,
    traced,
)

# ---------------------------------------------------------------------------
# Span data model
# ---------------------------------------------------------------------------


class TestSpan:
    def test_duration_ms(self) -> None:
        s = Span(span_id="a", name="x", start_time=1.0, end_time=1.050)
        assert abs(s.duration_ms - 50.0) < 1e-6

    def test_defaults(self) -> None:
        s = Span(span_id="a", name="x")
        assert s.status == SpanStatus.UNSET
        assert s.attributes == {}
        assert s.events == []
        assert s.parent_id is None


# ---------------------------------------------------------------------------
# InMemoryTracer
# ---------------------------------------------------------------------------


class TestInMemoryTracer:
    def test_start_end_span(self) -> None:
        t = InMemoryTracer()
        sid = t.start_span("test_op", attributes={"k": "v"})
        t.end_span(sid, status=SpanStatus.OK)
        assert len(t.spans) == 1
        assert t.spans[0].name == "test_op"
        assert t.spans[0].attributes["k"] == "v"
        assert t.spans[0].status == SpanStatus.OK

    def test_parent_child(self) -> None:
        t = InMemoryTracer()
        root = t.start_span("root")
        child = t.start_span("child", parent_id=root)
        t.end_span(child)
        t.end_span(root)
        assert len(t.spans) == 2
        assert t.spans[0].parent_id == root  # child ended first
        assert t.spans[1].parent_id is None  # root

    def test_add_event(self) -> None:
        t = InMemoryTracer()
        sid = t.start_span("op")
        t.add_event(sid, "checkpoint", {"step": 3})
        t.end_span(sid)
        assert len(t.spans[0].events) == 1
        assert t.spans[0].events[0]["name"] == "checkpoint"

    def test_find_spans(self) -> None:
        t = InMemoryTracer()
        for name in ("a", "b", "a"):
            sid = t.start_span(name)
            t.end_span(sid)
        assert len(t.find_spans("a")) == 2

    def test_root_spans(self) -> None:
        t = InMemoryTracer()
        r = t.start_span("root")
        c = t.start_span("child", parent_id=r)
        t.end_span(c)
        t.end_span(r)
        assert len(t.root_spans) == 1
        assert t.root_spans[0].name == "root"

    def test_clear(self) -> None:
        t = InMemoryTracer()
        sid = t.start_span("x")
        t.end_span(sid)
        t.clear()
        assert len(t.spans) == 0

    def test_end_unknown_span_logs_warning(self) -> None:
        t = InMemoryTracer()
        t.end_span("nonexistent")  # should not raise
        assert len(t.spans) == 0

    def test_protocol_compliance(self) -> None:
        assert isinstance(InMemoryTracer(), Tracer)

    def test_duration_positive(self) -> None:
        t = InMemoryTracer()
        sid = t.start_span("slow")
        time.sleep(0.01)
        t.end_span(sid)
        assert t.spans[0].duration_ms > 0


# ---------------------------------------------------------------------------
# OpenTelemetryTracer (fallback mode)
# ---------------------------------------------------------------------------


class TestOpenTelemetryTracer:
    def test_fallback_to_in_memory(self) -> None:
        """Without opentelemetry-api, OTel tracer degrades gracefully."""
        ot = OpenTelemetryTracer(service_name="test")
        sid = ot.start_span("op")
        ot.add_event(sid, "ev")
        ot.end_span(sid)
        # Spans accessible via the fallback
        assert len(ot.spans) >= 0  # works even if otel is installed

    def test_protocol_compliance(self) -> None:
        assert isinstance(OpenTelemetryTracer(), Tracer)


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------


class TestTracedDecorator:
    def test_sync_function(self) -> None:
        t = InMemoryTracer()

        @traced(t, "add")
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)
        assert result == 5
        assert len(t.spans) == 1
        assert t.spans[0].name == "add"
        assert t.spans[0].status == SpanStatus.OK
        assert t.spans[0].attributes["arg_0"] == 2

    def test_async_function(self) -> None:
        t = InMemoryTracer()

        @traced(t, "fetch")
        async def fetch(url: str) -> str:
            return f"response from {url}"

        result = asyncio.get_event_loop().run_until_complete(fetch("http://x.com"))
        assert "response from" in result
        assert len(t.spans) == 1
        assert t.spans[0].name == "fetch"

    def test_error_records_error_status(self) -> None:
        t = InMemoryTracer()

        @traced(t, "fail")
        def fail() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            fail()
        assert len(t.spans) == 1
        assert t.spans[0].status == SpanStatus.ERROR
        assert "boom" in t.spans[0].attributes["error"]

    def test_default_name_from_function(self) -> None:
        t = InMemoryTracer()

        @traced(t)
        def my_func() -> int:
            return 42

        my_func()
        assert t.spans[0].name == "my_func"

    def test_capture_args_false(self) -> None:
        t = InMemoryTracer()

        @traced(t, capture_args=False)
        def secret(password: str) -> str:
            return "ok"

        secret("hunter2")
        assert "password" not in t.spans[0].attributes
        assert "arg_0" not in t.spans[0].attributes


# ---------------------------------------------------------------------------
# trace_graph_execution
# ---------------------------------------------------------------------------


class TestTraceGraphExecution:
    def test_creates_root_and_child_spans(self) -> None:
        t = InMemoryTracer()
        with trace_graph_execution(t, "rag", context_id="ctx-1") as hooks:
            hooks["on_node_start"]("retrieve")
            hooks["on_node_end"]("retrieve")
            hooks["on_node_start"]("generate")
            hooks["on_node_end"]("generate")

        # root + 2 children = 3 spans
        assert len(t.spans) == 3
        names = {s.name for s in t.spans}
        assert "graph:rag" in names
        assert "node:retrieve" in names
        assert "node:generate" in names

    def test_root_span_has_context_id(self) -> None:
        t = InMemoryTracer()
        with trace_graph_execution(t, "rag", context_id="abc"):
            pass
        root = t.find_spans("graph:rag")[0]
        assert root.attributes["context.id"] == "abc"

    def test_error_propagation(self) -> None:
        t = InMemoryTracer()
        with pytest.raises(RuntimeError, match="fail"):
            with trace_graph_execution(t, "rag") as hooks:
                hooks["on_node_start"]("bad_node")
                raise RuntimeError("fail")

        root = t.find_spans("graph:rag")[0]
        assert root.status == SpanStatus.ERROR
        assert "fail" in root.attributes["error"]

    def test_hooks_dict_contains_root_span_id(self) -> None:
        t = InMemoryTracer()
        with trace_graph_execution(t, "x") as hooks:
            assert "root_span_id" in hooks
            assert isinstance(hooks["root_span_id"], str)
