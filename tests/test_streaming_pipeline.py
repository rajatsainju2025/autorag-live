"""Tests for the streaming pipeline."""

import pytest

from autorag_live.core.context import ContextStage, RAGContext, RetrievedDocument
from autorag_live.core.state_graph import END, StateGraph
from autorag_live.pipeline.streaming import (
    EventType,
    StreamEvent,
    StreamingPipeline,
    collect_stream_events,
)

# ---------------------------------------------------------------------------
# Helper nodes
# ---------------------------------------------------------------------------


async def retrieve(ctx: RAGContext) -> RAGContext:
    doc = RetrievedDocument(doc_id="d1", content="RLHF info", score=0.9)
    return ctx.add_documents([doc]).advance_stage(ContextStage.RETRIEVAL)


async def generate(ctx: RAGContext) -> RAGContext:
    return ctx.with_answer("RLHF is ...", confidence=0.85).advance_stage(ContextStage.GENERATION)


async def failing_node(ctx: RAGContext) -> RAGContext:
    raise RuntimeError("Boom!")


def _build_graph() -> StateGraph:
    g = StateGraph("test")
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreamEvent:
    def test_to_dict(self):
        event = StreamEvent(
            event_type=EventType.TOKEN,
            stage="generate",
            data={"token": "Hello"},
            context_id="abc",
        )
        d = event.to_dict()
        assert d["event"] == "token"
        assert d["stage"] == "generate"
        assert d["data"]["token"] == "Hello"

    def test_to_sse(self):
        event = StreamEvent(
            event_type=EventType.COMPLETE,
            data={"answer": "done"},
        )
        sse = event.to_sse()
        assert sse.startswith("event: complete\n")
        assert '"answer": "done"' in sse


class TestStreamingPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_events(self):
        graph = _build_graph().compile()
        pipeline = StreamingPipeline(graph)
        ctx = RAGContext.create(query="What is RLHF?")

        events = await collect_stream_events(pipeline, ctx)
        event_types = [e.event_type for e in events]

        assert EventType.PIPELINE_START in event_types
        assert EventType.STAGE_START in event_types
        assert EventType.STAGE_END in event_types
        assert EventType.COMPLETE in event_types

    @pytest.mark.asyncio
    async def test_pipeline_start_has_query(self):
        graph = _build_graph().compile()
        pipeline = StreamingPipeline(graph)
        ctx = RAGContext.create(query="What is RLHF?")

        events = await collect_stream_events(pipeline, ctx)
        start_event = next(e for e in events if e.event_type == EventType.PIPELINE_START)
        assert start_event.data["query"] == "What is RLHF?"

    @pytest.mark.asyncio
    async def test_complete_event_has_answer(self):
        graph = _build_graph().compile()
        pipeline = StreamingPipeline(graph)
        ctx = RAGContext.create(query="test")

        events = await collect_stream_events(pipeline, ctx)
        complete = next(e for e in events if e.event_type == EventType.COMPLETE)
        assert complete.data["answer"] == "RLHF is ..."
        assert complete.data["confidence"] == 0.85
        assert complete.data["node_trace"] == ["retrieve", "generate"]

    @pytest.mark.asyncio
    async def test_stage_events_per_node(self):
        graph = _build_graph().compile()
        pipeline = StreamingPipeline(graph)
        ctx = RAGContext.create(query="test")

        events = await collect_stream_events(pipeline, ctx)
        stage_starts = [e.stage for e in events if e.event_type == EventType.STAGE_START]
        stage_ends = [e.stage for e in events if e.event_type == EventType.STAGE_END]

        assert "retrieve" in stage_starts
        assert "generate" in stage_starts
        assert "retrieve" in stage_ends
        assert "generate" in stage_ends

    @pytest.mark.asyncio
    async def test_error_in_pipeline(self):
        g = StateGraph("fail")
        g.add_node("bad", failing_node)
        g.set_entry_point("bad")
        g.add_edge("bad", END)
        graph = g.compile()

        pipeline = StreamingPipeline(graph)
        ctx = RAGContext.create(query="test")

        events = await collect_stream_events(pipeline, ctx)
        complete = next(e for e in events if e.event_type == EventType.COMPLETE)
        assert complete.data["has_error"] is True

    @pytest.mark.asyncio
    async def test_context_id_propagated(self):
        graph = _build_graph().compile()
        pipeline = StreamingPipeline(graph)
        ctx = RAGContext.create(query="test")

        events = await collect_stream_events(pipeline, ctx)
        for event in events:
            assert event.context_id == ctx.context_id
