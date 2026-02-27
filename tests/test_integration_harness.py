"""
End-to-end integration test harness for the agentic RAG pipeline.

Wires together all the new modules added in this sprint:

    RAGContext → StateGraph → StreamingPipeline → Observability
    → Checkpoint → Guardrails → LLM Provider Factory

This test proves the modules compose correctly in a realistic
retrieve → grade → generate workflow with conditional routing,
streaming events, tracing spans, checkpoint persistence, and
guardrail enforcement.
"""

from __future__ import annotations

import pytest

from autorag_live.agent.guardrails import GuardrailEnforcer, PermissionPolicy, ToolPermission
from autorag_live.core.checkpoint import Checkpoint, InMemoryCheckpointStore
from autorag_live.core.context import ContextStage, RAGContext, RetrievedDocument
from autorag_live.core.observability import (
    InMemoryTracer,
    SpanStatus,
    trace_graph_execution,
    traced,
)
from autorag_live.core.state_graph import END, StateGraph
from autorag_live.llm.provider_factory import LLMProviderFactory
from autorag_live.pipeline.streaming import EventType, StreamingPipeline, collect_stream_events

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _build_rag_graph() -> StateGraph:
    """Build a realistic retrieve → grade → generate|rewrite graph."""

    async def retrieve(ctx: RAGContext) -> RAGContext:
        docs = [
            RetrievedDocument(
                doc_id="d1",
                content="RLHF uses human feedback to fine-tune LLMs.",
                score=0.95,
                source="wiki",
            ),
            RetrievedDocument(
                doc_id="d2",
                content="Reinforcement learning optimises a reward signal.",
                score=0.80,
                source="textbook",
            ),
        ]
        ctx = ctx.advance_stage(ContextStage.RETRIEVAL)
        return ctx.add_documents(docs)

    async def grade(ctx: RAGContext) -> RAGContext:
        relevant = any(d.score > 0.9 for d in ctx.documents)
        return ctx.set_metadata("docs_relevant", relevant)

    def route_after_grade(ctx: RAGContext) -> str:
        if ctx.metadata.get("docs_relevant"):
            return "generate"
        return "rewrite"

    async def generate(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.GENERATION)
        answer = "RLHF is a technique that uses human feedback to align LLMs."
        return ctx.with_answer(answer, confidence=0.92)

    async def rewrite(ctx: RAGContext) -> RAGContext:
        return ctx.set_metadata("rewritten", True)

    graph = StateGraph("integration_rag")
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("generate", generate)
    graph.add_node("rewrite", rewrite)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("rewrite", END)

    return graph


# ---------------------------------------------------------------------------
# 1. StateGraph + RAGContext (happy path)
# ---------------------------------------------------------------------------


class TestGraphContextIntegration:
    """Verify StateGraph correctly threads RAGContext through nodes."""

    @pytest.mark.asyncio
    async def test_retrieve_grade_generate_happy_path(self) -> None:
        compiled = _build_rag_graph().compile()
        ctx = RAGContext.create(query="What is RLHF?")
        result = await compiled.invoke(ctx)

        assert result.terminated_at == END
        assert result.context.answer is not None
        assert "RLHF" in result.context.answer
        assert result.context.confidence > 0.9
        assert result.context.document_count == 2
        assert result.node_trace == ["retrieve", "grade", "generate"]


# ---------------------------------------------------------------------------
# 2. StreamingPipeline + StateGraph
# ---------------------------------------------------------------------------


class TestStreamingIntegration:
    """Verify StreamingPipeline yields correct events from graph execution."""

    @pytest.mark.asyncio
    async def test_stream_events_sequence(self) -> None:
        compiled = _build_rag_graph().compile()
        pipeline = StreamingPipeline(compiled)
        ctx = RAGContext.create(query="What is RLHF?")

        events = await collect_stream_events(pipeline, ctx)
        types = [e.event_type for e in events]

        # Must start with PIPELINE_START and end with COMPLETE
        assert types[0] == EventType.PIPELINE_START
        assert types[-1] == EventType.COMPLETE

        # Must have STAGE_START/STAGE_END for each node
        stage_starts = [e.stage for e in events if e.event_type == EventType.STAGE_START]
        stage_ends = [e.stage for e in events if e.event_type == EventType.STAGE_END]
        assert stage_starts == ["retrieve", "grade", "generate"]
        assert stage_ends == ["retrieve", "grade", "generate"]

        # COMPLETE event should contain the answer
        final = events[-1]
        assert final.data["answer"] is not None
        assert final.data["confidence"] > 0.9
        assert final.data["node_trace"] == ["retrieve", "grade", "generate"]


# ---------------------------------------------------------------------------
# 3. Observability + StateGraph
# ---------------------------------------------------------------------------


class TestObservabilityIntegration:
    """Verify tracing hooks capture graph execution spans."""

    @pytest.mark.asyncio
    async def test_trace_graph_creates_root_and_child_spans(self) -> None:
        tracer = InMemoryTracer()
        compiled = _build_rag_graph().compile()
        ctx = RAGContext.create(query="What is RLHF?")

        with trace_graph_execution(tracer, "rag", context_id=ctx.context_id) as hooks:
            await compiled.invoke(
                ctx,
                on_node_start=lambda name, _ctx: hooks["on_node_start"](name),
                on_node_end=lambda name, _ctx, _lat: hooks["on_node_end"](name),
            )

        assert len(tracer.spans) == 4  # root + 3 nodes
        root = tracer.find_spans("graph:rag")[0]
        assert root.status == SpanStatus.OK
        assert root.attributes["context.id"] == ctx.context_id

        node_names = {s.name for s in tracer.spans if s.name.startswith("node:")}
        assert node_names == {"node:retrieve", "node:grade", "node:generate"}

    @pytest.mark.asyncio
    async def test_traced_decorator_on_pipeline_nodes(self) -> None:
        tracer = InMemoryTracer()

        @traced(tracer, "score_documents")
        async def score(query: str, doc_count: int) -> float:
            return 0.95

        result = await score("What is RLHF?", doc_count=2)
        assert result == 0.95
        assert len(tracer.spans) == 1
        assert tracer.spans[0].name == "score_documents"
        assert tracer.spans[0].attributes["arg_0"] == "What is RLHF?"
        assert tracer.spans[0].attributes["doc_count"] == 2


# ---------------------------------------------------------------------------
# 4. Checkpoint + RAGContext round-trip
# ---------------------------------------------------------------------------


class TestCheckpointIntegration:
    """Verify checkpoint save/restore preserves pipeline state."""

    @pytest.mark.asyncio
    async def test_checkpoint_round_trip_after_graph_execution(self) -> None:
        compiled = _build_rag_graph().compile()
        ctx = RAGContext.create(query="What is RLHF?")
        result = await compiled.invoke(ctx)

        # Save checkpoint (sync store)
        store = InMemoryCheckpointStore()
        cp = Checkpoint.from_context(result.context, node="generate")
        store.save(cp)

        # Load checkpoint and verify
        loaded = store.load(cp.checkpoint_id)
        assert loaded is not None
        restored = loaded.to_context()
        assert restored.query == "What is RLHF?"
        assert restored.answer == result.context.answer
        assert restored.confidence == result.context.confidence
        assert restored.document_count == 2

    def test_list_checkpoints(self) -> None:
        store = InMemoryCheckpointStore()
        ctx1 = RAGContext.create(query="q1")
        ctx2 = RAGContext.create(query="q2")

        cp1 = Checkpoint.from_context(ctx1, node="a")
        cp2 = Checkpoint.from_context(ctx2, node="b")
        store.save(cp1)
        store.save(cp2)

        # Each checkpoint belongs to a different context
        cps1 = store.list_for_context(ctx1.context_id)
        cps2 = store.list_for_context(ctx2.context_id)
        assert len(cps1) == 1
        assert len(cps2) == 1


# ---------------------------------------------------------------------------
# 5. Guardrails integration
# ---------------------------------------------------------------------------


class TestGuardrailsIntegration:
    """Verify guardrail enforcement in a pipeline context."""

    def test_permissive_allows_all(self) -> None:
        enforcer = GuardrailEnforcer(policy=PermissionPolicy.permissive())
        # Should not raise for any tool
        enforcer.check("web_search", {ToolPermission.NETWORK, ToolPermission.READ})
        enforcer.check("file_write", {ToolPermission.FILESYSTEM, ToolPermission.WRITE})

    def test_read_only_blocks_writes(self) -> None:
        enforcer = GuardrailEnforcer(policy=PermissionPolicy.read_only())
        # Read is allowed
        enforcer.check("db_query", {ToolPermission.READ})
        # Write is denied
        from autorag_live.agent.guardrails import PermissionDenied

        with pytest.raises(PermissionDenied):
            enforcer.check("db_write", {ToolPermission.WRITE})

    def test_restricted_blocks_sensitive(self) -> None:
        enforcer = GuardrailEnforcer(policy=PermissionPolicy.restricted())
        from autorag_live.agent.guardrails import PermissionDenied

        with pytest.raises(PermissionDenied):
            enforcer.check("admin_op", {ToolPermission.ADMIN})

    def test_audit_log_accumulates(self) -> None:
        from autorag_live.agent.guardrails import EnforcementAction

        # Use a policy with AUDIT enforcement so violations are logged but not raised
        audit_policy = PermissionPolicy(
            allowed={ToolPermission.READ},
            enforcement=EnforcementAction.AUDIT,
        )
        enforcer = GuardrailEnforcer(policy=audit_policy)
        enforcer.check("admin_op", {ToolPermission.ADMIN})
        assert len(enforcer.audit_log) == 1
        assert enforcer.audit_log[0]["tool"] == "admin_op"


# ---------------------------------------------------------------------------
# 6. LLM Provider Factory
# ---------------------------------------------------------------------------


class TestLLMProviderIntegration:
    """Verify provider factory produces working mock providers."""

    def test_mock_provider_generates(self) -> None:
        provider = LLMProviderFactory.create("mock")
        assert provider is not None
        assert provider.provider_name is not None
        assert provider.config.model == "mock"

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            LLMProviderFactory.create("nonexistent_provider_xyz")


# ---------------------------------------------------------------------------
# 7. Full end-to-end: all modules wired together
# ---------------------------------------------------------------------------


class TestFullEndToEnd:
    """The grand integration: query → graph → streaming → tracing → checkpoint."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_all_modules(self) -> None:
        # 1. Build graph
        compiled = _build_rag_graph().compile()

        # 2. Set up tracing
        tracer = InMemoryTracer()

        # 3. Set up guardrails
        enforcer = GuardrailEnforcer(policy=PermissionPolicy.permissive())
        enforcer.check("retrieve", {ToolPermission.READ, ToolPermission.NETWORK})

        # 4. Create context
        ctx = RAGContext.create(query="What is RLHF?", tags=frozenset(["test"]))

        # 5. Execute with tracing
        with trace_graph_execution(tracer, "full_pipeline", context_id=ctx.context_id) as hooks:
            result = await compiled.invoke(
                ctx,
                on_node_start=lambda name, _ctx: hooks["on_node_start"](name),
                on_node_end=lambda name, _ctx, _lat: hooks["on_node_end"](name),
            )

        # 6. Verify graph result
        assert result.terminated_at == END
        assert result.context.answer is not None
        assert result.context.confidence > 0.9
        assert result.node_trace == ["retrieve", "grade", "generate"]

        # 7. Verify tracing captured everything
        assert len(tracer.spans) == 4  # root + 3 nodes
        root = tracer.find_spans("graph:full_pipeline")[0]
        assert root.status == SpanStatus.OK

        # 8. Checkpoint the final state (sync store)
        store = InMemoryCheckpointStore()
        cp = Checkpoint.from_context(result.context, node="generate")
        store.save(cp)

        loaded = store.load(cp.checkpoint_id)
        assert loaded is not None
        restored = loaded.to_context()
        assert restored.answer == result.context.answer
        assert restored.query == "What is RLHF?"

        # 9. Stream the same pipeline and verify events
        pipeline = StreamingPipeline(compiled)
        events = await collect_stream_events(pipeline, RAGContext.create(query="What is RLHF?"))
        assert events[0].event_type == EventType.PIPELINE_START
        assert events[-1].event_type == EventType.COMPLETE
        assert events[-1].data["answer"] is not None

    @pytest.mark.asyncio
    async def test_conditional_routing_to_rewrite_branch(self) -> None:
        """Test the alternate path when docs are not relevant."""

        async def retrieve_bad(ctx: RAGContext) -> RAGContext:
            docs = [
                RetrievedDocument(
                    doc_id="d1",
                    content="Irrelevant content.",
                    score=0.3,
                    source="noise",
                ),
            ]
            return ctx.add_documents(docs)

        async def grade(ctx: RAGContext) -> RAGContext:
            relevant = any(d.score > 0.9 for d in ctx.documents)
            return ctx.set_metadata("docs_relevant", relevant)

        def route(ctx: RAGContext) -> str:
            return "generate" if ctx.metadata.get("docs_relevant") else "rewrite"

        async def generate(ctx: RAGContext) -> RAGContext:
            return ctx.with_answer("answer", confidence=0.9)

        async def rewrite(ctx: RAGContext) -> RAGContext:
            return ctx.set_metadata("rewritten", True)

        graph = StateGraph("alt_path")
        graph.add_node("retrieve", retrieve_bad)
        graph.add_node("grade", grade)
        graph.add_node("generate", generate)
        graph.add_node("rewrite", rewrite)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges("grade", route, {"generate": "generate", "rewrite": "rewrite"})
        graph.add_edge("generate", END)
        graph.add_edge("rewrite", END)

        compiled = graph.compile()
        result = await compiled.invoke(RAGContext.create(query="test"))

        # Should take the rewrite branch
        assert result.node_trace == ["retrieve", "grade", "rewrite"]
        assert result.context.metadata.get("rewritten") is True
        assert result.context.answer is None  # never reached generate
