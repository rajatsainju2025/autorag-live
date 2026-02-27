"""Tests for StateGraph — the graph-based workflow engine."""

import pytest

from autorag_live.core.context import ContextStage, RAGContext, RetrievedDocument
from autorag_live.core.state_graph import END, StateGraph

# ---------------------------------------------------------------------------
# Helper node functions
# ---------------------------------------------------------------------------


async def retrieve_node(ctx: RAGContext) -> RAGContext:
    doc = RetrievedDocument(doc_id="d1", content="RLHF explanation", score=0.9)
    return ctx.add_documents([doc]).advance_stage(ContextStage.RETRIEVAL)


async def grade_node(ctx: RAGContext) -> RAGContext:
    relevant = ctx.document_count > 0
    return ctx.set_metadata("docs_relevant", relevant)


async def generate_node(ctx: RAGContext) -> RAGContext:
    return ctx.with_answer("RLHF is ...", confidence=0.9).advance_stage(ContextStage.GENERATION)


async def rewrite_node(ctx: RAGContext) -> RAGContext:
    return ctx.set_metadata("rewritten", True)


async def failing_node(ctx: RAGContext) -> RAGContext:
    raise ValueError("Intentional failure")


def route_after_grade(ctx: RAGContext) -> str:
    if ctx.metadata.get("docs_relevant"):
        return "generate"
    return "rewrite"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStateGraphBuilder:
    """Test graph construction."""

    def test_add_nodes(self):
        g = StateGraph("test")
        g.add_node("a", retrieve_node)
        g.add_node("b", grade_node)
        assert "a" in g._nodes
        assert "b" in g._nodes

    def test_duplicate_node_raises(self):
        g = StateGraph()
        g.add_node("a", retrieve_node)
        with pytest.raises(ValueError, match="already exists"):
            g.add_node("a", grade_node)

    def test_reserved_end_name_raises(self):
        g = StateGraph()
        with pytest.raises(ValueError, match="reserved"):
            g.add_node(END, retrieve_node)

    def test_add_edge(self):
        g = StateGraph()
        g.add_node("a", retrieve_node)
        g.add_node("b", grade_node)
        g.add_edge("a", "b")
        assert g._edges["a"] == "b"

    def test_edge_to_end(self):
        g = StateGraph()
        g.add_node("a", retrieve_node)
        g.add_edge("a", END)
        assert g._edges["a"] == END

    def test_invalid_source_raises(self):
        g = StateGraph()
        with pytest.raises(ValueError, match="not found"):
            g.add_edge("missing", END)

    def test_duplicate_outgoing_edge_raises(self):
        g = StateGraph()
        g.add_node("a", retrieve_node)
        g.add_node("b", grade_node)
        g.add_node("c", generate_node)
        g.add_edge("a", "b")
        with pytest.raises(ValueError, match="already has"):
            g.add_edge("a", "c")

    def test_compile_without_entry_raises(self):
        g = StateGraph()
        g.add_node("a", retrieve_node)
        g.add_edge("a", END)
        with pytest.raises(ValueError, match="Entry point"):
            g.compile()

    def test_set_entry_point_invalid_raises(self):
        g = StateGraph()
        with pytest.raises(ValueError, match="not found"):
            g.set_entry_point("missing")


class TestCompiledGraphExecution:
    """Test graph execution via invoke()."""

    @pytest.mark.asyncio
    async def test_linear_pipeline(self):
        g = StateGraph("linear")
        g.add_node("retrieve", retrieve_node)
        g.add_node("generate", generate_node)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)

        compiled = g.compile()
        result = await compiled.invoke(RAGContext.create(query="What is RLHF?"))

        assert result.context.answer == "RLHF is ..."
        assert result.node_trace == ["retrieve", "generate"]
        assert result.terminated_at == END
        assert len(result.steps) == 2
        assert all(s.success for s in result.steps)

    @pytest.mark.asyncio
    async def test_conditional_branching(self):
        g = StateGraph("branching")
        g.add_node("retrieve", retrieve_node)
        g.add_node("grade", grade_node)
        g.add_node("generate", generate_node)
        g.add_node("rewrite", rewrite_node)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite"}
        )
        g.add_edge("generate", END)
        g.add_edge("rewrite", END)

        compiled = g.compile()
        result = await compiled.invoke(RAGContext.create(query="test"))

        # Should route to generate since retrieve_node adds a document
        assert "generate" in result.node_trace
        assert "rewrite" not in result.node_trace
        assert result.context.answer == "RLHF is ..."

    @pytest.mark.asyncio
    async def test_cycle_with_max_steps(self):
        """Test that cycles are handled and max_steps prevents infinite loops."""
        call_count = 0

        async def counting_node(ctx: RAGContext) -> RAGContext:
            nonlocal call_count
            call_count += 1
            return ctx.set_metadata("count", call_count)

        g = StateGraph("cycle")
        g.add_node("a", counting_node)
        g.set_entry_point("a")
        g.add_edge("a", "a")  # infinite cycle

        compiled = g.compile()
        result = await compiled.invoke(RAGContext.create(query="test"), max_steps=5)

        assert call_count == 5
        assert result.context.metadata.get("max_steps_exceeded") is True

    @pytest.mark.asyncio
    async def test_node_failure_marks_error(self):
        g = StateGraph("fail")
        g.add_node("bad", failing_node)
        g.set_entry_point("bad")
        g.add_edge("bad", END)

        compiled = g.compile()
        result = await compiled.invoke(RAGContext.create(query="test"))

        assert result.context.has_error
        assert not result.steps[0].success
        assert "Intentional failure" in (result.steps[0].error or "")

    @pytest.mark.asyncio
    async def test_callbacks_fired(self):
        starts = []
        ends = []

        def on_start(name, ctx):
            starts.append(name)

        def on_end(name, ctx, lat):
            ends.append((name, lat))

        g = StateGraph()
        g.add_node("a", retrieve_node)
        g.set_entry_point("a")
        g.add_edge("a", END)

        compiled = g.compile()
        await compiled.invoke(
            RAGContext.create(query="test"), on_node_start=on_start, on_node_end=on_end
        )

        assert starts == ["a"]
        assert len(ends) == 1
        assert ends[0][0] == "a"

    @pytest.mark.asyncio
    async def test_dead_end_node_terminates(self):
        """A node with no outgoing edge should terminate gracefully."""
        g = StateGraph()
        g.add_node("only", retrieve_node)
        g.set_entry_point("only")
        # Deliberately no edge added

        compiled = g.compile()
        result = await compiled.invoke(RAGContext.create(query="test"))

        assert result.terminated_at == END
        assert result.node_trace == ["only"]


class TestCompiledGraphIntrospection:
    """Test graph introspection methods."""

    def test_node_names(self):
        g = StateGraph()
        g.add_node("a", retrieve_node)
        g.add_node("b", grade_node)
        g.set_entry_point("a")
        g.add_edge("a", "b")
        g.add_edge("b", END)
        compiled = g.compile()
        assert set(compiled.node_names) == {"a", "b"}

    def test_entry_point(self):
        g = StateGraph()
        g.add_node("start", retrieve_node)
        g.set_entry_point("start")
        g.add_edge("start", END)
        compiled = g.compile()
        assert compiled.entry_point == "start"

    def test_get_edges_from(self):
        g = StateGraph()
        g.add_node("a", retrieve_node)
        g.add_node("b", grade_node)
        g.set_entry_point("a")
        g.add_edge("a", "b")
        g.add_edge("b", END)
        compiled = g.compile()

        assert compiled.get_edges_from("a") == {"type": "unconditional", "target": "b"}
        assert compiled.get_edges_from("b") == {"type": "unconditional", "target": END}

    def test_repr(self):
        g = StateGraph("my_graph")
        g.add_node("a", retrieve_node)
        g.set_entry_point("a")
        g.add_edge("a", END)
        compiled = g.compile()
        assert "my_graph" in repr(compiled)
