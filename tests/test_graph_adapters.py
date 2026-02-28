"""Tests for autorag_live.agent.graph_adapters."""

from __future__ import annotations

import pytest

from autorag_live.agent.graph_adapters import (
    agent_to_node,
    flare_node,
    lats_node,
    plan_execute_node,
    react_node,
    self_rag_node,
)
from autorag_live.core.context import ContextStage, RAGContext, RetrievedDocument
from autorag_live.core.state_graph import END, StateGraph

# ---------------------------------------------------------------------------
# Mock LLM / Retriever
# ---------------------------------------------------------------------------


async def mock_llm(prompt: str, **kwargs) -> str:
    """Deterministic mock LLM."""
    if "Which candidate" in prompt:
        return "1"
    if "rate the relevance" in prompt.lower():
        return "0.9"
    if "plan" in prompt.lower() and "step" in prompt.lower():
        return "Step 1: Analyze the question\nStep 2: Synthesize answer"
    return "Final Answer: RLHF uses human feedback to align LLMs."


async def mock_retriever(query: str, top_k: int = 5) -> list:
    return [
        {
            "id": "r1",
            "content": "RLHF is reinforcement learning from human feedback.",
            "score": 0.9,
        },
        {"id": "r2", "content": "Human feedback improves model alignment.", "score": 0.8},
    ]


def _ctx_with_docs(query: str = "What is RLHF?") -> RAGContext:
    return RAGContext.create(query=query).add_documents(
        [
            RetrievedDocument(doc_id="d1", content="RLHF uses human feedback.", score=0.95),
            RetrievedDocument(doc_id="d2", content="RL optimises rewards.", score=0.8),
        ]
    )


# ---------------------------------------------------------------------------
# ReAct adapter
# ---------------------------------------------------------------------------


class TestReactNode:
    @pytest.mark.asyncio
    async def test_produces_answer(self) -> None:
        node = react_node(mock_llm)
        ctx = _ctx_with_docs()
        result = await node(ctx)
        assert result.answer is not None
        assert "RLHF" in result.answer
        assert result.metadata["agent_type"] == "react"

    @pytest.mark.asyncio
    async def test_records_reasoning_traces(self) -> None:
        node = react_node(mock_llm, max_iterations=2)
        result = await node(_ctx_with_docs())
        assert len(result.reasoning_traces) >= 1

    @pytest.mark.asyncio
    async def test_advances_stage(self) -> None:
        node = react_node(mock_llm)
        result = await node(_ctx_with_docs())
        assert any(sl.stage == ContextStage.GENERATION for sl in result.latencies)


# ---------------------------------------------------------------------------
# Self-RAG adapter
# ---------------------------------------------------------------------------


class TestSelfRagNode:
    @pytest.mark.asyncio
    async def test_produces_answer(self) -> None:
        node = self_rag_node(mock_llm, mock_retriever)
        result = await node(_ctx_with_docs())
        assert result.answer is not None
        assert result.metadata["agent_type"] == "self_rag"

    @pytest.mark.asyncio
    async def test_low_relevance_triggers_reretrieval(self) -> None:
        async def low_relevance_llm(prompt: str, **kw) -> str:
            if "rate the relevance" in prompt.lower():
                return "0.2"
            return "Re-retrieved answer about RLHF."

        node = self_rag_node(low_relevance_llm, mock_retriever, relevance_threshold=0.5)
        result = await node(_ctx_with_docs())
        # Should have re-retrieved docs
        traces = [t.content for t in result.reasoning_traces]
        assert any("Re-retrieved" in t for t in traces)


# ---------------------------------------------------------------------------
# FLARE adapter
# ---------------------------------------------------------------------------


class TestFlareNode:
    @pytest.mark.asyncio
    async def test_produces_answer(self) -> None:
        node = flare_node(mock_llm, mock_retriever)
        result = await node(_ctx_with_docs())
        assert result.answer is not None
        assert result.metadata["agent_type"] == "flare"

    @pytest.mark.asyncio
    async def test_records_iterations(self) -> None:
        node = flare_node(mock_llm, mock_retriever, max_iterations=2)
        result = await node(_ctx_with_docs())
        assert len(result.reasoning_traces) >= 1


# ---------------------------------------------------------------------------
# LATS adapter
# ---------------------------------------------------------------------------


class TestLatsNode:
    @pytest.mark.asyncio
    async def test_produces_answer(self) -> None:
        node = lats_node(mock_llm, num_expansions=2)
        result = await node(_ctx_with_docs())
        assert result.answer is not None
        assert result.metadata["agent_type"] == "lats"

    @pytest.mark.asyncio
    async def test_multiple_candidates_evaluated(self) -> None:
        node = lats_node(mock_llm, num_expansions=3)
        result = await node(_ctx_with_docs())
        traces = [t.content for t in result.reasoning_traces]
        assert any("selected" in t.lower() for t in traces)


# ---------------------------------------------------------------------------
# Plan-and-Execute adapter
# ---------------------------------------------------------------------------


class TestPlanExecuteNode:
    @pytest.mark.asyncio
    async def test_produces_answer(self) -> None:
        node = plan_execute_node(mock_llm)
        result = await node(_ctx_with_docs())
        assert result.answer is not None
        assert result.metadata["agent_type"] == "plan_execute"

    @pytest.mark.asyncio
    async def test_plan_steps_metadata(self) -> None:
        node = plan_execute_node(mock_llm, max_steps=3)
        result = await node(_ctx_with_docs())
        assert "plan_steps" in result.metadata


# ---------------------------------------------------------------------------
# Generic agent_to_node adapter
# ---------------------------------------------------------------------------


class TestAgentToNode:
    @pytest.mark.asyncio
    async def test_wraps_callable(self) -> None:
        async def my_agent(query: str, docs: str) -> str:
            return f"Answer to: {query}"

        node = agent_to_node(my_agent, agent_name="custom_agent")
        result = await node(_ctx_with_docs())
        assert result.answer is not None
        assert "What is RLHF?" in result.answer
        assert result.metadata["agent_type"] == "custom_agent"


# ---------------------------------------------------------------------------
# Compose in a StateGraph
# ---------------------------------------------------------------------------


class TestGraphComposition:
    @pytest.mark.asyncio
    async def test_react_in_graph(self) -> None:
        """Prove a react adapter node composes into a full graph."""

        async def retrieve(ctx: RAGContext) -> RAGContext:
            return ctx.add_documents(
                [RetrievedDocument(doc_id="d1", content="RLHF info", score=0.9)]
            )

        graph = StateGraph("react_graph")
        graph.add_node("retrieve", retrieve)
        graph.add_node("reason", react_node(mock_llm))
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "reason")
        graph.add_edge("reason", END)

        compiled = graph.compile()
        result = await compiled.invoke(RAGContext.create(query="What is RLHF?"))
        assert result.terminated_at == END
        assert result.context.answer is not None
        assert result.node_trace == ["retrieve", "reason"]

    @pytest.mark.asyncio
    async def test_multiple_agents_conditional(self) -> None:
        """Multiple agent adapters with conditional routing."""

        async def retrieve(ctx: RAGContext) -> RAGContext:
            return ctx.add_documents(
                [RetrievedDocument(doc_id="d1", content="RLHF info", score=0.9)]
            ).set_metadata("complexity", "simple")

        def route(ctx: RAGContext) -> str:
            return "react" if ctx.metadata.get("complexity") == "simple" else "lats"

        graph = StateGraph("multi_agent")
        graph.add_node("retrieve", retrieve)
        graph.add_node("react", react_node(mock_llm))
        graph.add_node("lats", lats_node(mock_llm, num_expansions=2))
        graph.set_entry_point("retrieve")
        graph.add_conditional_edges("retrieve", route, {"react": "react", "lats": "lats"})
        graph.add_edge("react", END)
        graph.add_edge("lats", END)

        compiled = graph.compile()
        result = await compiled.invoke(RAGContext.create(query="Simple question"))
        assert result.node_trace == ["retrieve", "react"]
