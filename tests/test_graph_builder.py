"""Tests for autorag_live.pipeline.graph_builder."""

from __future__ import annotations

import pytest

from autorag_live.core.context import ContextStage, RAGContext, RetrievedDocument
from autorag_live.core.state_graph import END
from autorag_live.pipeline.graph_builder import PipelineGraphBuilder

# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------


async def mock_retrieve(ctx: RAGContext) -> RAGContext:
    docs = [
        RetrievedDocument(doc_id="d1", content="RLHF info", score=0.95),
        RetrievedDocument(doc_id="d2", content="RL basics", score=0.80),
    ]
    return ctx.advance_stage(ContextStage.RETRIEVAL).add_documents(docs)


async def mock_rerank(ctx: RAGContext) -> RAGContext:
    sorted_docs = sorted(ctx.documents, key=lambda d: d.score, reverse=True)
    return ctx.advance_stage(ContextStage.RERANKING).replace_documents(sorted_docs)


async def mock_generate(ctx: RAGContext) -> RAGContext:
    return ctx.advance_stage(ContextStage.GENERATION).with_answer(
        "RLHF uses human feedback.", confidence=0.9
    )


async def mock_evaluate(ctx: RAGContext) -> RAGContext:
    return ctx.advance_stage(ContextStage.EVALUATION).add_eval_score("faithfulness", 0.95)


async def mock_safety(ctx: RAGContext) -> RAGContext:
    return ctx.advance_stage(ContextStage.SAFETY).set_metadata("safe", True)


async def mock_custom(ctx: RAGContext) -> RAGContext:
    return ctx.set_metadata("custom_ran", True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMinimalPipeline:
    @pytest.mark.asyncio
    async def test_defaults_produce_result(self) -> None:
        """Minimal builder with defaults still compiles and runs."""
        builder = PipelineGraphBuilder()
        compiled = builder.build()
        result = await compiled.invoke(RAGContext.create(query="test"))
        assert result.terminated_at == END
        assert result.context.answer is not None

    @pytest.mark.asyncio
    async def test_custom_retrieve_generate(self) -> None:
        builder = PipelineGraphBuilder()
        builder.set_retriever(mock_retrieve)
        builder.set_generator(mock_generate)
        compiled = builder.build()
        result = await compiled.invoke(RAGContext.create(query="What is RLHF?"))
        assert result.context.answer == "RLHF uses human feedback."
        assert result.context.document_count == 2
        assert result.node_trace == ["retrieve", "generate"]


class TestWithReranking:
    @pytest.mark.asyncio
    async def test_rerank_stage_inserted(self) -> None:
        builder = PipelineGraphBuilder()
        builder.set_retriever(mock_retrieve)
        builder.enable_reranking(mock_rerank)
        builder.set_generator(mock_generate)
        compiled = builder.build()
        result = await compiled.invoke(RAGContext.create(query="test"))
        assert result.node_trace == ["retrieve", "rerank", "generate"]


class TestWithEvaluation:
    @pytest.mark.asyncio
    async def test_eval_stage_runs(self) -> None:
        builder = PipelineGraphBuilder()
        builder.set_retriever(mock_retrieve)
        builder.set_generator(mock_generate)
        builder.enable_evaluation(mock_evaluate)
        compiled = builder.build()
        result = await compiled.invoke(RAGContext.create(query="test"))
        assert result.node_trace == ["retrieve", "generate", "evaluate"]
        assert len(result.context.eval_scores) == 1


class TestWithSafety:
    @pytest.mark.asyncio
    async def test_safety_stage_runs(self) -> None:
        builder = PipelineGraphBuilder()
        builder.set_retriever(mock_retrieve)
        builder.set_generator(mock_generate)
        builder.enable_safety(mock_safety)
        compiled = builder.build()
        result = await compiled.invoke(RAGContext.create(query="test"))
        assert result.node_trace == ["retrieve", "generate", "safety"]
        assert result.context.metadata.get("safe") is True


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_all_stages(self) -> None:
        builder = PipelineGraphBuilder(name="full")
        builder.set_retriever(mock_retrieve)
        builder.enable_reranking(mock_rerank)
        builder.set_generator(mock_generate)
        builder.enable_evaluation(mock_evaluate)
        builder.enable_safety(mock_safety)
        compiled = builder.build()
        result = await compiled.invoke(RAGContext.create(query="test"))
        assert result.node_trace == ["retrieve", "rerank", "generate", "evaluate", "safety"]
        assert result.terminated_at == END

    @pytest.mark.asyncio
    async def test_custom_node_injection(self) -> None:
        builder = PipelineGraphBuilder()
        builder.set_retriever(mock_retrieve)
        builder.set_generator(mock_generate)
        builder.add_custom_node("post_process", mock_custom, after="generate")
        compiled = builder.build()
        result = await compiled.invoke(RAGContext.create(query="test"))
        assert "post_process" in result.node_trace
        assert result.context.metadata.get("custom_ran") is True


class TestDescribe:
    def test_describe_minimal(self) -> None:
        builder = PipelineGraphBuilder(name="test_pipe")
        desc = builder.describe()
        assert desc["name"] == "test_pipe"
        assert "retrieve" in desc["stages"]
        assert "generate" in desc["stages"]

    def test_describe_full(self) -> None:
        builder = PipelineGraphBuilder()
        builder.enable_reranking()
        builder.enable_evaluation()
        builder.enable_safety()
        builder.add_custom_node("kgnode", mock_custom)
        desc = builder.describe()
        assert "rerank" in desc["stages"]
        assert "evaluate" in desc["stages"]
        assert "safety" in desc["stages"]
        assert "kgnode" in desc["custom_nodes"]
