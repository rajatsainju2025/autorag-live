"""Tests for human-in-the-loop hooks."""

import pytest

from autorag_live.agent.hitl import (
    AutoApprovePolicy,
    CallbackApprovalPolicy,
    HumanAction,
    HumanDecision,
    HumanInterrupt,
    InterruptPoint,
    RejectAllPolicy,
    hitl_node,
)
from autorag_live.core.context import RAGContext

# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------


class TestAutoApprovePolicy:
    def test_always_approves(self):
        policy = AutoApprovePolicy()
        ctx = RAGContext.create(query="test")
        decision = policy.request_approval("node_x", ctx, InterruptPoint.BEFORE)
        assert decision.action == HumanAction.APPROVE


class TestRejectAllPolicy:
    def test_always_rejects(self):
        policy = RejectAllPolicy()
        ctx = RAGContext.create(query="test")
        decision = policy.request_approval("node_x", ctx, InterruptPoint.BEFORE)
        assert decision.action == HumanAction.REJECT
        assert "dry-run" in decision.reason


class TestCallbackApprovalPolicy:
    def test_delegates_to_callback(self):
        def my_callback(node, ctx, point):
            return HumanDecision(action=HumanAction.EDIT, reason="fixed typo")

        policy = CallbackApprovalPolicy(my_callback)
        ctx = RAGContext.create(query="test")
        decision = policy.request_approval("node_x", ctx, InterruptPoint.AFTER)
        assert decision.action == HumanAction.EDIT
        assert decision.reason == "fixed typo"


# ---------------------------------------------------------------------------
# hitl_node decorator tests
# ---------------------------------------------------------------------------


class TestHitlNodeDecorator:
    @pytest.mark.asyncio
    async def test_auto_approve_before(self):
        @hitl_node(interrupt="before", policy=AutoApprovePolicy())
        async def my_node(ctx: RAGContext) -> RAGContext:
            return ctx.with_answer("result", confidence=0.9)

        ctx = RAGContext.create(query="test")
        result = await my_node(ctx)
        assert result.answer == "result"

    @pytest.mark.asyncio
    async def test_reject_before_skips_execution(self):
        executed = False

        @hitl_node(interrupt="before", policy=RejectAllPolicy())
        async def my_node(ctx: RAGContext) -> RAGContext:
            nonlocal executed
            executed = True
            return ctx.with_answer("should not reach", confidence=1.0)

        ctx = RAGContext.create(query="test")
        result = await my_node(ctx)

        assert not executed
        assert result.answer is None
        assert result.metadata.get("hitl_rejected_my_node") is True

    @pytest.mark.asyncio
    async def test_abort_marks_error(self):
        def abort_callback(node, ctx, point):
            return HumanDecision(action=HumanAction.ABORT, reason="too risky")

        @hitl_node(interrupt="before", policy=CallbackApprovalPolicy(abort_callback))
        async def risky_node(ctx: RAGContext) -> RAGContext:
            return ctx.with_answer("dangerous", confidence=1.0)

        ctx = RAGContext.create(query="test")
        result = await risky_node(ctx)

        assert result.has_error
        assert "Aborted" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_edit_replaces_context(self):
        edited_ctx = RAGContext.create(query="edited query")

        def edit_callback(node, ctx, point):
            return HumanDecision(
                action=HumanAction.EDIT,
                modified_context=edited_ctx.with_answer("human-edited answer"),
            )

        @hitl_node(interrupt="after", policy=CallbackApprovalPolicy(edit_callback))
        async def my_node(ctx: RAGContext) -> RAGContext:
            return ctx.with_answer("original answer", confidence=0.5)

        ctx = RAGContext.create(query="test")
        result = await my_node(ctx)

        assert result.answer == "human-edited answer"

    @pytest.mark.asyncio
    async def test_both_interrupts(self):
        calls = []

        def tracking_callback(node, ctx, point):
            calls.append(point.value)
            return HumanDecision(action=HumanAction.APPROVE)

        @hitl_node(interrupt="both", policy=CallbackApprovalPolicy(tracking_callback))
        async def my_node(ctx: RAGContext) -> RAGContext:
            return ctx.with_answer("ok")

        ctx = RAGContext.create(query="test")
        await my_node(ctx)

        assert calls == ["before", "after"]

    def test_hitl_config_metadata(self):
        @hitl_node(interrupt="before")
        async def my_node(ctx: RAGContext) -> RAGContext:
            return ctx

        assert hasattr(my_node, "_hitl_config")
        assert my_node._hitl_config["interrupt"] == InterruptPoint.BEFORE


# ---------------------------------------------------------------------------
# HumanInterrupt exception tests
# ---------------------------------------------------------------------------


class TestHumanInterrupt:
    def test_exception_attributes(self):
        ctx = RAGContext.create(query="test")
        exc = HumanInterrupt("grade_docs", ctx, InterruptPoint.BEFORE)

        assert exc.node == "grade_docs"
        assert exc.point == InterruptPoint.BEFORE
        assert "grade_docs" in str(exc)

    def test_protocol_compliance(self):
        from autorag_live.agent.hitl import ApprovalPolicy

        assert isinstance(AutoApprovePolicy(), ApprovalPolicy)
        assert isinstance(RejectAllPolicy(), ApprovalPolicy)
