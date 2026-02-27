"""
Human-in-the-loop (HITL) hooks for agentic RAG workflows.

Provides ``interrupt_before`` / ``interrupt_after`` node decorators and an
``ApprovalPolicy`` protocol so human operators can review, approve, modify,
or reject agent actions before they take effect.

This is a critical missing piece vs LangGraph and CrewAI. Modern agentic
systems need explicit pause points where humans can:
    1. Approve/reject a tool invocation (e.g. web search, code execution)
    2. Edit the agent's planned answer before delivery
    3. Inject corrections into the reasoning trace
    4. Abort a runaway agent loop

Architecture:
    - ``HumanInterrupt`` exception — signals a pause to the graph engine
    - ``InterruptConfig`` — per-node interrupt settings
    - ``HumanDecision`` — the human's response (approve/reject/edit/abort)
    - ``ApprovalPolicy`` protocol — pluggable policy backends
    - ``AutoApprovePolicy`` — always approve (for CI / non-interactive)
    - ``CallbackApprovalPolicy`` — delegates to a callback function
    - ``hitl_node()`` — decorator that wraps a graph node with HITL logic

Example:
    >>> from autorag_live.agent.hitl import hitl_node, AutoApprovePolicy
    >>>
    >>> @hitl_node(interrupt="before", policy=AutoApprovePolicy())
    ... async def execute_code(ctx: RAGContext) -> RAGContext:
    ...     return ctx.set_metadata("executed", True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, runtime_checkable

from autorag_live.core.context import RAGContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class InterruptPoint(str, Enum):
    """When to interrupt relative to node execution."""

    BEFORE = "before"
    AFTER = "after"
    BOTH = "both"


class HumanAction(str, Enum):
    """Actions a human can take at an interrupt point."""

    APPROVE = "approve"  # Continue as planned
    REJECT = "reject"  # Skip this node, continue pipeline
    EDIT = "edit"  # Modify context, then continue
    ABORT = "abort"  # Stop the entire pipeline


@dataclass(frozen=True)
class HumanDecision:
    """
    The human operator's decision at an interrupt point.

    Attributes:
        action:          What the human decided.
        modified_context: If action is EDIT, the updated RAGContext.
        reason:          Optional explanation for the decision.
        metadata:        Arbitrary extra data.
    """

    action: HumanAction
    modified_context: Optional[RAGContext] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class HumanInterrupt(Exception):
    """
    Raised when execution should pause for human review.

    The graph engine catches this and persists a checkpoint so the
    workflow can be resumed after the human responds.
    """

    def __init__(self, node: str, context: RAGContext, point: InterruptPoint):
        self.node = node
        self.context = context
        self.point = point
        super().__init__(f"HITL interrupt at node '{node}' ({point.value})")


# ---------------------------------------------------------------------------
# ApprovalPolicy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ApprovalPolicy(Protocol):
    """Protocol for human approval backends."""

    def request_approval(
        self,
        node: str,
        context: RAGContext,
        point: InterruptPoint,
    ) -> HumanDecision:
        """
        Request human approval.

        Implementations may block (CLI prompt), poll (web UI), or auto-approve.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------


class AutoApprovePolicy:
    """Always approve — for CI, testing, and non-interactive pipelines."""

    def request_approval(
        self,
        node: str,
        context: RAGContext,
        point: InterruptPoint,
    ) -> HumanDecision:
        logger.debug("Auto-approving node '%s' at %s", node, point.value)
        return HumanDecision(action=HumanAction.APPROVE)


class RejectAllPolicy:
    """Always reject — useful for dry-run / preview modes."""

    def request_approval(
        self,
        node: str,
        context: RAGContext,
        point: InterruptPoint,
    ) -> HumanDecision:
        return HumanDecision(action=HumanAction.REJECT, reason="dry-run mode")


class CallbackApprovalPolicy:
    """
    Delegates to a user-supplied callback for maximum flexibility.

    The callback receives ``(node, context, point)`` and must return a
    ``HumanDecision``.
    """

    def __init__(
        self,
        callback: Callable[[str, RAGContext, InterruptPoint], HumanDecision],
    ):
        self._callback = callback

    def request_approval(
        self,
        node: str,
        context: RAGContext,
        point: InterruptPoint,
    ) -> HumanDecision:
        return self._callback(node, context, point)


# ---------------------------------------------------------------------------
# hitl_node decorator
# ---------------------------------------------------------------------------


def hitl_node(
    interrupt: str = "before",
    policy: Optional[ApprovalPolicy] = None,
) -> Callable:
    """
    Decorator that wraps a graph node function with HITL interrupt logic.

    Args:
        interrupt: When to interrupt — "before", "after", or "both".
        policy:    The approval policy to use. Defaults to AutoApprovePolicy.

    Returns:
        Decorated async function that checks approval before/after execution.
    """
    point = InterruptPoint(interrupt)
    _policy = policy or AutoApprovePolicy()

    def decorator(
        fn: Callable[[RAGContext], Awaitable[RAGContext]],
    ) -> Callable[[RAGContext], Awaitable[RAGContext]]:
        @wraps(fn)
        async def wrapper(ctx: RAGContext) -> RAGContext:
            node_name = fn.__name__

            # --- BEFORE interrupt ---
            if point in (InterruptPoint.BEFORE, InterruptPoint.BOTH):
                decision = _policy.request_approval(node_name, ctx, InterruptPoint.BEFORE)
                ctx = _apply_decision(decision, ctx, node_name, "before")
                if decision.action in (HumanAction.REJECT, HumanAction.ABORT):
                    return ctx

            # --- Execute the node ---
            ctx = await fn(ctx)

            # --- AFTER interrupt ---
            if point in (InterruptPoint.AFTER, InterruptPoint.BOTH):
                decision = _policy.request_approval(node_name, ctx, InterruptPoint.AFTER)
                ctx = _apply_decision(decision, ctx, node_name, "after")

            return ctx

        # Store metadata so the graph engine can introspect
        wrapper._hitl_config = {  # type: ignore[attr-defined]
            "interrupt": point,
            "policy": _policy,
        }
        return wrapper

    return decorator


def _apply_decision(
    decision: HumanDecision,
    ctx: RAGContext,
    node: str,
    phase: str,
) -> RAGContext:
    """Apply a human decision to the context."""
    if decision.action == HumanAction.APPROVE:
        logger.debug("Human approved node '%s' (%s)", node, phase)
        return ctx

    if decision.action == HumanAction.EDIT and decision.modified_context is not None:
        logger.info("Human edited context at node '%s' (%s)", node, phase)
        return decision.modified_context

    if decision.action == HumanAction.REJECT:
        logger.info("Human rejected node '%s' (%s): %s", node, phase, decision.reason)
        return ctx.set_metadata(f"hitl_rejected_{node}", True).add_reasoning_trace(
            f"Human rejected node '{node}': {decision.reason}",
            stage="hitl",
        )

    if decision.action == HumanAction.ABORT:
        logger.warning("Human aborted pipeline at node '%s'", node)
        return ctx.mark_error(f"Aborted by human at node '{node}': {decision.reason}")

    return ctx
