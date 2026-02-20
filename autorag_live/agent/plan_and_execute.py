"""
Plan-and-Execute Agentic RAG.

Implements a two-phase agent that cleanly separates *planning* from
*execution*, enabling better long-horizon task handling than the
interleaved ReAct loop.

Phase 1 — PLAN
    The LLM receives the user query and generates an ordered list of
    sub-steps.  Each step specifies: ``step_id``, ``description``,
    ``tool`` (optional), ``depends_on``.

Phase 2 — EXECUTE
    Steps are executed respecting dependency order.  Independent steps
    run concurrently.  Each step's result is fed back to dependent steps
    as context.  The final step synthesises all results into an answer.

Phase 3 — REPLAN (optional)
    If execution fails or the plan is clearly incomplete, the LLM can
    be asked to replan based on what has been learned.

References
----------
- "Plan-and-Solve Prompting: Improving Zero-Shot CoT with Planning"
  Wang et al., 2023 (https://arxiv.org/abs/2305.04091)
- "HuggingGPT: Solving AI Tasks with ChatGPT"
  Shen et al., 2023
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
LLMFn = Callable[[str], Coroutine[Any, Any, str]]
ToolFn = Callable[[str, Dict[str, Any]], Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# Plan data structures
# ---------------------------------------------------------------------------


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in an execution plan."""

    step_id: str
    description: str
    tool: Optional[str] = None
    tool_input: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0.0

    def is_ready(self, completed_ids: set[str]) -> bool:
        """True when all dependencies are satisfied."""
        return all(dep in completed_ids for dep in self.depends_on)


@dataclass
class ExecutionPlan:
    """An ordered list of steps to execute."""

    steps: List[PlanStep] = field(default_factory=list)
    original_query: str = ""

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        return next((s for s in self.steps if s.step_id == step_id), None)

    def completed_ids(self) -> set[str]:
        return {s.step_id for s in self.steps if s.status == StepStatus.DONE}

    def pending_ready(self) -> List[PlanStep]:
        """Return pending steps whose dependencies are all met."""
        done = self.completed_ids()
        return [s for s in self.steps if s.status == StepStatus.PENDING and s.is_ready(done)]

    def is_complete(self) -> bool:
        return all(
            s.status in (StepStatus.DONE, StepStatus.SKIPPED, StepStatus.FAILED) for s in self.steps
        )

    def summary(self) -> str:
        lines = [f"Plan for: {self.original_query[:80]}"]
        for s in self.steps:
            deps = f" (needs: {', '.join(s.depends_on)})" if s.depends_on else ""
            lines.append(f"  [{s.step_id}] {s.status.value:8s} — {s.description[:60]}{deps}")
        return "\n".join(lines)


@dataclass
class PlanAndExecuteResult:
    """Final result from Plan-and-Execute agent."""

    query: str
    answer: str
    plan: ExecutionPlan
    replanned: bool = False
    total_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer[:500],
            "steps_executed": len(self.plan.steps),
            "replanned": self.replanned,
            "total_latency_ms": round(self.total_latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PLAN_PROMPT = (
    "You are an expert planner for a question-answering system.\n\n"
    "Given the following query, create a concise step-by-step plan to answer it.\n"
    "Each step must be a JSON object with these fields:\n"
    '  "step_id": unique string like "s1", "s2"\n'
    '  "description": what to do in this step\n'
    '  "tool": one of ["retrieve", "reason", "synthesise"] or null\n'
    '  "depends_on": list of step_ids this step depends on\n\n'
    "Output a JSON array of steps only. No other text.\n\n"
    "Query: {query}\n\n"
    "Plan (JSON array):"
)

_EXECUTE_STEP_PROMPT = (
    "You are executing step {step_id} of a plan.\n\n"
    "Original query: {query}\n"
    "Step description: {description}\n"
    "Context from previous steps:\n{context}\n\n"
    "Execute this step and provide its result:"
)

_SYNTHESISE_PROMPT = (
    "You have completed a multi-step reasoning plan.\n\n"
    "Original query: {query}\n\n"
    "Step results:\n{step_results}\n\n"
    "Using the step results above, write a final comprehensive answer to the query:"
)

_REPLAN_PROMPT = (
    "The original plan for the following query encountered issues.\n\n"
    "Query: {query}\n"
    "Original plan:\n{original_plan}\n"
    "Completed steps:\n{completed}\n"
    "Failed steps:\n{failed}\n\n"
    "Create a revised plan to complete the query. "
    "Reuse completed results where possible.\n"
    "Output a JSON array of remaining steps only:"
)


# ---------------------------------------------------------------------------
# Plan parser
# ---------------------------------------------------------------------------


def _parse_plan(raw: str, query: str) -> ExecutionPlan:
    """Parse LLM JSON output into an ExecutionPlan."""
    # Extract JSON array from response
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    raw_json = match.group(0) if match else raw

    try:
        items = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("PlanAndExecute: failed to parse plan JSON; using single-step fallback")
        items = [
            {"step_id": "s1", "description": f"Answer: {query}", "tool": "reason", "depends_on": []}
        ]

    steps: List[PlanStep] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        steps.append(
            PlanStep(
                step_id=str(item.get("step_id", f"s{len(steps)+1}")),
                description=str(item.get("description", "")),
                tool=item.get("tool"),
                tool_input=item.get("tool_input", {}),
                depends_on=list(item.get("depends_on", [])),
            )
        )

    return ExecutionPlan(steps=steps, original_query=query)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PlanAndExecuteAgent:
    """
    Plan-and-Execute agentic RAG agent.

    Args:
        llm_fn: Async ``(prompt: str) → str`` callable.
        tools: Dict mapping tool name → async ``(description, inputs) → str`` callable.
        max_replan_attempts: How many times to attempt replanning on failure.
        step_concurrency: Max concurrent step executions.
        step_timeout_s: Per-step timeout in seconds.

    Example::

        agent = PlanAndExecuteAgent(llm_fn=my_llm, tools={"retrieve": retrieve_fn})
        result = await agent.run("What caused the 2008 financial crisis?")
    """

    def __init__(
        self,
        llm_fn: LLMFn,
        tools: Optional[Dict[str, ToolFn]] = None,
        max_replan_attempts: int = 1,
        step_concurrency: int = 4,
        step_timeout_s: float = 30.0,
    ) -> None:
        self.llm_fn = llm_fn
        self.tools = tools or {}
        self.max_replan_attempts = max_replan_attempts
        self._semaphore = asyncio.Semaphore(step_concurrency)
        self.step_timeout_s = step_timeout_s

    async def run(self, query: str) -> PlanAndExecuteResult:
        """
        Run the full Plan → Execute → (Replan) → Synthesise cycle.

        Args:
            query: User question.

        Returns:
            :class:`PlanAndExecuteResult` with the final answer and full plan.
        """
        t0 = time.monotonic()
        plan = await self._plan(query)
        logger.info("PlanAndExecute: generated %d steps", len(plan.steps))

        replanned = False
        for attempt in range(self.max_replan_attempts + 1):
            await self._execute_plan(query, plan)
            failed = [s for s in plan.steps if s.status == StepStatus.FAILED]
            if not failed or attempt == self.max_replan_attempts:
                break
            logger.info("PlanAndExecute: replanning after %d failed steps", len(failed))
            plan = await self._replan(query, plan)
            replanned = True

        answer = await self._synthesise(query, plan)
        total_ms = (time.monotonic() - t0) * 1000

        return PlanAndExecuteResult(
            query=query,
            answer=answer,
            plan=plan,
            replanned=replanned,
            total_latency_ms=round(total_ms, 1),
        )

    async def _plan(self, query: str) -> ExecutionPlan:
        prompt = _PLAN_PROMPT.format(query=query)
        raw = await self.llm_fn(prompt)
        return _parse_plan(raw, query)

    async def _execute_plan(self, query: str, plan: ExecutionPlan) -> None:
        """Execute plan steps respecting dependency order, with concurrency."""
        while not plan.is_complete():
            ready = plan.pending_ready()
            if not ready:
                # Check for deadlock
                still_pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
                if still_pending:
                    logger.warning("PlanAndExecute: deadlock detected — skipping remaining steps")
                    for s in still_pending:
                        s.status = StepStatus.SKIPPED
                break

            tasks = [asyncio.create_task(self._execute_step(query, step, plan)) for step in ready]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_step(self, query: str, step: PlanStep, plan: ExecutionPlan) -> None:
        step.status = StepStatus.RUNNING
        t0 = time.monotonic()

        context = self._build_context(plan)
        try:
            async with asyncio.timeout(self.step_timeout_s):
                async with self._semaphore:
                    if step.tool and step.tool in self.tools:
                        result = await self.tools[step.tool](step.description, step.tool_input)
                    else:
                        prompt = _EXECUTE_STEP_PROMPT.format(
                            step_id=step.step_id,
                            query=query,
                            description=step.description,
                            context=context,
                        )
                        result = await self.llm_fn(prompt)

            step.result = result
            step.status = StepStatus.DONE
        except TimeoutError:
            step.error = "timeout"
            step.status = StepStatus.FAILED
            logger.warning("PlanAndExecute: step %s timed out", step.step_id)
        except Exception as exc:
            step.error = str(exc)
            step.status = StepStatus.FAILED
            logger.warning("PlanAndExecute: step %s failed: %s", step.step_id, exc)
        finally:
            step.latency_ms = (time.monotonic() - t0) * 1000

    async def _replan(self, query: str, old_plan: ExecutionPlan) -> ExecutionPlan:
        completed = "\n".join(
            f"  [{s.step_id}] {s.description[:60]}: {str(s.result)[:100]}"
            for s in old_plan.steps
            if s.status == StepStatus.DONE
        )
        failed = "\n".join(
            f"  [{s.step_id}] {s.description[:60]}: {s.error}"
            for s in old_plan.steps
            if s.status == StepStatus.FAILED
        )
        prompt = _REPLAN_PROMPT.format(
            query=query,
            original_plan=old_plan.summary(),
            completed=completed or "(none)",
            failed=failed or "(none)",
        )
        raw = await self.llm_fn(prompt)
        new_plan = _parse_plan(raw, query)
        # Carry over completed steps from old plan
        for s in old_plan.steps:
            if s.status == StepStatus.DONE:
                new_plan.steps.insert(0, s)
        return new_plan

    async def _synthesise(self, query: str, plan: ExecutionPlan) -> str:
        step_results = "\n".join(
            f"Step {s.step_id} ({s.description[:50]}): {str(s.result)[:300]}"
            for s in plan.steps
            if s.result
        )
        if not step_results:
            return "Unable to generate an answer — all plan steps failed."

        prompt = _SYNTHESISE_PROMPT.format(query=query, step_results=step_results)
        return await self.llm_fn(prompt)

    @staticmethod
    def _build_context(plan: ExecutionPlan) -> str:
        parts = []
        for step in plan.steps:
            if step.status == StepStatus.DONE and step.result:
                parts.append(f"[{step.step_id}] {step.description[:50]}: {step.result[:300]}")
        return "\n".join(parts) or "(no prior results)"
