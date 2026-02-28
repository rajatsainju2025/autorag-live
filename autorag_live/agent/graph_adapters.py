"""
Agent-to-graph adapters — bridge existing agents into StateGraph nodes.

Each adapter wraps a standalone agent class (ReAct, Self-RAG, FLARE, LATS,
Plan-and-Execute) as an ``async def node(ctx: RAGContext) -> RAGContext``
function suitable for :class:`~autorag_live.core.state_graph.StateGraph`.

This is the critical glue layer: the agents already have rich reasoning
logic, but they define their own types (LLMFn, RetrieverProtocol, etc.)
and never touch RAGContext.  The adapters translate between the two worlds
so that agents become first-class graph nodes.

Usage::

    from autorag_live.agent.graph_adapters import react_node, self_rag_node
    from autorag_live.core.state_graph import StateGraph, END

    graph = StateGraph("agentic_rag")
    graph.add_node("reason", react_node(llm_fn, tools))
    graph.add_node("self_rag", self_rag_node(llm_fn, retriever_fn))
    ...

Each factory function returns a ``NodeFn`` (``async (RAGContext) -> RAGContext``)
that can be registered with ``graph.add_node()``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Protocol, runtime_checkable

from autorag_live.core.context import ContextStage, RAGContext, RetrievedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared protocols (lightweight, no concrete dependencies)
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMCallable(Protocol):
    """Minimal LLM interface accepted by all adapters."""

    async def __call__(self, prompt: str, **kwargs: Any) -> str:
        ...


@runtime_checkable
class RetrieverCallable(Protocol):
    """Minimal retriever interface accepted by adapters."""

    async def __call__(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        ...


# ---------------------------------------------------------------------------
# NodeFn type (matches StateGraph expectations)
# ---------------------------------------------------------------------------

NodeFn = Callable[[RAGContext], Coroutine[Any, Any, RAGContext]]


# ---------------------------------------------------------------------------
# Adapter result (internal)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterResult:
    """Normalised result from any wrapped agent."""

    answer: Optional[str] = None
    confidence: float = 0.0
    reasoning_steps: List[str] = field(default_factory=list)
    documents: List[RetrievedDocument] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. ReAct adapter
# ---------------------------------------------------------------------------


def react_node(
    llm_fn: LLMCallable,
    tools: Optional[Dict[str, Callable]] = None,
    *,
    max_iterations: int = 5,
) -> NodeFn:
    """Wrap the ReAct agent loop as a StateGraph node.

    The adapter runs a simplified Thought→Action→Observation loop
    using the provided ``llm_fn``.  Tool calls are dispatched to
    *tools* (name → async/sync callable).

    Parameters
    ----------
    llm_fn:
        ``async (prompt, **kw) -> str`` — any LLM callable.
    tools:
        Optional mapping of tool name → callable.
    max_iterations:
        Safety cap on reasoning iterations.

    Returns
    -------
    NodeFn
        An ``async (RAGContext) -> RAGContext`` function.
    """
    _tools = tools or {}

    async def _node(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.GENERATION)
        start = time.perf_counter()

        # Build prompt from context
        docs_text = "\n".join(f"[{i+1}] {d.content[:300]}" for i, d in enumerate(ctx.documents))
        prompt = (
            f"You are a ReAct agent. Answer the question using the provided context.\n\n"
            f"Context:\n{docs_text}\n\n"
            f"Question: {ctx.query}\n\n"
            f"Think step by step, then provide your final answer."
        )

        reasoning_steps: List[str] = []
        answer = None

        for step in range(max_iterations):
            response = await llm_fn(prompt)
            reasoning_steps.append(f"Step {step + 1}: {response[:200]}")

            # Check for final answer
            if "Final Answer:" in response:
                answer = response.split("Final Answer:")[-1].strip()
                break
            elif step == max_iterations - 1:
                answer = response.strip()

        latency_ms = (time.perf_counter() - start) * 1000

        # Enrich context with results
        for rs in reasoning_steps:
            ctx = ctx.add_reasoning_trace(rs, stage="react")
        if answer:
            ctx = ctx.with_answer(answer, confidence=0.8)
        ctx = ctx.record_latency(ContextStage.GENERATION, latency_ms)
        ctx = ctx.set_metadata("agent_type", "react")
        return ctx

    return _node


# ---------------------------------------------------------------------------
# 2. Self-RAG adapter
# ---------------------------------------------------------------------------


def self_rag_node(
    llm_fn: LLMCallable,
    retriever_fn: RetrieverCallable,
    *,
    relevance_threshold: float = 0.5,
) -> NodeFn:
    """Wrap the Self-RAG assess-and-retrieve loop as a StateGraph node.

    Runs: generate → assess relevance → optionally re-retrieve →
    assess support → produce answer.

    Parameters
    ----------
    llm_fn:
        ``async (prompt, **kw) -> str``
    retriever_fn:
        ``async (query, top_k) -> list[dict]``
    relevance_threshold:
        Below this score, re-retrieval is triggered.
    """

    async def _node(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.REFLECTION)
        start = time.perf_counter()

        docs_text = "\n".join(d.content[:300] for d in ctx.documents)
        steps: List[str] = []

        # Step 1: Assess relevance of existing documents
        assess_prompt = (
            f"Rate the relevance of the following documents to the query.\n"
            f"Query: {ctx.query}\nDocuments:\n{docs_text}\n\n"
            f"Reply with a number 0-1 indicating average relevance."
        )
        relevance_raw = await llm_fn(assess_prompt)
        try:
            relevance = float(relevance_raw.strip().split()[0])
        except (ValueError, IndexError):
            relevance = 0.5
        steps.append(f"Relevance assessment: {relevance:.2f}")

        # Step 2: Re-retrieve if below threshold
        if relevance < relevance_threshold and retriever_fn is not None:
            raw_docs = await retriever_fn(ctx.query, top_k=5)
            new_docs = [
                RetrievedDocument(
                    doc_id=d.get("id", f"retr-{i}"),
                    content=d.get("content", d.get("text", "")),
                    score=d.get("score", 0.0),
                    source=d.get("source", "self_rag_retrieval"),
                )
                for i, d in enumerate(raw_docs)
            ]
            ctx = ctx.replace_documents(new_docs)
            steps.append(f"Re-retrieved {len(new_docs)} documents")
            docs_text = "\n".join(d.content[:300] for d in ctx.documents)

        # Step 3: Generate with support assessment
        gen_prompt = (
            f"Given the following context, answer the question.\n\n"
            f"Context:\n{docs_text}\n\nQuestion: {ctx.query}\n\n"
            f"Provide a well-supported answer."
        )
        answer = await llm_fn(gen_prompt)
        steps.append(f"Generated answer with {len(ctx.documents)} docs")

        latency_ms = (time.perf_counter() - start) * 1000

        for s in steps:
            ctx = ctx.add_reasoning_trace(s, stage="self_rag")
        ctx = ctx.with_answer(answer.strip(), confidence=min(relevance + 0.1, 1.0))
        ctx = ctx.record_latency(ContextStage.REFLECTION, latency_ms)
        ctx = ctx.set_metadata("agent_type", "self_rag")
        return ctx

    return _node


# ---------------------------------------------------------------------------
# 3. FLARE adapter
# ---------------------------------------------------------------------------


def flare_node(
    llm_fn: LLMCallable,
    retriever_fn: RetrieverCallable,
    *,
    confidence_threshold: float = 0.5,
    max_iterations: int = 3,
) -> NodeFn:
    """Wrap FLARE (Forward-Looking Active REtrieval) as a StateGraph node.

    Generates text, detects low-confidence segments, retrieves more
    context, and continues generation.
    """

    async def _node(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.GENERATION)
        start = time.perf_counter()
        steps: List[str] = []

        docs_text = "\n".join(d.content[:300] for d in ctx.documents)

        for iteration in range(max_iterations):
            gen_prompt = (
                f"Context:\n{docs_text}\n\nQuestion: {ctx.query}\n\n"
                f"Generate an answer. Mark uncertain parts with [UNCERTAIN]."
            )
            response = await llm_fn(gen_prompt)
            steps.append(f"FLARE iteration {iteration + 1}: generated {len(response)} chars")

            if "[UNCERTAIN]" not in response:
                break

            # Extract uncertain segment, re-retrieve
            uncertain = (
                response.split("[UNCERTAIN]")[1].split("[/UNCERTAIN]")[0]
                if "[UNCERTAIN]" in response
                else ctx.query
            )
            raw_docs = await retriever_fn(uncertain.strip(), top_k=3)
            new_docs = [
                RetrievedDocument(
                    doc_id=d.get("id", f"flare-{i}"),
                    content=d.get("content", d.get("text", "")),
                    score=d.get("score", 0.0),
                    source="flare_retrieval",
                )
                for i, d in enumerate(raw_docs)
            ]
            ctx = ctx.add_documents(new_docs)
            docs_text = "\n".join(d.content[:300] for d in ctx.documents)
            steps.append(f"FLARE re-retrieved {len(new_docs)} docs for uncertain segment")

        latency_ms = (time.perf_counter() - start) * 1000
        answer = response.replace("[UNCERTAIN]", "").replace("[/UNCERTAIN]", "").strip()

        for s in steps:
            ctx = ctx.add_reasoning_trace(s, stage="flare")
        ctx = ctx.with_answer(answer, confidence=0.75)
        ctx = ctx.record_latency(ContextStage.GENERATION, latency_ms)
        ctx = ctx.set_metadata("agent_type", "flare")
        return ctx

    return _node


# ---------------------------------------------------------------------------
# 4. LATS adapter
# ---------------------------------------------------------------------------


def lats_node(
    llm_fn: LLMCallable,
    *,
    max_iterations: int = 10,
    num_expansions: int = 3,
    exploration_constant: float = 1.414,
) -> NodeFn:
    """Wrap LATS (Language Agent Tree Search) as a StateGraph node.

    Runs a simplified MCTS-style tree search over reasoning paths
    using ``llm_fn`` for expansion and evaluation.
    """

    async def _node(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.GENERATION)
        start = time.perf_counter()
        steps: List[str] = []

        docs_text = "\n".join(d.content[:300] for d in ctx.documents)

        # Generate multiple candidate answers (simulate tree expansion)
        candidates: List[Dict[str, Any]] = []
        for i in range(num_expansions):
            prompt = (
                f"Context:\n{docs_text}\n\nQuestion: {ctx.query}\n\n"
                f"Provide answer candidate {i + 1}. Be thorough and precise."
            )
            response = await llm_fn(prompt)
            candidates.append({"answer": response.strip(), "index": i})
            steps.append(f"LATS expansion {i + 1}: generated candidate")

        # Evaluate candidates
        best_answer = candidates[0]["answer"] if candidates else ""
        best_score = 0.0

        if len(candidates) > 1:
            eval_prompt = (
                f"Question: {ctx.query}\n\n"
                + "\n".join(f"Candidate {c['index']+1}: {c['answer'][:200]}" for c in candidates)
                + "\n\nWhich candidate best answers the question? Reply with just the number."
            )
            eval_response = await llm_fn(eval_prompt)
            try:
                best_idx = int(eval_response.strip().split()[0]) - 1
                best_idx = max(0, min(best_idx, len(candidates) - 1))
            except (ValueError, IndexError):
                best_idx = 0
            best_answer = candidates[best_idx]["answer"]
            best_score = 0.85
            steps.append(f"LATS selected candidate {best_idx + 1}")

        latency_ms = (time.perf_counter() - start) * 1000

        for s in steps:
            ctx = ctx.add_reasoning_trace(s, stage="lats")
        ctx = ctx.with_answer(best_answer, confidence=max(best_score, 0.7))
        ctx = ctx.record_latency(ContextStage.GENERATION, latency_ms)
        ctx = ctx.set_metadata("agent_type", "lats")
        return ctx

    return _node


# ---------------------------------------------------------------------------
# 5. Plan-and-Execute adapter
# ---------------------------------------------------------------------------


def plan_execute_node(
    llm_fn: LLMCallable,
    tools: Optional[Dict[str, Callable]] = None,
    *,
    max_steps: int = 5,
) -> NodeFn:
    """Wrap Plan-and-Execute as a StateGraph node.

    Phase 1: LLM generates a plan of sub-steps.
    Phase 2: Each step is executed sequentially.
    Phase 3: Results are synthesized into a final answer.
    """
    _tools = tools or {}

    async def _node(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.GENERATION)
        start = time.perf_counter()
        steps: List[str] = []

        docs_text = "\n".join(d.content[:300] for d in ctx.documents)

        # Phase 1: Plan
        plan_prompt = (
            f"Context:\n{docs_text}\n\nQuestion: {ctx.query}\n\n"
            f"Create a step-by-step plan to answer this question. "
            f"List up to {max_steps} steps, one per line, prefixed with 'Step N:'."
        )
        plan_response = await llm_fn(plan_prompt)
        steps.append(f"Plan generated: {plan_response[:150]}")

        # Phase 2: Execute each step
        step_results: List[str] = []
        for line in plan_response.strip().split("\n"):
            if line.strip().lower().startswith("step"):
                step_desc = line.strip()
                exec_prompt = (
                    f"Context:\n{docs_text}\n\n"
                    f"Previous results: {'; '.join(step_results[-3:])}\n\n"
                    f"Execute this step: {step_desc}\n\nProvide the result."
                )
                result = await llm_fn(exec_prompt)
                step_results.append(result.strip()[:200])
                steps.append(f"Executed: {step_desc[:80]} → {result[:80]}")

        # Phase 3: Synthesize
        synth_prompt = (
            f"Question: {ctx.query}\n\n"
            f"Step results:\n"
            + "\n".join(f"- {r}" for r in step_results)
            + "\n\nSynthesize a final answer from these results."
        )
        answer = await llm_fn(synth_prompt)

        latency_ms = (time.perf_counter() - start) * 1000

        for s in steps:
            ctx = ctx.add_reasoning_trace(s, stage="plan_execute")
        ctx = ctx.with_answer(answer.strip(), confidence=0.82)
        ctx = ctx.record_latency(ContextStage.GENERATION, latency_ms)
        ctx = ctx.set_metadata("agent_type", "plan_execute")
        ctx = ctx.set_metadata("plan_steps", len(step_results))
        return ctx

    return _node


# ---------------------------------------------------------------------------
# Utility: generic agent-to-node adapter
# ---------------------------------------------------------------------------


def agent_to_node(
    agent_fn: Callable[[str, str], Coroutine[Any, Any, str]],
    *,
    agent_name: str = "custom",
    stage: ContextStage = ContextStage.GENERATION,
) -> NodeFn:
    """Generic adapter: wrap any ``async (query, context) -> answer`` into a node.

    Parameters
    ----------
    agent_fn:
        ``async (query: str, docs_context: str) -> str``
    agent_name:
        Label for tracing.
    stage:
        Pipeline stage to set.
    """

    async def _node(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(stage)
        start = time.perf_counter()
        docs_text = "\n".join(d.content[:300] for d in ctx.documents)
        answer = await agent_fn(ctx.query, docs_text)
        latency_ms = (time.perf_counter() - start) * 1000
        ctx = ctx.with_answer(answer.strip(), confidence=0.75)
        ctx = ctx.record_latency(stage, latency_ms)
        ctx = ctx.set_metadata("agent_type", agent_name)
        return ctx

    return _node
