"""
Pre-built graph patterns for common agentic RAG workflows.

Provides factory functions that return ready-to-run
:class:`~autorag_live.core.state_graph.CompiledGraph` instances for:

1. **Corrective RAG** — retrieve → grade → (generate | rewrite+re-retrieve)
2. **Self-RAG**       — retrieve → assess_relevance → assess_support → generate
3. **Adaptive RAG**   — route (simple→direct | complex→multi-hop)
4. **Plan-Execute**   — plan → execute_steps → synthesize → reflect

Each pattern is parameterised by lightweight callables (``LLMCallable``,
``RetrieverCallable``) so they are vendor-agnostic and testable with mocks.

Example::

    from autorag_live.pipeline.patterns import build_corrective_rag

    graph = build_corrective_rag(llm_fn, retriever_fn)
    result = await graph.invoke(RAGContext.create(query="What is RLHF?"))
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from autorag_live.core.context import ContextStage, RAGContext, RetrievedDocument
from autorag_live.core.state_graph import END, CompiledGraph, StateGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMCallable(Protocol):
    async def __call__(self, prompt: str, **kwargs: Any) -> str:
        ...


@runtime_checkable
class RetrieverCallable(Protocol):
    async def __call__(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        ...


# ---------------------------------------------------------------------------
# Helpers: convert raw retriever dicts → RetrievedDocument
# ---------------------------------------------------------------------------


def _raw_to_docs(raw: List[Dict[str, Any]], source: str = "retrieval") -> List[RetrievedDocument]:
    return [
        RetrievedDocument(
            doc_id=d.get("id", f"{source}-{i}"),
            content=d.get("content", d.get("text", "")),
            score=d.get("score", 0.0),
            source=source,
        )
        for i, d in enumerate(raw)
    ]


# =========================================================================
# 1. Corrective RAG
# =========================================================================


def build_corrective_rag(
    llm_fn: LLMCallable,
    retriever_fn: RetrieverCallable,
    *,
    relevance_threshold: float = 0.5,
    max_rewrites: int = 2,
    top_k: int = 5,
) -> CompiledGraph:
    """Build a Corrective RAG graph.

    Flow::

        retrieve → grade ─┬── (relevant)  → generate → END
                           └── (irrelevant) → rewrite → retrieve  (cycle)

    The ``grade`` node uses the LLM to score document relevance.
    If the average relevance is below *relevance_threshold*, the query
    is rewritten and retrieval is retried (up to *max_rewrites* times).
    """

    async def retrieve(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.RETRIEVAL)
        query = ctx.metadata.get("rewritten_query", ctx.query)
        raw = await retriever_fn(query, top_k=top_k)
        docs = _raw_to_docs(raw, source="corrective_rag")
        return ctx.add_documents(docs).add_reasoning_trace(
            f"Retrieved {len(docs)} documents for: {query[:80]}", stage="retrieve"
        )

    async def grade(ctx: RAGContext) -> RAGContext:
        docs_text = "\n".join(f"- {d.content[:200]}" for d in ctx.documents[-top_k:])
        prompt = (
            f"Rate the relevance of these documents to the query on a scale 0-1.\n\n"
            f"Query: {ctx.query}\n\nDocuments:\n{docs_text}\n\n"
            f"Reply with a single number between 0 and 1."
        )
        raw_score = await llm_fn(prompt)
        try:
            score = float(raw_score.strip().split()[0])
        except (ValueError, IndexError):
            score = 0.5
        ctx = ctx.set_metadata("relevance_score", score)
        rewrites = ctx.metadata.get("rewrite_count", 0)
        ctx = ctx.set_metadata("rewrite_count", rewrites)
        return ctx.add_reasoning_trace(
            f"Graded relevance: {score:.2f} (threshold: {relevance_threshold})", stage="grade"
        )

    def grade_router(ctx: RAGContext) -> str:
        score = ctx.metadata.get("relevance_score", 0.0)
        rewrites = ctx.metadata.get("rewrite_count", 0)
        if score >= relevance_threshold or rewrites >= max_rewrites:
            return "generate"
        return "rewrite"

    async def rewrite(ctx: RAGContext) -> RAGContext:
        prompt = (
            f"The following query did not retrieve relevant documents.\n"
            f"Original query: {ctx.query}\n\n"
            f"Rewrite the query to improve retrieval. Reply with the rewritten query only."
        )
        new_query = await llm_fn(prompt)
        count = ctx.metadata.get("rewrite_count", 0) + 1
        ctx = ctx.set_metadata("rewritten_query", new_query.strip())
        ctx = ctx.set_metadata("rewrite_count", count)
        return ctx.add_reasoning_trace(
            f"Rewrote query (attempt {count}): {new_query.strip()[:80]}", stage="rewrite"
        )

    async def generate(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.GENERATION)
        docs_text = "\n".join(d.content[:300] for d in ctx.documents)
        prompt = (
            f"Context:\n{docs_text}\n\nQuestion: {ctx.query}\n\n"
            f"Provide a comprehensive answer based on the context."
        )
        answer = await llm_fn(prompt)
        return ctx.with_answer(answer.strip(), confidence=0.85).add_reasoning_trace(
            "Generated final answer", stage="generate"
        )

    graph = StateGraph("corrective_rag")
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", grade_router, {"generate": "generate", "rewrite": "rewrite"}
    )
    graph.add_edge("rewrite", "retrieve")  # cycle
    graph.add_edge("generate", END)
    return graph.compile()


# =========================================================================
# 2. Self-RAG graph
# =========================================================================


def build_self_rag(
    llm_fn: LLMCallable,
    retriever_fn: RetrieverCallable,
    *,
    relevance_threshold: float = 0.6,
    support_threshold: float = 0.5,
    top_k: int = 5,
) -> CompiledGraph:
    """Build a Self-RAG graph with reflection tokens.

    Flow::

        retrieve → assess_relevance ─┬── (relevant)  → generate → assess_support → END
                                       └── (irrelevant) → re_retrieve → generate → ...
    """

    async def retrieve(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.RETRIEVAL)
        raw = await retriever_fn(ctx.query, top_k=top_k)
        return ctx.add_documents(_raw_to_docs(raw, "self_rag"))

    async def assess_relevance(ctx: RAGContext) -> RAGContext:
        docs_text = "\n".join(d.content[:200] for d in ctx.documents)
        prompt = (
            f"Assess document relevance. Query: {ctx.query}\n"
            f"Documents:\n{docs_text}\n\nRate 0-1:"
        )
        raw = await llm_fn(prompt)
        try:
            score = float(raw.strip().split()[0])
        except (ValueError, IndexError):
            score = 0.5
        return ctx.set_metadata("relevance", score).add_reasoning_trace(
            f"[ISREL] relevance={score:.2f}", stage="assess_relevance"
        )

    def relevance_router(ctx: RAGContext) -> str:
        return (
            "generate" if ctx.metadata.get("relevance", 0) >= relevance_threshold else "re_retrieve"
        )

    async def re_retrieve(ctx: RAGContext) -> RAGContext:
        raw = await retriever_fn(f"more detail: {ctx.query}", top_k=top_k)
        return ctx.replace_documents(_raw_to_docs(raw, "self_rag_retry")).add_reasoning_trace(
            "Re-retrieved after low relevance", stage="re_retrieve"
        )

    async def generate(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.GENERATION)
        docs_text = "\n".join(d.content[:300] for d in ctx.documents)
        answer = await llm_fn(f"Context:\n{docs_text}\n\nQ: {ctx.query}\nA:")
        return ctx.with_answer(answer.strip(), confidence=0.8)

    async def assess_support(ctx: RAGContext) -> RAGContext:
        prompt = (
            f"Does the answer '{ctx.answer[:200]}' have support from these docs?\n"
            + "\n".join(d.content[:200] for d in ctx.documents)
            + "\nRate support 0-1:"
        )
        raw = await llm_fn(prompt)
        try:
            score = float(raw.strip().split()[0])
        except (ValueError, IndexError):
            score = 0.5
        return ctx.set_metadata("support", score).add_reasoning_trace(
            f"[ISSUP] support={score:.2f}", stage="assess_support"
        )

    graph = StateGraph("self_rag")
    graph.add_node("retrieve", retrieve)
    graph.add_node("assess_relevance", assess_relevance)
    graph.add_node("re_retrieve", re_retrieve)
    graph.add_node("generate", generate)
    graph.add_node("assess_support", assess_support)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "assess_relevance")
    graph.add_conditional_edges(
        "assess_relevance",
        relevance_router,
        {"generate": "generate", "re_retrieve": "re_retrieve"},
    )
    graph.add_edge("re_retrieve", "generate")
    graph.add_edge("generate", "assess_support")
    graph.add_edge("assess_support", END)
    return graph.compile()


# =========================================================================
# 3. Adaptive RAG
# =========================================================================


def build_adaptive_rag(
    llm_fn: LLMCallable,
    retriever_fn: RetrieverCallable,
    *,
    top_k: int = 5,
) -> CompiledGraph:
    """Build an Adaptive RAG graph.

    Flow::

        classify → simple_generate  (for easy queries → END)
                 → retrieve → generate (for complex queries → END)
    """

    async def classify(ctx: RAGContext) -> RAGContext:
        prompt = (
            f"Classify this query as 'simple' or 'complex'.\n"
            f"Query: {ctx.query}\nReply with one word: simple or complex."
        )
        label = (await llm_fn(prompt)).strip().lower()
        complexity = "simple" if "simple" in label else "complex"
        return ctx.set_metadata("complexity", complexity).add_reasoning_trace(
            f"Query classified as: {complexity}", stage="classify"
        )

    def complexity_router(ctx: RAGContext) -> str:
        return ctx.metadata.get("complexity", "complex")

    async def simple_generate(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.GENERATION)
        answer = await llm_fn(f"Answer this directly: {ctx.query}")
        return ctx.with_answer(answer.strip(), confidence=0.7)

    async def retrieve(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.RETRIEVAL)
        raw = await retriever_fn(ctx.query, top_k=top_k)
        return ctx.add_documents(_raw_to_docs(raw, "adaptive"))

    async def generate(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.GENERATION)
        docs_text = "\n".join(d.content[:300] for d in ctx.documents)
        answer = await llm_fn(f"Context:\n{docs_text}\n\nQ: {ctx.query}\nA:")
        return ctx.with_answer(answer.strip(), confidence=0.85)

    graph = StateGraph("adaptive_rag")
    graph.add_node("classify", classify)
    graph.add_node("simple_generate", simple_generate)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        complexity_router,
        {"simple": "simple_generate", "complex": "retrieve"},
    )
    graph.add_edge("simple_generate", END)
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# =========================================================================
# 4. Plan-and-Execute graph
# =========================================================================


def build_plan_execute(
    llm_fn: LLMCallable,
    retriever_fn: Optional[RetrieverCallable] = None,
    *,
    max_steps: int = 5,
) -> CompiledGraph:
    """Build a Plan-and-Execute graph.

    Flow::

        plan → execute → synthesize → reflect ─┬── (good)   → END
                                                 └── (revise) → plan  (cycle)
    """

    async def plan(ctx: RAGContext) -> RAGContext:
        prompt = (
            f"Create a concise step-by-step plan to answer: {ctx.query}\n"
            f"List up to {max_steps} steps, one per line."
        )
        plan_text = await llm_fn(prompt)
        steps = [l.strip() for l in plan_text.strip().split("\n") if l.strip()]
        return ctx.set_metadata("plan", steps).add_reasoning_trace(
            f"Plan created: {len(steps)} steps", stage="plan"
        )

    async def execute(ctx: RAGContext) -> RAGContext:
        steps = ctx.metadata.get("plan", [])
        results: List[str] = []
        for step in steps[:max_steps]:
            result = await llm_fn(f"Execute: {step}\nContext: {ctx.query}")
            results.append(result.strip()[:200])
        return ctx.set_metadata("step_results", results).add_reasoning_trace(
            f"Executed {len(results)} steps", stage="execute"
        )

    async def synthesize(ctx: RAGContext) -> RAGContext:
        ctx = ctx.advance_stage(ContextStage.GENERATION)
        results = ctx.metadata.get("step_results", [])
        prompt = (
            f"Question: {ctx.query}\n\n"
            f"Step results:\n"
            + "\n".join(f"- {r}" for r in results)
            + "\n\nSynthesize a final answer."
        )
        answer = await llm_fn(prompt)
        return ctx.with_answer(answer.strip(), confidence=0.8)

    async def reflect(ctx: RAGContext) -> RAGContext:
        prompt = (
            f"Question: {ctx.query}\nAnswer: {(ctx.answer or '')[:300]}\n\n"
            f"Rate answer quality 0-1. Reply with a single number."
        )
        raw = await llm_fn(prompt)
        try:
            score = float(raw.strip().split()[0])
        except (ValueError, IndexError):
            score = 0.7
        return ctx.set_metadata("reflection_score", score).add_reasoning_trace(
            f"Reflection score: {score:.2f}", stage="reflect"
        )

    def reflect_router(ctx: RAGContext) -> str:
        score = ctx.metadata.get("reflection_score", 1.0)
        revisions = ctx.metadata.get("revision_count", 0)
        if score >= 0.7 or revisions >= 2:
            return "done"
        return "revise"

    graph = StateGraph("plan_execute")
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("synthesize", synthesize)
    graph.add_node("reflect", reflect)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "synthesize")
    graph.add_edge("synthesize", "reflect")
    graph.add_conditional_edges("reflect", reflect_router, {"done": END, "revise": "plan"})
    return graph.compile()
