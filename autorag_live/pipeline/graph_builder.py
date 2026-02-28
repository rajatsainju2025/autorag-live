"""
Graph-based pipeline builder — composable RAG pipeline as a StateGraph.

Replaces the linear ``PipelineOrchestrator`` with a declarative builder
that composes retrieval, reranking, generation, evaluation, and safety
nodes into a :class:`~autorag_live.core.state_graph.CompiledGraph`.

The builder provides sensible defaults while allowing full customisation
of node functions and edge routing.

Usage::

    from autorag_live.pipeline.graph_builder import PipelineGraphBuilder

    builder = PipelineGraphBuilder()
    builder.set_retriever(my_retrieve_fn)
    builder.set_generator(my_generate_fn)
    compiled = builder.build()
    result = await compiled.invoke(RAGContext.create(query="What is RLHF?"))

Each ``set_*`` method accepts an ``async (RAGContext) -> RAGContext`` function,
or one of the built-in node factories from ``agent.graph_adapters``.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from autorag_live.core.context import ContextStage, RAGContext
from autorag_live.core.state_graph import END, CompiledGraph, StateGraph

logger = logging.getLogger(__name__)

# Type alias matching StateGraph expectations
NodeFn = Callable[[RAGContext], Awaitable[RAGContext]]
RouterFn = Callable[[RAGContext], Union[str, Awaitable[str]]]


# ---------------------------------------------------------------------------
# Default node implementations
# ---------------------------------------------------------------------------


async def _default_retrieve(ctx: RAGContext) -> RAGContext:
    """Placeholder retrieve node — returns context unchanged."""
    return ctx.advance_stage(ContextStage.RETRIEVAL)


async def _default_generate(ctx: RAGContext) -> RAGContext:
    """Placeholder generate node — echoes the query."""
    ctx = ctx.advance_stage(ContextStage.GENERATION)
    return ctx.with_answer(
        f"[No generator configured] Query was: {ctx.query}",
        confidence=0.0,
    )


async def _default_rerank(ctx: RAGContext) -> RAGContext:
    """Passthrough rerank — sorts documents by score descending."""
    ctx = ctx.advance_stage(ContextStage.RERANKING)
    sorted_docs = sorted(ctx.documents, key=lambda d: d.score, reverse=True)
    return ctx.replace_documents(sorted_docs)


async def _default_evaluate(ctx: RAGContext) -> RAGContext:
    """Passthrough evaluate — no-op."""
    return ctx.advance_stage(ContextStage.EVALUATION)


async def _default_safety(ctx: RAGContext) -> RAGContext:
    """Passthrough safety — no-op."""
    return ctx.advance_stage(ContextStage.SAFETY)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class PipelineGraphBuilder:
    """
    Declarative builder for graph-based RAG pipelines.

    Composes retrieval → rerank → generate → evaluate → safety as a
    :class:`CompiledGraph` with optional conditional routing and custom
    node injection.

    Example::

        builder = PipelineGraphBuilder(name="my_rag")
        builder.set_retriever(my_retrieve_fn)
        builder.set_generator(react_node(llm_fn))
        builder.enable_reranking(my_rerank_fn)
        builder.enable_evaluation(my_eval_fn)
        compiled = builder.build()
    """

    def __init__(self, name: str = "rag_pipeline") -> None:
        self._name = name
        self._retrieve_fn: NodeFn = _default_retrieve
        self._generate_fn: NodeFn = _default_generate
        self._rerank_fn: Optional[NodeFn] = None
        self._evaluate_fn: Optional[NodeFn] = None
        self._safety_fn: Optional[NodeFn] = None
        self._custom_nodes: Dict[str, NodeFn] = {}
        self._pre_retrieve_nodes: List[str] = []
        self._post_generate_nodes: List[str] = []
        self._router_fn: Optional[RouterFn] = None
        self._router_map: Optional[Dict[str, str]] = None

    # ---- Setters -----------------------------------------------------------

    def set_retriever(self, fn: NodeFn) -> "PipelineGraphBuilder":
        """Set the retrieval node function."""
        self._retrieve_fn = fn
        return self

    def set_generator(self, fn: NodeFn) -> "PipelineGraphBuilder":
        """Set the generation node function."""
        self._generate_fn = fn
        return self

    def enable_reranking(self, fn: Optional[NodeFn] = None) -> "PipelineGraphBuilder":
        """Enable the reranking stage with an optional custom function."""
        self._rerank_fn = fn or _default_rerank
        return self

    def enable_evaluation(self, fn: Optional[NodeFn] = None) -> "PipelineGraphBuilder":
        """Enable the evaluation stage."""
        self._evaluate_fn = fn or _default_evaluate
        return self

    def enable_safety(self, fn: Optional[NodeFn] = None) -> "PipelineGraphBuilder":
        """Enable the safety-check stage."""
        self._safety_fn = fn or _default_safety
        return self

    def add_custom_node(
        self,
        name: str,
        fn: NodeFn,
        *,
        after: str = "generate",
    ) -> "PipelineGraphBuilder":
        """Inject a custom node into the pipeline.

        Parameters
        ----------
        name:
            Unique node name.
        fn:
            Async node function.
        after:
            Node after which to insert (``"retrieve"``, ``"rerank"``,
            ``"generate"``).  Default is after generation.
        """
        self._custom_nodes[name] = fn
        if after in ("retrieve", "rerank"):
            self._pre_retrieve_nodes.append(name)
        else:
            self._post_generate_nodes.append(name)
        return self

    def set_query_router(
        self,
        router_fn: RouterFn,
        path_map: Dict[str, str],
    ) -> "PipelineGraphBuilder":
        """Add conditional routing before retrieval.

        ``router_fn(ctx)`` returns a key from ``path_map``, which maps
        to a target node name already registered.
        """
        self._router_fn = router_fn
        self._router_map = path_map
        return self

    # ---- Build -------------------------------------------------------------

    def build(self) -> CompiledGraph:
        """Compile the pipeline into an executable :class:`CompiledGraph`."""
        graph = StateGraph(self._name)

        # Core nodes
        graph.add_node("retrieve", self._retrieve_fn)
        graph.add_node("generate", self._generate_fn)

        # Optional rerank
        if self._rerank_fn is not None:
            graph.add_node("rerank", self._rerank_fn)

        # Optional stages
        if self._evaluate_fn is not None:
            graph.add_node("evaluate", self._evaluate_fn)
        if self._safety_fn is not None:
            graph.add_node("safety", self._safety_fn)

        # Custom nodes
        for name, fn in self._custom_nodes.items():
            graph.add_node(name, fn)

        # ---- Wire edges ----
        graph.set_entry_point("retrieve")

        # retrieve → (rerank?) → generate
        if self._rerank_fn is not None:
            graph.add_edge("retrieve", "rerank")
            graph.add_edge("rerank", "generate")
        else:
            graph.add_edge("retrieve", "generate")

        # generate → post_generate_nodes → evaluate → safety → END
        chain: List[str] = ["generate"]
        chain.extend(self._post_generate_nodes)
        if self._evaluate_fn is not None:
            chain.append("evaluate")
        if self._safety_fn is not None:
            chain.append("safety")

        for i in range(len(chain) - 1):
            graph.add_edge(chain[i], chain[i + 1])
        graph.add_edge(chain[-1], END)

        return graph.compile()

    # ---- Introspection -----------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """Return a summary of the configured pipeline."""
        stages = ["retrieve"]
        if self._rerank_fn is not None:
            stages.append("rerank")
        stages.append("generate")
        stages.extend(self._post_generate_nodes)
        if self._evaluate_fn is not None:
            stages.append("evaluate")
        if self._safety_fn is not None:
            stages.append("safety")
        return {
            "name": self._name,
            "stages": stages,
            "custom_nodes": list(self._custom_nodes.keys()),
            "has_router": self._router_fn is not None,
        }
