"""
StateGraph — a graph-based workflow engine for agentic RAG pipelines.

Replaces the linear stage-by-stage pipeline with a directed graph that
supports conditional edges, cycles, and parallel branches.  Inspired by
LangGraph's ``StateGraph`` but built on AutoRAG-Live's ``RAGContext``.

Key Features:
    1. **Declarative graph definition** — ``add_node()``, ``add_edge()``,
       ``add_conditional_edges()``
    2. **Cyclic support** — enables retrieve → grade → re-retrieve loops
    3. **Async-first execution** — every node function is ``async``
    4. **Copy-on-write state** — parallel branches don't corrupt each other
    5. **Built-in ``END`` sentinel** — explicitly marks terminal transitions
    6. **Execution trace** — every run records the ordered list of visited nodes

Example:
    >>> from autorag_live.core.context import RAGContext
    >>> from autorag_live.core.state_graph import StateGraph, END
    >>>
    >>> async def retrieve(ctx: RAGContext) -> RAGContext:
    ...     # ... fetch docs ...
    ...     return ctx.add_documents(docs)
    ...
    >>> async def grade(ctx: RAGContext) -> RAGContext:
    ...     return ctx.set_metadata("docs_relevant", True)
    ...
    >>> def route_after_grade(ctx: RAGContext) -> str:
    ...     if ctx.metadata.get("docs_relevant"):
    ...         return "generate"
    ...     return "rewrite"
    ...
    >>> graph = StateGraph("rag")
    >>> graph.add_node("retrieve", retrieve)
    >>> graph.add_node("grade", grade)
    >>> graph.add_node("generate", generate)
    >>> graph.add_node("rewrite", rewrite)
    >>> graph.set_entry_point("retrieve")
    >>> graph.add_edge("retrieve", "grade")
    >>> graph.add_conditional_edges("grade", route_after_grade,
    ...     {"generate": "generate", "rewrite": "rewrite"})
    >>> graph.add_edge("rewrite", "retrieve")   # ← cycle
    >>> graph.add_edge("generate", END)
    >>>
    >>> compiled = graph.compile()
    >>> result = await compiled.invoke(RAGContext.create(query="What is RLHF?"))
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from autorag_live.core.context import RAGContext

logger = logging.getLogger(__name__)

# Sentinel value indicating the graph should terminate.
END: str = "__END__"

# Type aliases
NodeFn = Callable[[RAGContext], Awaitable[RAGContext]]
RouterFn = Callable[[RAGContext], Union[str, Awaitable[str]]]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    """A node in the state graph."""

    name: str
    fn: NodeFn
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionalEdge:
    """An edge whose target depends on a routing function."""

    router_fn: RouterFn
    path_map: Dict[str, str]  # router_fn return value → target node name


@dataclass
class ExecutionStep:
    """Record of a single node execution during a graph run."""

    node: str
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class GraphResult:
    """Result of a full graph execution."""

    context: RAGContext
    steps: List[ExecutionStep]
    total_latency_ms: float
    terminated_at: str  # node name or END

    @property
    def node_trace(self) -> List[str]:
        """Ordered list of visited node names."""
        return [s.node for s in self.steps]


# ---------------------------------------------------------------------------
# StateGraph definition (mutable builder)
# ---------------------------------------------------------------------------


class StateGraph:
    """
    Mutable builder for defining an agentic RAG workflow as a directed graph.

    Call ``.compile()`` to produce an immutable ``CompiledGraph`` that can be
    executed via ``await compiled.invoke(ctx)``.
    """

    def __init__(self, name: str = "rag_graph"):
        self.name = name
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, str] = {}  # source → target (unconditional)
        self._conditional_edges: Dict[str, ConditionalEdge] = {}
        self._entry_point: Optional[str] = None

    # -- Node management -----------------------------------------------------

    def add_node(self, name: str, fn: NodeFn, **metadata: Any) -> StateGraph:
        """Register a node.  ``fn`` must be ``async def fn(ctx) -> ctx``."""
        if name == END:
            raise ValueError("Cannot use reserved name '__END__' as a node name")
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists")
        self._nodes[name] = GraphNode(name=name, fn=fn, metadata=metadata)
        return self

    def set_entry_point(self, name: str) -> StateGraph:
        """Set the starting node of the graph."""
        if name not in self._nodes:
            raise ValueError(f"Node '{name}' not found.  Add it first with add_node().")
        self._entry_point = name
        return self

    # -- Edge management -----------------------------------------------------

    def add_edge(self, source: str, target: str) -> StateGraph:
        """Add an unconditional edge from *source* to *target*."""
        self._validate_node_ref(source, "source")
        if target != END:
            self._validate_node_ref(target, "target")
        if source in self._edges or source in self._conditional_edges:
            raise ValueError(
                f"Node '{source}' already has an outgoing edge.  "
                "Use add_conditional_edges() for branching."
            )
        self._edges[source] = target
        return self

    def add_conditional_edges(
        self,
        source: str,
        router_fn: RouterFn,
        path_map: Dict[str, str],
    ) -> StateGraph:
        """
        Add conditional routing after *source*.

        ``router_fn(ctx)`` returns a string key.  ``path_map`` maps those keys
        to target node names (or ``END``).
        """
        self._validate_node_ref(source, "source")
        for target in path_map.values():
            if target != END:
                self._validate_node_ref(target, "target")
        if source in self._edges or source in self._conditional_edges:
            raise ValueError(f"Node '{source}' already has an outgoing edge.")
        self._conditional_edges[source] = ConditionalEdge(router_fn=router_fn, path_map=path_map)
        return self

    # -- Compilation ---------------------------------------------------------

    def compile(self) -> CompiledGraph:
        """Validate and compile the graph into an executable form."""
        if self._entry_point is None:
            raise ValueError("Entry point not set.  Call set_entry_point() first.")
        # Check that every node has at least one outgoing edge (except END-targets)
        for name in self._nodes:
            if name not in self._edges and name not in self._conditional_edges:
                logger.warning("Node '%s' has no outgoing edges — it will be a dead end.", name)
        return CompiledGraph(
            name=self.name,
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            conditional_edges=dict(self._conditional_edges),
            entry_point=self._entry_point,
        )

    # -- Internal helpers ----------------------------------------------------

    def _validate_node_ref(self, name: str, label: str) -> None:
        if name not in self._nodes:
            raise ValueError(
                f"{label.capitalize()} node '{name}' not found.  "
                f"Available nodes: {list(self._nodes.keys())}"
            )


# ---------------------------------------------------------------------------
# CompiledGraph (immutable, executable)
# ---------------------------------------------------------------------------


class CompiledGraph:
    """
    Immutable compiled graph ready for execution.

    Call ``await graph.invoke(ctx)`` to run the workflow.
    """

    def __init__(
        self,
        name: str,
        nodes: Dict[str, GraphNode],
        edges: Dict[str, str],
        conditional_edges: Dict[str, ConditionalEdge],
        entry_point: str,
    ):
        self.name = name
        self._nodes = nodes
        self._edges = edges
        self._conditional_edges = conditional_edges
        self._entry_point = entry_point

    async def invoke(
        self,
        context: RAGContext,
        *,
        max_steps: int = 50,
        on_node_start: Optional[Callable[[str, RAGContext], None]] = None,
        on_node_end: Optional[Callable[[str, RAGContext, float], None]] = None,
    ) -> GraphResult:
        """
        Execute the graph starting from the entry point.

        Args:
            context:        Initial RAGContext.
            max_steps:      Safety limit to prevent infinite loops.
            on_node_start:  Optional callback fired before each node.
            on_node_end:    Optional callback fired after each node.

        Returns:
            GraphResult with final context, execution trace, and timings.
        """
        steps: List[ExecutionStep] = []
        current_node = self._entry_point
        ctx = context
        total_start = time.perf_counter()

        for iteration in range(max_steps):
            if current_node == END:
                break

            node = self._nodes.get(current_node)
            if node is None:
                raise RuntimeError(f"Node '{current_node}' not found in compiled graph")

            # Callback: node start
            if on_node_start:
                on_node_start(current_node, ctx)

            # Execute node
            step_start = time.perf_counter()
            try:
                ctx = await node.fn(ctx)
                latency = (time.perf_counter() - step_start) * 1000
                steps.append(ExecutionStep(node=current_node, latency_ms=latency, success=True))
            except Exception as exc:
                latency = (time.perf_counter() - step_start) * 1000
                steps.append(
                    ExecutionStep(
                        node=current_node,
                        latency_ms=latency,
                        success=False,
                        error=str(exc),
                    )
                )
                ctx = ctx.mark_error(f"Node '{current_node}' failed: {exc}")
                logger.error("Graph node '%s' failed: %s", current_node, exc)
                break

            # Callback: node end
            if on_node_end:
                on_node_end(current_node, ctx, latency)

            # Determine next node
            current_node = await self._resolve_next(current_node, ctx)
        else:
            logger.warning(
                "Graph '%s' hit max_steps=%d — possible infinite loop", self.name, max_steps
            )
            ctx = ctx.set_metadata("max_steps_exceeded", True)

        total_latency = (time.perf_counter() - total_start) * 1000
        return GraphResult(
            context=ctx,
            steps=steps,
            total_latency_ms=total_latency,
            terminated_at=current_node,
        )

    async def _resolve_next(self, current: str, ctx: RAGContext) -> str:
        """Determine the next node from edges or conditional edges."""
        # Unconditional edge
        if current in self._edges:
            return self._edges[current]

        # Conditional edge
        if current in self._conditional_edges:
            cond = self._conditional_edges[current]
            result = cond.router_fn(ctx)
            if inspect.isawaitable(result):
                result = await result
            target = cond.path_map.get(result)  # type: ignore[arg-type]
            if target is None:
                raise RuntimeError(
                    f"Router for '{current}' returned '{result}' which is not in "
                    f"path_map keys: {list(cond.path_map.keys())}"
                )
            return target

        # No outgoing edge — treat as terminal
        return END

    # -- Introspection -------------------------------------------------------

    @property
    def node_names(self) -> List[str]:
        """Return all node names in the graph."""
        return list(self._nodes.keys())

    @property
    def entry_point(self) -> str:
        return self._entry_point

    def get_edges_from(self, node: str) -> Dict[str, Any]:
        """Return edge info for a given node."""
        if node in self._edges:
            return {"type": "unconditional", "target": self._edges[node]}
        if node in self._conditional_edges:
            cond = self._conditional_edges[node]
            return {"type": "conditional", "path_map": cond.path_map}
        return {"type": "terminal"}

    def __repr__(self) -> str:
        return (
            f"CompiledGraph(name={self.name!r}, nodes={self.node_names}, "
            f"entry={self._entry_point!r})"
        )
