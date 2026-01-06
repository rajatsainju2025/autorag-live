"""
Advanced Query Planner V2 with Hierarchical Planning.

Implements state-of-the-art query planning techniques for complex
multi-step retrieval and reasoning tasks.

Key Features:
1. RAPTOR-style hierarchical query decomposition
2. Self-Ask recursive sub-question generation
3. Tree-structured query plans
4. Adaptive depth control
5. Parallel sub-query execution

References:
- RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (Sarthi et al., 2024)
- Self-Ask: Measuring and Narrowing the Compositionality Gap (Press et al., 2022)
- Least-to-Most Prompting (Zhou et al., 2022)

Example:
    >>> planner = HierarchicalQueryPlanner(llm)
    >>> plan = await planner.plan("How did the Roman Empire's fall affect medieval Europe?")
    >>> print(plan.tree_structure())
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class LLMProtocol(Protocol):
    """Protocol for LLM interface."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response from prompt."""
        ...


class RetrieverProtocol(Protocol):
    """Protocol for retriever interface."""

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents for query."""
        ...


# =============================================================================
# Enums and Data Structures
# =============================================================================


class DecompositionStrategy(str, Enum):
    """Strategy for query decomposition."""

    SELF_ASK = "self_ask"  # Recursive follow-up questions
    LEAST_TO_MOST = "least_to_most"  # Simple to complex
    TREE_OF_THOUGHT = "tree_of_thought"  # Branching exploration
    RAPTOR = "raptor"  # Hierarchical abstraction
    SEQUENTIAL = "sequential"  # Linear step-by-step


class NodeType(str, Enum):
    """Type of query node in the plan tree."""

    ROOT = "root"  # Original query
    SUBQUERY = "subquery"  # Decomposed sub-question
    FACT = "fact"  # Atomic fact retrieval
    SYNTHESIS = "synthesis"  # Aggregation node
    VERIFICATION = "verification"  # Fact-checking node


class NodeStatus(str, Enum):
    """Execution status of a node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class QueryNode:
    """
    A node in the hierarchical query plan tree.

    Attributes:
        node_id: Unique identifier
        query: The query text
        node_type: Type of node
        parent_id: Parent node ID (None for root)
        children: Child node IDs
        depth: Depth in the tree
        status: Execution status
        result: Execution result
        evidence: Retrieved evidence
        confidence: Confidence score
        metadata: Additional metadata
    """

    node_id: str
    query: str
    node_type: NodeType = NodeType.SUBQUERY

    # Tree structure
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    depth: int = 0

    # Execution
    status: NodeStatus = NodeStatus.PENDING
    result: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Scoring
    confidence: float = 0.0
    relevance: float = 0.0

    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def execution_time(self) -> float:
        """Get execution time in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0

    @property
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0

    @property
    def is_ready(self) -> bool:
        """Check if node is ready for execution."""
        return self.status == NodeStatus.PENDING


@dataclass
class QueryPlanTree:
    """
    Hierarchical query plan as a tree structure.

    Attributes:
        root_id: ID of the root node
        nodes: All nodes indexed by ID
        strategy: Decomposition strategy used
        max_depth: Maximum depth reached
        original_query: The original query
    """

    root_id: str
    nodes: Dict[str, QueryNode]
    strategy: DecompositionStrategy
    max_depth: int = 0
    original_query: str = ""

    # Execution tracking
    total_nodes: int = 0
    completed_nodes: int = 0

    # Metadata
    created_at: float = field(default_factory=time.time)

    def get_node(self, node_id: str) -> Optional[QueryNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_root(self) -> QueryNode:
        """Get root node."""
        return self.nodes[self.root_id]

    def get_children(self, node_id: str) -> List[QueryNode]:
        """Get children of a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[c] for c in node.children if c in self.nodes]

    def get_leaves(self) -> List[QueryNode]:
        """Get all leaf nodes."""
        return [n for n in self.nodes.values() if n.is_leaf]

    def get_by_depth(self, depth: int) -> List[QueryNode]:
        """Get all nodes at a specific depth."""
        return [n for n in self.nodes.values() if n.depth == depth]

    def get_ready_nodes(self) -> List[QueryNode]:
        """Get nodes ready for execution (leaves first, bottom-up)."""
        ready = []

        # Get pending leaf nodes first
        for node in self.nodes.values():
            if node.status != NodeStatus.PENDING:
                continue

            if node.is_leaf:
                ready.append(node)
            else:
                # Check if all children are completed
                children_done = all(
                    self.nodes[c].status == NodeStatus.COMPLETED
                    for c in node.children
                    if c in self.nodes
                )
                if children_done:
                    ready.append(node)

        return ready

    def tree_structure(self, node_id: Optional[str] = None, indent: int = 0) -> str:
        """Get tree structure as formatted string."""
        if node_id is None:
            node_id = self.root_id

        node = self.nodes.get(node_id)
        if not node:
            return ""

        prefix = "  " * indent
        status_icon = {
            NodeStatus.PENDING: "○",
            NodeStatus.RUNNING: "◐",
            NodeStatus.COMPLETED: "●",
            NodeStatus.FAILED: "✗",
            NodeStatus.SKIPPED: "◌",
        }.get(node.status, "?")

        lines = [f"{prefix}{status_icon} [{node.node_type.value}] {node.query[:60]}..."]

        for child_id in node.children:
            lines.append(self.tree_structure(child_id, indent + 1))

        return "\n".join(lines)

    @property
    def is_complete(self) -> bool:
        """Check if all nodes are completed."""
        return all(
            n.status in (NodeStatus.COMPLETED, NodeStatus.SKIPPED) for n in self.nodes.values()
        )


# =============================================================================
# Query Decomposition Strategies
# =============================================================================


class DecomposerBase(ABC):
    """Base class for query decomposition strategies."""

    @abstractmethod
    async def decompose(
        self,
        query: str,
        context: Optional[str] = None,
        max_subqueries: int = 5,
    ) -> List[str]:
        """Decompose query into sub-queries."""
        pass


class SelfAskDecomposer(DecomposerBase):
    """
    Self-Ask style decomposition.

    Recursively generates follow-up questions until reaching
    atomic questions that can be answered directly.
    """

    def __init__(self, llm: LLMProtocol):
        """Initialize decomposer."""
        self.llm = llm

    async def decompose(
        self,
        query: str,
        context: Optional[str] = None,
        max_subqueries: int = 5,
    ) -> List[str]:
        """Decompose using Self-Ask pattern."""
        prompt = f"""Given this question, generate follow-up questions that would help answer it.
Only generate questions if the original question requires multiple steps to answer.
If the question is simple and can be answered directly, respond with "NONE".

Question: {query}

Context (if available): {context or 'None'}

Generate up to {max_subqueries} follow-up questions, one per line:"""

        try:
            response = await self.llm.generate(prompt)

            if "NONE" in response.upper():
                return []

            subqueries = []
            for line in response.strip().split("\n"):
                line = line.strip()
                # Remove numbering
                line = re.sub(r"^\d+[\.\)]\s*", "", line)
                if line and "?" in line:
                    subqueries.append(line)

            return subqueries[:max_subqueries]
        except Exception as e:
            logger.warning(f"Self-Ask decomposition failed: {e}")
            return []


class LeastToMostDecomposer(DecomposerBase):
    """
    Least-to-Most prompting decomposition.

    Breaks down complex questions from simple to complex,
    building up to the final answer.
    """

    def __init__(self, llm: LLMProtocol):
        """Initialize decomposer."""
        self.llm = llm

    async def decompose(
        self,
        query: str,
        context: Optional[str] = None,
        max_subqueries: int = 5,
    ) -> List[str]:
        """Decompose from simple to complex."""
        prompt = f"""Break down this complex question into simpler sub-questions.
Start with the simplest foundational question and progress to more complex ones.
Each subsequent question should build on the answers to previous ones.

Complex Question: {query}

List sub-questions from simplest to most complex:"""

        try:
            response = await self.llm.generate(prompt)

            subqueries = []
            for line in response.strip().split("\n"):
                line = line.strip()
                line = re.sub(r"^\d+[\.\)]\s*", "", line)
                if line and len(line) > 10:
                    subqueries.append(line)

            return subqueries[:max_subqueries]
        except Exception as e:
            logger.warning(f"Least-to-Most decomposition failed: {e}")
            return []


class RaptorDecomposer(DecomposerBase):
    """
    RAPTOR-style hierarchical decomposition.

    Creates a tree of queries with different abstraction levels,
    from specific facts to high-level summaries.
    """

    def __init__(self, llm: LLMProtocol):
        """Initialize decomposer."""
        self.llm = llm

    async def decompose(
        self,
        query: str,
        context: Optional[str] = None,
        max_subqueries: int = 5,
    ) -> List[str]:
        """Decompose into hierarchical sub-queries."""
        prompt = f"""Analyze this question and create a hierarchy of sub-questions at different abstraction levels.

Original Question: {query}

Generate sub-questions at different levels:
1. HIGH-LEVEL: Broad conceptual questions
2. MID-LEVEL: Specific topic questions
3. LOW-LEVEL: Atomic fact questions

Format each with its level prefix (HIGH/MID/LOW):"""

        try:
            response = await self.llm.generate(prompt)

            subqueries = []
            for line in response.strip().split("\n"):
                line = line.strip()
                # Extract query after level prefix
                for prefix in ["HIGH-LEVEL:", "MID-LEVEL:", "LOW-LEVEL:", "HIGH:", "MID:", "LOW:"]:
                    if prefix in line.upper():
                        query_text = line.split(":", 1)[-1].strip()
                        if query_text:
                            subqueries.append(query_text)
                        break

            return subqueries[:max_subqueries]
        except Exception as e:
            logger.warning(f"RAPTOR decomposition failed: {e}")
            return []


# =============================================================================
# Hierarchical Query Planner
# =============================================================================


class HierarchicalQueryPlanner:
    """
    Advanced query planner with hierarchical decomposition.

    Creates tree-structured query plans for complex multi-hop questions.

    Example:
        >>> planner = HierarchicalQueryPlanner(llm)
        >>> plan = await planner.plan("What caused the 2008 financial crisis?")
        >>> for node in plan.get_leaves():
        ...     print(f"Subquery: {node.query}")
    """

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        strategy: DecompositionStrategy = DecompositionStrategy.SELF_ASK,
        max_depth: int = 3,
        max_children: int = 4,
    ):
        """
        Initialize planner.

        Args:
            llm: Language model for decomposition
            strategy: Decomposition strategy
            max_depth: Maximum tree depth
            max_children: Maximum children per node
        """
        self.llm = llm
        self.strategy = strategy
        self.max_depth = max_depth
        self.max_children = max_children

        # Initialize decomposers
        self._decomposers: Dict[DecompositionStrategy, DecomposerBase] = {}
        if llm:
            self._decomposers[DecompositionStrategy.SELF_ASK] = SelfAskDecomposer(llm)
            self._decomposers[DecompositionStrategy.LEAST_TO_MOST] = LeastToMostDecomposer(llm)
            self._decomposers[DecompositionStrategy.RAPTOR] = RaptorDecomposer(llm)

    async def plan(
        self,
        query: str,
        *,
        strategy: Optional[DecompositionStrategy] = None,
        max_depth: Optional[int] = None,
    ) -> QueryPlanTree:
        """
        Create hierarchical query plan.

        Args:
            query: Original query
            strategy: Override default strategy
            max_depth: Override default max depth

        Returns:
            QueryPlanTree with decomposed queries
        """
        strategy = strategy or self.strategy
        max_depth = max_depth or self.max_depth

        # Create root node
        root_id = str(uuid.uuid4())[:8]
        root = QueryNode(
            node_id=root_id,
            query=query,
            node_type=NodeType.ROOT,
            depth=0,
        )

        nodes: Dict[str, QueryNode] = {root_id: root}

        # Build tree recursively
        if self.llm:
            await self._build_tree(
                nodes=nodes,
                parent_id=root_id,
                strategy=strategy,
                current_depth=0,
                max_depth=max_depth,
            )

        # Calculate max depth reached
        actual_max_depth = max(n.depth for n in nodes.values())

        return QueryPlanTree(
            root_id=root_id,
            nodes=nodes,
            strategy=strategy,
            max_depth=actual_max_depth,
            original_query=query,
            total_nodes=len(nodes),
        )

    async def _build_tree(
        self,
        nodes: Dict[str, QueryNode],
        parent_id: str,
        strategy: DecompositionStrategy,
        current_depth: int,
        max_depth: int,
    ) -> None:
        """Recursively build query tree."""
        if current_depth >= max_depth:
            return

        parent = nodes[parent_id]
        decomposer = self._decomposers.get(strategy)

        if not decomposer:
            return

        # Decompose parent query
        subqueries = await decomposer.decompose(
            parent.query,
            max_subqueries=self.max_children,
        )

        # Create child nodes
        for subquery in subqueries:
            child_id = str(uuid.uuid4())[:8]
            child = QueryNode(
                node_id=child_id,
                query=subquery,
                node_type=NodeType.SUBQUERY,
                parent_id=parent_id,
                depth=current_depth + 1,
            )

            nodes[child_id] = child
            parent.children.append(child_id)

            # Recursively decompose children
            await self._build_tree(
                nodes=nodes,
                parent_id=child_id,
                strategy=strategy,
                current_depth=current_depth + 1,
                max_depth=max_depth,
            )

    def estimate_complexity(self, query: str) -> Dict[str, Any]:
        """
        Estimate query complexity without full decomposition.

        Args:
            query: Query to analyze

        Returns:
            Complexity metrics
        """
        # Heuristic indicators
        multi_hop_words = [
            "how",
            "why",
            "compare",
            "contrast",
            "relationship",
            "cause",
            "effect",
            "influence",
            "impact",
        ]
        comparison_words = ["vs", "versus", "difference", "better", "worse"]
        temporal_words = ["before", "after", "during", "when", "timeline"]

        query_lower = query.lower()

        multi_hop_count = sum(1 for w in multi_hop_words if w in query_lower)
        comparison_count = sum(1 for w in comparison_words if w in query_lower)
        temporal_count = sum(1 for w in temporal_words if w in query_lower)

        # Question count (nested questions)
        question_count = query.count("?")

        # Word count
        word_count = len(query.split())

        # Calculate complexity score
        complexity_score = (
            multi_hop_count * 2
            + comparison_count * 3
            + temporal_count * 1.5
            + (question_count - 1) * 2
            + (word_count / 20)
        )

        # Determine recommended depth
        if complexity_score < 2:
            recommended_depth = 1
        elif complexity_score < 5:
            recommended_depth = 2
        else:
            recommended_depth = 3

        return {
            "complexity_score": complexity_score,
            "recommended_depth": recommended_depth,
            "is_multi_hop": multi_hop_count > 0,
            "is_comparison": comparison_count > 0,
            "is_temporal": temporal_count > 0,
            "word_count": word_count,
        }


# =============================================================================
# Query Plan Executor
# =============================================================================


class QueryPlanExecutor:
    """
    Executes hierarchical query plans.

    Processes query trees bottom-up, aggregating results
    from leaf nodes to root.
    """

    def __init__(
        self,
        retriever: Optional[RetrieverProtocol] = None,
        llm: Optional[LLMProtocol] = None,
        max_concurrent: int = 5,
    ):
        """
        Initialize executor.

        Args:
            retriever: Retriever for evidence
            llm: LLM for synthesis
            max_concurrent: Max concurrent operations
        """
        self.retriever = retriever
        self.llm = llm
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(
        self,
        plan: QueryPlanTree,
        *,
        retrieve_top_k: int = 5,
    ) -> QueryPlanTree:
        """
        Execute query plan.

        Args:
            plan: Query plan tree
            retrieve_top_k: Number of documents to retrieve per query

        Returns:
            Executed plan with results
        """
        # Process bottom-up: leaves first
        while not plan.is_complete:
            ready_nodes = plan.get_ready_nodes()

            if not ready_nodes:
                logger.warning("No ready nodes but plan incomplete")
                break

            # Execute ready nodes in parallel
            tasks = [self._execute_node(node, plan, retrieve_top_k) for node in ready_nodes]

            await asyncio.gather(*tasks, return_exceptions=True)

            plan.completed_nodes = sum(
                1 for n in plan.nodes.values() if n.status == NodeStatus.COMPLETED
            )

        return plan

    async def _execute_node(
        self,
        node: QueryNode,
        plan: QueryPlanTree,
        top_k: int,
    ) -> None:
        """Execute a single node."""
        async with self._semaphore:
            node.status = NodeStatus.RUNNING
            node.started_at = time.time()

            try:
                if node.is_leaf:
                    # Leaf nodes: retrieve evidence
                    await self._execute_leaf(node, top_k)
                else:
                    # Internal nodes: synthesize from children
                    await self._execute_synthesis(node, plan)

                node.status = NodeStatus.COMPLETED

            except Exception as e:
                logger.error(f"Node execution failed: {e}")
                node.status = NodeStatus.FAILED
                node.metadata["error"] = str(e)

            finally:
                node.completed_at = time.time()

    async def _execute_leaf(self, node: QueryNode, top_k: int) -> None:
        """Execute leaf node (retrieval)."""
        if self.retriever:
            docs = await self.retriever.retrieve(node.query, top_k=top_k)
            node.evidence = docs
            node.result = self._format_evidence(docs)
            node.confidence = self._calculate_confidence(docs)
        else:
            node.result = f"[No retriever] Query: {node.query}"
            node.confidence = 0.0

    async def _execute_synthesis(
        self,
        node: QueryNode,
        plan: QueryPlanTree,
    ) -> None:
        """Execute synthesis node."""
        # Collect child results
        children = plan.get_children(node.node_id)
        child_results = []

        for child in children:
            if child.status == NodeStatus.COMPLETED and child.result:
                child_results.append(
                    {
                        "query": child.query,
                        "result": child.result,
                        "confidence": child.confidence,
                    }
                )

        if not child_results:
            node.result = "No child results to synthesize"
            node.confidence = 0.0
            return

        if self.llm:
            # Use LLM to synthesize
            node.result = await self._llm_synthesize(node.query, child_results)
        else:
            # Simple concatenation
            node.result = "\n\n".join(f"Q: {r['query']}\nA: {r['result']}" for r in child_results)

        # Average confidence from children
        node.confidence = sum(r["confidence"] for r in child_results) / len(child_results)

    async def _llm_synthesize(
        self,
        query: str,
        child_results: List[Dict[str, Any]],
    ) -> str:
        """Synthesize results using LLM."""
        context = "\n\n".join(
            f"Sub-question: {r['query']}\nAnswer: {r['result']}" for r in child_results
        )

        prompt = f"""Based on the following sub-questions and their answers, provide a comprehensive answer to the main question.

Main Question: {query}

Sub-questions and Answers:
{context}

Synthesized Answer:"""

        return await self.llm.generate(prompt)

    def _format_evidence(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as text."""
        if not docs:
            return "No relevant documents found."

        parts = []
        for i, doc in enumerate(docs[:3], 1):
            content = doc.get("content", doc.get("text", str(doc)))
            parts.append(f"[{i}] {content[:500]}")

        return "\n\n".join(parts)

    def _calculate_confidence(self, docs: List[Dict[str, Any]]) -> float:
        """Calculate confidence from retrieved documents."""
        if not docs:
            return 0.0

        scores = [doc.get("score", 0.5) for doc in docs[:3]]
        return sum(scores) / len(scores)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_planner(
    llm: Optional[LLMProtocol] = None,
    strategy: DecompositionStrategy = DecompositionStrategy.SELF_ASK,
) -> HierarchicalQueryPlanner:
    """
    Create a hierarchical query planner.

    Args:
        llm: Language model
        strategy: Decomposition strategy

    Returns:
        HierarchicalQueryPlanner instance
    """
    return HierarchicalQueryPlanner(llm=llm, strategy=strategy)


async def decompose_query(
    query: str,
    llm: LLMProtocol,
    strategy: DecompositionStrategy = DecompositionStrategy.SELF_ASK,
) -> List[str]:
    """
    Quick query decomposition.

    Args:
        query: Query to decompose
        llm: Language model
        strategy: Decomposition strategy

    Returns:
        List of sub-queries
    """
    planner = HierarchicalQueryPlanner(llm=llm, strategy=strategy, max_depth=1)
    plan = await planner.plan(query)
    return [node.query for node in plan.get_leaves()]
