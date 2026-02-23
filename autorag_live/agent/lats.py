"""
Language Agent Tree Search (LATS).

Implements LATS from "Language Agent Tree Search Unifies Reasoning, Acting,
and Planning in Language Models" (Zhou et al., 2023).

LATS combines Monte Carlo Tree Search (MCTS) with LLM-based agents:
  - Each tree node represents a reasoning/action state.
  - The LLM generates candidate next steps (expansion).
  - A value function (also LLM-based) scores node quality.
  - UCT (Upper Confidence bound for Trees) balances exploration vs exploitation.
  - Backpropagation updates ancestor values on success/failure.

This enables significantly better reasoning over long-horizon tasks compared
to vanilla ReAct or CoT, as it can backtrack and explore alternative paths.

References:
    https://arxiv.org/abs/2310.04406
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
LLMFn = Callable[[str], Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# Enums & Dataclasses
# ---------------------------------------------------------------------------


class NodeState(str, Enum):
    """Lifecycle of a LATS tree node."""

    UNEXPANDED = "unexpanded"
    EXPANDED = "expanded"
    TERMINAL = "terminal"
    FAILED = "failed"


@dataclass
class LATSNode:
    """A single node in the LATS search tree."""

    state: str  # The reasoning/action state accumulated so far
    parent: Optional["LATSNode"] = None
    children: List["LATSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0  # Cumulative value from backprop
    depth: int = 0
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_state: NodeState = NodeState.UNEXPANDED
    is_terminal: bool = False
    reward: float = 0.0  # Terminal reward from environment

    @property
    def q_value(self) -> float:
        """Mean value of this node."""
        return self.value / self.visits if self.visits > 0 else 0.0

    def uct_score(self, exploration_constant: float, parent_visits: int) -> float:
        """UCT selection score balancing exploitation and exploration."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.q_value
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def best_child(self, exploration_constant: float) -> Optional["LATSNode"]:
        """Select the child with the highest UCT score."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.uct_score(exploration_constant, self.visits))


@dataclass
class LATSConfig:
    """Configuration for the LATS agent."""

    max_iterations: int = 50
    max_depth: int = 8
    num_expansions: int = 3  # Candidate children per expansion
    exploration_constant: float = 1.414  # sqrt(2) is classic UCT default
    value_threshold: float = 0.8  # Early-stop if value >= threshold
    rollout_depth: int = 3  # Steps for simulation/rollout


@dataclass
class LATSResult:
    """Result of a LATS search."""

    answer: str
    best_path: List[str]  # State chain from root to best leaf
    root_value: float
    total_iterations: int
    nodes_explored: int


# ---------------------------------------------------------------------------
# Core LATS Agent
# ---------------------------------------------------------------------------


class LATSAgent:
    """
    Language Agent Tree Search.

    Uses MCTS with LLM expansions and valuations to solve complex,
    multi-step reasoning and retrieval tasks.

    Example::

        async def my_llm(prompt: str) -> str:
            # call your LLM here
            return "..."

        agent = LATSAgent(llm_fn=my_llm)
        result = await agent.search("What caused the 2008 financial crisis?")
        print(result.answer)
    """

    _EXPAND_PROMPT = """\
You are solving a problem step by step using reasoning and actions.

Current state:
{state}

Generate {num_expansions} distinct next reasoning steps or actions.
Each step should meaningfully advance toward solving the original problem.

Format as:
STEP 1: <reasoning or action>
STEP 2: <reasoning or action>
...

Original problem: {problem}
"""

    _VALUE_PROMPT = """\
Evaluate the quality of this reasoning chain for solving the problem.

Problem: {problem}

Reasoning chain:
{state}

Score the chain from 0.0 (completely wrong/stuck) to 1.0 (correct answer found).
Provide a brief justification, then output: SCORE: <float>
"""

    _TERMINAL_PROMPT = """\
Based on the following reasoning chain, provide a final concise answer.

Problem: {problem}

Reasoning chain:
{state}

Final Answer:"""

    def __init__(self, llm_fn: LLMFn, config: Optional[LATSConfig] = None) -> None:
        self._llm = llm_fn
        self.config = config or LATSConfig()
        self._nodes_explored = 0

    async def _expand(self, node: LATSNode, problem: str) -> List[LATSNode]:
        """Generate candidate child nodes via LLM."""
        prompt = self._EXPAND_PROMPT.format(
            state=node.state,
            num_expansions=self.config.num_expansions,
            problem=problem,
        )
        response = await self._llm(prompt)

        children: List[LATSNode] = []
        for line in response.splitlines():
            line = line.strip()
            if line.lower().startswith("step ") and ":" in line:
                step_text = line.split(":", 1)[1].strip()
                new_state = f"{node.state}\n{step_text}" if node.state else step_text
                child = LATSNode(
                    state=new_state,
                    parent=node,
                    depth=node.depth + 1,
                )
                children.append(child)
                if len(children) >= self.config.num_expansions:
                    break

        # Fallback: if parsing failed, create one child with raw response
        if not children:
            new_state = f"{node.state}\n{response.strip()}" if node.state else response.strip()
            children.append(LATSNode(state=new_state, parent=node, depth=node.depth + 1))

        node.children.extend(children)
        node.node_state = NodeState.EXPANDED
        return children

    async def _evaluate(self, node: LATSNode, problem: str) -> float:
        """Use LLM to assign a value to a node's reasoning state."""
        prompt = self._VALUE_PROMPT.format(problem=problem, state=node.state)
        response = await self._llm(prompt)

        for line in reversed(response.splitlines()):
            if "SCORE:" in line.upper():
                try:
                    score_str = line.upper().split("SCORE:")[-1].strip()
                    return max(0.0, min(1.0, float(score_str)))
                except ValueError:
                    pass
        return 0.0

    def _backpropagate(self, node: LATSNode, value: float) -> None:
        """Propagate a value up the tree, updating visit counts."""
        current: Optional[LATSNode] = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent

    def _select(self, root: LATSNode) -> LATSNode:
        """
        Tree policy: traverse from root to a leaf using UCT,
        then return the leaf for expansion.
        """
        node = root
        while node.children and node.node_state == NodeState.EXPANDED:
            best = node.best_child(self.config.exploration_constant)
            if best is None:
                break
            node = best
        return node

    def _best_leaf(self, root: LATSNode) -> LATSNode:
        """Return the highest-value leaf found during search."""
        best = root
        stack = [root]
        while stack:
            current = stack.pop()
            if current.q_value > best.q_value:
                best = current
            stack.extend(current.children)
        return best

    def _extract_path(self, node: LATSNode) -> List[str]:
        """Trace back from a node to root, returning state chain."""
        path: List[str] = []
        current: Optional[LATSNode] = node
        while current is not None:
            if current.state:
                path.append(current.state)
            current = current.parent
        return list(reversed(path))

    async def search(self, problem: str) -> LATSResult:
        """
        Run LATS search to solve the given problem.

        Args:
            problem: The question or task to solve.

        Returns:
            LATSResult containing the best answer and search metadata.
        """
        logger.info("Starting LATS search for: %s", problem[:80])
        self._nodes_explored = 0
        root = LATSNode(state="", depth=0)

        for iteration in range(self.config.max_iterations):
            # 1. Selection
            selected = self._select(root)

            # 2. Expansion (skip if max depth reached)
            if selected.depth >= self.config.max_depth:
                selected.node_state = NodeState.TERMINAL
                selected.is_terminal = True
                value = await self._evaluate(selected, problem)
                self._backpropagate(selected, value)
                continue

            new_children = await self._expand(selected, problem)
            self._nodes_explored += len(new_children)

            # 3. Simulation / Evaluation
            eval_tasks = [self._evaluate(child, problem) for child in new_children]
            values = await asyncio.gather(*eval_tasks)

            # 4. Backpropagation
            for child, value in zip(new_children, values):
                child.visits = 0  # Will be set during backprop
                self._backpropagate(child, value)

            # 5. Early stopping if a high-value state is found
            best_value = max(values)
            if best_value >= self.config.value_threshold:
                logger.info("LATS early stop at iteration %d (value=%.3f)", iteration, best_value)
                break

        # Extract final answer from the best leaf
        best_leaf = self._best_leaf(root)
        path = self._extract_path(best_leaf)

        terminal_prompt = self._TERMINAL_PROMPT.format(problem=problem, state=best_leaf.state)
        answer = await self._llm(terminal_prompt)

        logger.info(
            "LATS complete: %d iterations, %d nodes explored, best_value=%.3f",
            iteration + 1,
            self._nodes_explored,
            best_leaf.q_value,
        )

        return LATSResult(
            answer=answer.strip(),
            best_path=path,
            root_value=root.q_value,
            total_iterations=iteration + 1,
            nodes_explored=self._nodes_explored,
        )
