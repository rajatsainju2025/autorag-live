"""
Advanced Reasoning Engine for Agentic RAG.

Implements state-of-the-art reasoning strategies:
- Tree of Thoughts (ToT): Explore multiple reasoning paths
- Reflexion: Self-reflection and iterative improvement
- Chain of Thought (CoT): Step-by-step reasoning
- Least-to-Most: Decompose complex problems
- Self-Consistency: Multiple reasoning paths with voting

References:
- Tree of Thoughts (Yao et al., 2023)
- Reflexion (Shinn et al., 2023)
- Chain of Thought (Wei et al., 2022)
- Self-Consistency (Wang et al., 2023)

Example:
    >>> reasoner = TreeOfThoughtsReasoner(llm, branching_factor=3, max_depth=3)
    >>> result = await reasoner.reason("Complex question here")
    >>> print(result.best_path)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM interactions."""

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Generate response from prompt."""
        ...


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Protocol for thought evaluation."""

    async def evaluate(self, thought: str, context: Dict[str, Any]) -> float:
        """Evaluate quality of a thought (0-1)."""
        ...


# =============================================================================
# Data Structures
# =============================================================================


class ReasoningStrategy(Enum):
    """Available reasoning strategies."""

    CHAIN_OF_THOUGHT = auto()
    TREE_OF_THOUGHTS = auto()
    REFLEXION = auto()
    SELF_CONSISTENCY = auto()
    LEAST_TO_MOST = auto()


@dataclass
class ThoughtNode:
    """A single thought in the reasoning tree."""

    id: str
    content: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children_ids) == 0

    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent_id is None


@dataclass
class ReasoningPath:
    """A complete reasoning path from root to leaf."""

    nodes: List[ThoughtNode]
    total_score: float = 0.0
    is_complete: bool = False
    final_answer: Optional[str] = None

    @property
    def depth(self) -> int:
        """Depth of this path."""
        return len(self.nodes)

    def get_trace(self) -> str:
        """Get formatted reasoning trace."""
        lines = []
        for i, node in enumerate(self.nodes):
            indent = "  " * i
            lines.append(f"{indent}Step {i + 1}: {node.content}")
        if self.final_answer:
            lines.append(f"\nAnswer: {self.final_answer}")
        return "\n".join(lines)


@dataclass
class ReasoningResult:
    """Result from reasoning process."""

    query: str
    strategy: ReasoningStrategy
    best_path: Optional[ReasoningPath] = None
    all_paths: List[ReasoningPath] = field(default_factory=list)
    answer: str = ""
    confidence: float = 0.0
    iterations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionFeedback:
    """Feedback from self-reflection."""

    is_correct: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    improved_reasoning: Optional[str] = None
    confidence: float = 0.0


# =============================================================================
# Base Reasoner
# =============================================================================


class BaseReasoner(ABC):
    """Base class for reasoning strategies."""

    def __init__(self, llm: LLMProtocol, verbose: bool = False):
        """Initialize reasoner."""
        self.llm = llm
        self.verbose = verbose

    @abstractmethod
    async def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute reasoning for a query."""
        ...

    def _log(self, message: str) -> None:
        """Log if verbose mode enabled."""
        if self.verbose:
            logger.info(message)


# =============================================================================
# Chain of Thought Reasoner
# =============================================================================


class ChainOfThoughtReasoner(BaseReasoner):
    """
    Chain of Thought (CoT) reasoning.

    Generates step-by-step reasoning to solve problems.
    """

    COT_PROMPT = """Solve the following problem step by step.
Think through each step carefully before providing your final answer.

Problem: {query}

{context}

Let's think step by step:"""

    def __init__(
        self,
        llm: LLMProtocol,
        verbose: bool = False,
        num_steps: int = 5,
    ):
        """Initialize CoT reasoner."""
        super().__init__(llm, verbose)
        self.num_steps = num_steps

    async def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute chain of thought reasoning."""
        context_str = self._format_context(context) if context else ""

        prompt = self.COT_PROMPT.format(query=query, context=context_str)
        response = await self.llm.generate(prompt, temperature=0.7)

        # Parse steps from response
        steps = self._parse_steps(response)

        # Create path
        nodes = []
        for i, step in enumerate(steps):
            node = ThoughtNode(
                id=f"step_{i}",
                content=step,
                parent_id=f"step_{i-1}" if i > 0 else None,
                depth=i,
                score=1.0,
            )
            nodes.append(node)

        path = ReasoningPath(
            nodes=nodes,
            total_score=1.0,
            is_complete=True,
            final_answer=self._extract_answer(response),
        )

        return ReasoningResult(
            query=query,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            best_path=path,
            all_paths=[path],
            answer=path.final_answer or "",
            confidence=0.8,
            iterations=1,
        )

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt."""
        if "documents" in context:
            docs = context["documents"]
            return "Context:\n" + "\n".join(f"- {d}" for d in docs[:5])
        return ""

    def _parse_steps(self, response: str) -> List[str]:
        """Parse reasoning steps from response."""
        lines = response.strip().split("\n")
        steps = []
        current_step = []

        for line in lines:
            if line.strip().startswith(("Step", "1.", "2.", "3.", "4.", "5.")):
                if current_step:
                    steps.append(" ".join(current_step))
                current_step = [line.strip()]
            elif current_step:
                current_step.append(line.strip())

        if current_step:
            steps.append(" ".join(current_step))

        return steps if steps else [response]

    def _extract_answer(self, response: str) -> str:
        """Extract final answer from response."""
        # Look for explicit answer markers
        markers = ["Therefore", "Thus", "Answer:", "Final Answer:", "In conclusion"]
        for marker in markers:
            if marker.lower() in response.lower():
                idx = response.lower().find(marker.lower())
                return response[idx:].strip()

        # Return last paragraph
        paragraphs = response.strip().split("\n\n")
        return paragraphs[-1] if paragraphs else response


# =============================================================================
# Tree of Thoughts Reasoner
# =============================================================================


class TreeOfThoughtsReasoner(BaseReasoner):
    """
    Tree of Thoughts (ToT) reasoning.

    Explores multiple reasoning paths and evaluates them to find the best solution.
    """

    THOUGHT_PROMPT = """Given the problem and current reasoning path, generate the next thought.

Problem: {query}

Current path:
{path}

Generate a single next step in the reasoning. Be specific and logical."""

    EVALUATE_PROMPT = """Evaluate how promising this reasoning path is for solving the problem.

Problem: {query}

Reasoning path:
{path}

Rate the path from 0-10:
- 0: Completely wrong or irrelevant
- 5: On the right track but incomplete
- 10: Likely leads to correct answer

Score (just the number):"""

    def __init__(
        self,
        llm: LLMProtocol,
        evaluator: Optional[EvaluatorProtocol] = None,
        branching_factor: int = 3,
        max_depth: int = 4,
        beam_width: int = 2,
        verbose: bool = False,
    ):
        """Initialize ToT reasoner."""
        super().__init__(llm, verbose)
        self.evaluator = evaluator
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.beam_width = beam_width
        self._node_counter = 0

    async def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute tree of thoughts reasoning."""
        self._node_counter = 0

        # Initialize root
        root = ThoughtNode(
            id=self._next_id(),
            content=f"Problem: {query}",
            depth=0,
            score=1.0,
        )

        # Build tree using beam search
        all_nodes: Dict[str, ThoughtNode] = {root.id: root}
        current_beam = [root]

        for depth in range(self.max_depth):
            self._log(f"ToT Depth {depth + 1}/{self.max_depth}")

            # Generate children for each node in beam
            candidates = []
            for node in current_beam:
                children = await self._expand_node(node, query, all_nodes)
                candidates.extend(children)

            if not candidates:
                break

            # Evaluate and select top candidates
            scored_candidates = []
            for child in candidates:
                score = await self._evaluate_path(child, query, all_nodes)
                child.score = score
                scored_candidates.append((score, child))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            current_beam = [c for _, c in scored_candidates[: self.beam_width]]

            # Add to tree
            for node in current_beam:
                all_nodes[node.id] = node

        # Extract all complete paths
        all_paths = self._extract_paths(root, all_nodes)

        # Find best path
        best_path = max(all_paths, key=lambda p: p.total_score) if all_paths else None

        # Generate final answer from best path
        answer = ""
        if best_path:
            answer = await self._generate_answer(query, best_path)
            best_path.final_answer = answer

        return ReasoningResult(
            query=query,
            strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            best_path=best_path,
            all_paths=all_paths,
            answer=answer,
            confidence=best_path.total_score if best_path else 0.0,
            iterations=len(all_nodes),
            metadata={"tree_size": len(all_nodes)},
        )

    async def _expand_node(
        self,
        node: ThoughtNode,
        query: str,
        all_nodes: Dict[str, ThoughtNode],
    ) -> List[ThoughtNode]:
        """Generate child thoughts for a node."""
        path_str = self._get_path_string(node, all_nodes)

        children = []
        for _ in range(self.branching_factor):
            prompt = self.THOUGHT_PROMPT.format(query=query, path=path_str)
            thought = await self.llm.generate(prompt, temperature=0.9)

            child = ThoughtNode(
                id=self._next_id(),
                content=thought.strip(),
                parent_id=node.id,
                depth=node.depth + 1,
            )
            children.append(child)
            node.children_ids.append(child.id)

        return children

    async def _evaluate_path(
        self,
        node: ThoughtNode,
        query: str,
        all_nodes: Dict[str, ThoughtNode],
    ) -> float:
        """Evaluate a reasoning path."""
        if self.evaluator:
            path_str = self._get_path_string(node, all_nodes)
            return await self.evaluator.evaluate(path_str, {"query": query})

        # Use LLM for evaluation
        path_str = self._get_path_string(node, all_nodes)
        prompt = self.EVALUATE_PROMPT.format(query=query, path=path_str)
        response = await self.llm.generate(prompt, temperature=0.3)

        try:
            score = float(response.strip().split()[0]) / 10.0
            return min(1.0, max(0.0, score))
        except (ValueError, IndexError):
            return 0.5

    def _get_path_string(
        self,
        node: ThoughtNode,
        all_nodes: Dict[str, ThoughtNode],
    ) -> str:
        """Get string representation of path to node."""
        path = []
        current = node

        while current:
            path.append(current.content)
            if current.parent_id:
                current = all_nodes.get(current.parent_id)
            else:
                break

        path.reverse()
        return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(path))

    def _extract_paths(
        self,
        root: ThoughtNode,
        all_nodes: Dict[str, ThoughtNode],
    ) -> List[ReasoningPath]:
        """Extract all complete paths from tree."""
        paths = []

        def dfs(node: ThoughtNode, current_path: List[ThoughtNode]):
            current_path.append(node)

            if node.is_leaf() or node.depth >= self.max_depth:
                # Complete path
                path = ReasoningPath(
                    nodes=list(current_path),
                    total_score=sum(n.score for n in current_path) / len(current_path),
                    is_complete=True,
                )
                paths.append(path)
            else:
                for child_id in node.children_ids:
                    child = all_nodes.get(child_id)
                    if child:
                        dfs(child, current_path)

            current_path.pop()

        dfs(root, [])
        return paths

    async def _generate_answer(self, query: str, path: ReasoningPath) -> str:
        """Generate final answer from reasoning path."""
        path_str = "\n".join(f"{i + 1}. {n.content}" for i, n in enumerate(path.nodes))
        prompt = f"""Based on the following reasoning, provide a concise final answer.

Problem: {query}

Reasoning:
{path_str}

Final Answer:"""

        return await self.llm.generate(prompt, temperature=0.3)

    def _next_id(self) -> str:
        """Generate next node ID."""
        self._node_counter += 1
        return f"node_{self._node_counter}"


# =============================================================================
# Reflexion Reasoner
# =============================================================================


class ReflexionReasoner(BaseReasoner):
    """
    Reflexion reasoning with self-reflection and iterative improvement.

    Generates answer, reflects on it, and improves iteratively.
    """

    INITIAL_PROMPT = """Answer the following question step by step.

Question: {query}

{context}

Provide a detailed answer:"""

    REFLECT_PROMPT = """Reflect on your previous answer and identify any issues.

Question: {query}

Your previous answer:
{answer}

Reflect on:
1. Is the answer correct and complete?
2. Are there any logical errors?
3. What could be improved?

Reflection:"""

    IMPROVE_PROMPT = """Based on your reflection, provide an improved answer.

Question: {query}

Previous answer:
{answer}

Reflection:
{reflection}

Improved answer:"""

    def __init__(
        self,
        llm: LLMProtocol,
        max_iterations: int = 3,
        improvement_threshold: float = 0.1,
        verbose: bool = False,
    ):
        """Initialize Reflexion reasoner."""
        super().__init__(llm, verbose)
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold

    async def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute reflexion reasoning."""
        context_str = self._format_context(context) if context else ""

        # Initial answer
        prompt = self.INITIAL_PROMPT.format(query=query, context=context_str)
        current_answer = await self.llm.generate(prompt, temperature=0.7)

        all_paths = []
        iterations = 0

        for i in range(self.max_iterations):
            iterations = i + 1
            self._log(f"Reflexion iteration {iterations}")

            # Create path for current answer
            path = ReasoningPath(
                nodes=[
                    ThoughtNode(
                        id=f"answer_{i}",
                        content=current_answer,
                        depth=i,
                    )
                ],
                total_score=0.0,
                is_complete=True,
                final_answer=current_answer,
            )
            all_paths.append(path)

            # Reflect on answer
            feedback = await self._reflect(query, current_answer)

            # Score based on reflection
            path.total_score = feedback.confidence

            # Check if improvement needed
            if feedback.is_correct or i == self.max_iterations - 1:
                break

            # Improve answer
            improved = await self._improve(query, current_answer, feedback)

            # Check if actually improved
            if improved.strip() == current_answer.strip():
                break

            current_answer = improved

        # Best path is the last one (most refined)
        best_path = all_paths[-1] if all_paths else None

        return ReasoningResult(
            query=query,
            strategy=ReasoningStrategy.REFLEXION,
            best_path=best_path,
            all_paths=all_paths,
            answer=current_answer,
            confidence=best_path.total_score if best_path else 0.0,
            iterations=iterations,
        )

    async def _reflect(self, query: str, answer: str) -> ReflectionFeedback:
        """Generate reflection on answer."""
        prompt = self.REFLECT_PROMPT.format(query=query, answer=answer)
        reflection = await self.llm.generate(prompt, temperature=0.5)

        # Analyze reflection
        is_correct = self._analyze_correctness(reflection)
        issues = self._extract_issues(reflection)
        suggestions = self._extract_suggestions(reflection)

        # Estimate confidence
        confidence = 0.9 if is_correct else 0.5 - (len(issues) * 0.1)
        confidence = max(0.0, min(1.0, confidence))

        return ReflectionFeedback(
            is_correct=is_correct,
            issues=issues,
            suggestions=suggestions,
            improved_reasoning=reflection,
            confidence=confidence,
        )

    async def _improve(
        self,
        query: str,
        answer: str,
        feedback: ReflectionFeedback,
    ) -> str:
        """Generate improved answer based on reflection."""
        prompt = self.IMPROVE_PROMPT.format(
            query=query,
            answer=answer,
            reflection=feedback.improved_reasoning,
        )
        return await self.llm.generate(prompt, temperature=0.5)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt."""
        if "documents" in context:
            docs = context["documents"]
            return "Context:\n" + "\n".join(f"- {d}" for d in docs[:5])
        return ""

    def _analyze_correctness(self, reflection: str) -> bool:
        """Analyze if reflection indicates correctness."""
        negative_indicators = [
            "incorrect",
            "wrong",
            "error",
            "mistake",
            "issue",
            "problem",
            "missing",
            "incomplete",
        ]
        reflection_lower = reflection.lower()
        return not any(ind in reflection_lower for ind in negative_indicators)

    def _extract_issues(self, reflection: str) -> List[str]:
        """Extract issues from reflection."""
        issues = []
        lines = reflection.split("\n")
        for line in lines:
            if any(kw in line.lower() for kw in ["issue", "error", "wrong", "problem"]):
                issues.append(line.strip())
        return issues

    def _extract_suggestions(self, reflection: str) -> List[str]:
        """Extract suggestions from reflection."""
        suggestions = []
        lines = reflection.split("\n")
        for line in lines:
            if any(kw in line.lower() for kw in ["should", "could", "improve", "better"]):
                suggestions.append(line.strip())
        return suggestions


# =============================================================================
# Self-Consistency Reasoner
# =============================================================================


class SelfConsistencyReasoner(BaseReasoner):
    """
    Self-Consistency reasoning with voting.

    Generates multiple reasoning paths and uses majority voting for the answer.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        num_samples: int = 5,
        temperature: float = 0.7,
        verbose: bool = False,
    ):
        """Initialize self-consistency reasoner."""
        super().__init__(llm, verbose)
        self.num_samples = num_samples
        self.temperature = temperature
        self._cot_reasoner = ChainOfThoughtReasoner(llm, verbose)

    async def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute self-consistency reasoning."""
        # Generate multiple reasoning paths
        tasks = [self._cot_reasoner.reason(query, context) for _ in range(self.num_samples)]
        results = await asyncio.gather(*tasks)

        # Collect all paths and answers
        all_paths = []
        answers: Dict[str, List[ReasoningPath]] = {}

        for result in results:
            if result.best_path:
                all_paths.append(result.best_path)
                answer = self._normalize_answer(result.answer)
                if answer not in answers:
                    answers[answer] = []
                answers[answer].append(result.best_path)

        # Find majority answer
        best_answer = ""
        best_count = 0
        best_paths: List[ReasoningPath] = []

        for answer, paths in answers.items():
            if len(paths) > best_count:
                best_count = len(paths)
                best_answer = answer
                best_paths = paths

        # Calculate confidence based on agreement
        confidence = best_count / self.num_samples if self.num_samples > 0 else 0.0

        # Select best path from majority
        best_path = max(best_paths, key=lambda p: p.total_score) if best_paths else None

        return ReasoningResult(
            query=query,
            strategy=ReasoningStrategy.SELF_CONSISTENCY,
            best_path=best_path,
            all_paths=all_paths,
            answer=best_answer,
            confidence=confidence,
            iterations=self.num_samples,
            metadata={
                "vote_distribution": {a: len(p) for a, p in answers.items()},
                "majority_count": best_count,
            },
        )

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Simple normalization - could be more sophisticated
        return answer.strip().lower()


# =============================================================================
# Unified Reasoning Engine
# =============================================================================


class ReasoningEngine:
    """
    Unified reasoning engine that selects and applies appropriate strategies.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        default_strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT,
        verbose: bool = False,
    ):
        """Initialize reasoning engine."""
        self.llm = llm
        self.default_strategy = default_strategy
        self.verbose = verbose

        # Initialize reasoners
        self._reasoners: Dict[ReasoningStrategy, BaseReasoner] = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: ChainOfThoughtReasoner(llm, verbose),
            ReasoningStrategy.TREE_OF_THOUGHTS: TreeOfThoughtsReasoner(llm, verbose=verbose),
            ReasoningStrategy.REFLEXION: ReflexionReasoner(llm, verbose=verbose),
            ReasoningStrategy.SELF_CONSISTENCY: SelfConsistencyReasoner(llm, verbose=verbose),
        }

    async def reason(
        self,
        query: str,
        strategy: Optional[ReasoningStrategy] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """
        Execute reasoning with specified or auto-selected strategy.

        Args:
            query: Query to reason about
            strategy: Reasoning strategy (auto-selected if None)
            context: Additional context

        Returns:
            ReasoningResult with answer and reasoning trace
        """
        # Select strategy
        selected = strategy or self._select_strategy(query, context)

        # Get reasoner
        reasoner = self._reasoners.get(selected)
        if not reasoner:
            reasoner = self._reasoners[ReasoningStrategy.CHAIN_OF_THOUGHT]

        # Execute reasoning
        return await reasoner.reason(query, context)

    def _select_strategy(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> ReasoningStrategy:
        """Auto-select appropriate strategy based on query characteristics."""
        query_lower = query.lower()

        # Complex multi-step problems -> Tree of Thoughts
        if any(kw in query_lower for kw in ["complex", "multi-step", "analyze"]):
            return ReasoningStrategy.TREE_OF_THOUGHTS

        # Questions needing verification -> Reflexion
        if any(kw in query_lower for kw in ["verify", "check", "correct"]):
            return ReasoningStrategy.REFLEXION

        # Factual questions needing confidence -> Self-Consistency
        if any(kw in query_lower for kw in ["what is", "who is", "when"]):
            return ReasoningStrategy.SELF_CONSISTENCY

        return self.default_strategy

    def add_reasoner(
        self,
        strategy: ReasoningStrategy,
        reasoner: BaseReasoner,
    ) -> None:
        """Add or replace a reasoner for a strategy."""
        self._reasoners[strategy] = reasoner

    def get_reasoner(self, strategy: ReasoningStrategy) -> Optional[BaseReasoner]:
        """Get reasoner for a strategy."""
        return self._reasoners.get(strategy)
