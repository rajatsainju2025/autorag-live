"""
Chain-of-thought planning and reasoning module for agents.

Implements reasoning traces, action planning, and goal decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ReasoningType(Enum):
    """Types of reasoning steps."""

    ANALYSIS = "analysis"
    PLANNING = "planning"
    DECISION = "decision"
    REFLECTION = "reflection"
    EXECUTION = "execution"


@dataclass
class ReasoningStep:
    """Single step in reasoning chain."""

    step_num: int
    step_type: ReasoningType
    content: str
    confidence: float = 1.0  # 0-1 confidence in this step
    dependencies: List[int] = field(default_factory=list)  # depends on step nums
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Export step as dictionary."""
        return {
            "step_num": self.step_num,
            "type": self.step_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }


@dataclass
class ReasoningTrace:
    """Complete reasoning chain for a query."""

    query: str
    steps: List[ReasoningStep] = field(default_factory=list)
    final_action: Optional[str] = None
    success: bool = False
    error: Optional[str] = None

    def add_step(
        self,
        step_type: ReasoningType,
        content: str,
        confidence: float = 1.0,
        dependencies: Optional[List[int]] = None,
    ) -> ReasoningStep:
        """Add reasoning step."""
        step = ReasoningStep(
            step_num=len(self.steps),
            step_type=step_type,
            content=content,
            confidence=confidence,
            dependencies=dependencies or [],
        )
        self.steps.append(step)
        return step

    def get_reasoning_chain(self) -> str:
        """Get formatted reasoning chain."""
        lines = [f"Query: {self.query}\n"]
        for step in self.steps:
            indent = "  " * len(step.dependencies)
            confidence_str = (
                f" (confidence: {step.confidence:.2f})" if step.confidence < 1.0 else ""
            )
            lines.append(f"{indent}[{step.step_type.value}] {step.content}{confidence_str}")
        if self.final_action:
            lines.append(f"\nFinal Action: {self.final_action}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export trace as dictionary."""
        return {
            "query": self.query,
            "steps": [step.to_dict() for step in self.steps],
            "final_action": self.final_action,
            "success": self.success,
            "error": self.error,
        }


class Reasoner:
    """
    Chain-of-thought reasoning engine.

    Performs structured reasoning and generates action plans.
    """

    def __init__(self, verbose: bool = False):
        """Initialize reasoner."""
        self.verbose = verbose
        self.reasoning_traces: List[ReasoningTrace] = []

    def reason_about_query(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """
        Generate reasoning trace for a query.

        Args:
            query: User query
            context: Optional context information

        Returns:
            Reasoning trace
        """
        trace = ReasoningTrace(query=query)
        context = context or {}

        # Step 1: Analyze query
        trace.add_step(
            ReasoningType.ANALYSIS,
            self._analyze_query(query),
            confidence=0.95,
        )

        # Step 2: Identify intent
        intent = self._identify_intent(query)
        trace.add_step(
            ReasoningType.ANALYSIS,
            f"Intent identified: {intent}",
            confidence=0.9,
            dependencies=[0],
        )

        # Step 3: Decompose into subgoals
        subgoals = self._decompose_goal(query, intent, context)
        for subgoal in subgoals:
            trace.add_step(
                ReasoningType.PLANNING,
                f"Subgoal: {subgoal}",
                confidence=0.85,
                dependencies=[1],
            )

        # Step 4: Plan actions
        planned_actions = self._plan_actions(query, intent, subgoals)
        for action in planned_actions:
            trace.add_step(
                ReasoningType.DECISION,
                f"Execute: {action}",
                confidence=0.8,
                dependencies=[2],
            )

        # Set final action as the first planned action
        if planned_actions:
            trace.final_action = planned_actions[0]
            trace.success = True

        self.reasoning_traces.append(trace)
        return trace

    def _analyze_query(self, query: str) -> str:
        """Analyze query structure and content."""
        # Extract key information
        words = query.lower().split()
        has_how = "how" in words
        has_why = "why" in words
        has_when = "when" in words

        question_type = "factual"
        if has_how:
            question_type = "procedural"
        elif has_why:
            question_type = "causal"
        elif has_when:
            question_type = "temporal"

        return f"Query type: {question_type}. Length: {len(words)} words. Key terms: {', '.join(words[:5])}"

    def _identify_intent(self, query: str) -> str:
        """Identify user's intent."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["summarize", "overview", "explain"]):
            return "get_summary"
        elif any(word in query_lower for word in ["list", "find", "search"]):
            return "find_items"
        elif any(word in query_lower for word in ["compare", "difference"]):
            return "compare"
        elif any(word in query_lower for word in ["how"]):
            return "learn_process"
        else:
            return "get_information"

    def _decompose_goal(self, query: str, intent: str, context: Dict[str, Any]) -> List[str]:
        """Decompose goal into subgoals."""
        subgoals = []

        # Common subgoals for any query
        subgoals.append("Search for relevant documents")

        if intent == "get_summary":
            subgoals.append("Extract key information")
            subgoals.append("Synthesize summary")
        elif intent == "find_items":
            subgoals.append("Filter results by relevance")
            subgoals.append("Rank by importance")
        elif intent == "compare":
            subgoals.append("Retrieve information for each item")
            subgoals.append("Identify comparison dimensions")
            subgoals.append("Compare point-by-point")
        elif intent == "learn_process":
            subgoals.append("Identify step sequence")
            subgoals.append("Extract details for each step")

        subgoals.append("Format final answer")

        return subgoals

    def _plan_actions(self, query: str, intent: str, subgoals: List[str]) -> List[str]:
        """Plan concrete actions for subgoals."""
        actions = []

        # Always start with retrieval
        actions.append("retrieve_documents")

        if intent == "get_summary":
            actions.append("synthesize_answer")
        elif intent == "find_items":
            actions.append("filter_and_rank_results")
        elif intent == "compare":
            actions.append("compare_items")
            actions.append("synthesize_comparison")
        elif intent == "learn_process":
            actions.append("extract_process_steps")
            actions.append("synthesize_process_guide")

        return actions

    def generate_action_plan(self, query: str, max_depth: int = 3) -> Dict[str, Any]:
        """
        Generate hierarchical action plan.

        Args:
            query: User query
            max_depth: Maximum planning depth

        Returns:
            Action plan structure
        """
        trace = self.reason_about_query(query)

        plan = {
            "query": query,
            "reasoning_trace": trace.to_dict(),
            "action_sequence": trace.final_action,
            "max_depth": max_depth,
            "estimated_steps": len(trace.steps),
        }

        return plan

    def validate_action_sequence(
        self, actions: List[str], available_tools: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate if action sequence is executable.

        Args:
            actions: Planned actions
            available_tools: Available tools

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check all actions exist
        available_set = set(available_tools)
        for action in actions:
            if action not in available_set:
                errors.append(f"Action '{action}' not available")

        # Check for circular dependencies (simplified)
        if len(actions) != len(set(actions)):
            errors.append("Action sequence contains duplicates")

        return len(errors) == 0, errors

    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get summary of all reasoning traces."""
        successful = sum(1 for t in self.reasoning_traces if t.success)
        failed = sum(1 for t in self.reasoning_traces if not t.success)
        avg_steps = (
            sum(len(t.steps) for t in self.reasoning_traces) / len(self.reasoning_traces)
            if self.reasoning_traces
            else 0
        )

        return {
            "total_traces": len(self.reasoning_traces),
            "successful": successful,
            "failed": failed,
            "avg_steps_per_trace": avg_steps,
            "traces": [t.to_dict() for t in self.reasoning_traces[-5:]],  # Last 5
        }


@dataclass
class PlanExecutionState:
    """State of plan execution."""

    plan_id: str
    step_index: int
    total_steps: int
    completed_steps: List[str] = field(default_factory=list)
    current_step: Optional[str] = None
    failed_steps: List[Tuple[str, str]] = field(default_factory=list)
    progress_percentage: float = 0.0

    def step_completed(self, step_name: str) -> None:
        """Mark step as completed."""
        self.completed_steps.append(step_name)
        self.step_index += 1
        self.progress_percentage = (self.step_index / self.total_steps) * 100

    def step_failed(self, step_name: str, error: str) -> None:
        """Mark step as failed."""
        self.failed_steps.append((step_name, error))

    def to_dict(self) -> Dict[str, Any]:
        """Export execution state."""
        return {
            "plan_id": self.plan_id,
            "progress": self.progress_percentage,
            "steps_completed": len(self.completed_steps),
            "total_steps": self.total_steps,
            "failed_steps": len(self.failed_steps),
            "current_step": self.current_step,
        }


class PlanExecutor:
    """Executes action plans generated by reasoner."""

    def __init__(self):
        """Initialize plan executor."""
        self.execution_states: Dict[str, PlanExecutionState] = {}

    def create_execution_state(self, plan_id: str, num_steps: int) -> PlanExecutionState:
        """Create new execution state for plan."""
        state = PlanExecutionState(plan_id=plan_id, step_index=0, total_steps=num_steps)
        self.execution_states[plan_id] = state
        return state

    def get_execution_state(self, plan_id: str) -> Optional[PlanExecutionState]:
        """Get execution state by ID."""
        return self.execution_states.get(plan_id)
