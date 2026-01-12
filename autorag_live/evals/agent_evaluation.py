"""
Agent-specific evaluation framework for Agentic RAG systems.

This module provides comprehensive evaluation capabilities specifically designed
for agentic RAG systems, including:
- Multi-hop reasoning evaluation
- Tool usage effectiveness metrics
- Agent trajectory analysis
- Memory utilization assessment
- Multi-agent coordination metrics
- CRUD operation benchmarks
- End-to-end agent quality metrics
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, TypeVar

import numpy as np

# =============================================================================
# Core Evaluation Types
# =============================================================================


class EvaluationDimension(Enum):
    """Dimensions for agent evaluation."""

    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    REASONING = "reasoning"
    TOOL_USAGE = "tool_usage"
    EFFICIENCY = "efficiency"
    MEMORY = "memory"
    COORDINATION = "coordination"
    FAITHFULNESS = "faithfulness"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"


class HopType(Enum):
    """Types of reasoning hops in multi-hop QA."""

    BRIDGE = "bridge"  # A implies B, B implies C
    COMPARISON = "comparison"  # Compare entities
    COMPOSITIONAL = "compositional"  # Compose multiple facts
    INFERENCE = "inference"  # Logical inference required
    TEMPORAL = "temporal"  # Time-based reasoning


@dataclass
class EvaluationMetric:
    """A single evaluation metric result."""

    name: str
    value: float
    dimension: EvaluationDimension
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate metric value."""
        if not 0.0 <= self.value <= 1.0:
            # Allow values outside [0,1] but warn
            self.metadata["out_of_range"] = True


@dataclass
class AgentTrajectory:
    """Represents an agent's execution trajectory."""

    query: str
    steps: List[Dict[str, Any]]
    final_answer: str
    ground_truth: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chains: List[str] = field(default_factory=list)
    memory_accesses: List[Dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""

    overall_score: float
    metrics: List[EvaluationMetric]
    dimension_scores: Dict[EvaluationDimension, float]
    trajectories_analyzed: int
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "overall_score": self.overall_score,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "dimension": m.dimension.value,
                    "confidence": m.confidence,
                }
                for m in self.metrics
            ],
            "dimension_scores": {k.value: v for k, v in self.dimension_scores.items()},
            "trajectories_analyzed": self.trajectories_analyzed,
            "timestamp": self.timestamp.isoformat(),
            "recommendations": self.recommendations,
            "detailed_analysis": self.detailed_analysis,
        }


# =============================================================================
# Evaluation Protocols
# =============================================================================


class MetricEvaluator(Protocol):
    """Protocol for metric evaluators."""

    def evaluate(self, trajectory: AgentTrajectory) -> List[EvaluationMetric]:
        """Evaluate a single trajectory."""
        ...

    def batch_evaluate(self, trajectories: List[AgentTrajectory]) -> List[List[EvaluationMetric]]:
        """Evaluate multiple trajectories."""
        ...


T = TypeVar("T")


class BaseEvaluator(ABC, Generic[T]):
    """Abstract base class for evaluators."""

    @abstractmethod
    def evaluate(self, data: T) -> List[EvaluationMetric]:
        """Evaluate input data."""
        pass

    def batch_evaluate(self, data_list: List[T]) -> List[List[EvaluationMetric]]:
        """Evaluate multiple inputs."""
        return [self.evaluate(data) for data in data_list]


# =============================================================================
# Multi-Hop Reasoning Evaluation
# =============================================================================


@dataclass
class MultiHopQuestion:
    """Multi-hop question with ground truth."""

    question: str
    answer: str
    supporting_facts: List[str]
    hop_types: List[HopType]
    num_hops: int
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiHopEvaluator(BaseEvaluator[AgentTrajectory]):
    """Evaluator for multi-hop reasoning capabilities."""

    def __init__(
        self,
        hop_weight: float = 0.3,
        path_weight: float = 0.3,
        answer_weight: float = 0.4,
    ):
        """
        Initialize multi-hop evaluator.

        Args:
            hop_weight: Weight for hop completion score
            path_weight: Weight for reasoning path quality
            answer_weight: Weight for final answer correctness
        """
        self.hop_weight = hop_weight
        self.path_weight = path_weight
        self.answer_weight = answer_weight

    def evaluate(self, trajectory: AgentTrajectory) -> List[EvaluationMetric]:
        """Evaluate multi-hop reasoning in a trajectory."""
        metrics = []

        # Hop completion score
        hop_score = self._evaluate_hop_completion(trajectory)
        metrics.append(
            EvaluationMetric(
                name="hop_completion",
                value=hop_score,
                dimension=EvaluationDimension.REASONING,
                metadata={"description": "Fraction of reasoning hops completed"},
            )
        )

        # Reasoning path quality
        path_score = self._evaluate_reasoning_path(trajectory)
        metrics.append(
            EvaluationMetric(
                name="reasoning_path_quality",
                value=path_score,
                dimension=EvaluationDimension.REASONING,
                metadata={"description": "Quality of reasoning chain"},
            )
        )

        # Answer correctness
        answer_score = self._evaluate_answer_correctness(trajectory)
        metrics.append(
            EvaluationMetric(
                name="answer_correctness",
                value=answer_score,
                dimension=EvaluationDimension.ACCURACY,
                metadata={"description": "Correctness of final answer"},
            )
        )

        # Supporting fact coverage
        fact_coverage = self._evaluate_fact_coverage(trajectory)
        metrics.append(
            EvaluationMetric(
                name="supporting_fact_coverage",
                value=fact_coverage,
                dimension=EvaluationDimension.COMPLETENESS,
                metadata={"description": "Coverage of supporting facts"},
            )
        )

        return metrics

    def _evaluate_hop_completion(self, trajectory: AgentTrajectory) -> float:
        """Evaluate how many reasoning hops were completed."""
        if not trajectory.reasoning_chains:
            return 0.0

        # Count distinct reasoning steps
        num_chains = len(trajectory.reasoning_chains)
        expected_hops = trajectory.metadata.get("expected_hops", 2)

        return min(num_chains / expected_hops, 1.0)

    def _evaluate_reasoning_path(self, trajectory: AgentTrajectory) -> float:
        """Evaluate quality of reasoning path."""
        if not trajectory.reasoning_chains:
            return 0.0

        # Score based on reasoning chain characteristics
        scores = []

        for chain in trajectory.reasoning_chains:
            chain_score = 0.0

            # Check for logical connectors
            logical_connectors = [
                "therefore",
                "because",
                "since",
                "thus",
                "hence",
                "implies",
            ]
            if any(conn in chain.lower() for conn in logical_connectors):
                chain_score += 0.3

            # Check for evidence citation
            if "according to" in chain.lower() or "based on" in chain.lower():
                chain_score += 0.3

            # Check for conclusion
            if "conclude" in chain.lower() or "answer" in chain.lower():
                chain_score += 0.2

            # Length penalty for very short or very long chains
            words = len(chain.split())
            if 10 <= words <= 100:
                chain_score += 0.2
            elif words < 5:
                chain_score -= 0.1

            scores.append(max(0.0, min(1.0, chain_score)))

        return float(np.mean(scores)) if scores else 0.0

    def _evaluate_answer_correctness(self, trajectory: AgentTrajectory) -> float:
        """Evaluate correctness of the final answer."""
        if not trajectory.ground_truth:
            return 0.5  # Neutral if no ground truth

        answer = trajectory.final_answer.lower().strip()
        ground_truth = trajectory.ground_truth.lower().strip()

        # Exact match
        if answer == ground_truth:
            return 1.0

        # Fuzzy match - check if ground truth is in answer
        if ground_truth in answer:
            return 0.8

        # Token overlap
        answer_tokens = set(answer.split())
        truth_tokens = set(ground_truth.split())

        if not truth_tokens:
            return 0.0

        overlap = len(answer_tokens & truth_tokens) / len(truth_tokens)
        return overlap * 0.6

    def _evaluate_fact_coverage(self, trajectory: AgentTrajectory) -> float:
        """Evaluate coverage of supporting facts."""
        supporting_facts = trajectory.metadata.get("supporting_facts", [])
        if not supporting_facts:
            return 1.0  # No facts to cover

        # Check how many facts appear in reasoning
        all_reasoning = " ".join(trajectory.reasoning_chains).lower()
        covered = sum(1 for fact in supporting_facts if fact.lower() in all_reasoning)

        return covered / len(supporting_facts)


# =============================================================================
# Tool Usage Evaluation
# =============================================================================


@dataclass
class ToolCall:
    """Represents a tool call made by an agent."""

    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None


class ToolUsageEvaluator(BaseEvaluator[AgentTrajectory]):
    """Evaluator for tool usage effectiveness."""

    def __init__(self, tool_importance: Optional[Dict[str, float]] = None):
        """
        Initialize tool usage evaluator.

        Args:
            tool_importance: Optional weights for different tools
        """
        self.tool_importance = tool_importance or {}

    def evaluate(self, trajectory: AgentTrajectory) -> List[EvaluationMetric]:
        """Evaluate tool usage in trajectory."""
        metrics = []

        # Tool selection accuracy
        selection_score = self._evaluate_tool_selection(trajectory)
        metrics.append(
            EvaluationMetric(
                name="tool_selection_accuracy",
                value=selection_score,
                dimension=EvaluationDimension.TOOL_USAGE,
            )
        )

        # Tool success rate
        success_rate = self._evaluate_tool_success_rate(trajectory)
        metrics.append(
            EvaluationMetric(
                name="tool_success_rate",
                value=success_rate,
                dimension=EvaluationDimension.TOOL_USAGE,
            )
        )

        # Tool efficiency
        efficiency = self._evaluate_tool_efficiency(trajectory)
        metrics.append(
            EvaluationMetric(
                name="tool_efficiency",
                value=efficiency,
                dimension=EvaluationDimension.EFFICIENCY,
            )
        )

        # Tool result utilization
        utilization = self._evaluate_result_utilization(trajectory)
        metrics.append(
            EvaluationMetric(
                name="tool_result_utilization",
                value=utilization,
                dimension=EvaluationDimension.TOOL_USAGE,
            )
        )

        return metrics

    def _evaluate_tool_selection(self, trajectory: AgentTrajectory) -> float:
        """Evaluate whether appropriate tools were selected."""
        if not trajectory.tool_calls:
            # Check if tools should have been used
            query_lower = trajectory.query.lower()
            should_use_tools = any(
                kw in query_lower for kw in ["search", "find", "calculate", "retrieve", "look up"]
            )
            return 0.0 if should_use_tools else 1.0

        # Score based on tool relevance to query
        query_keywords = set(trajectory.query.lower().split())
        relevant_tools = 0

        for call in trajectory.tool_calls:
            tool_name = call.get("tool_name", "").lower()
            # Simple relevance check
            if any(kw in tool_name for kw in query_keywords):
                relevant_tools += 1
            elif call.get("success", False):
                # Assume successful calls were relevant
                relevant_tools += 0.5

        return min(relevant_tools / len(trajectory.tool_calls), 1.0)

    def _evaluate_tool_success_rate(self, trajectory: AgentTrajectory) -> float:
        """Calculate tool call success rate."""
        if not trajectory.tool_calls:
            return 1.0

        successful = sum(1 for call in trajectory.tool_calls if call.get("success", False))
        return successful / len(trajectory.tool_calls)

    def _evaluate_tool_efficiency(self, trajectory: AgentTrajectory) -> float:
        """Evaluate efficiency of tool usage."""
        if not trajectory.tool_calls:
            return 1.0

        # Penalize redundant calls
        unique_calls = set()
        for call in trajectory.tool_calls:
            key = (call.get("tool_name"), str(call.get("arguments", {})))
            unique_calls.add(key)

        redundancy_ratio = len(unique_calls) / len(trajectory.tool_calls)

        # Penalize excessive tool calls
        expected_calls = trajectory.metadata.get("expected_tool_calls", 3)
        call_ratio = min(expected_calls / len(trajectory.tool_calls), 1.0)

        return (redundancy_ratio + call_ratio) / 2

    def _evaluate_result_utilization(self, trajectory: AgentTrajectory) -> float:
        """Evaluate how well tool results were utilized."""
        if not trajectory.tool_calls:
            return 1.0

        utilized = 0
        final_answer = trajectory.final_answer.lower()

        for call in trajectory.tool_calls:
            result = str(call.get("result", "")).lower()
            if result and any(token in final_answer for token in result.split()[:5]):
                utilized += 1

        return utilized / len(trajectory.tool_calls)


# =============================================================================
# Memory Utilization Evaluation
# =============================================================================


class MemoryEvaluator(BaseEvaluator[AgentTrajectory]):
    """Evaluator for agent memory system effectiveness."""

    def evaluate(self, trajectory: AgentTrajectory) -> List[EvaluationMetric]:
        """Evaluate memory utilization."""
        metrics = []

        # Memory access efficiency
        access_efficiency = self._evaluate_access_efficiency(trajectory)
        metrics.append(
            EvaluationMetric(
                name="memory_access_efficiency",
                value=access_efficiency,
                dimension=EvaluationDimension.MEMORY,
            )
        )

        # Context relevance
        context_relevance = self._evaluate_context_relevance(trajectory)
        metrics.append(
            EvaluationMetric(
                name="memory_context_relevance",
                value=context_relevance,
                dimension=EvaluationDimension.MEMORY,
            )
        )

        # Memory consistency
        consistency = self._evaluate_memory_consistency(trajectory)
        metrics.append(
            EvaluationMetric(
                name="memory_consistency",
                value=consistency,
                dimension=EvaluationDimension.COHERENCE,
            )
        )

        return metrics

    def _evaluate_access_efficiency(self, trajectory: AgentTrajectory) -> float:
        """Evaluate efficiency of memory accesses."""
        if not trajectory.memory_accesses:
            return 0.5  # Neutral if no memory system

        # Check for cache hits and retrieval efficiency
        hits = sum(1 for access in trajectory.memory_accesses if access.get("hit", False))
        return hits / len(trajectory.memory_accesses) if trajectory.memory_accesses else 0.0

    def _evaluate_context_relevance(self, trajectory: AgentTrajectory) -> float:
        """Evaluate relevance of retrieved memory contexts."""
        if not trajectory.memory_accesses:
            return 0.5

        relevant = sum(
            1 for access in trajectory.memory_accesses if access.get("relevance_score", 0) > 0.5
        )
        return relevant / len(trajectory.memory_accesses)

    def _evaluate_memory_consistency(self, trajectory: AgentTrajectory) -> float:
        """Evaluate consistency of memory state."""
        # Check for contradictions in memory accesses
        contents = [
            access.get("content", "")
            for access in trajectory.memory_accesses
            if access.get("content")
        ]

        if len(contents) < 2:
            return 1.0

        # Simple consistency check - no exact contradictions
        contradictions = 0
        for i, content1 in enumerate(contents):
            for content2 in contents[i + 1 :]:
                if self._are_contradictory(content1, content2):
                    contradictions += 1

        max_contradictions = len(contents) * (len(contents) - 1) / 2
        return 1.0 - (contradictions / max_contradictions) if max_contradictions > 0 else 1.0

    def _are_contradictory(self, text1: str, text2: str) -> bool:
        """Check if two texts are contradictory."""
        # Simple negation check
        negations = ["not", "no", "never", "false", "incorrect"]
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # If one has negation and they share key terms, might be contradictory
        has_neg1 = any(neg in words1 for neg in negations)
        has_neg2 = any(neg in words2 for neg in negations)

        if has_neg1 != has_neg2:
            # One is negated - check for significant overlap
            overlap = len(words1 & words2) / max(len(words1), len(words2))
            return overlap > 0.5

        return False


# =============================================================================
# Multi-Agent Coordination Evaluation
# =============================================================================


@dataclass
class AgentInteraction:
    """Represents an interaction between agents."""

    source_agent: str
    target_agent: str
    message_type: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiAgentEvaluator(BaseEvaluator[List[AgentTrajectory]]):
    """Evaluator for multi-agent coordination."""

    def evaluate(self, trajectories: List[AgentTrajectory]) -> List[EvaluationMetric]:
        """Evaluate multi-agent coordination."""
        metrics = []

        # Consensus quality
        consensus_score = self._evaluate_consensus(trajectories)
        metrics.append(
            EvaluationMetric(
                name="consensus_quality",
                value=consensus_score,
                dimension=EvaluationDimension.COORDINATION,
            )
        )

        # Role specialization
        specialization = self._evaluate_specialization(trajectories)
        metrics.append(
            EvaluationMetric(
                name="role_specialization",
                value=specialization,
                dimension=EvaluationDimension.COORDINATION,
            )
        )

        # Communication efficiency
        comm_efficiency = self._evaluate_communication(trajectories)
        metrics.append(
            EvaluationMetric(
                name="communication_efficiency",
                value=comm_efficiency,
                dimension=EvaluationDimension.EFFICIENCY,
            )
        )

        # Conflict resolution
        conflict_resolution = self._evaluate_conflict_resolution(trajectories)
        metrics.append(
            EvaluationMetric(
                name="conflict_resolution",
                value=conflict_resolution,
                dimension=EvaluationDimension.COORDINATION,
            )
        )

        return metrics

    def _evaluate_consensus(self, trajectories: List[AgentTrajectory]) -> float:
        """Evaluate quality of consensus among agents."""
        if len(trajectories) < 2:
            return 1.0

        answers = [t.final_answer.lower().strip() for t in trajectories]

        # Calculate agreement ratio
        answer_counts: Dict[str, int] = {}
        for answer in answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1

        max_agreement = max(answer_counts.values())
        return max_agreement / len(trajectories)

    def _evaluate_specialization(self, trajectories: List[AgentTrajectory]) -> float:
        """Evaluate how well agents specialized in their roles."""
        if not trajectories:
            return 0.0

        # Check tool usage diversity
        all_tools = set()
        agent_tools: List[set] = []

        for trajectory in trajectories:
            tools = {call.get("tool_name") for call in trajectory.tool_calls}
            agent_tools.append(tools)
            all_tools.update(tools)

        if not all_tools:
            return 0.5  # No tools used

        # Calculate specialization - agents should use different tools
        if len(agent_tools) < 2:
            return 1.0

        overlaps = []
        for i, tools1 in enumerate(agent_tools):
            for tools2 in agent_tools[i + 1 :]:
                if tools1 or tools2:
                    overlap = len(tools1 & tools2) / max(len(tools1 | tools2), 1)
                    overlaps.append(overlap)

        # Lower overlap = better specialization
        avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0
        return 1.0 - avg_overlap

    def _evaluate_communication(self, trajectories: List[AgentTrajectory]) -> float:
        """Evaluate communication efficiency between agents."""
        total_messages = 0
        useful_messages = 0

        for trajectory in trajectories:
            interactions = trajectory.metadata.get("interactions", [])
            total_messages += len(interactions)

            for interaction in interactions:
                # Consider message useful if it influenced the final answer
                if interaction.get("influenced_answer", False):
                    useful_messages += 1

        if total_messages == 0:
            return 1.0

        return useful_messages / total_messages

    def _evaluate_conflict_resolution(self, trajectories: List[AgentTrajectory]) -> float:
        """Evaluate how well conflicts were resolved."""
        conflicts = []
        resolutions = []

        for trajectory in trajectories:
            debate_rounds = trajectory.metadata.get("debate_rounds", [])
            for round_data in debate_rounds:
                if round_data.get("conflict"):
                    conflicts.append(round_data)
                    if round_data.get("resolved"):
                        resolutions.append(round_data)

        if not conflicts:
            return 1.0

        return len(resolutions) / len(conflicts)


# =============================================================================
# CRUD Operation Benchmarks
# =============================================================================


@dataclass
class CRUDBenchmarkResult:
    """Result of a CRUD operation benchmark."""

    operation: str
    success: bool
    latency_ms: float
    accuracy: float
    consistency: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CRUDEvaluator:
    """Evaluator for CRUD operations in agentic RAG."""

    def __init__(self):
        """Initialize CRUD evaluator."""
        self.operations = ["create", "read", "update", "delete"]

    def evaluate_create(
        self, trajectory: AgentTrajectory, expected_document: Dict[str, Any]
    ) -> CRUDBenchmarkResult:
        """Evaluate document creation capability."""
        created_doc = trajectory.metadata.get("created_document", {})

        # Check field accuracy
        correct_fields = 0
        total_fields = len(expected_document)

        for key, value in expected_document.items():
            if key in created_doc and created_doc[key] == value:
                correct_fields += 1

        accuracy = correct_fields / total_fields if total_fields > 0 else 0.0

        return CRUDBenchmarkResult(
            operation="create",
            success=accuracy > 0.8,
            latency_ms=trajectory.execution_time_ms,
            accuracy=accuracy,
            consistency=1.0,  # Single operation
        )

    def evaluate_read(
        self, trajectory: AgentTrajectory, expected_content: str
    ) -> CRUDBenchmarkResult:
        """Evaluate document retrieval capability."""
        retrieved = trajectory.final_answer

        # Calculate content similarity
        expected_tokens = set(expected_content.lower().split())
        retrieved_tokens = set(retrieved.lower().split())

        if not expected_tokens:
            accuracy = 1.0 if not retrieved_tokens else 0.0
        else:
            accuracy = len(expected_tokens & retrieved_tokens) / len(expected_tokens)

        return CRUDBenchmarkResult(
            operation="read",
            success=accuracy > 0.7,
            latency_ms=trajectory.execution_time_ms,
            accuracy=accuracy,
            consistency=1.0,
        )

    def evaluate_update(
        self,
        trajectory: AgentTrajectory,
        original: Dict[str, Any],
        expected_after: Dict[str, Any],
    ) -> CRUDBenchmarkResult:
        """Evaluate document update capability."""
        updated_doc = trajectory.metadata.get("updated_document", {})

        # Check that updates were applied correctly
        correct_updates = 0
        total_updates = 0

        for key in expected_after:
            if expected_after[key] != original.get(key):
                total_updates += 1
                if key in updated_doc and updated_doc[key] == expected_after[key]:
                    correct_updates += 1

        accuracy = correct_updates / total_updates if total_updates > 0 else 1.0

        # Check consistency - unchanged fields should remain
        unchanged_correct = 0
        unchanged_total = 0
        for key in original:
            if original[key] == expected_after.get(key):
                unchanged_total += 1
                if key in updated_doc and updated_doc[key] == original[key]:
                    unchanged_correct += 1

        consistency = unchanged_correct / unchanged_total if unchanged_total > 0 else 1.0

        return CRUDBenchmarkResult(
            operation="update",
            success=accuracy > 0.8 and consistency > 0.9,
            latency_ms=trajectory.execution_time_ms,
            accuracy=accuracy,
            consistency=consistency,
        )

    def evaluate_delete(self, trajectory: AgentTrajectory, document_id: str) -> CRUDBenchmarkResult:
        """Evaluate document deletion capability."""
        deleted_ids = trajectory.metadata.get("deleted_documents", [])

        success = document_id in deleted_ids
        accuracy = 1.0 if success else 0.0

        # Check that only target was deleted
        extra_deletions = len([d for d in deleted_ids if d != document_id])
        consistency = 1.0 if extra_deletions == 0 else max(0.0, 1.0 - extra_deletions * 0.2)

        return CRUDBenchmarkResult(
            operation="delete",
            success=success,
            latency_ms=trajectory.execution_time_ms,
            accuracy=accuracy,
            consistency=consistency,
        )

    def run_benchmark(
        self, trajectories: Dict[str, AgentTrajectory], test_data: Dict[str, Any]
    ) -> Dict[str, CRUDBenchmarkResult]:
        """Run full CRUD benchmark suite."""
        results = {}

        if "create" in trajectories:
            results["create"] = self.evaluate_create(
                trajectories["create"], test_data.get("expected_document", {})
            )

        if "read" in trajectories:
            results["read"] = self.evaluate_read(
                trajectories["read"], test_data.get("expected_content", "")
            )

        if "update" in trajectories:
            results["update"] = self.evaluate_update(
                trajectories["update"],
                test_data.get("original", {}),
                test_data.get("expected_after", {}),
            )

        if "delete" in trajectories:
            results["delete"] = self.evaluate_delete(
                trajectories["delete"], test_data.get("document_id", "")
            )

        return results


# =============================================================================
# Comprehensive Agent Evaluation Suite
# =============================================================================


class AgentEvaluationSuite:
    """Comprehensive evaluation suite for agentic RAG systems."""

    def __init__(
        self,
        evaluators: Optional[List[BaseEvaluator]] = None,
        weights: Optional[Dict[EvaluationDimension, float]] = None,
    ):
        """
        Initialize evaluation suite.

        Args:
            evaluators: List of evaluators to use
            weights: Dimension weights for overall score
        """
        self.evaluators = evaluators or [
            MultiHopEvaluator(),
            ToolUsageEvaluator(),
            MemoryEvaluator(),
        ]
        self.weights = weights or {
            EvaluationDimension.ACCURACY: 0.2,
            EvaluationDimension.RELEVANCE: 0.15,
            EvaluationDimension.REASONING: 0.2,
            EvaluationDimension.TOOL_USAGE: 0.15,
            EvaluationDimension.EFFICIENCY: 0.1,
            EvaluationDimension.MEMORY: 0.1,
            EvaluationDimension.COORDINATION: 0.1,
        }

    def evaluate(self, trajectory: AgentTrajectory) -> EvaluationReport:
        """Run full evaluation on a trajectory."""
        all_metrics = []

        for evaluator in self.evaluators:
            try:
                metrics = evaluator.evaluate(trajectory)
                all_metrics.extend(metrics)
            except Exception as e:
                # Log error but continue
                all_metrics.append(
                    EvaluationMetric(
                        name=f"error_{type(evaluator).__name__}",
                        value=0.0,
                        dimension=EvaluationDimension.ACCURACY,
                        metadata={"error": str(e)},
                    )
                )

        # Aggregate by dimension
        dimension_scores = self._aggregate_by_dimension(all_metrics)

        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_scores, all_metrics)

        return EvaluationReport(
            overall_score=overall_score,
            metrics=all_metrics,
            dimension_scores=dimension_scores,
            trajectories_analyzed=1,
            recommendations=recommendations,
            detailed_analysis={
                "trajectory_id": trajectory.metadata.get("id"),
                "query": trajectory.query,
                "num_steps": len(trajectory.steps),
                "num_tool_calls": len(trajectory.tool_calls),
            },
        )

    def batch_evaluate(self, trajectories: List[AgentTrajectory]) -> EvaluationReport:
        """Evaluate multiple trajectories and aggregate results."""
        all_metrics = []
        all_dimension_scores: Dict[EvaluationDimension, List[float]] = {
            dim: [] for dim in EvaluationDimension
        }

        for trajectory in trajectories:
            report = self.evaluate(trajectory)
            all_metrics.extend(report.metrics)
            for dim, score in report.dimension_scores.items():
                all_dimension_scores[dim].append(score)

        # Average dimension scores
        avg_dimension_scores = {
            dim: float(np.mean(scores)) if scores else 0.0
            for dim, scores in all_dimension_scores.items()
        }

        # Calculate overall score
        overall_score = self._calculate_overall_score(avg_dimension_scores)

        # Generate aggregate recommendations
        recommendations = self._generate_recommendations(avg_dimension_scores, all_metrics)

        return EvaluationReport(
            overall_score=overall_score,
            metrics=all_metrics,
            dimension_scores=avg_dimension_scores,
            trajectories_analyzed=len(trajectories),
            recommendations=recommendations,
            detailed_analysis={
                "num_trajectories": len(trajectories),
                "avg_steps": float(np.mean([len(t.steps) for t in trajectories])),
                "avg_tool_calls": float(np.mean([len(t.tool_calls) for t in trajectories])),
            },
        )

    def _aggregate_by_dimension(
        self, metrics: List[EvaluationMetric]
    ) -> Dict[EvaluationDimension, float]:
        """Aggregate metrics by dimension."""
        dimension_values: Dict[EvaluationDimension, List[float]] = {
            dim: [] for dim in EvaluationDimension
        }

        for metric in metrics:
            dimension_values[metric.dimension].append(metric.value)

        return {
            dim: float(np.mean(values)) if values else 0.0
            for dim, values in dimension_values.items()
        }

    def _calculate_overall_score(self, dimension_scores: Dict[EvaluationDimension, float]) -> float:
        """Calculate weighted overall score."""
        total_weight = sum(self.weights.values())
        weighted_sum = sum(
            dimension_scores.get(dim, 0.0) * weight for dim, weight in self.weights.items()
        )
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _generate_recommendations(
        self,
        dimension_scores: Dict[EvaluationDimension, float],
        metrics: List[EvaluationMetric],
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Find low-scoring dimensions
        for dim, score in dimension_scores.items():
            if score < 0.5:
                recommendations.append(
                    f"Improve {dim.value}: Current score {score:.2f} is below threshold"
                )

        # Specific metric recommendations
        low_metrics = [m for m in metrics if m.value < 0.3]
        for metric in low_metrics[:5]:  # Top 5 lowest
            recommendations.append(
                f"Address low {metric.name}: {metric.value:.2f} - {metric.metadata.get('description', '')}"
            )

        return recommendations


# =============================================================================
# Faithfulness and Hallucination Detection
# =============================================================================


class FaithfulnessEvaluator(BaseEvaluator[AgentTrajectory]):
    """Evaluator for faithfulness and hallucination detection."""

    def __init__(self, llm_judge: Optional[Callable] = None):
        """
        Initialize faithfulness evaluator.

        Args:
            llm_judge: Optional LLM function for advanced evaluation
        """
        self.llm_judge = llm_judge

    def evaluate(self, trajectory: AgentTrajectory) -> List[EvaluationMetric]:
        """Evaluate faithfulness of the response."""
        metrics = []

        # Source attribution
        attribution_score = self._evaluate_attribution(trajectory)
        metrics.append(
            EvaluationMetric(
                name="source_attribution",
                value=attribution_score,
                dimension=EvaluationDimension.FAITHFULNESS,
            )
        )

        # Claim verification
        verification_score = self._evaluate_claim_verification(trajectory)
        metrics.append(
            EvaluationMetric(
                name="claim_verification",
                value=verification_score,
                dimension=EvaluationDimension.FAITHFULNESS,
            )
        )

        # Hallucination detection
        hallucination_score = self._detect_hallucination(trajectory)
        metrics.append(
            EvaluationMetric(
                name="hallucination_free",
                value=hallucination_score,
                dimension=EvaluationDimension.FAITHFULNESS,
            )
        )

        return metrics

    def _evaluate_attribution(self, trajectory: AgentTrajectory) -> float:
        """Evaluate whether claims are properly attributed to sources."""
        retrieved_content = " ".join(
            str(step.get("retrieved_docs", "")) for step in trajectory.steps
        )
        answer = trajectory.final_answer

        if not retrieved_content:
            return 0.0

        # Check for attribution markers
        attribution_markers = ["according to", "based on", "source:", "from"]
        has_attribution = any(marker in answer.lower() for marker in attribution_markers)

        # Check content overlap
        answer_tokens = set(answer.lower().split())
        source_tokens = set(retrieved_content.lower().split())
        overlap = len(answer_tokens & source_tokens) / max(len(answer_tokens), 1)

        base_score = 0.5 if has_attribution else 0.2
        return min(base_score + overlap * 0.5, 1.0)

    def _evaluate_claim_verification(self, trajectory: AgentTrajectory) -> float:
        """Verify claims in the answer against retrieved documents."""
        # Extract claims from answer
        claims = self._extract_claims(trajectory.final_answer)
        if not claims:
            return 1.0

        # Get source content
        source_content = " ".join(str(step.get("retrieved_docs", "")) for step in trajectory.steps)

        verified = 0
        for claim in claims:
            if self._verify_claim(claim, source_content):
                verified += 1

        return verified / len(claims)

    def _detect_hallucination(self, trajectory: AgentTrajectory) -> float:
        """Detect potential hallucinations in the response."""
        answer = trajectory.final_answer.lower()
        source_content = " ".join(
            str(step.get("retrieved_docs", "")) for step in trajectory.steps
        ).lower()

        # Identify entities and facts in answer
        answer_entities = self._extract_entities(answer)
        source_entities = self._extract_entities(source_content)

        if not answer_entities:
            return 1.0

        # Check how many answer entities appear in sources
        grounded = sum(1 for entity in answer_entities if entity in source_entities)
        return grounded / len(answer_entities)

    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from text."""
        # Simple sentence-based extraction
        sentences = text.replace(".", ". ").split(". ")
        claims = []

        for sentence in sentences:
            # Filter to likely factual claims
            if len(sentence.split()) > 5:
                # Skip questions and commands
                if not sentence.strip().endswith("?") and not sentence.strip().startswith(
                    ("please", "can you", "could you")
                ):
                    claims.append(sentence.strip())

        return claims

    def _verify_claim(self, claim: str, source: str) -> bool:
        """Verify a single claim against source content."""
        claim_tokens = set(claim.lower().split())
        source_tokens = set(source.lower().split())

        # Require significant overlap
        overlap = len(claim_tokens & source_tokens) / max(len(claim_tokens), 1)
        return overlap > 0.6

    def _extract_entities(self, text: str) -> set:
        """Extract entities from text."""
        # Simple capitalized word extraction
        words = text.split()
        entities = set()

        for word in words:
            # Numbers
            if word.replace(".", "").replace(",", "").isdigit():
                entities.add(word)
            # Names (simple heuristic)
            elif word and word[0].isupper() and len(word) > 1:
                entities.add(word.lower())

        return entities


# =============================================================================
# Factory Functions
# =============================================================================


def create_evaluation_suite(
    include_multi_hop: bool = True,
    include_tool_usage: bool = True,
    include_memory: bool = True,
    include_faithfulness: bool = True,
    custom_weights: Optional[Dict[EvaluationDimension, float]] = None,
) -> AgentEvaluationSuite:
    """
    Create a configured evaluation suite.

    Args:
        include_multi_hop: Include multi-hop reasoning evaluation
        include_tool_usage: Include tool usage evaluation
        include_memory: Include memory evaluation
        include_faithfulness: Include faithfulness evaluation
        custom_weights: Custom dimension weights

    Returns:
        Configured AgentEvaluationSuite
    """
    evaluators = []

    if include_multi_hop:
        evaluators.append(MultiHopEvaluator())

    if include_tool_usage:
        evaluators.append(ToolUsageEvaluator())

    if include_memory:
        evaluators.append(MemoryEvaluator())

    if include_faithfulness:
        evaluators.append(FaithfulnessEvaluator())

    return AgentEvaluationSuite(evaluators=evaluators, weights=custom_weights)


def evaluate_agent_run(
    query: str,
    answer: str,
    ground_truth: Optional[str] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    **metadata: Any,
) -> EvaluationReport:
    """
    Convenience function to evaluate a single agent run.

    Args:
        query: The input query
        answer: The agent's answer
        ground_truth: Optional correct answer
        steps: Agent execution steps
        tool_calls: Tool calls made
        **metadata: Additional metadata

    Returns:
        EvaluationReport with results
    """
    trajectory = AgentTrajectory(
        query=query,
        steps=steps or [],
        final_answer=answer,
        ground_truth=ground_truth,
        tool_calls=tool_calls or [],
        metadata=metadata,
    )

    suite = create_evaluation_suite()
    return suite.evaluate(trajectory)
