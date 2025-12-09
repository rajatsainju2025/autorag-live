"""
Agent reflection and self-critique system for AutoRAG-Live.

Enables agent self-evaluation, error analysis, and adaptive strategy adjustment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ReflectionType(Enum):
    """Types of reflection analysis."""

    ANSWER_QUALITY = "answer_quality"
    ERROR_ANALYSIS = "error_analysis"
    STRATEGY_ASSESSMENT = "strategy_assessment"
    CONSISTENCY_CHECK = "consistency_check"
    CONFIDENCE_ESTIMATION = "confidence_estimation"


@dataclass
class ReflectionCriteria:
    """Criteria for evaluating agent responses."""

    relevance: float = 0.0  # Is answer relevant to query?
    correctness: float = 0.0  # Is answer factually correct?
    completeness: float = 0.0  # Does it answer all parts?
    clarity: float = 0.0  # Is it clearly explained?
    grounding: float = 0.0  # Is it grounded in sources?

    def average_score(self) -> float:
        """Calculate average quality score."""
        scores = [
            self.relevance,
            self.correctness,
            self.completeness,
            self.clarity,
            self.grounding,
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def get_lowest_dimension(self) -> Tuple[str, float]:
        """Get lowest scoring dimension."""
        dimensions = {
            "relevance": self.relevance,
            "correctness": self.correctness,
            "completeness": self.completeness,
            "clarity": self.clarity,
            "grounding": self.grounding,
        }
        return min(dimensions.items(), key=lambda x: x[1])


@dataclass
class ErrorAnalysis:
    """Analysis of error or failure."""

    error_type: str  # "retrieval_failure", "reasoning_error", "synthesis_error"
    root_cause: str  # Description of root cause
    severity: float  # 0-1 scale
    suggested_strategy: str  # How to address this error
    learning_point: str  # What agent should learn

    def is_recoverable(self) -> bool:
        """Check if error is recoverable."""
        recoverable_types = [
            "incomplete_retrieval",
            "poor_ranking",
            "synthesis_gap",
        ]
        return self.error_type in recoverable_types


@dataclass
class ReflectionResult:
    """Result of reflection analysis."""

    reflection_type: ReflectionType
    query: str
    response: str
    timestamp: str

    criteria: Optional[ReflectionCriteria] = None
    error_analysis: Optional[ErrorAnalysis] = None
    confidence_score: float = 0.5
    uncertainty: float = 0.3
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_high_quality(self, threshold: float = 0.75) -> bool:
        """Check if response meets quality threshold."""
        if self.criteria:
            return self.criteria.average_score() >= threshold
        return False

    def should_retry(self) -> bool:
        """Determine if response should trigger retry."""
        if self.error_analysis and self.error_analysis.is_recoverable():
            return True
        if self.criteria and self.criteria.average_score() < 0.6:
            return True
        return False


class ReflectionEngine:
    """
    Enables agent self-reflection and self-critique.

    Analyzes agent responses for quality, detects errors, and recommends
    strategy adjustments.
    """

    def __init__(self, verbose: bool = False):
        """Initialize reflection engine."""
        self.verbose = verbose
        self.logger = logging.getLogger("ReflectionEngine")
        self.reflection_history: List[ReflectionResult] = []

    def evaluate_answer_quality(
        self,
        query: str,
        response: str,
        sources: Optional[List[str]] = None,
    ) -> ReflectionResult:
        """
        Evaluate quality of agent's answer.

        Args:
            query: Original user query
            response: Agent's response
            sources: Retrieved documents used

        Returns:
            ReflectionResult with quality assessment
        """
        from datetime import datetime

        criteria = ReflectionCriteria()

        # Relevance: Does answer address query?
        criteria.relevance = self._assess_relevance(query, response)

        # Correctness: Heuristic check (simple keyword matching)
        criteria.correctness = self._assess_correctness(response)

        # Completeness: Does it address all aspects?
        criteria.completeness = self._assess_completeness(query, response)

        # Clarity: Is explanation clear?
        criteria.clarity = self._assess_clarity(response)

        # Grounding: Is it based on sources?
        criteria.grounding = self._assess_grounding(response, sources or [])

        result = ReflectionResult(
            reflection_type=ReflectionType.ANSWER_QUALITY,
            query=query,
            response=response,
            criteria=criteria,
            timestamp=datetime.utcnow().isoformat(),
            confidence_score=criteria.average_score(),
        )

        # Generate recommendations
        dimension, score = criteria.get_lowest_dimension()
        if score < 0.7:
            result.recommendations.append(f"Improve {dimension}: currently at {score:.2f}")

        if criteria.grounding < 0.6:
            result.recommendations.append("Strengthen grounding with explicit citations")

        self.reflection_history.append(result)
        return result

    def analyze_error(
        self,
        query: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReflectionResult:
        """
        Analyze error or failure in agent execution.

        Args:
            query: Query that caused error
            error_message: Error description
            context: Additional context about failure

        Returns:
            ReflectionResult with error analysis
        """
        from datetime import datetime

        context = context or {}

        # Classify error type
        error_type = self._classify_error(error_message)

        # Determine root cause
        root_cause = self._determine_root_cause(error_type, error_message, context)

        # Assess severity
        severity = self._assess_severity(error_type)

        # Suggest recovery strategy
        strategy = self._suggest_recovery_strategy(error_type)

        analysis = ErrorAnalysis(
            error_type=error_type,
            root_cause=root_cause,
            severity=severity,
            suggested_strategy=strategy,
            learning_point=f"Avoid {root_cause} in future",
        )

        result = ReflectionResult(
            reflection_type=ReflectionType.ERROR_ANALYSIS,
            query=query,
            response=error_message,
            error_analysis=analysis,
            timestamp=datetime.utcnow().isoformat(),
            confidence_score=1.0 - severity,
            uncertainty=severity,
        )

        # Generate recommendations
        if analysis.is_recoverable():
            result.recommendations.append(f"Try: {strategy}")
        else:
            result.recommendations.append("Consider alternative approach")

        self.reflection_history.append(result)
        return result

    def assess_strategy(
        self,
        query: str,
        strategy_used: str,
        success: bool,
        metrics: Optional[Dict[str, float]] = None,
    ) -> ReflectionResult:
        """
        Assess effectiveness of strategy used.

        Args:
            query: Query being processed
            strategy_used: Strategy applied
            success: Whether strategy succeeded
            metrics: Performance metrics

        Returns:
            ReflectionResult with strategy assessment
        """
        from datetime import datetime

        metrics = metrics or {}

        result = ReflectionResult(
            reflection_type=ReflectionType.STRATEGY_ASSESSMENT,
            query=query,
            response=f"Strategy: {strategy_used}",
            confidence_score=1.0 if success else 0.3,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metrics,
        )

        if success:
            result.recommendations.append(f"Reinforce {strategy_used}")
            result.recommendations.append(f"Apply to similar queries (metrics: {metrics})")
        else:
            result.recommendations.append(f"Adjust or replace {strategy_used}")
            result.recommendations.append("Analyze why strategy failed for this query type")

        self.reflection_history.append(result)
        return result

    def estimate_confidence(
        self,
        response: str,
        num_supporting_sources: int = 1,
        agreement_score: float = 0.5,
    ) -> Tuple[float, float]:
        """
        Estimate confidence and uncertainty in response.

        Args:
            response: Generated response
            num_supporting_sources: How many sources support answer
            agreement_score: How much sources agree (0-1)

        Returns:
            Tuple of (confidence, uncertainty)
        """
        # Base confidence from response length and structure
        length_score = min(1.0, len(response.split()) / 100)

        # Source support
        source_score = min(1.0, num_supporting_sources / 3)

        # Agreement between sources
        agreement_weight = agreement_score

        # Combined confidence
        confidence = length_score * 0.3 + source_score * 0.4 + agreement_weight * 0.3

        # Uncertainty is inverse of confidence with additional factors
        uncertainty = (1.0 - confidence) * 0.6 + (1.0 - agreement_weight) * 0.4

        return min(1.0, confidence), min(1.0, uncertainty)

    def get_reflection_summary(self) -> Dict[str, Any]:
        """Get summary of all reflections."""
        if not self.reflection_history:
            return {
                "total_reflections": 0,
                "avg_confidence": 0.0,
                "improvement_trend": [],
            }

        total = len(self.reflection_history)
        avg_confidence = sum(r.confidence_score for r in self.reflection_history) / total

        # Calculate improvement trend
        recent = self.reflection_history[-10:]
        trend = [r.confidence_score for r in recent]

        return {
            "total_reflections": total,
            "avg_confidence": avg_confidence,
            "improvement_trend": trend,
            "high_quality_rate": sum(1 for r in self.reflection_history if r.is_high_quality())
            / total,
        }

    # Private helper methods

    def _assess_relevance(self, query: str, response: str) -> float:
        """Assess answer relevance to query."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        overlap = len(query_words & response_words)
        return min(1.0, overlap / max(len(query_words), 1) * 2)

    def _assess_correctness(self, response: str) -> float:
        """Assess answer correctness (heuristic)."""
        # Check for hedging language (might indicate uncertainty)
        hedges = [
            "might",
            "possibly",
            "unclear",
            "uncertain",
            "appears",
            "seems",
        ]
        hedge_count = sum(1 for h in hedges if h in response.lower())

        # Check for evidence of reasoning
        reasoning_markers = ["because", "therefore", "thus", "as a result"]
        reasoning_count = sum(1 for m in reasoning_markers if m in response.lower())

        score = 1.0 - (hedge_count * 0.1) + min(0.3, reasoning_count * 0.1)
        return max(0.0, min(1.0, score))

    def _assess_completeness(self, query: str, response: str) -> float:
        """Assess answer completeness."""
        # Simple heuristic: response should be substantial
        min_words = 20
        word_count = len(response.split())

        if word_count < min_words:
            return 0.3

        # Check for question marks in query (multiple parts)
        question_count = query.count("?")
        if question_count > 1:
            # Should address multiple aspects
            return min(1.0, word_count / (min_words * question_count))

        return min(1.0, word_count / 50)

    def _assess_clarity(self, response: str) -> float:
        """Assess answer clarity."""
        # Heuristics: sentence length, structure, etc.
        sentences = response.split(".")
        if not sentences:
            return 0.3

        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / len(sentences)

        # Optimal sentence length is 15-20 words
        if avg_sentence_length > 40:
            return 0.5  # Too long
        elif avg_sentence_length < 5:
            return 0.4  # Too short
        else:
            return 0.8  # Optimal

    def _assess_grounding(self, response: str, sources: List[str]) -> float:
        """Assess grounding in sources."""
        if not sources:
            return 0.3  # No sources

        response_words = set(response.lower().split())
        source_words = set()

        for source in sources:
            source_words.update(source.lower().split())

        overlap = len(response_words & source_words)
        return min(1.0, overlap / max(len(response_words), 1))

    def _classify_error(self, error_message: str) -> str:
        """Classify error into categories."""
        msg_lower = error_message.lower()

        if "retrieval" in msg_lower or "not found" in msg_lower:
            return "retrieval_failure"
        elif "reasoning" in msg_lower or "logic" in msg_lower:
            return "reasoning_error"
        elif "synthesis" in msg_lower or "answer" in msg_lower:
            return "synthesis_error"
        elif "timeout" in msg_lower:
            return "timeout"
        else:
            return "unknown_error"

    def _determine_root_cause(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any],
    ) -> str:
        """Determine root cause of error."""
        if error_type == "retrieval_failure":
            return "Insufficient matching documents in corpus"
        elif error_type == "reasoning_error":
            return "Reasoning logic did not produce valid result"
        elif error_type == "synthesis_error":
            return "Failed to synthesize answer from retrieved content"
        else:
            return error_message

    def _assess_severity(self, error_type: str) -> float:
        """Assess error severity (0-1)."""
        severity_map = {
            "retrieval_failure": 0.4,
            "reasoning_error": 0.7,
            "synthesis_error": 0.6,
            "timeout": 0.5,
            "unknown_error": 0.8,
        }
        return severity_map.get(error_type, 0.5)

    def _suggest_recovery_strategy(self, error_type: str) -> str:
        """Suggest recovery strategy for error."""
        strategies = {
            "retrieval_failure": "Broaden query, try different retrieval strategy",
            "reasoning_error": "Re-decompose goal, check reasoning steps",
            "synthesis_error": "Retrieve more sources, simplify synthesis",
            "timeout": "Reduce search scope, use cached results",
            "unknown_error": "Log error details and try alternative approach",
        }
        return strategies.get(error_type, "Retry with adjusted parameters")
