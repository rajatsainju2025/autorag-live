"""
Comprehensive evaluation framework with RAGAS metrics and benchmarking.

Enables data-driven optimization through modern RAG evaluation metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MetricType(Enum):
    """Types of evaluation metrics."""

    FAITHFULNESS = "faithfulness"  # Is answer true to source?
    RELEVANCE = "relevance"  # Is answer relevant to query?
    COMPLETENESS = "completeness"  # Does it answer all parts?
    COHERENCE = "coherence"  # Is answer well-structured?
    LATENCY = "latency"  # Response time
    COST = "cost"  # Computational cost


@dataclass
class MetricScore:
    """Single metric evaluation."""

    metric_type: MetricType
    score: float  # 0-1 range
    confidence: float = 0.5
    explanation: str = ""
    benchmark_value: Optional[float] = None

    def is_above_benchmark(self) -> bool:
        """Check if score meets benchmark."""
        if self.benchmark_value is None:
            return True

        return self.score >= self.benchmark_value


@dataclass
class EvaluationResult:
    """Complete evaluation result for a query-response pair."""

    query: str
    response: str
    reference: Optional[str] = None
    metrics: Dict[MetricType, MetricScore] = field(default_factory=dict)
    overall_score: float = 0.0
    passed: bool = True
    timestamp: str = ""

    def add_metric(
        self,
        metric_type: MetricType,
        score: float,
        confidence: float = 0.5,
        explanation: str = "",
        benchmark: Optional[float] = None,
    ) -> None:
        """Add metric score."""
        self.metrics[metric_type] = MetricScore(
            metric_type=metric_type,
            score=score,
            confidence=confidence,
            explanation=explanation,
            benchmark_value=benchmark,
        )

    def calculate_overall_score(self) -> float:
        """Calculate overall evaluation score."""
        if not self.metrics:
            return 0.0

        weighted_sum = sum(m.score * m.confidence for m in self.metrics.values())

        total_confidence = sum(m.confidence for m in self.metrics.values())

        self.overall_score = weighted_sum / total_confidence if total_confidence > 0 else 0

        return self.overall_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "response": self.response[:100],
            "metrics": {k.value: v.score for k, v in self.metrics.items()},
            "overall_score": round(self.overall_score, 3),
            "passed": self.passed,
        }


class RAGASEvaluator:
    """
    Implements RAGAS (Retrieval-Augmented Generation Assessment) metrics.

    Faithfulness, Relevance, Completeness evaluation.
    """

    def __init__(self):
        """Initialize RAGAS evaluator."""
        self.logger = logging.getLogger("RAGASEvaluator")

    def evaluate_faithfulness(self, response: str, sources: List[str]) -> MetricScore:
        """
        Evaluate faithfulness of response to sources.

        Uses a pre-built source word set to avoid re-joining/lowering per claim.
        """
        if not sources:
            return MetricScore(
                metric_type=MetricType.FAITHFULNESS,
                score=0.0,
                confidence=0.7,
                explanation="No sources available",
                benchmark_value=0.85,
            )

        # Build source word set ONCE
        source_text = " ".join(sources).lower()
        source_words = frozenset(source_text.split())

        claims = [c.strip() for c in response.split(".") if len(c.strip()) > 10]
        if not claims:
            return MetricScore(
                metric_type=MetricType.FAITHFULNESS,
                score=0.0,
                confidence=0.7,
                explanation="No substantive claims found",
                benchmark_value=0.85,
            )

        supported = 0
        for claim in claims:
            key_words = [w for w in claim.lower().split() if len(w) > 4]
            if not key_words:
                continue
            # Word-set intersection instead of linear scan per keyword
            matches = sum(1 for w in key_words if w in source_words)
            if matches >= len(key_words) * 0.6:
                supported += 1

        score = supported / len(claims)
        explanation = f"{int(score * 100)}% of claims are grounded"

        return MetricScore(
            metric_type=MetricType.FAITHFULNESS,
            score=min(1.0, score),
            confidence=0.7,
            explanation=explanation,
            benchmark_value=0.85,
        )

    def evaluate_relevance(self, response: str, query: str) -> MetricScore:
        """
        Evaluate relevance of response to query.

        Args:
            response: Generated response
            query: Original query

        Returns:
            MetricScore for relevance
        """
        query_words = frozenset(w.lower() for w in query.split() if len(w) > 3)
        response_words = frozenset(w.lower() for w in response.split() if len(w) > 3)
        return self.evaluate_relevance_fast(query_words, response_words)

    def evaluate_relevance_fast(
        self, query_words: frozenset, response_words: frozenset
    ) -> MetricScore:
        """Evaluate relevance using pre-built frozen word sets (zero-copy)."""
        if not query_words:
            score = 0.5
        else:
            overlap = len(query_words & response_words)
            score = min(1.0, overlap / len(query_words))

        return MetricScore(
            metric_type=MetricType.RELEVANCE,
            score=score,
            confidence=0.7,
            explanation=f"Key term coverage: {int(score * 100)}%",
            benchmark_value=0.88,
        )

    def evaluate_completeness(self, response: str, query: str) -> MetricScore:
        """
        Evaluate completeness of response.

        Args:
            response: Generated response
            query: Original query

        Returns:
            MetricScore for completeness
        """
        return self.evaluate_completeness_fast(query, len(response.split()))

    def evaluate_completeness_fast(self, query: str, response_word_count: int) -> MetricScore:
        """Evaluate completeness using a pre-computed word count."""
        question_count = max(1, query.count("?"))
        expected_length = question_count * 30

        score = min(
            1.0,
            response_word_count / expected_length if expected_length > 0 else 0,
        )

        return MetricScore(
            metric_type=MetricType.COMPLETENESS,
            score=score,
            confidence=0.6,
            explanation=(
                f"Response length: {response_word_count} words " f"(expected ~{expected_length})"
            ),
            benchmark_value=0.85,
        )


class BenchmarkSuite:
    """Manages benchmarks and comparison tests."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.logger = logging.getLogger("BenchmarkSuite")
        self.evaluator = RAGASEvaluator()
        self.results: List[EvaluationResult] = []

        # Default benchmarks
        self.benchmarks = {
            MetricType.FAITHFULNESS: 0.85,
            MetricType.RELEVANCE: 0.88,
            MetricType.COMPLETENESS: 0.85,
            MetricType.COHERENCE: 0.80,
        }

    def evaluate_response(
        self,
        query: str,
        response: str,
        sources: List[str],
        reference: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single query-response pair.

        Pre-computes shared word sets once and passes them to metrics,
        avoiding redundant tokenisation across faithfulness/relevance/completeness.
        """
        from datetime import datetime

        result = EvaluationResult(
            query=query,
            response=response,
            reference=reference,
            timestamp=datetime.utcnow().isoformat(),
        )

        # --- Pre-compute shared features ONCE ---
        query_words = frozenset(w.lower() for w in query.split() if len(w) > 3)
        response_words = frozenset(w.lower() for w in response.split() if len(w) > 3)
        response_word_count = len(response.split())

        # Evaluate faithfulness (already uses frozenset internally)
        faithfulness = self.evaluator.evaluate_faithfulness(response, sources)
        faithfulness.benchmark_value = self.benchmarks[MetricType.FAITHFULNESS]
        result.add_metric(
            MetricType.FAITHFULNESS,
            faithfulness.score,
            faithfulness.confidence,
            faithfulness.explanation,
            faithfulness.benchmark_value,
        )

        # Evaluate relevance — pass pre-built word sets
        relevance = self.evaluator.evaluate_relevance_fast(query_words, response_words)
        relevance.benchmark_value = self.benchmarks[MetricType.RELEVANCE]
        result.add_metric(
            MetricType.RELEVANCE,
            relevance.score,
            relevance.confidence,
            relevance.explanation,
            relevance.benchmark_value,
        )

        # Evaluate completeness — pass pre-computed word count
        completeness = self.evaluator.evaluate_completeness_fast(query, response_word_count)
        completeness.benchmark_value = self.benchmarks[MetricType.COMPLETENESS]
        result.add_metric(
            MetricType.COMPLETENESS,
            completeness.score,
            completeness.confidence,
            completeness.explanation,
            completeness.benchmark_value,
        )

        # Calculate overall score
        result.calculate_overall_score()

        # Determine pass/fail
        result.passed = all(m.is_above_benchmark() for m in result.metrics.values())

        self.results.append(result)

        return result

    def batch_evaluate(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate multiple test cases.

        Args:
            test_cases: List of {query, response, sources, reference}

        Returns:
            Benchmark results summary
        """
        results = []

        for test_case in test_cases:
            result = self.evaluate_response(
                query=test_case.get("query", ""),
                response=test_case.get("response", ""),
                sources=test_case.get("sources", []),
                reference=test_case.get("reference"),
            )
            results.append(result)

        return self._summarize_results(results)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.results:
            return {}

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)

        metrics_avg = {}

        for metric_type in MetricType:
            scores = [
                r.metrics[metric_type].score for r in self.results if metric_type in r.metrics
            ]

            if scores:
                metrics_avg[metric_type.value] = {
                    "avg": round(sum(scores) / len(scores), 3),
                    "min": round(min(scores), 3),
                    "max": round(max(scores), 3),
                }

        return {
            "total_evaluations": total,
            "pass_rate": round(passed / total if total > 0 else 0, 3),
            "metrics": metrics_avg,
            "avg_overall_score": round(sum(r.overall_score for r in self.results) / total, 3)
            if total > 0
            else 0,
        }

    def _summarize_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Summarize batch evaluation results."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        metrics_summary = {}

        for metric_type in MetricType:
            scores = [r.metrics[metric_type].score for r in results if metric_type in r.metrics]

            if scores:
                metrics_summary[metric_type.value] = {
                    "avg": round(sum(scores) / len(scores), 3),
                    "pass_rate": round(sum(1 for s in scores if s >= 0.8) / len(scores), 3),
                }

        return {
            "total_tests": total,
            "passed": passed,
            "pass_rate": round(passed / total if total > 0 else 0, 3),
            "metrics": metrics_summary,
            "recommendations": self._generate_recommendations(results),
        }

    def _generate_recommendations(self, results: List[EvaluationResult]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []

        # Analyze weaknesses
        metric_scores = {}

        for metric_type in MetricType:
            scores = [r.metrics[metric_type].score for r in results if metric_type in r.metrics]

            if scores:
                avg_score = sum(scores) / len(scores)
                metric_scores[metric_type] = avg_score

        # Find worst-performing metric
        if metric_scores:
            worst_metric = min(
                metric_scores,
                key=lambda x: metric_scores[x],
            )
            worst_score = metric_scores[worst_metric]

            if worst_score < 0.8:
                if worst_metric == MetricType.FAITHFULNESS:
                    recommendations.append("Improve source grounding: use citations, verify claims")
                elif worst_metric == MetricType.RELEVANCE:
                    recommendations.append("Improve query understanding: refine retrieval strategy")
                elif worst_metric == MetricType.COMPLETENESS:
                    recommendations.append("Provide more comprehensive answers: expand synthesis")

        return recommendations
