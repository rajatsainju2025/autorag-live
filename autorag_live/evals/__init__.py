"""
Evaluation module for AutoRAG-Live.

This module provides evaluation and benchmarking components for:
- Small evaluation suites: Quick performance checks
- Advanced metrics: Comprehensive evaluation with multiple metrics
- LLM judges: AI-powered evaluation of answer quality
- Performance benchmarks: System performance testing
"""

from .advanced_metrics import aggregate_metrics, comprehensive_evaluation
from .llm_judge import DeterministicJudge, LLMJudge
from .performance_benchmarks import run_full_benchmark_suite
from .small import run_small_suite

__all__ = [
    "run_small_suite",
    "comprehensive_evaluation",
    "aggregate_metrics",
    "LLMJudge",
    "DeterministicJudge",
    "run_full_benchmark_suite",
]
