"""
Evaluation module for AutoRAG-Live.

This module provides evaluation and benchmarking components for:
- Small evaluation suites: Quick performance checks
- Advanced metrics: Comprehensive evaluation with multiple metrics
- LLM judges: AI-powered evaluation of answer quality
- Performance benchmarks: System performance testing
"""

from .small import run_small_suite
from .advanced_metrics import comprehensive_evaluation, aggregate_metrics
from .llm_judge import LLMJudge, DeterministicJudge
from .performance_benchmarks import run_full_benchmark_suite

__all__ = [
    "run_small_suite",
    "comprehensive_evaluation",
    "aggregate_metrics",
    "LLMJudge",
    "DeterministicJudge",
    "run_full_benchmark_suite"
]