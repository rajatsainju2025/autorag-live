"""
Disagreement analysis module for AutoRAG-Live.

This module provides disagreement analysis components for:
- Metrics: Jaccard similarity and Kendall's Tau rank correlation
- Report generation: HTML reports for disagreement visualization
- Analysis tools: Understanding retriever disagreements for optimization
"""

from .metrics import (
    jaccard_at_k,
    kendall_tau_at_k
)
from .report import generate_disagreement_report

__all__ = [
    "jaccard_at_k",
    "kendall_tau_at_k",
    "generate_disagreement_report"
]