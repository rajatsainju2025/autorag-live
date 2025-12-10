"""
Pipeline optimization module for AutoRAG-Live.

This module provides optimization and policy components for:
- Hybrid weight optimization: Grid search and bandit algorithms
- Acceptance policies: Safe configuration updates with rollback
- Bandit optimization: Online learning for parameter tuning
- Pipeline orchestration: Unified execution of all RAG components
"""

from .acceptance_policy import AcceptancePolicy, safe_config_update
from .bandit_optimizer import BanditArm, BanditHybridOptimizer, UCB1Bandit
from .hybrid_optimizer import (
    HybridWeights,
    grid_search_hybrid_weights,
    load_hybrid_config,
    save_hybrid_config,
)
from .orchestrator import (
    PipelineConfig,
    PipelineOrchestrator,
    PipelineResult,
    PipelineStage,
    StageResult,
)

__all__ = [
    "grid_search_hybrid_weights",
    "save_hybrid_config",
    "load_hybrid_config",
    "HybridWeights",
    "BanditHybridOptimizer",
    "UCB1Bandit",
    "BanditArm",
    "AcceptancePolicy",
    "safe_config_update",
    # Orchestrator
    "PipelineOrchestrator",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    "StageResult",
]
