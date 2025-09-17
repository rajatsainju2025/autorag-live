"""
Pipeline optimization module for AutoRAG-Live.

This module provides optimization and policy components for:
- Hybrid weight optimization: Grid search and bandit algorithms
- Acceptance policies: Safe configuration updates with rollback
- Bandit optimization: Online learning for parameter tuning
"""

from .hybrid_optimizer import (
    grid_search_hybrid_weights,
    save_hybrid_config,
    load_hybrid_config,
    HybridWeights
)
from .bandit_optimizer import (
    BanditHybridOptimizer,
    UCB1Bandit,
    BanditArm
)
from .acceptance_policy import AcceptancePolicy, safe_config_update

__all__ = [
    "grid_search_hybrid_weights",
    "save_hybrid_config",
    "load_hybrid_config",
    "HybridWeights",
    "BanditHybridOptimizer",
    "UCB1Bandit",
    "BanditArm",
    "AcceptancePolicy",
    "safe_config_update"
]