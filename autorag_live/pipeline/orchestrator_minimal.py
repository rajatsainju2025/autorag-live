from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class PipelineStage(Enum):
    ROUTING = "routing"
    RETRIEVAL = "retrieval"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    MULTI_AGENT = "multi_agent"
    SYNTHESIS = "synthesis"
    SAFETY = "safety"
    EVALUATION = "evaluation"


@dataclass
class StageResult:
    stage: PipelineStage
    success: bool
    data: Any
    latency_ms: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    query: str
    answer: str
    sources: List[str]
    confidence: float
    stage_results: Dict[PipelineStage, StageResult] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    safety_passed: bool = True
    evaluation: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    enable_routing: bool = True
    enable_knowledge_graph: bool = True
    enable_multi_agent: bool = True
    enable_safety: bool = True
    enable_evaluation: bool = False

    max_retrieval_results: int = 10
    multi_agent_timeout: float = 30.0
    safety_block_on_failure: bool = True

    parallel_stages: bool = False
    cache_intermediate_results: bool = True

    corpus: Optional[List[str]] = None

    on_stage_complete: Optional[Callable[[StageResult], None]] = None


class PipelineOrchestrator:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        return PipelineResult(query=query, answer="", sources=[], confidence=0.0)


__all__ = [
    "PipelineOrchestrator",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    "StageResult",
]
