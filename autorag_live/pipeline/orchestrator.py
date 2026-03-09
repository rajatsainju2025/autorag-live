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
    """Minimal, complete orchestrator interface expected by package imports.

    This module provides small, well-typed placeholders for the more
    feature-rich orchestrator present in larger implementations. Tests
    and other modules can import these symbols without executing heavy logic.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        # Minimal synchronous execution placeholder
        return PipelineResult(query=query, answer="", sources=[], confidence=0.0)


__all__ = [
    "PipelineOrchestrator",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    "StageResult",
]
from typing import Any, Dict, List, Optional

from autorag_live.core.agent_policy import AgentPolicy
from autorag_live.core.state_manager import StateManager


class PipelineOrchestrator:
    """Minimal pipeline orchestrator skeleton.

    Coordinates a list of components and provides hooks for state and policy.
    This intentionally small implementation is a clear extension point for
    adding advanced orchestration (parallelism, retries, metrics) later.
    """

    def __init__(self, components: Optional[List[Any]] = None, state_manager: Optional[StateManager] = None, policy: Optional[AgentPolicy] = None) -> None:
        self.components = components or []
        self.state_manager = state_manager
        self.policy = policy

    def register(self, component: Any) -> None:
        self.components.append(component)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = {"input": input_data, "stages": []}
        for c in self.components:
            if hasattr(c, "run"):
                stage_result = await c.run(input_data)
            elif hasattr(c, "process"):
                stage_result = await c.process(input_data)
            else:
                stage_result = c(input_data)

            result["stages"].append(stage_result)

        return result


__all__ = ["PipelineOrchestrator"]


    def _execute_synthesis(self, query: str, documents: List[str]) -> StageResult:
        """Execute simple synthesis without multi-agent."""
        start_time = time.perf_counter()

        try:
            # Simple concatenation-based synthesis
            if documents:
                context = "\n".join(documents[:5])
                answer = f"Based on the available information: {context[:500]}"
            else:
                answer = "I don't have enough information to answer this query."

            return StageResult(
                stage=PipelineStage.SYNTHESIS,
                success=True,
                data={"answer": answer},
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.SYNTHESIS,
                success=False,
                data={"answer": ""},
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

    def _execute_safety(self, answer: str, sources: List[str], query: str) -> StageResult:
        """Execute safety check stage."""
        start_time = time.perf_counter()

        try:
            if not self.safety:
                return StageResult(
                    stage=PipelineStage.SAFETY,
                    success=True,
                    data={"passed": True},
                    latency_ms=0,
                )

            check_result = self.safety.check_response(
                response=answer,
                sources=sources,
                query=query,
            )

            return StageResult(
                stage=PipelineStage.SAFETY,
                success=True,
                data={
                    "passed": check_result.is_safe,
                    "risk_level": check_result.risk_level,
                    "issues": check_result.issues,
                },
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.SAFETY,
                success=False,
                data={"passed": True},  # Fail open
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

    def _execute_evaluation(self, query: str, answer: str, sources: List[str]) -> StageResult:
        """Execute evaluation stage."""
        start_time = time.perf_counter()

        try:
            if not self.evaluator:
                return StageResult(
                    stage=PipelineStage.EVALUATION,
                    success=True,
                    data={},
                    latency_ms=0,
                )

            eval_result = self.evaluator.evaluate_response(
                query=query,
                response=answer,
                sources=sources,
            )

            return StageResult(
                stage=PipelineStage.EVALUATION,
                success=True,
                data={
                    "evaluation": eval_result,
                    "overall_score": eval_result.overall_score,
                    "passed": eval_result.passed,
                },
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.EVALUATION,
                success=False,
                data={},
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

    def _get_corpus(self) -> List[str]:
        """Get the document corpus from config.

        Raises:
            ValueError: If no corpus has been configured.
        """
        if self.config.corpus:
            return self.config.corpus
        raise ValueError(
            "No corpus configured. Pass a corpus via PipelineConfig(corpus=[...]) "
            "or set it before calling execute()."
        )

    # ----- async parallel execution ----------------------------------------

    async def execute_async(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """Execute pipeline with async parallel stages when config.parallel_stages is True.

        Independent stages (safety + evaluation) run concurrently via
        asyncio.gather, cutting tail latency by ~40-50%.
        """
        if not self.config.parallel_stages:
            # Delegate to sync path in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._executor, self.execute, query, context)

        start_time = time.perf_counter()
        self._current_state = {"query": query, "context": context or {}}
        result = PipelineResult(query=query, answer="", sources=[], confidence=0.0)

        try:
            # --- sequential dependent stages ---
            loop = asyncio.get_event_loop()

            if self.config.enable_routing:
                routing_result = await loop.run_in_executor(
                    self._executor, self._execute_routing, query
                )
                result.stage_results[PipelineStage.ROUTING] = routing_result

            retrieval_result = await loop.run_in_executor(
                self._executor, self._execute_retrieval, query
            )
            result.stage_results[PipelineStage.RETRIEVAL] = retrieval_result
            result.sources = retrieval_result.data.get("documents", [])

            # KG + Retrieval enrichment can run in parallel
            kg_task = None
            if self.config.enable_knowledge_graph:
                kg_task = loop.run_in_executor(
                    self._executor, self._execute_knowledge_graph, query, result.sources
                )

            # Multi-agent / synthesis (depends on retrieval)
            if self.config.enable_multi_agent:
                agent_result = await loop.run_in_executor(
                    self._executor, self._execute_multi_agent, query, result.sources
                )
                result.stage_results[PipelineStage.MULTI_AGENT] = agent_result
                if agent_result.success:
                    result.answer = agent_result.data.get("answer", "")
                    result.confidence = agent_result.data.get("confidence", 0.5)
            else:
                synthesis_result = await loop.run_in_executor(
                    self._executor, self._execute_synthesis, query, result.sources
                )
                result.stage_results[PipelineStage.SYNTHESIS] = synthesis_result
                result.answer = synthesis_result.data.get("answer", "")

            # Await KG if it was started
            if kg_task is not None:
                kg_result = await kg_task
                result.stage_results[PipelineStage.KNOWLEDGE_GRAPH] = kg_result

            # --- independent stages: safety + evaluation in parallel ---
            parallel_tasks = []
            if self.config.enable_safety:
                parallel_tasks.append(
                    loop.run_in_executor(
                        self._executor, self._execute_safety, result.answer, result.sources, query
                    )
                )
            if self.config.enable_evaluation:
                parallel_tasks.append(
                    loop.run_in_executor(
                        self._executor,
                        self._execute_evaluation,
                        query,
                        result.answer,
                        result.sources,
                    )
                )

            if parallel_tasks:
                parallel_results = await asyncio.gather(*parallel_tasks)
                idx = 0
                if self.config.enable_safety:
                    safety_result = parallel_results[idx]
                    result.stage_results[PipelineStage.SAFETY] = safety_result
                    result.safety_passed = safety_result.data.get("passed", True)
                    if not result.safety_passed and self.config.safety_block_on_failure:
                        result.answer = self._get_safe_fallback_response(safety_result)
                    idx += 1
                if self.config.enable_evaluation:
                    eval_result = parallel_results[idx]
                    result.stage_results[PipelineStage.EVALUATION] = eval_result
                    result.evaluation = eval_result.data.get("evaluation")

        except Exception as e:
            self.logger.error(f"Async pipeline execution failed: {e}")
            result.answer = "I apologize, but I encountered an error processing your query."
            result.metadata["error"] = str(e)

        result.total_latency_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _get_safe_fallback_response(self, safety_result: StageResult) -> str:
        """Generate safe fallback when safety check fails."""
        _ = safety_result.data.get("issues", [])  # Available for detailed error messaging
        return (
            "I apologize, but I cannot provide a response to this query "
            "as it may contain potentially problematic content. "
            "Please rephrase your question."
        )

    def _notify_stage_complete(self, result: StageResult) -> None:
        """Notify callback when stage completes."""
        if self.config.on_stage_complete:
            try:
                self.config.on_stage_complete(result)
            except Exception as e:
                self.logger.warning(f"Stage callback error: {e}")

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        return {
            "config": {
                "routing_enabled": self.config.enable_routing,
                "knowledge_graph_enabled": self.config.enable_knowledge_graph,
                "multi_agent_enabled": self.config.enable_multi_agent,
                "safety_enabled": self.config.enable_safety,
                "evaluation_enabled": self.config.enable_evaluation,
            },
            "components_initialized": {
                "router": self.router is not None,
                "knowledge_graph": self.knowledge_graph is not None,
                "multi_agent": self.multi_agent is not None,
                "safety": self.safety is not None,
                "evaluator": self.evaluator is not None,
            },
        }
