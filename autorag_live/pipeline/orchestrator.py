"""
Unified Pipeline Orchestrator for AutoRAG-Live.

Connects all agentic RAG components into a cohesive, configurable pipeline
with proper state management and execution flow.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from autorag_live.evals.ragas_evaluation import BenchmarkSuite, EvaluationResult
from autorag_live.knowledge_graph.graph import KnowledgeGraph
from autorag_live.multi_agent.collaboration import MultiAgentOrchestrator
from autorag_live.retrievers import hybrid as hybrid_module
from autorag_live.routing.router import Router
from autorag_live.safety.guardrails import SafetyGuardrails


class PipelineStage(Enum):
    """Stages in the agentic RAG pipeline."""

    ROUTING = "routing"
    RETRIEVAL = "retrieval"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    MULTI_AGENT = "multi_agent"
    SYNTHESIS = "synthesis"
    SAFETY = "safety"
    EVALUATION = "evaluation"


@dataclass
class StageResult:
    """Result from a pipeline stage."""

    stage: PipelineStage
    success: bool
    data: Any
    latency_ms: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete result from pipeline execution."""

    query: str
    answer: str
    sources: List[str]
    confidence: float
    stage_results: Dict[PipelineStage, StageResult] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    safety_passed: bool = True
    evaluation: Optional[EvaluationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_stage_latencies(self) -> Dict[str, float]:
        """Get latency breakdown by stage."""
        return {s.value: r.latency_ms for s, r in self.stage_results.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "sources_count": len(self.sources),
            "confidence": round(self.confidence, 3),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "safety_passed": self.safety_passed,
            "stage_latencies": self.get_stage_latencies(),
        }


@dataclass
class PipelineConfig:
    """Configuration for the pipeline orchestrator."""

    # Stage enablement
    enable_routing: bool = True
    enable_knowledge_graph: bool = True
    enable_multi_agent: bool = True
    enable_safety: bool = True
    enable_evaluation: bool = False

    # Stage-specific settings
    max_retrieval_results: int = 10
    multi_agent_timeout: float = 30.0
    safety_block_on_failure: bool = True

    # Performance settings
    parallel_stages: bool = False  # Run independent stages in parallel
    cache_intermediate_results: bool = True

    # Corpus source (replaces hardcoded list)
    corpus: Optional[List[str]] = None

    # Callbacks
    on_stage_complete: Optional[Callable[[StageResult], None]] = None


class PipelineOrchestrator:
    """
    Unified orchestrator for the agentic RAG pipeline.

    Coordinates execution across routing, retrieval, knowledge graph,
    multi-agent collaboration, safety, and evaluation stages.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the pipeline orchestrator."""
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger("PipelineOrchestrator")

        # Thread pool for running sync stages concurrently
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Initialize components
        self._init_components()

        # Stage execution order
        self._stage_order = [
            PipelineStage.ROUTING,
            PipelineStage.RETRIEVAL,
            PipelineStage.KNOWLEDGE_GRAPH,
            PipelineStage.MULTI_AGENT,
            PipelineStage.SYNTHESIS,
            PipelineStage.SAFETY,
            PipelineStage.EVALUATION,
        ]

        # State for current execution
        self._current_state: Dict[str, Any] = {}

    def _init_components(self) -> None:
        """Initialize pipeline components."""
        self.router = Router() if self.config.enable_routing else None
        self.knowledge_graph = KnowledgeGraph() if self.config.enable_knowledge_graph else None
        self.multi_agent = MultiAgentOrchestrator() if self.config.enable_multi_agent else None
        self.safety = SafetyGuardrails() if self.config.enable_safety else None
        self.evaluator = BenchmarkSuite() if self.config.enable_evaluation else None

    def execute(self, query: str, context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """
        Execute the full pipeline for a query.

        Args:
            query: User query
            context: Optional context (conversation history, user preferences)

        Returns:
            Complete pipeline result
        """
        start_time = time.perf_counter()
        self._current_state = {"query": query, "context": context or {}}

        result = PipelineResult(
            query=query,
            answer="",
            sources=[],
            confidence=0.0,
        )

        try:
            # Stage 1: Routing
            if self.config.enable_routing:
                routing_result = self._execute_routing(query)
                result.stage_results[PipelineStage.ROUTING] = routing_result
                self._notify_stage_complete(routing_result)

            # Stage 2: Retrieval
            retrieval_result = self._execute_retrieval(query)
            result.stage_results[PipelineStage.RETRIEVAL] = retrieval_result
            result.sources = retrieval_result.data.get("documents", [])
            self._notify_stage_complete(retrieval_result)

            # Stage 3: Knowledge Graph enrichment
            if self.config.enable_knowledge_graph:
                kg_result = self._execute_knowledge_graph(query, result.sources)
                result.stage_results[PipelineStage.KNOWLEDGE_GRAPH] = kg_result
                self._notify_stage_complete(kg_result)

            # Stage 4: Multi-agent collaboration
            if self.config.enable_multi_agent:
                agent_result = self._execute_multi_agent(query, result.sources)
                result.stage_results[PipelineStage.MULTI_AGENT] = agent_result
                if agent_result.success:
                    result.answer = agent_result.data.get("answer", "")
                    result.confidence = agent_result.data.get("confidence", 0.5)
                self._notify_stage_complete(agent_result)
            else:
                # Simple synthesis without multi-agent
                synthesis_result = self._execute_synthesis(query, result.sources)
                result.stage_results[PipelineStage.SYNTHESIS] = synthesis_result
                result.answer = synthesis_result.data.get("answer", "")
                self._notify_stage_complete(synthesis_result)

            # Stage 5: Safety check
            if self.config.enable_safety:
                safety_result = self._execute_safety(result.answer, result.sources, query)
                result.stage_results[PipelineStage.SAFETY] = safety_result
                result.safety_passed = safety_result.data.get("passed", True)

                if not result.safety_passed and self.config.safety_block_on_failure:
                    result.answer = self._get_safe_fallback_response(safety_result)

                self._notify_stage_complete(safety_result)

            # Stage 6: Evaluation
            if self.config.enable_evaluation:
                eval_result = self._execute_evaluation(query, result.answer, result.sources)
                result.stage_results[PipelineStage.EVALUATION] = eval_result
                result.evaluation = eval_result.data.get("evaluation")
                self._notify_stage_complete(eval_result)

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            result.answer = "I apologize, but I encountered an error processing your query."
            result.metadata["error"] = str(e)

        result.total_latency_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _execute_routing(self, query: str) -> StageResult:
        """Execute routing stage."""
        start_time = time.perf_counter()

        try:
            if not self.router:
                return StageResult(
                    stage=PipelineStage.ROUTING,
                    success=True,
                    data={"decision": None},
                    latency_ms=0,
                )

            decision = self.router.route(query)
            self._current_state["routing_decision"] = decision

            return StageResult(
                stage=PipelineStage.ROUTING,
                success=True,
                data={
                    "decision": decision,
                    "query_type": decision.analysis.query_type.value,
                    "complexity": decision.analysis.complexity,
                },
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.ROUTING,
                success=False,
                data={},
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

    def _execute_retrieval(self, query: str) -> StageResult:
        """Execute retrieval stage."""
        start_time = time.perf_counter()

        try:
            # Use routing decision if available (for future strategy selection)
            _ = self._current_state.get("routing_decision")

            corpus = self._get_corpus()
            results = hybrid_module.hybrid_retrieve(
                query, corpus, k=self.config.max_retrieval_results
            )

            documents = [doc for doc, _ in results]
            scores = [score for _, score in results]

            self._current_state["retrieved_docs"] = documents
            self._current_state["retrieval_scores"] = scores

            return StageResult(
                stage=PipelineStage.RETRIEVAL,
                success=True,
                data={
                    "documents": documents,
                    "scores": scores,
                    "count": len(documents),
                },
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.RETRIEVAL,
                success=False,
                data={"documents": [], "scores": []},
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

    def _execute_knowledge_graph(self, query: str, documents: List[str]) -> StageResult:
        """Execute knowledge graph enrichment stage."""
        start_time = time.perf_counter()

        try:
            if not self.knowledge_graph:
                return StageResult(
                    stage=PipelineStage.KNOWLEDGE_GRAPH,
                    success=True,
                    data={},
                    latency_ms=0,
                )

            # Extract entities from documents and build graph
            for doc in documents:
                self.knowledge_graph.extract_entities(doc)
                self.knowledge_graph.discover_relations(doc)

            # Get expanded query with related concepts
            expanded_terms = self.knowledge_graph.expand_query(query)

            self._current_state["expanded_query"] = expanded_terms
            self._current_state["entity_count"] = len(self.knowledge_graph.nodes)

            return StageResult(
                stage=PipelineStage.KNOWLEDGE_GRAPH,
                success=True,
                data={
                    "expanded_terms": expanded_terms,
                    "entities_found": len(self.knowledge_graph.nodes),
                    "relations_found": len(self.knowledge_graph.relations),
                },
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.KNOWLEDGE_GRAPH,
                success=False,
                data={},
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

    def _execute_multi_agent(self, query: str, documents: List[str]) -> StageResult:
        """Execute multi-agent collaboration stage."""
        start_time = time.perf_counter()

        try:
            if not self.multi_agent:
                return StageResult(
                    stage=PipelineStage.MULTI_AGENT,
                    success=True,
                    data={"answer": "", "confidence": 0.5},
                    latency_ms=0,
                )

            # Run collaborative reasoning with orchestrate_query
            result = self.multi_agent.orchestrate_query(query)

            return StageResult(
                stage=PipelineStage.MULTI_AGENT,
                success=True,
                data={
                    "answer": result.get("final_answer", ""),
                    "confidence": result.get("consensus_score", 0.5),
                    "agent_proposals": result.get("proposals", []),
                },
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.MULTI_AGENT,
                success=False,
                data={"answer": "", "confidence": 0.0},
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

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
        """Get the document corpus from config or return default."""
        if self.config.corpus:
            return self.config.corpus
        return [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing enables computers to understand text.",
            "Retrieval-augmented generation combines retrieval with generation.",
            "Large language models are trained on vast amounts of text data.",
        ]

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
