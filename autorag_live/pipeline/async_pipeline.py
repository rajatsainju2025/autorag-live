"""
Advanced Async Pipeline for Agentic RAG Systems.

This module provides a fully asynchronous, streaming-capable pipeline
for agentic RAG with:
- True async/await throughout the pipeline
- Streaming responses with incremental updates
- Parallel stage execution where possible
- Backpressure handling and rate limiting
- Circuit breaker pattern for fault tolerance
- Event-driven progress tracking
- Composable pipeline stages
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
)

# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar("T")
Input = TypeVar("Input")
Output = TypeVar("Output")


class PipelineEventType(Enum):
    """Types of pipeline events."""

    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    STAGE_STREAMING = "stage_streaming"
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    TOKEN_GENERATED = "token_generated"
    DOCUMENT_RETRIEVED = "document_retrieved"
    PROGRESS_UPDATE = "progress_update"


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PipelineEvent:
    """Event emitted during pipeline execution."""

    event_type: PipelineEventType
    stage_name: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A chunk of streaming output."""

    content: str
    chunk_type: str = "text"
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageMetrics:
    """Metrics for a pipeline stage."""

    stage_name: str
    execution_count: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0
    last_execution: Optional[datetime] = None
    avg_latency_ms: float = 0.0

    def record_execution(self, latency_ms: float, success: bool = True) -> None:
        """Record a stage execution."""
        self.execution_count += 1
        self.total_latency_ms += latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.execution_count
        self.last_execution = datetime.now()
        if not success:
            self.error_count += 1


@dataclass
class PipelineContext:
    """Execution context passed through pipeline stages."""

    query: str
    session_id: str = ""
    user_id: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    retrieved_documents: List[str] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def set_result(self, stage_name: str, result: Any) -> None:
        """Store an intermediate result."""
        self.intermediate_results[stage_name] = result

    def get_result(self, stage_name: str, default: Any = None) -> Any:
        """Get an intermediate result."""
        return self.intermediate_results.get(stage_name, default)


@dataclass
class AsyncStageResult(Generic[T]):
    """Result from an async pipeline stage."""

    success: bool
    data: Optional[T]
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AsyncPipelineResult:
    """Complete result from async pipeline execution."""

    query: str
    answer: str
    sources: List[str]
    confidence: float
    stage_results: Dict[str, AsyncStageResult] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    stream_complete: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Event System
# =============================================================================


class EventEmitter:
    """Async event emitter for pipeline events."""

    def __init__(self):
        """Initialize event emitter."""
        self._listeners: Dict[
            PipelineEventType, List[Callable[[PipelineEvent], Awaitable[None]]]
        ] = {event_type: [] for event_type in PipelineEventType}
        self._global_listeners: List[Callable[[PipelineEvent], Awaitable[None]]] = []

    def on(
        self,
        event_type: PipelineEventType,
        callback: Callable[[PipelineEvent], Awaitable[None]],
    ) -> None:
        """Register an event listener."""
        self._listeners[event_type].append(callback)

    def on_all(self, callback: Callable[[PipelineEvent], Awaitable[None]]) -> None:
        """Register a listener for all events."""
        self._global_listeners.append(callback)

    async def emit(self, event: PipelineEvent) -> None:
        """Emit an event to all listeners."""
        tasks = []

        # Type-specific listeners
        for listener in self._listeners.get(event.event_type, []):
            tasks.append(asyncio.create_task(listener(event)))

        # Global listeners
        for listener in self._global_listeners:
            tasks.append(asyncio.create_task(listener(event)))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_successes: int = 2,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_successes: Successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_successes = half_open_successes

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_success_count = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    async def can_execute(self) -> bool:
        """Check if execution is allowed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time and (
                    time.time() - self._last_failure_time > self.recovery_timeout
                ):
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_success_count = 0
                    return True
                return False

            # HALF_OPEN - allow execution to test
            return True

    async def record_success(self) -> None:
        """Record a successful execution."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_success_count += 1
                if self._half_open_success_count >= self.half_open_successes:
                    self._state = CircuitState.CLOSED
                    self._failures = 0

            self._failures = 0

    async def record_failure(self) -> None:
        """Record a failed execution."""
        async with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
            elif self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN


# =============================================================================
# Rate Limiter
# =============================================================================


class AsyncRateLimiter:
    """Token bucket rate limiter for async operations."""

    def __init__(self, rate: float, burst: int):
        """
        Initialize rate limiter.

        Args:
            rate: Tokens per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_time = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, waiting if necessary."""
        async with self._lock:
            while True:
                now = time.time()
                # Add tokens based on elapsed time
                self._tokens = min(self.burst, self._tokens + (now - self._last_time) * self.rate)
                self._last_time = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                # Calculate wait time
                wait_time = (tokens - self._tokens) / self.rate
                await asyncio.sleep(wait_time)


# =============================================================================
# Pipeline Stages
# =============================================================================


class AsyncPipelineStage(ABC, Generic[Input, Output]):
    """Abstract base class for async pipeline stages."""

    def __init__(
        self,
        name: str,
        timeout: float = 30.0,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        """
        Initialize pipeline stage.

        Args:
            name: Stage name
            timeout: Execution timeout in seconds
            circuit_breaker: Optional circuit breaker
        """
        self.name = name
        self.timeout = timeout
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.metrics = StageMetrics(stage_name=name)
        self.logger = logging.getLogger(f"AsyncStage.{name}")

    @abstractmethod
    async def process(
        self, input_data: Input, context: PipelineContext
    ) -> AsyncStageResult[Output]:
        """Process input and return result."""
        pass

    async def execute(
        self, input_data: Input, context: PipelineContext
    ) -> AsyncStageResult[Output]:
        """Execute stage with timeout and circuit breaker."""
        start_time = time.perf_counter()

        # Check circuit breaker
        if not await self.circuit_breaker.can_execute():
            self.logger.warning(f"Circuit open for stage {self.name}")
            return AsyncStageResult(
                success=False,
                data=None,
                error="Circuit breaker open",
                latency_ms=0.0,
            )

        try:
            # Execute with timeout
            result = await asyncio.wait_for(self.process(input_data, context), timeout=self.timeout)
            latency_ms = (time.perf_counter() - start_time) * 1000

            if result.success:
                await self.circuit_breaker.record_success()
            else:
                await self.circuit_breaker.record_failure()

            result.latency_ms = latency_ms
            self.metrics.record_execution(latency_ms, result.success)
            return result

        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            await self.circuit_breaker.record_failure()
            self.metrics.record_execution(latency_ms, success=False)
            self.logger.error(f"Stage {self.name} timed out")
            return AsyncStageResult(
                success=False,
                data=None,
                error=f"Timeout after {self.timeout}s",
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            await self.circuit_breaker.record_failure()
            self.metrics.record_execution(latency_ms, success=False)
            self.logger.error(f"Stage {self.name} failed: {e}")
            return AsyncStageResult(success=False, data=None, error=str(e), latency_ms=latency_ms)


class StreamingStage(AsyncPipelineStage[Input, Output]):
    """Pipeline stage with streaming output support."""

    @abstractmethod
    async def stream(
        self, input_data: Input, context: PipelineContext
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream output chunks."""
        pass  # pragma: no cover
        yield  # Make this a generator

    async def process(
        self, input_data: Input, context: PipelineContext
    ) -> AsyncStageResult[Output]:
        """Process by collecting all streamed chunks."""
        chunks: List[str] = []

        async for chunk in self.stream(input_data, context):
            chunks.append(chunk.content)

        combined = "".join(chunks)
        return AsyncStageResult(success=True, data=combined)  # type: ignore


# =============================================================================
# Concrete Pipeline Stages
# =============================================================================


class AsyncRetrievalStage(AsyncPipelineStage[str, List[str]]):
    """Async document retrieval stage."""

    def __init__(
        self,
        retriever: Optional[Any] = None,
        top_k: int = 10,
        **kwargs: Any,
    ):
        """Initialize retrieval stage."""
        super().__init__(name="retrieval", **kwargs)
        self.retriever = retriever
        self.top_k = top_k

    async def process(self, query: str, context: PipelineContext) -> AsyncStageResult[List[str]]:
        """Retrieve relevant documents."""
        try:
            # Simulate async retrieval
            await asyncio.sleep(0.01)  # Yield to event loop

            if self.retriever:
                # Use actual retriever if available
                if hasattr(self.retriever, "aretrieve"):
                    docs = await self.retriever.aretrieve(query, k=self.top_k)
                elif hasattr(self.retriever, "retrieve"):
                    # Run sync retriever in thread pool
                    loop = asyncio.get_event_loop()
                    docs = await loop.run_in_executor(
                        None, lambda: self.retriever.retrieve(query, k=self.top_k)
                    )
                else:
                    docs = []
            else:
                # Mock documents for demonstration
                docs = [f"Document {i} about {query}" for i in range(self.top_k)]

            context.retrieved_documents = docs
            context.set_result(self.name, docs)

            return AsyncStageResult(success=True, data=docs, metadata={"count": len(docs)})

        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return AsyncStageResult(success=False, data=[], error=str(e))


class AsyncGenerationStage(StreamingStage[Dict[str, Any], str]):
    """Async text generation stage with streaming."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        max_tokens: int = 1024,
        **kwargs: Any,
    ):
        """Initialize generation stage."""
        super().__init__(name="generation", **kwargs)
        self.llm = llm
        self.max_tokens = max_tokens

    async def stream(
        self, input_data: Dict[str, Any], context: PipelineContext
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream generated text."""
        query = input_data.get("query", context.query)
        documents = input_data.get("documents", context.retrieved_documents)

        # Build prompt
        docs_text = "\n\n".join(documents[:5])
        prompt = f"Based on the following documents:\n{docs_text}\n\nAnswer: {query}"

        if self.llm and hasattr(self.llm, "astream"):
            async for token in self.llm.astream(prompt):
                yield StreamChunk(content=token, chunk_type="token")
        else:
            # Mock streaming for demonstration
            words = f"Based on the retrieved documents, the answer to '{query}' is: "
            words += "This is a comprehensive response generated from the relevant context."
            for word in words.split():
                await asyncio.sleep(0.01)
                yield StreamChunk(content=word + " ", chunk_type="token")

        yield StreamChunk(content="", chunk_type="end", is_final=True)


class AsyncRerankStage(AsyncPipelineStage[Dict[str, Any], List[str]]):
    """Async reranking stage."""

    def __init__(self, reranker: Optional[Any] = None, top_k: int = 5, **kwargs: Any):
        """Initialize rerank stage."""
        super().__init__(name="rerank", **kwargs)
        self.reranker = reranker
        self.top_k = top_k

    async def process(
        self, input_data: Dict[str, Any], context: PipelineContext
    ) -> AsyncStageResult[List[str]]:
        """Rerank documents."""
        query = input_data.get("query", context.query)
        documents = input_data.get("documents", context.retrieved_documents)

        if not documents:
            return AsyncStageResult(success=True, data=[])

        try:
            if self.reranker and hasattr(self.reranker, "arerank"):
                reranked = await self.reranker.arerank(query, documents, k=self.top_k)
            else:
                # Simple mock rerank - just take top k
                reranked = documents[: self.top_k]

            context.retrieved_documents = reranked
            context.set_result(self.name, reranked)

            return AsyncStageResult(success=True, data=reranked, metadata={"count": len(reranked)})

        except Exception as e:
            return AsyncStageResult(success=False, data=documents, error=str(e))


class AsyncSafetyStage(AsyncPipelineStage[str, Dict[str, Any]]):
    """Async safety checking stage."""

    def __init__(self, safety_checker: Optional[Any] = None, **kwargs: Any):
        """Initialize safety stage."""
        super().__init__(name="safety", **kwargs)
        self.safety_checker = safety_checker

    async def process(
        self, text: str, context: PipelineContext
    ) -> AsyncStageResult[Dict[str, Any]]:
        """Check text for safety issues."""
        try:
            if self.safety_checker and hasattr(self.safety_checker, "acheck"):
                result = await self.safety_checker.acheck(text)
            else:
                # Mock safety check
                result = {"safe": True, "issues": []}

            return AsyncStageResult(
                success=True,
                data=result,
                metadata={"safe": result.get("safe", True)},
            )

        except Exception as e:
            return AsyncStageResult(
                success=False,
                data={"safe": True, "issues": []},
                error=str(e),
            )


# =============================================================================
# Main Async Pipeline
# =============================================================================


class AsyncRAGPipeline:
    """
    Fully asynchronous RAG pipeline with streaming support.

    Features:
    - Parallel stage execution where possible
    - Streaming responses
    - Circuit breakers for fault tolerance
    - Rate limiting
    - Event-driven progress tracking
    """

    def __init__(
        self,
        stages: Optional[List[AsyncPipelineStage]] = None,
        rate_limit: Optional[AsyncRateLimiter] = None,
        max_concurrent_stages: int = 3,
    ):
        """
        Initialize async pipeline.

        Args:
            stages: List of pipeline stages
            rate_limit: Optional rate limiter
            max_concurrent_stages: Max stages to run in parallel
        """
        self.stages = stages or self._create_default_stages()
        self.rate_limiter = rate_limit
        self.max_concurrent = max_concurrent_stages
        self.events = EventEmitter()
        self.logger = logging.getLogger("AsyncRAGPipeline")

        # Stage dependency graph for parallel execution
        self._stage_dependencies: Dict[str, Set[str]] = {}

    def _create_default_stages(self) -> List[AsyncPipelineStage]:
        """Create default pipeline stages."""
        return [
            AsyncRetrievalStage(top_k=10),
            AsyncRerankStage(top_k=5),
            AsyncGenerationStage(max_tokens=1024),
            AsyncSafetyStage(),
        ]

    def add_stage(self, stage: AsyncPipelineStage, depends_on: Optional[List[str]] = None) -> None:
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        if depends_on:
            self._stage_dependencies[stage.name] = set(depends_on)

    async def execute(
        self,
        query: str,
        context: Optional[PipelineContext] = None,
        stream: bool = False,
    ) -> Union[AsyncPipelineResult, AsyncGenerator[StreamChunk, None]]:
        """
        Execute the pipeline.

        Args:
            query: Input query
            context: Optional execution context
            stream: Whether to stream the response

        Returns:
            Pipeline result or stream of chunks
        """
        if stream:
            return self._execute_streaming(query, context)
        else:
            return await self._execute_batch(query, context)

    async def _execute_batch(
        self, query: str, context: Optional[PipelineContext] = None
    ) -> AsyncPipelineResult:
        """Execute pipeline in batch mode."""
        start_time = time.perf_counter()
        ctx = context or PipelineContext(query=query)

        await self.events.emit(
            PipelineEvent(
                event_type=PipelineEventType.PIPELINE_STARTED,
                stage_name="pipeline",
                data={"query": query},
            )
        )

        result = AsyncPipelineResult(query=query, answer="", sources=[], confidence=0.0)

        current_data: Any = query

        for stage in self.stages:
            # Rate limiting
            if self.rate_limiter:
                await self.rate_limiter.acquire()

            await self.events.emit(
                PipelineEvent(
                    event_type=PipelineEventType.STAGE_STARTED,
                    stage_name=stage.name,
                    data={"input_type": type(current_data).__name__},
                )
            )

            stage_result = await stage.execute(current_data, ctx)
            result.stage_results[stage.name] = stage_result

            if stage_result.success:
                await self.events.emit(
                    PipelineEvent(
                        event_type=PipelineEventType.STAGE_COMPLETED,
                        stage_name=stage.name,
                        data={"latency_ms": stage_result.latency_ms},
                    )
                )

                # Prepare input for next stage
                if stage.name == "retrieval":
                    current_data = {"query": query, "documents": stage_result.data}
                    result.sources = stage_result.data or []
                elif stage.name == "rerank":
                    current_data = {"query": query, "documents": stage_result.data}
                    result.sources = stage_result.data or []
                elif stage.name == "generation":
                    result.answer = stage_result.data or ""
                    current_data = result.answer
                else:
                    current_data = stage_result.data
            else:
                await self.events.emit(
                    PipelineEvent(
                        event_type=PipelineEventType.STAGE_FAILED,
                        stage_name=stage.name,
                        data={"error": stage_result.error},
                    )
                )
                self.logger.warning(f"Stage {stage.name} failed: {stage_result.error}")

        result.total_latency_ms = (time.perf_counter() - start_time) * 1000
        result.confidence = self._calculate_confidence(result)

        await self.events.emit(
            PipelineEvent(
                event_type=PipelineEventType.PIPELINE_COMPLETED,
                stage_name="pipeline",
                data={"total_latency_ms": result.total_latency_ms},
            )
        )

        return result

    async def _execute_streaming(
        self, query: str, context: Optional[PipelineContext] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Execute pipeline with streaming output."""
        ctx = context or PipelineContext(query=query)

        # Run non-streaming stages first
        current_data: Any = query

        for stage in self.stages:
            if isinstance(stage, StreamingStage) and stage.name == "generation":
                # Stream the generation stage
                await self.events.emit(
                    PipelineEvent(
                        event_type=PipelineEventType.STAGE_STARTED,
                        stage_name=stage.name,
                        data={"streaming": True},
                    )
                )

                async for chunk in stage.stream(current_data, ctx):
                    await self.events.emit(
                        PipelineEvent(
                            event_type=PipelineEventType.TOKEN_GENERATED,
                            stage_name=stage.name,
                            data={"content": chunk.content},
                        )
                    )
                    yield chunk
            else:
                # Execute non-streaming stages
                stage_result = await stage.execute(current_data, ctx)

                if stage_result.success:
                    if stage.name == "retrieval":
                        current_data = {"query": query, "documents": stage_result.data}
                        # Emit document retrieval events
                        for doc in stage_result.data or []:
                            await self.events.emit(
                                PipelineEvent(
                                    event_type=PipelineEventType.DOCUMENT_RETRIEVED,
                                    stage_name=stage.name,
                                    data={"document": doc[:100]},
                                )
                            )
                    elif stage.name == "rerank":
                        current_data = {"query": query, "documents": stage_result.data}
                    else:
                        current_data = stage_result.data

    def _calculate_confidence(self, result: AsyncPipelineResult) -> float:
        """Calculate confidence score for the result."""
        # Base confidence on stage success rate
        successful_stages = sum(1 for sr in result.stage_results.values() if sr.success)
        total_stages = len(result.stage_results)

        if total_stages == 0:
            return 0.0

        stage_confidence = successful_stages / total_stages

        # Adjust based on number of sources
        source_confidence = min(len(result.sources) / 5, 1.0)

        # Adjust based on answer length
        answer_confidence = min(len(result.answer.split()) / 50, 1.0)

        return stage_confidence * 0.5 + source_confidence * 0.3 + answer_confidence * 0.2


# =============================================================================
# Parallel Pipeline Execution
# =============================================================================


class ParallelAsyncPipeline(AsyncRAGPipeline):
    """Pipeline with parallel stage execution support."""

    def __init__(
        self,
        stages: Optional[List[AsyncPipelineStage]] = None,
        dependencies: Optional[Dict[str, List[str]]] = None,
        **kwargs: Any,
    ):
        """
        Initialize parallel pipeline.

        Args:
            stages: Pipeline stages
            dependencies: Stage dependency graph
            **kwargs: Additional arguments
        """
        super().__init__(stages=stages, **kwargs)
        self.dependencies = dependencies or {}

    async def execute_parallel(
        self, query: str, context: Optional[PipelineContext] = None
    ) -> AsyncPipelineResult:
        """Execute pipeline with parallel stages where possible."""
        start_time = time.perf_counter()
        ctx = context or PipelineContext(query=query)

        result = AsyncPipelineResult(query=query, answer="", sources=[], confidence=0.0)

        # Build stage lookup
        stage_by_name = {stage.name: stage for stage in self.stages}
        completed: Set[str] = set()
        stage_results: Dict[str, AsyncStageResult] = {}

        async def execute_stage(stage_name: str) -> None:
            """Execute a single stage."""
            stage = stage_by_name[stage_name]

            # Get input from dependencies
            deps = self.dependencies.get(stage_name, [])
            if deps:
                input_data = {dep: stage_results[dep].data for dep in deps if dep in stage_results}
            else:
                input_data = query

            stage_result = await stage.execute(input_data, ctx)
            stage_results[stage_name] = stage_result
            completed.add(stage_name)

        # Execute stages in dependency order
        remaining = set(stage_by_name.keys())

        while remaining:
            # Find stages whose dependencies are satisfied
            ready = [
                name
                for name in remaining
                if all(dep in completed for dep in self.dependencies.get(name, []))
            ]

            if not ready:
                self.logger.error("Circular dependency detected")
                break

            # Execute ready stages in parallel
            tasks = [execute_stage(name) for name in ready]
            await asyncio.gather(*tasks)

            remaining -= set(ready)

        # Build final result
        result.stage_results = stage_results

        if "generation" in stage_results and stage_results["generation"].data:
            result.answer = stage_results["generation"].data
        if "retrieval" in stage_results and stage_results["retrieval"].data:
            result.sources = stage_results["retrieval"].data

        result.total_latency_ms = (time.perf_counter() - start_time) * 1000
        result.confidence = self._calculate_confidence(result)

        return result


# =============================================================================
# Pipeline Builder
# =============================================================================


class AsyncPipelineBuilder:
    """Builder for constructing async pipelines."""

    def __init__(self):
        """Initialize builder."""
        self._stages: List[AsyncPipelineStage] = []
        self._dependencies: Dict[str, List[str]] = {}
        self._rate_limit: Optional[AsyncRateLimiter] = None
        self._max_concurrent: int = 3

    def add_retrieval(
        self,
        retriever: Optional[Any] = None,
        top_k: int = 10,
        timeout: float = 30.0,
    ) -> "AsyncPipelineBuilder":
        """Add retrieval stage."""
        self._stages.append(AsyncRetrievalStage(retriever=retriever, top_k=top_k, timeout=timeout))
        return self

    def add_rerank(
        self,
        reranker: Optional[Any] = None,
        top_k: int = 5,
        depends_on: Optional[List[str]] = None,
    ) -> "AsyncPipelineBuilder":
        """Add reranking stage."""
        stage = AsyncRerankStage(reranker=reranker, top_k=top_k)
        self._stages.append(stage)
        if depends_on:
            self._dependencies[stage.name] = depends_on
        return self

    def add_generation(
        self, llm: Optional[Any] = None, max_tokens: int = 1024
    ) -> "AsyncPipelineBuilder":
        """Add generation stage."""
        self._stages.append(AsyncGenerationStage(llm=llm, max_tokens=max_tokens))
        return self

    def add_safety(self, safety_checker: Optional[Any] = None) -> "AsyncPipelineBuilder":
        """Add safety checking stage."""
        self._stages.append(AsyncSafetyStage(safety_checker=safety_checker))
        return self

    def add_custom_stage(
        self,
        stage: AsyncPipelineStage,
        depends_on: Optional[List[str]] = None,
    ) -> "AsyncPipelineBuilder":
        """Add a custom stage."""
        self._stages.append(stage)
        if depends_on:
            self._dependencies[stage.name] = depends_on
        return self

    def with_rate_limit(self, rate: float, burst: int) -> "AsyncPipelineBuilder":
        """Configure rate limiting."""
        self._rate_limit = AsyncRateLimiter(rate=rate, burst=burst)
        return self

    def with_max_concurrent(self, max_concurrent: int) -> "AsyncPipelineBuilder":
        """Configure max concurrent stages."""
        self._max_concurrent = max_concurrent
        return self

    def build(self) -> AsyncRAGPipeline:
        """Build the pipeline."""
        return AsyncRAGPipeline(
            stages=self._stages,
            rate_limit=self._rate_limit,
            max_concurrent_stages=self._max_concurrent,
        )

    def build_parallel(self) -> ParallelAsyncPipeline:
        """Build a parallel pipeline."""
        return ParallelAsyncPipeline(
            stages=self._stages,
            dependencies=self._dependencies,
            rate_limit=self._rate_limit,
            max_concurrent_stages=self._max_concurrent,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_async_pipeline(
    retriever: Optional[Any] = None,
    reranker: Optional[Any] = None,
    llm: Optional[Any] = None,
    safety_checker: Optional[Any] = None,
    rate_limit: Optional[tuple] = None,
) -> AsyncRAGPipeline:
    """
    Create a configured async pipeline.

    Args:
        retriever: Document retriever
        reranker: Document reranker
        llm: Language model for generation
        safety_checker: Safety checker
        rate_limit: Optional (rate, burst) tuple for rate limiting

    Returns:
        Configured AsyncRAGPipeline
    """
    builder = AsyncPipelineBuilder()
    builder.add_retrieval(retriever=retriever)
    builder.add_rerank(reranker=reranker, depends_on=["retrieval"])
    builder.add_generation(llm=llm)
    builder.add_safety(safety_checker=safety_checker)

    if rate_limit:
        builder.with_rate_limit(rate_limit[0], rate_limit[1])

    return builder.build()


async def run_async_query(
    query: str,
    pipeline: Optional[AsyncRAGPipeline] = None,
    stream: bool = False,
) -> Union[AsyncPipelineResult, AsyncIterator[StreamChunk]]:
    """
    Run a query through the async pipeline.

    Args:
        query: Input query
        pipeline: Optional pipeline (creates default if None)
        stream: Whether to stream the response

    Returns:
        Pipeline result or stream of chunks
    """
    pipe = pipeline or create_async_pipeline()
    return await pipe.execute(query, stream=stream)


# =============================================================================
# Adaptive Backpressure Control - State-of-the-Art Optimization
# =============================================================================


@dataclass
class BackpressureMetrics:
    """Metrics for backpressure monitoring."""

    queue_depth: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    current_rate: float = 100.0
    target_rate: float = 100.0
    throttle_events: int = 0
    total_requests: int = 0


class AdaptiveBackpressureController:
    """
    Adaptive rate limiting with backpressure signaling for production RAG systems.

    Implements feedback-based rate control that:
    1. Monitors queue depth and latency
    2. Automatically adjusts rate based on downstream capacity
    3. Applies exponential backoff under pressure
    4. Prevents cascading failures under load

    Based on: "Serving Large Language Models" (NVIDIA/Microsoft, 2024)
    patterns for production LLM inference systems.

    Example:
        >>> controller = AdaptiveBackpressureController(
        ...     base_rate=100.0,
        ...     target_latency_ms=100.0
        ... )
        >>> async with controller.acquire() as token:
        ...     result = await process_request()
        ...     # Latency automatically recorded on exit
    """

    def __init__(
        self,
        base_rate: float = 100.0,
        max_queue_depth: int = 1000,
        target_latency_ms: float = 100.0,
        latency_window_size: int = 100,
        rate_increase_factor: float = 1.1,
        rate_decrease_factor: float = 0.8,
        min_rate: float = 1.0,
        max_rate_multiplier: float = 2.0,
    ):
        """
        Initialize adaptive backpressure controller.

        Args:
            base_rate: Base request rate (requests/second)
            max_queue_depth: Maximum queue depth before rejection
            target_latency_ms: Target p50 latency in milliseconds
            latency_window_size: Number of samples for latency calculation
            rate_increase_factor: Factor to increase rate when under target
            rate_decrease_factor: Factor to decrease rate when over target
            min_rate: Minimum allowed rate
            max_rate_multiplier: Maximum rate as multiplier of base_rate
        """
        self.base_rate = base_rate
        self.max_queue_depth = max_queue_depth
        self.target_latency_ms = target_latency_ms
        self.latency_window_size = latency_window_size
        self.rate_increase_factor = rate_increase_factor
        self.rate_decrease_factor = rate_decrease_factor
        self.min_rate = min_rate
        self.max_rate = base_rate * max_rate_multiplier

        self._current_rate = base_rate
        self._queue: asyncio.Queue[float] = asyncio.Queue(maxsize=max_queue_depth)
        self._latency_window: List[float] = []
        self._lock = asyncio.Lock()

        self._metrics = BackpressureMetrics(current_rate=base_rate, target_rate=base_rate)

        # Background task for periodic rate adjustment
        self._adjustment_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the background rate adjustment task."""
        if not self._running:
            self._running = True
            self._adjustment_task = asyncio.create_task(self._rate_adjustment_loop())

    async def stop(self) -> None:
        """Stop the background rate adjustment task."""
        self._running = False
        if self._adjustment_task:
            self._adjustment_task.cancel()
            try:
                await self._adjustment_task
            except asyncio.CancelledError:
                pass

    @property
    def is_overloaded(self) -> bool:
        """Check if system is overloaded based on queue depth."""
        return self._queue.qsize() > self.max_queue_depth * 0.8

    @property
    def current_rate(self) -> float:
        """Get current adaptive rate."""
        return self._current_rate

    @property
    def queue_depth(self) -> int:
        """Get current queue depth."""
        return self._queue.qsize()

    def acquire(self) -> "BackpressureToken":
        """
        Acquire a backpressure token for request processing.

        Returns a context manager that tracks latency automatically.

        Returns:
            BackpressureToken for use in async with statement
        """
        return BackpressureToken(self)

    async def _acquire_internal(self) -> bool:
        """Internal acquire with backpressure logic."""
        self._metrics.total_requests += 1

        # Apply backpressure if queue is filling up
        if self.is_overloaded:
            await self._apply_backpressure()
            self._metrics.throttle_events += 1

        # Try to add to queue
        try:
            self._queue.put_nowait(time.time())
            return True
        except asyncio.QueueFull:
            return False

    async def _apply_backpressure(self) -> None:
        """Apply exponential backoff based on queue depth."""
        queue_ratio = self._queue.qsize() / self.max_queue_depth

        # Exponential backoff: backoff increases sharply as queue fills
        backoff = min(5.0, 0.1 * (2 ** (queue_ratio * 4)))
        await asyncio.sleep(backoff)

    async def _release_internal(self, start_time: float) -> None:
        """Internal release that records latency."""
        latency_ms = (time.time() - start_time) * 1000

        async with self._lock:
            self._latency_window.append(latency_ms)
            if len(self._latency_window) > self.latency_window_size:
                self._latency_window.pop(0)

        # Try to get from queue
        try:
            self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

    def _calculate_percentile(self, percentile: float) -> float:
        """Calculate latency percentile from window."""
        if not self._latency_window:
            return 0.0

        sorted_latencies = sorted(self._latency_window)
        index = int(len(sorted_latencies) * percentile / 100)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    async def _adjust_rate(self) -> None:
        """Adjust rate based on latency feedback."""
        if len(self._latency_window) < 10:
            return  # Not enough data

        async with self._lock:
            avg_latency = sum(self._latency_window) / len(self._latency_window)
            p99_latency = self._calculate_percentile(99)

            # Adjust based on average latency vs target
            if avg_latency > self.target_latency_ms * 1.5:
                # Significantly over target - decrease rate
                self._current_rate = max(
                    self.min_rate, self._current_rate * self.rate_decrease_factor
                )
            elif avg_latency > self.target_latency_ms:
                # Slightly over target - gentle decrease
                self._current_rate = max(
                    self.min_rate,
                    self._current_rate * (1 - (1 - self.rate_decrease_factor) * 0.5),
                )
            elif avg_latency < self.target_latency_ms * 0.5:
                # Well under target - increase rate
                self._current_rate = min(
                    self.max_rate, self._current_rate * self.rate_increase_factor
                )
            elif avg_latency < self.target_latency_ms * 0.8:
                # Slightly under target - gentle increase
                self._current_rate = min(
                    self.max_rate,
                    self._current_rate * (1 + (self.rate_increase_factor - 1) * 0.5),
                )

            # Update metrics
            self._metrics.avg_latency_ms = avg_latency
            self._metrics.p99_latency_ms = p99_latency
            self._metrics.current_rate = self._current_rate
            self._metrics.queue_depth = self._queue.qsize()

    async def _rate_adjustment_loop(self) -> None:
        """Background loop for periodic rate adjustment."""
        while self._running:
            await asyncio.sleep(1.0)  # Adjust every second
            await self._adjust_rate()

    def get_metrics(self) -> BackpressureMetrics:
        """Get current backpressure metrics."""
        return self._metrics


class BackpressureToken:
    """
    Token for tracking request lifecycle with backpressure.

    Use as async context manager:
        async with controller.acquire() as token:
            await process_request()
    """

    def __init__(self, controller: AdaptiveBackpressureController):
        """Initialize token."""
        self.controller = controller
        self.start_time: float = 0.0
        self.acquired: bool = False

    async def __aenter__(self) -> "BackpressureToken":
        """Acquire token on context entry."""
        self.start_time = time.time()
        self.acquired = await self.controller._acquire_internal()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release token and record latency on context exit."""
        if self.acquired:
            await self.controller._release_internal(self.start_time)


class LoadShedder:
    """
    Load shedding for graceful degradation under extreme load.

    Implements priority-based request rejection when system is overloaded.
    """

    def __init__(
        self,
        backpressure_controller: AdaptiveBackpressureController,
        shed_threshold: float = 0.9,
        priority_levels: int = 3,
    ):
        """
        Initialize load shedder.

        Args:
            backpressure_controller: Backpressure controller for metrics
            shed_threshold: Queue depth ratio to start shedding
            priority_levels: Number of priority levels (higher = more important)
        """
        self.controller = backpressure_controller
        self.shed_threshold = shed_threshold
        self.priority_levels = priority_levels

        self._shed_counts: Dict[int, int] = {i: 0 for i in range(priority_levels)}

    def should_shed(self, priority: int = 0) -> bool:
        """
        Check if request should be shed based on priority.

        Args:
            priority: Request priority (0 = lowest, higher = more important)

        Returns:
            True if request should be rejected
        """
        if priority >= self.priority_levels:
            return False  # Highest priority never shed

        queue_ratio = self.controller.queue_depth / self.controller.max_queue_depth

        if queue_ratio < self.shed_threshold:
            return False  # Not overloaded

        # Progressive shedding based on priority
        # Lower priority requests are shed first
        shed_probability = (queue_ratio - self.shed_threshold) / (1 - self.shed_threshold)
        priority_factor = (self.priority_levels - priority) / self.priority_levels

        import random

        if random.random() < shed_probability * priority_factor:
            self._shed_counts[priority] += 1
            return True

        return False

    def get_shed_stats(self) -> Dict[str, int]:
        """Get load shedding statistics."""
        return {f"priority_{p}_shed": count for p, count in self._shed_counts.items()}
