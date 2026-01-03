"""
Async Agent Execution Module.

Provides fully asynchronous agent execution with proper concurrency primitives
for parallel tool execution, cancellation support, and timeout handling.

Key Features:
1. Async-first agent execution
2. Parallel tool execution with asyncio.gather
3. Semaphore-based concurrency limiting
4. Proper cancellation and timeout handling
5. Background task management
6. Streaming with async generators

Example:
    >>> executor = AsyncAgentExecutor(agent, max_concurrent_tools=5)
    >>> result = await executor.run(query, timeout=30.0)
    >>> async for event in executor.stream(query):
    ...     print(event)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Coroutine, Dict, List, Optional, Set, TypeVar

from autorag_live.core.protocols import (
    AgentState,
    AgentStatus,
    BaseLLM,
    Message,
    ToolCall,
    ToolResult,
)
from autorag_live.llm.tool_calling import ToolManager

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Execution Types
# =============================================================================


class ExecutionStatus(str, Enum):
    """Status of async execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    ERROR = "error"


class StreamEventType(str, Enum):
    """Types of streaming events."""

    START = "start"
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    TOKEN = "token"
    PROGRESS = "progress"
    ERROR = "error"
    DONE = "done"


@dataclass
class StreamEvent:
    """
    Event in streaming execution.

    Attributes:
        type: Event type
        data: Event data
        timestamp: When event occurred
        step: Current step number
    """

    type: StreamEventType
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
        }


@dataclass
class ExecutionResult:
    """
    Result of async agent execution.

    Attributes:
        status: Execution status
        result: Final result if successful
        error: Error message if failed
        duration_ms: Total execution time
        steps_completed: Number of steps completed
        tool_calls_made: Number of tool calls made
    """

    status: ExecutionStatus
    result: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    steps_completed: int = 0
    tool_calls_made: int = 0

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "steps_completed": self.steps_completed,
            "tool_calls_made": self.tool_calls_made,
        }


@dataclass
class TaskHandle:
    """
    Handle for tracking async task.

    Attributes:
        task_id: Unique task identifier
        task: The asyncio Task
        created_at: When task was created
        status: Current status
    """

    task_id: str
    task: asyncio.Task[Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: ExecutionStatus = ExecutionStatus.PENDING

    def cancel(self) -> bool:
        """Cancel the task."""
        if not self.task.done():
            self.task.cancel()
            self.status = ExecutionStatus.CANCELLED
            return True
        return False

    @property
    def is_done(self) -> bool:
        """Check if task is done."""
        return self.task.done()


# =============================================================================
# Concurrency Utilities
# =============================================================================


class ConcurrencyLimiter:
    """
    Limits concurrent async operations using semaphore.

    Example:
        >>> limiter = ConcurrencyLimiter(max_concurrent=5)
        >>> async with limiter:
        ...     await some_operation()
    """

    def __init__(self, max_concurrent: int = 10):
        """Initialize limiter."""
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._max_concurrent = max_concurrent

    @property
    def active_count(self) -> int:
        """Number of currently active operations."""
        return self._active_count

    @property
    def available_slots(self) -> int:
        """Number of available slots."""
        return self._max_concurrent - self._active_count

    async def __aenter__(self) -> "ConcurrencyLimiter":
        """Enter context - acquire semaphore."""
        await self._semaphore.acquire()
        self._active_count += 1
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context - release semaphore."""
        self._active_count -= 1
        self._semaphore.release()


class TaskGroup:
    """
    Manages a group of related async tasks.

    Provides cleanup on context exit and task tracking.

    Example:
        >>> async with TaskGroup() as group:
        ...     group.create_task(coro1())
        ...     group.create_task(coro2())
        ...     # Tasks run concurrently
    """

    def __init__(self, name: str = ""):
        """Initialize task group."""
        self.name = name or f"group_{uuid.uuid4().hex[:8]}"
        self._tasks: Set[asyncio.Task[Any]] = set()
        self._closed = False

    def create_task(self, coro: Coroutine[Any, Any, T], name: str = "") -> asyncio.Task[T]:
        """Create and track a task."""
        if self._closed:
            raise RuntimeError("TaskGroup is closed")

        task = asyncio.create_task(coro, name=name or f"{self.name}_task")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    @property
    def pending_count(self) -> int:
        """Number of pending tasks."""
        return len([t for t in self._tasks if not t.done()])

    async def cancel_all(self) -> None:
        """Cancel all pending tasks."""
        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def __aenter__(self) -> "TaskGroup":
        """Enter context."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context - cancel remaining tasks."""
        self._closed = True
        await self.cancel_all()


async def run_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout: float,
    *,
    default: Optional[T] = None,
) -> tuple[T | None, bool]:
    """
    Run coroutine with timeout.

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        default: Default value if timeout

    Returns:
        (result, timed_out) tuple
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        return result, False
    except asyncio.TimeoutError:
        return default, True


async def gather_with_concurrency(
    coros: List[Coroutine[Any, Any, T]],
    max_concurrent: int = 10,
    *,
    return_exceptions: bool = False,
) -> List[T | BaseException]:
    """
    Gather coroutines with concurrency limit.

    Args:
        coros: Coroutines to run
        max_concurrent: Max concurrent operations
        return_exceptions: Whether to return exceptions

    Returns:
        List of results in order
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *[run_with_semaphore(c) for c in coros],
        return_exceptions=return_exceptions,
    )


# =============================================================================
# Async Tool Executor
# =============================================================================


class AsyncToolExecutor:
    """
    Executes tools asynchronously with concurrency control.

    Features:
    - Concurrent tool execution
    - Timeout per tool
    - Retry on failure
    - Result caching

    Example:
        >>> executor = AsyncToolExecutor(tools, max_concurrent=5)
        >>> results = await executor.execute_batch(tool_calls)
    """

    def __init__(
        self,
        tool_manager: ToolManager,
        *,
        max_concurrent: int = 5,
        default_timeout: float = 30.0,
        max_retries: int = 2,
    ):
        """Initialize executor."""
        self.tool_manager = tool_manager
        self.max_concurrent = max_concurrent
        self.default_timeout = default_timeout
        self.max_retries = max_retries

        self._limiter = ConcurrencyLimiter(max_concurrent)
        self._cache: Dict[str, ToolResult] = {}

    def _cache_key(self, call: ToolCall) -> str:
        """Generate cache key for tool call."""
        import json

        args = call.get_arguments()
        return f"{call.name}:{json.dumps(args, sort_keys=True)}"

    async def execute_single(
        self,
        call: ToolCall,
        *,
        timeout: Optional[float] = None,
        use_cache: bool = True,
    ) -> ToolResult:
        """
        Execute single tool call.

        Args:
            call: Tool call to execute
            timeout: Override default timeout
            use_cache: Whether to use caching

        Returns:
            Tool result
        """
        # Check cache
        if use_cache:
            cache_key = self._cache_key(call)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for {call.name}")
                return self._cache[cache_key]

        # Execute with concurrency limit
        async with self._limiter:
            for attempt in range(self.max_retries + 1):
                result = await self.tool_manager.execute_tool_call(
                    call,
                    timeout=timeout or self.default_timeout,
                )

                if result.success:
                    # Cache successful result
                    if use_cache:
                        self._cache[self._cache_key(call)] = result
                    return result

                # Retry on failure
                if attempt < self.max_retries:
                    logger.warning(
                        f"Tool {call.name} failed, retrying ({attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))

            return result

    async def execute_batch(
        self,
        calls: List[ToolCall],
        *,
        timeout: Optional[float] = None,
    ) -> List[ToolResult]:
        """
        Execute multiple tool calls concurrently.

        Args:
            calls: Tool calls to execute
            timeout: Per-call timeout

        Returns:
            Results in same order as calls
        """
        coros = [self.execute_single(call, timeout=timeout) for call in calls]
        return await asyncio.gather(*coros)

    def clear_cache(self) -> None:
        """Clear result cache."""
        self._cache.clear()


# =============================================================================
# Async Agent Executor
# =============================================================================


class AsyncAgentExecutor:
    """
    Fully async agent executor with streaming support.

    Features:
    - Async agent loop execution
    - Streaming events during execution
    - Cancellation and timeout support
    - Background task management

    Example:
        >>> executor = AsyncAgentExecutor(agent)
        >>> result = await executor.run("What is 2+2?", timeout=60.0)
        >>>
        >>> # Or with streaming
        >>> async for event in executor.stream("What is 2+2?"):
        ...     print(event)
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[ToolManager] = None,
        *,
        max_iterations: int = 10,
        max_concurrent_tools: int = 5,
        default_timeout: float = 60.0,
        verbose: bool = False,
    ):
        """Initialize executor."""
        self.llm = llm
        self.tools = tools or ToolManager()
        self.max_iterations = max_iterations
        self.default_timeout = default_timeout
        self.verbose = verbose

        self._tool_executor = AsyncToolExecutor(
            self.tools,
            max_concurrent=max_concurrent_tools,
        )
        self._active_tasks: Dict[str, TaskHandle] = {}

    async def _execute_step(
        self,
        state: AgentState,
        event_queue: Optional[asyncio.Queue[StreamEvent]] = None,
    ) -> tuple[AgentState, bool]:
        """
        Execute single agent step.

        Args:
            state: Current agent state
            event_queue: Optional queue for streaming events

        Returns:
            (new_state, is_done) tuple
        """
        state.iteration += 1
        state.status = AgentStatus.THINKING

        # Emit progress
        if event_queue:
            await event_queue.put(
                StreamEvent(
                    type=StreamEventType.PROGRESS,
                    data={"step": state.iteration, "status": "thinking"},
                    step=state.iteration,
                )
            )

        # Build messages for LLM
        messages = [
            Message.system(
                "You are a helpful assistant. Use tools when needed. "
                "When you have the answer, respond with FINAL_ANSWER: <answer>"
            ),
        ]
        messages.extend(state.messages)

        # Generate response
        tools_schema = self.tools.get_openai_tools() if self.tools.list_tools() else None

        result = await self.llm.generate(
            messages,
            tools=tools_schema,
            temperature=0.7,
        )

        # Emit thought
        if event_queue:
            await event_queue.put(
                StreamEvent(
                    type=StreamEventType.THOUGHT,
                    data=result.content,
                    step=state.iteration,
                )
            )

        # Check for final answer
        content = result.content or ""
        if "FINAL_ANSWER:" in content:
            answer = content.split("FINAL_ANSWER:", 1)[1].strip()
            state.current_thought = answer
            state.status = AgentStatus.COMPLETE

            if event_queue:
                await event_queue.put(
                    StreamEvent(
                        type=StreamEventType.DONE,
                        data=answer,
                        step=state.iteration,
                    )
                )
            return state, True

        # Handle tool calls
        if result.has_tool_calls and result.tool_calls:
            state.status = AgentStatus.ACTING

            if event_queue:
                await event_queue.put(
                    StreamEvent(
                        type=StreamEventType.ACTION,
                        data=[tc.to_dict() for tc in result.tool_calls],
                        step=state.iteration,
                    )
                )

            # Execute tools in parallel
            tool_results = await self._tool_executor.execute_batch(result.tool_calls)

            state.status = AgentStatus.OBSERVING

            # Add results to state
            for tc, tr in zip(result.tool_calls, tool_results):
                state.messages.append(Message.assistant(content, tool_calls=[tc]))
                state.messages.append(tr.to_message())

                if event_queue:
                    await event_queue.put(
                        StreamEvent(
                            type=StreamEventType.OBSERVATION,
                            data={"tool": tr.name, "result": tr.result, "success": tr.success},
                            step=state.iteration,
                        )
                    )

            return state, False

        # No tool calls and no final answer - add response and continue
        state.messages.append(Message.assistant(content))
        state.current_thought = content

        # Check if we should treat as final answer
        if state.iteration >= self.max_iterations:
            state.status = AgentStatus.COMPLETE
            return state, True

        return state, False

    async def run(
        self,
        query: str,
        *,
        timeout: Optional[float] = None,
        state: Optional[AgentState] = None,
    ) -> ExecutionResult:
        """
        Run agent to completion.

        Args:
            query: User query
            timeout: Execution timeout
            state: Initial state (optional)

        Returns:
            ExecutionResult with answer
        """
        start_time = time.time()
        timeout = timeout or self.default_timeout

        # Initialize state
        if state is None:
            state = AgentState()
            state.messages.append(Message.user(query))

        tool_calls_count = 0

        try:
            # Run with timeout
            async with asyncio.timeout(timeout):
                for _ in range(self.max_iterations):
                    state, is_done = await self._execute_step(state)

                    # Count tool calls from this step
                    tool_calls_count += len([m for m in state.messages if m.role.value == "tool"])

                    if is_done:
                        break

        except asyncio.TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=f"Execution timed out after {timeout}s",
                duration_ms=(time.time() - start_time) * 1000,
                steps_completed=state.iteration,
                tool_calls_made=tool_calls_count,
            )
        except asyncio.CancelledError:
            return ExecutionResult(
                status=ExecutionStatus.CANCELLED,
                error="Execution was cancelled",
                duration_ms=(time.time() - start_time) * 1000,
                steps_completed=state.iteration,
                tool_calls_made=tool_calls_count,
            )
        except Exception as e:
            logger.exception("Agent execution error")
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
                steps_completed=state.iteration,
                tool_calls_made=tool_calls_count,
            )

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            result=state.current_thought,
            duration_ms=(time.time() - start_time) * 1000,
            steps_completed=state.iteration,
            tool_calls_made=tool_calls_count,
        )

    async def stream(
        self,
        query: str,
        *,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream agent execution events.

        Args:
            query: User query
            timeout: Execution timeout

        Yields:
            StreamEvent for each step
        """
        event_queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        state = AgentState()
        state.messages.append(Message.user(query))

        # Emit start event
        yield StreamEvent(type=StreamEventType.START, data={"query": query})

        timeout = timeout or self.default_timeout

        try:
            async with asyncio.timeout(timeout):
                for _ in range(self.max_iterations):
                    state, is_done = await self._execute_step(state, event_queue)

                    # Yield all events from queue
                    while not event_queue.empty():
                        yield await event_queue.get()

                    if is_done:
                        break

        except asyncio.TimeoutError:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": f"Timeout after {timeout}s"},
                step=state.iteration,
            )
        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
                step=state.iteration,
            )

    def submit(
        self,
        query: str,
        *,
        timeout: Optional[float] = None,
    ) -> TaskHandle:
        """
        Submit query for background execution.

        Args:
            query: User query
            timeout: Execution timeout

        Returns:
            TaskHandle for tracking
        """
        task_id = f"task_{uuid.uuid4().hex[:12]}"

        async def wrapped() -> ExecutionResult:
            return await self.run(query, timeout=timeout)

        task = asyncio.create_task(wrapped(), name=task_id)
        handle = TaskHandle(task_id=task_id, task=task, status=ExecutionStatus.RUNNING)
        self._active_tasks[task_id] = handle

        def cleanup(t: asyncio.Task[Any]) -> None:
            if task_id in self._active_tasks:
                if t.cancelled():
                    self._active_tasks[task_id].status = ExecutionStatus.CANCELLED
                elif t.exception():
                    self._active_tasks[task_id].status = ExecutionStatus.ERROR
                else:
                    self._active_tasks[task_id].status = ExecutionStatus.COMPLETED

        task.add_done_callback(cleanup)
        return handle

    def get_task(self, task_id: str) -> Optional[TaskHandle]:
        """Get task by ID."""
        return self._active_tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        handle = self._active_tasks.get(task_id)
        if handle:
            return handle.cancel()
        return False

    async def cancel_all(self) -> int:
        """Cancel all running tasks."""
        cancelled = 0
        for handle in self._active_tasks.values():
            if handle.cancel():
                cancelled += 1
        return cancelled


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_agent_async(
    llm: BaseLLM,
    query: str,
    tools: Optional[ToolManager] = None,
    *,
    max_iterations: int = 10,
    timeout: float = 60.0,
) -> str:
    """
    Run agent query asynchronously.

    Args:
        llm: Language model
        query: User query
        tools: Tool manager
        max_iterations: Max steps
        timeout: Timeout in seconds

    Returns:
        Answer string
    """
    executor = AsyncAgentExecutor(
        llm,
        tools,
        max_iterations=max_iterations,
    )
    result = await executor.run(query, timeout=timeout)
    return result.result or f"Error: {result.error}"


@asynccontextmanager
async def managed_executor(
    llm: BaseLLM,
    tools: Optional[ToolManager] = None,
    **kwargs: Any,
) -> AsyncGenerator[AsyncAgentExecutor, None]:
    """
    Context manager for executor with cleanup.

    Example:
        >>> async with managed_executor(llm, tools) as executor:
        ...     result = await executor.run("Hello")
    """
    executor = AsyncAgentExecutor(llm, tools, **kwargs)
    try:
        yield executor
    finally:
        await executor.cancel_all()
