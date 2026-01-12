"""
Modular Agent Base with Lifecycle Hooks and Plugin Architecture.

This module provides a state-of-the-art base agent class with:
- Proper lifecycle management (init, start, stop, cleanup)
- Plugin architecture for extensibility
- Event-driven communication
- Standardized state machine
- Observable patterns for debugging

Example:
    >>> class MyAgent(ModularAgent):
    ...     async def _execute_step(self, context):
    ...         # Custom step logic
    ...         return StepResult(...)
    ...
    >>> agent = MyAgent(config=AgentConfig())
    >>> agent.use(LoggingPlugin())
    >>> result = await agent.run("What is RAG?")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Agent State Machine
# =============================================================================


class AgentLifecycleState(Enum):
    """Agent lifecycle states."""

    CREATED = auto()  # Agent instantiated
    INITIALIZED = auto()  # Resources allocated
    RUNNING = auto()  # Processing queries
    PAUSED = auto()  # Temporarily suspended
    STOPPING = auto()  # Cleanup in progress
    STOPPED = auto()  # Fully terminated
    ERROR = auto()  # Error state


class AgentExecutionState(Enum):
    """States during query execution."""

    IDLE = "idle"
    PLANNING = "planning"
    REASONING = "reasoning"
    RETRIEVING = "retrieving"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    FAILED = "failed"


# =============================================================================
# Events and Hooks
# =============================================================================


class AgentEvent(Enum):
    """Events emitted during agent execution."""

    # Lifecycle events
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_ERROR = "agent_error"

    # Execution events
    QUERY_RECEIVED = "query_received"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    THOUGHT_GENERATED = "thought_generated"
    ACTION_PLANNED = "action_planned"
    ACTION_EXECUTED = "action_executed"
    OBSERVATION_RECEIVED = "observation_received"
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    GENERATION_STARTED = "generation_started"
    GENERATION_COMPLETED = "generation_completed"
    REFLECTION_STARTED = "reflection_started"
    REFLECTION_COMPLETED = "reflection_completed"
    ANSWER_GENERATED = "answer_generated"


@dataclass
class EventPayload:
    """Payload for agent events."""

    event: AgentEvent
    agent_id: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[EventPayload], None]
AsyncEventHandler = Callable[[EventPayload], asyncio.Future]


# =============================================================================
# Plugin System
# =============================================================================


@runtime_checkable
class AgentPlugin(Protocol):
    """Protocol for agent plugins."""

    @property
    def name(self) -> str:
        """Plugin identifier."""
        ...

    async def on_event(self, event: EventPayload) -> None:
        """Handle agent events."""
        ...


class BasePlugin(ABC):
    """Base class for plugins with default implementations."""

    def __init__(self, name: str):
        self._name = name
        self._enabled = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    async def on_event(self, event: EventPayload) -> None:
        """Override to handle specific events."""
        pass

    async def on_agent_start(self, agent: "ModularAgent") -> None:
        """Called when agent starts."""
        pass

    async def on_agent_stop(self, agent: "ModularAgent") -> None:
        """Called when agent stops."""
        pass


class LoggingPlugin(BasePlugin):
    """Plugin that logs all agent events."""

    def __init__(self, log_level: int = logging.INFO):
        super().__init__("logging")
        self.log_level = log_level
        self.logger = logging.getLogger("agent.events")

    async def on_event(self, event: EventPayload) -> None:
        self.logger.log(
            self.log_level,
            f"[{event.agent_id}] {event.event.value}: {event.data}",
        )


class MetricsPlugin(BasePlugin):
    """Plugin that collects agent metrics."""

    def __init__(self):
        super().__init__("metrics")
        self.metrics: Dict[str, Any] = {
            "total_queries": 0,
            "total_steps": 0,
            "total_retrievals": 0,
            "total_generations": 0,
            "latencies": [],
            "errors": 0,
        }
        self._query_start_times: Dict[str, float] = {}

    async def on_event(self, event: EventPayload) -> None:
        if event.event == AgentEvent.QUERY_RECEIVED:
            self.metrics["total_queries"] += 1
            self._query_start_times[event.session_id] = time.time()
        elif event.event == AgentEvent.STEP_COMPLETED:
            self.metrics["total_steps"] += 1
        elif event.event == AgentEvent.RETRIEVAL_COMPLETED:
            self.metrics["total_retrievals"] += 1
        elif event.event == AgentEvent.GENERATION_COMPLETED:
            self.metrics["total_generations"] += 1
        elif event.event == AgentEvent.ANSWER_GENERATED:
            if event.session_id in self._query_start_times:
                latency = time.time() - self._query_start_times[event.session_id]
                self.metrics["latencies"].append(latency)
        elif event.event == AgentEvent.AGENT_ERROR:
            self.metrics["errors"] += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        latencies = self.metrics["latencies"]
        return {
            **self.metrics,
            "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
            "p50_latency": sorted(latencies)[len(latencies) // 2] if latencies else 0,
            "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
        }


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AgentConfig:
    """Configuration for modular agent."""

    # Identity
    name: str = "modular_agent"
    version: str = "1.0.0"

    # Execution limits
    max_iterations: int = 10
    max_tokens_per_step: int = 4096
    timeout_seconds: float = 300.0

    # Behavior
    enable_reflection: bool = True
    enable_planning: bool = True
    enable_memory: bool = True
    verbose: bool = False

    # LLM settings
    temperature: float = 0.7
    model: str = "gpt-4"

    # Advanced
    retry_on_error: bool = True
    max_retries: int = 3
    backoff_factor: float = 1.5


@dataclass
class StepResult:
    """Result from a single agent step."""

    step_num: int
    state: AgentExecutionState
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    reflection: Optional[str] = None
    is_final: bool = False
    answer: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Complete result from agent execution."""

    query: str
    answer: str
    steps: List[StepResult] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    total_latency_ms: float = 0.0
    tokens_used: int = 0
    retrieval_count: int = 0
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_trace(self) -> str:
        """Get human-readable trace."""
        lines = [f"Query: {self.query}\n"]
        for step in self.steps:
            lines.append(f"Step {step.step_num}:")
            if step.thought:
                lines.append(f"  Thought: {step.thought}")
            if step.action:
                lines.append(f"  Action: {step.action}")
            if step.observation:
                lines.append(f"  Observation: {step.observation[:200]}...")
            if step.reflection:
                lines.append(f"  Reflection: {step.reflection}")
            lines.append("")
        lines.append(f"Answer: {self.answer}")
        return "\n".join(lines)


@dataclass
class ExecutionContext:
    """Context passed through agent execution."""

    query: str
    session_id: str
    step_num: int = 0
    history: List[StepResult] = field(default_factory=list)
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, result: StepResult) -> None:
        """Add step result to history."""
        self.history.append(result)
        self.step_num += 1


# =============================================================================
# Modular Agent Base
# =============================================================================


class ModularAgent(ABC):
    """
    Base class for modular agentic RAG agents.

    Provides lifecycle management, plugin architecture, and event system.
    Subclasses implement _execute_step for custom agent logic.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the agent."""
        self.config = config or AgentConfig()
        self.agent_id = str(uuid.uuid4())[:8]

        # State
        self._lifecycle_state = AgentLifecycleState.CREATED
        self._execution_state = AgentExecutionState.IDLE

        # Plugins and events
        self._plugins: Dict[str, AgentPlugin] = {}
        self._event_handlers: Dict[AgentEvent, List[EventHandler]] = {}
        self._async_event_handlers: Dict[AgentEvent, List[AsyncEventHandler]] = {}

        # Internal state
        self._current_session: Optional[str] = None
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize agent resources."""
        if self._lifecycle_state != AgentLifecycleState.CREATED:
            raise RuntimeError(f"Cannot initialize from state {self._lifecycle_state}")

        await self._on_initialize()
        self._lifecycle_state = AgentLifecycleState.INITIALIZED
        logger.info(f"Agent {self.agent_id} initialized")

    async def start(self) -> None:
        """Start the agent."""
        if self._lifecycle_state == AgentLifecycleState.CREATED:
            await self.initialize()

        if self._lifecycle_state not in (
            AgentLifecycleState.INITIALIZED,
            AgentLifecycleState.PAUSED,
        ):
            raise RuntimeError(f"Cannot start from state {self._lifecycle_state}")

        await self._on_start()
        self._lifecycle_state = AgentLifecycleState.RUNNING

        # Notify plugins
        for plugin in self._plugins.values():
            if hasattr(plugin, "on_agent_start"):
                await plugin.on_agent_start(self)

        await self._emit_event(AgentEvent.AGENT_STARTED)
        logger.info(f"Agent {self.agent_id} started")

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        if self._lifecycle_state not in (
            AgentLifecycleState.RUNNING,
            AgentLifecycleState.PAUSED,
            AgentLifecycleState.ERROR,
        ):
            return

        self._lifecycle_state = AgentLifecycleState.STOPPING
        await self._on_stop()

        # Notify plugins
        for plugin in self._plugins.values():
            if hasattr(plugin, "on_agent_stop"):
                await plugin.on_agent_stop(self)

        await self._emit_event(AgentEvent.AGENT_STOPPED)
        self._lifecycle_state = AgentLifecycleState.STOPPED
        logger.info(f"Agent {self.agent_id} stopped")

    async def pause(self) -> None:
        """Pause agent execution."""
        if self._lifecycle_state == AgentLifecycleState.RUNNING:
            self._lifecycle_state = AgentLifecycleState.PAUSED
            logger.info(f"Agent {self.agent_id} paused")

    async def resume(self) -> None:
        """Resume agent execution."""
        if self._lifecycle_state == AgentLifecycleState.PAUSED:
            self._lifecycle_state = AgentLifecycleState.RUNNING
            logger.info(f"Agent {self.agent_id} resumed")

    @asynccontextmanager
    async def managed_execution(self) -> AsyncIterator["ModularAgent"]:
        """Context manager for agent lifecycle."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()

    # -------------------------------------------------------------------------
    # Plugin Management
    # -------------------------------------------------------------------------

    def use(self, plugin: AgentPlugin) -> "ModularAgent":
        """Register a plugin."""
        self._plugins[plugin.name] = plugin
        logger.debug(f"Plugin {plugin.name} registered")
        return self

    def remove_plugin(self, name: str) -> None:
        """Remove a plugin."""
        if name in self._plugins:
            del self._plugins[name]

    def get_plugin(self, name: str) -> Optional[AgentPlugin]:
        """Get a registered plugin."""
        return self._plugins.get(name)

    # -------------------------------------------------------------------------
    # Event System
    # -------------------------------------------------------------------------

    def on(self, event: AgentEvent, handler: Union[EventHandler, AsyncEventHandler]) -> None:
        """Register event handler."""
        if asyncio.iscoroutinefunction(handler):
            if event not in self._async_event_handlers:
                self._async_event_handlers[event] = []
            self._async_event_handlers[event].append(handler)
        else:
            if event not in self._event_handlers:
                self._event_handlers[event] = []
            self._event_handlers[event].append(handler)

    async def _emit_event(
        self,
        event: AgentEvent,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit an event to all handlers and plugins."""
        payload = EventPayload(
            event=event,
            agent_id=self.agent_id,
            session_id=self._current_session or "",
            data=data or {},
            metadata=metadata or {},
        )

        # Call sync handlers
        for handler in self._event_handlers.get(event, []):
            try:
                handler(payload)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

        # Call async handlers
        for handler in self._async_event_handlers.get(event, []):
            try:
                await handler(payload)
            except Exception as e:
                logger.error(f"Async event handler error: {e}")

        # Notify plugins
        for plugin in self._plugins.values():
            if hasattr(plugin, "enabled") and not plugin.enabled:
                continue
            try:
                await plugin.on_event(payload)
            except Exception as e:
                logger.error(f"Plugin {plugin.name} error: {e}")

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def run(
        self,
        query: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> AgentResult:
        """
        Execute the agent on a query.

        Args:
            query: User query to process
            context: Optional additional context
            stream: Whether to stream results

        Returns:
            AgentResult with answer and execution trace
        """
        # Ensure agent is running
        if self._lifecycle_state != AgentLifecycleState.RUNNING:
            await self.start()

        async with self._lock:
            session_id = str(uuid.uuid4())
            self._current_session = session_id
            start_time = time.time()

            # Create execution context
            exec_context = ExecutionContext(
                query=query,
                session_id=session_id,
                metadata=context or {},
            )

            await self._emit_event(
                AgentEvent.QUERY_RECEIVED,
                {"query": query},
            )

            try:
                self._execution_state = AgentExecutionState.PLANNING
                result = await self._execute_loop(exec_context)

                result.total_latency_ms = (time.time() - start_time) * 1000
                await self._emit_event(
                    AgentEvent.ANSWER_GENERATED,
                    {"answer": result.answer, "steps": len(result.steps)},
                )

                return result

            except Exception as e:
                logger.error(f"Agent execution error: {e}")
                self._lifecycle_state = AgentLifecycleState.ERROR
                await self._emit_event(AgentEvent.AGENT_ERROR, {"error": str(e)})

                return AgentResult(
                    query=query,
                    answer="",
                    success=False,
                    error=str(e),
                    session_id=session_id,
                    total_latency_ms=(time.time() - start_time) * 1000,
                )

            finally:
                self._execution_state = AgentExecutionState.IDLE
                self._current_session = None
                if self._lifecycle_state == AgentLifecycleState.ERROR:
                    self._lifecycle_state = AgentLifecycleState.RUNNING

    async def _execute_loop(self, context: ExecutionContext) -> AgentResult:
        """Main execution loop."""
        steps: List[StepResult] = []
        final_answer = ""

        for iteration in range(self.config.max_iterations):
            await self._emit_event(
                AgentEvent.STEP_STARTED,
                {"step": iteration},
            )

            # Execute single step
            step_result = await self._execute_step(context)
            steps.append(step_result)
            context.add_step(step_result)

            await self._emit_event(
                AgentEvent.STEP_COMPLETED,
                {
                    "step": iteration,
                    "state": step_result.state.value,
                    "is_final": step_result.is_final,
                },
            )

            # Check if done
            if step_result.is_final:
                final_answer = step_result.answer or ""
                break

            # Optional reflection
            if self.config.enable_reflection and iteration > 0:
                await self._reflect(context, step_result)

        return AgentResult(
            query=context.query,
            answer=final_answer,
            steps=steps,
            success=bool(final_answer),
            session_id=context.session_id,
        )

    # -------------------------------------------------------------------------
    # Abstract Methods (implement in subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def _execute_step(self, context: ExecutionContext) -> StepResult:
        """
        Execute a single agent step.

        Override this in subclasses to implement custom agent logic.

        Args:
            context: Current execution context

        Returns:
            StepResult with step outcome
        """
        ...

    # -------------------------------------------------------------------------
    # Hooks (override in subclasses)
    # -------------------------------------------------------------------------

    async def _on_initialize(self) -> None:
        """Hook called during initialization. Override for setup."""
        pass

    async def _on_start(self) -> None:
        """Hook called when agent starts. Override for startup logic."""
        pass

    async def _on_stop(self) -> None:
        """Hook called when agent stops. Override for cleanup."""
        pass

    async def _reflect(self, context: ExecutionContext, step: StepResult) -> None:
        """Optional reflection after each step. Override for custom logic."""
        pass

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def lifecycle_state(self) -> AgentLifecycleState:
        """Current lifecycle state."""
        return self._lifecycle_state

    @property
    def execution_state(self) -> AgentExecutionState:
        """Current execution state."""
        return self._execution_state

    @property
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._lifecycle_state == AgentLifecycleState.RUNNING

    @property
    def plugins(self) -> List[str]:
        """List registered plugin names."""
        return list(self._plugins.keys())


# =============================================================================
# Convenience Decorators
# =============================================================================


def agent_step(state: AgentExecutionState):
    """Decorator to track agent step state transitions."""

    def decorator(func):
        async def wrapper(self: ModularAgent, *args, **kwargs):
            self._execution_state = state
            try:
                return await func(self, *args, **kwargs)
            finally:
                pass  # State restored by caller

        return wrapper

    return decorator
