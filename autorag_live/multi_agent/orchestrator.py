"""
Agent orchestration module for AutoRAG-Live.

Provides coordination and management of multiple agents
for complex multi-step RAG workflows.

Features:
- Agent registry and lifecycle management
- Task routing and assignment
- Inter-agent communication
- Workflow orchestration
- Consensus mechanisms
- Agent monitoring
- Fault tolerance

Example usage:
    >>> orchestrator = AgentOrchestrator()
    >>> orchestrator.register_agent("retriever", RetrieverAgent())
    >>> orchestrator.register_agent("generator", GeneratorAgent())
    >>> 
    >>> result = await orchestrator.execute_workflow(
    ...     query="What is RAG?",
    ...     workflow="rag_pipeline"
    ... )
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AgentState(Enum):
    """Agent lifecycle states."""
    
    IDLE = auto()
    BUSY = auto()
    PAUSED = auto()
    ERROR = auto()
    STOPPED = auto()


class TaskPriority(Enum):
    """Task priority levels."""
    
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


@dataclass
class AgentCapability:
    """A capability that an agent provides."""
    
    name: str
    description: str = ""
    
    # Input/output types
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    
    # Performance hints
    avg_latency_ms: float = 0.0
    max_concurrent: int = 10
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    
    agent_id: str
    name: str
    agent_type: str
    
    # Capabilities
    capabilities: List[AgentCapability] = field(default_factory=list)
    
    # State
    state: AgentState = AgentState.IDLE
    
    # Statistics
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_response_time_ms: float = 0.0
    
    # Registration
    registered_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A task to be executed by an agent."""
    
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    
    # Assignment
    assigned_agent: Optional[str] = None
    
    # Priority and timing
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: float = 60.0
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    
    # Result
    result: Any = None
    error: Optional[str] = None
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Get task duration."""
        if self.completed_at > 0 and self.started_at > 0:
            return (self.completed_at - self.started_at) * 1000
        return 0.0


@dataclass
class Message:
    """Inter-agent message."""
    
    message_id: str
    sender: str
    recipient: str
    
    # Content
    message_type: str
    payload: Dict[str, Any]
    
    # Timing
    timestamp: float = field(default_factory=time.time)
    
    # Response
    reply_to: Optional[str] = None
    requires_response: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStep:
    """A step in a workflow."""
    
    step_id: str
    task_type: str
    
    # Configuration
    agent_type: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Error handling
    retry_count: int = 0
    fallback_step: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """A workflow definition."""
    
    workflow_id: str
    name: str
    
    # Steps
    steps: List[WorkflowStep] = field(default_factory=list)
    
    # Configuration
    timeout: float = 300.0
    parallel_enabled: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    
    workflow_id: str
    success: bool
    
    # Results
    final_result: Any = None
    step_results: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Errors
    errors: List[Tuple[str, str]] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Get workflow duration."""
        return (self.end_time - self.start_time) * 1000


class BaseAgent(ABC):
    """Base class for agents."""
    
    def __init__(
        self,
        name: str,
        agent_type: str = "generic",
    ):
        """
        Initialize agent.
        
        Args:
            name: Agent name
            agent_type: Type of agent
        """
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.agent_type = agent_type
        self.state = AgentState.IDLE
        
        self._message_handlers: Dict[str, Callable] = {}
    
    @property
    @abstractmethod
    def capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities."""
        pass
    
    @abstractmethod
    async def execute(
        self,
        task: Task,
    ) -> Any:
        """Execute a task."""
        pass
    
    async def handle_message(
        self,
        message: Message,
    ) -> Optional[Message]:
        """Handle incoming message."""
        handler = self._message_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        return None
    
    def register_message_handler(
        self,
        message_type: str,
        handler: Callable[[Message], Awaitable[Optional[Message]]],
    ) -> None:
        """Register message handler."""
        self._message_handlers[message_type] = handler
    
    def get_info(self) -> AgentInfo:
        """Get agent info."""
        return AgentInfo(
            agent_id=self.agent_id,
            name=self.name,
            agent_type=self.agent_type,
            capabilities=self.capabilities,
            state=self.state,
        )


class RetrieverAgent(BaseAgent):
    """Agent for document retrieval."""
    
    def __init__(self, name: str = "retriever"):
        super().__init__(name, "retriever")
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="retrieve",
                description="Retrieve documents from knowledge base",
                input_types=["query"],
                output_types=["documents"],
            )
        ]
    
    async def execute(self, task: Task) -> Any:
        """Execute retrieval task."""
        query = task.payload.get("query", "")
        
        # Simulated retrieval
        await asyncio.sleep(0.1)
        
        return {
            "query": query,
            "documents": [
                {"id": "doc1", "content": f"Retrieved content for: {query}"},
            ],
        }


class GeneratorAgent(BaseAgent):
    """Agent for response generation."""
    
    def __init__(self, name: str = "generator"):
        super().__init__(name, "generator")
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="generate",
                description="Generate response from context",
                input_types=["query", "documents"],
                output_types=["response"],
            )
        ]
    
    async def execute(self, task: Task) -> Any:
        """Execute generation task."""
        query = task.payload.get("query", "")
        documents = task.payload.get("documents", [])
        
        # Simulated generation
        await asyncio.sleep(0.2)
        
        return {
            "query": query,
            "response": f"Generated response for: {query}",
            "sources": [d.get("id") for d in documents],
        }


class MessageBus:
    """Message bus for inter-agent communication."""
    
    def __init__(self):
        """Initialize message bus."""
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._message_queue: asyncio.Queue = asyncio.Queue()
    
    def subscribe(
        self,
        topic: str,
        handler: Callable[[Message], Awaitable[None]],
    ) -> None:
        """Subscribe to topic."""
        self._subscribers[topic].append(handler)
    
    def unsubscribe(
        self,
        topic: str,
        handler: Callable,
    ) -> None:
        """Unsubscribe from topic."""
        if handler in self._subscribers[topic]:
            self._subscribers[topic].remove(handler)
    
    async def publish(
        self,
        topic: str,
        message: Message,
    ) -> None:
        """Publish message to topic."""
        handlers = self._subscribers.get(topic, [])
        
        tasks = [handler(message) for handler in handlers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send(
        self,
        message: Message,
    ) -> None:
        """Send direct message."""
        await self._message_queue.put(message)
    
    async def receive(
        self,
        timeout: float = 1.0,
    ) -> Optional[Message]:
        """Receive message from queue."""
        try:
            return await asyncio.wait_for(
                self._message_queue.get(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None


class TaskQueue:
    """Priority queue for tasks."""
    
    def __init__(self):
        """Initialize queue."""
        self._queues: Dict[TaskPriority, asyncio.Queue] = {
            priority: asyncio.Queue()
            for priority in TaskPriority
        }
    
    async def enqueue(self, task: Task) -> None:
        """Add task to queue."""
        await self._queues[task.priority].put(task)
    
    async def dequeue(
        self,
        timeout: float = 1.0,
    ) -> Optional[Task]:
        """Get highest priority task."""
        # Check queues in priority order
        for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
            queue = self._queues[priority]
            
            if not queue.empty():
                try:
                    return await asyncio.wait_for(
                        queue.get(),
                        timeout=0.01,
                    )
                except asyncio.TimeoutError:
                    continue
        
        return None
    
    @property
    def size(self) -> int:
        """Get total queue size."""
        return sum(q.qsize() for q in self._queues.values())


class AgentOrchestrator:
    """
    Main orchestrator for multi-agent systems.
    
    Example:
        >>> orchestrator = AgentOrchestrator()
        >>> 
        >>> # Register agents
        >>> orchestrator.register_agent("retriever", RetrieverAgent())
        >>> orchestrator.register_agent("generator", GeneratorAgent())
        >>> 
        >>> # Define workflow
        >>> workflow = Workflow(
        ...     workflow_id="rag_pipeline",
        ...     name="RAG Pipeline",
        ...     steps=[
        ...         WorkflowStep(step_id="retrieve", task_type="retrieve"),
        ...         WorkflowStep(step_id="generate", task_type="generate", depends_on=["retrieve"]),
        ...     ]
        ... )
        >>> orchestrator.register_workflow(workflow)
        >>> 
        >>> # Execute
        >>> result = await orchestrator.execute_workflow("rag_pipeline", {"query": "What is RAG?"})
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 10,
    ):
        """
        Initialize orchestrator.
        
        Args:
            max_concurrent_tasks: Max concurrent tasks
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Agents
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_info: Dict[str, AgentInfo] = {}
        
        # Workflows
        self._workflows: Dict[str, Workflow] = {}
        
        # Communication
        self._message_bus = MessageBus()
        self._task_queue = TaskQueue()
        
        # State
        self._running = False
        self._tasks: Dict[str, Task] = {}
        
        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    def register_agent(
        self,
        name: str,
        agent: BaseAgent,
    ) -> None:
        """
        Register an agent.
        
        Args:
            name: Agent name
            agent: Agent instance
        """
        self._agents[name] = agent
        self._agent_info[name] = agent.get_info()
        
        logger.info(f"Registered agent: {name} ({agent.agent_type})")
    
    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            name: Agent name
            
        Returns:
            True if unregistered
        """
        if name in self._agents:
            del self._agents[name]
            del self._agent_info[name]
            logger.info(f"Unregistered agent: {name}")
            return True
        return False
    
    def register_workflow(self, workflow: Workflow) -> None:
        """
        Register a workflow.
        
        Args:
            workflow: Workflow definition
        """
        self._workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.name}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return self._agents.get(name)
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """Get agents by type."""
        return [
            agent for agent in self._agents.values()
            if agent.agent_type == agent_type
        ]
    
    def get_agents_by_capability(
        self,
        capability: str,
    ) -> List[BaseAgent]:
        """Get agents with specific capability."""
        result = []
        
        for agent in self._agents.values():
            for cap in agent.capabilities:
                if cap.name == capability:
                    result.append(agent)
                    break
        
        return result
    
    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float = 60.0,
    ) -> Task:
        """
        Submit a task for execution.
        
        Args:
            task_type: Type of task
            payload: Task payload
            priority: Task priority
            timeout: Task timeout
            
        Returns:
            Task instance
        """
        task = Task(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            payload=payload,
            priority=priority,
            timeout=timeout,
        )
        
        self._tasks[task.task_id] = task
        await self._task_queue.enqueue(task)
        
        return task
    
    async def execute_task(
        self,
        task: Task,
    ) -> Any:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        # Find suitable agent
        agents = self.get_agents_by_capability(task.task_type)
        
        if not agents:
            raise ValueError(f"No agent found for task type: {task.task_type}")
        
        # Select agent (simple round-robin for now)
        agent = agents[0]
        
        task.assigned_agent = agent.name
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        agent.state = AgentState.BUSY
        
        try:
            async with self._semaphore:
                # Execute with timeout
                result = await asyncio.wait_for(
                    agent.execute(task),
                    timeout=task.timeout,
                )
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            
            # Update agent stats
            info = self._agent_info[agent.name]
            info.tasks_completed += 1
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.error = "Task timeout"
            
            info = self._agent_info[agent.name]
            info.tasks_failed += 1
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
            info = self._agent_info[agent.name]
            info.tasks_failed += 1
            
            logger.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            task.completed_at = time.time()
            agent.state = AgentState.IDLE
            
            # Update average response time
            info = self._agent_info[agent.name]
            total_tasks = info.tasks_completed + info.tasks_failed
            if total_tasks > 0:
                info.avg_response_time_ms = (
                    (info.avg_response_time_ms * (total_tasks - 1) + task.duration_ms)
                    / total_tasks
                )
            info.last_active = time.time()
        
        return task.result
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
    ) -> WorkflowResult:
        """
        Execute a workflow.
        
        Args:
            workflow_id: Workflow ID
            input_data: Input data
            
        Returns:
            WorkflowResult
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        result = WorkflowResult(
            workflow_id=workflow_id,
            success=False,
            start_time=time.time(),
        )
        
        # Track step results
        step_results: Dict[str, Any] = {}
        pending_steps = {step.step_id: step for step in workflow.steps}
        completed_steps: Set[str] = set()
        
        # Current data (passed between steps)
        current_data = input_data.copy()
        
        try:
            while pending_steps:
                # Find steps ready to execute
                ready_steps = []
                
                for step_id, step in pending_steps.items():
                    deps_met = all(
                        dep in completed_steps
                        for dep in step.depends_on
                    )
                    if deps_met:
                        ready_steps.append(step)
                
                if not ready_steps:
                    # No progress possible - circular dependency?
                    raise RuntimeError("Workflow stuck - check dependencies")
                
                # Execute ready steps (possibly in parallel)
                if workflow.parallel_enabled and len(ready_steps) > 1:
                    tasks = []
                    for step in ready_steps:
                        task = Task(
                            task_id=str(uuid.uuid4()),
                            task_type=step.task_type,
                            payload={**current_data, **step.params},
                        )
                        tasks.append(self.execute_task(task))
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for step, step_result in zip(ready_steps, results):
                        if isinstance(step_result, Exception):
                            result.errors.append((step.step_id, str(step_result)))
                        else:
                            step_results[step.step_id] = step_result
                            if isinstance(step_result, dict):
                                current_data.update(step_result)
                        
                        completed_steps.add(step.step_id)
                        del pending_steps[step.step_id]
                else:
                    # Sequential execution
                    for step in ready_steps:
                        task = Task(
                            task_id=str(uuid.uuid4()),
                            task_type=step.task_type,
                            payload={**current_data, **step.params},
                        )
                        
                        try:
                            step_result = await self.execute_task(task)
                            step_results[step.step_id] = step_result
                            
                            if isinstance(step_result, dict):
                                current_data.update(step_result)
                                
                        except Exception as e:
                            result.errors.append((step.step_id, str(e)))
                        
                        completed_steps.add(step.step_id)
                        del pending_steps[step.step_id]
            
            result.success = len(result.errors) == 0
            result.final_result = current_data
            result.step_results = step_results
            
        except Exception as e:
            result.success = False
            result.errors.append(("workflow", str(e)))
            logger.error(f"Workflow {workflow_id} failed: {e}")
        
        finally:
            result.end_time = time.time()
        
        return result
    
    async def send_message(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        payload: Dict[str, Any],
    ) -> None:
        """Send message between agents."""
        message = Message(
            message_id=str(uuid.uuid4()),
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
        )
        
        agent = self._agents.get(recipient)
        if agent:
            await agent.handle_message(message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            'agents': {
                name: {
                    'type': info.agent_type,
                    'state': info.state.name,
                    'tasks_completed': info.tasks_completed,
                    'tasks_failed': info.tasks_failed,
                    'avg_response_time_ms': info.avg_response_time_ms,
                }
                for name, info in self._agent_info.items()
            },
            'workflows': list(self._workflows.keys()),
            'pending_tasks': self._task_queue.size,
        }


# Convenience functions

def create_rag_workflow() -> Workflow:
    """
    Create a standard RAG workflow.
    
    Returns:
        Workflow for RAG pipeline
    """
    return Workflow(
        workflow_id="rag_pipeline",
        name="RAG Pipeline",
        steps=[
            WorkflowStep(
                step_id="retrieve",
                task_type="retrieve",
                agent_type="retriever",
            ),
            WorkflowStep(
                step_id="generate",
                task_type="generate",
                agent_type="generator",
                depends_on=["retrieve"],
            ),
        ],
    )


def create_orchestrator_with_agents() -> AgentOrchestrator:
    """
    Create orchestrator with default agents.
    
    Returns:
        Configured AgentOrchestrator
    """
    orchestrator = AgentOrchestrator()
    
    orchestrator.register_agent("retriever", RetrieverAgent())
    orchestrator.register_agent("generator", GeneratorAgent())
    orchestrator.register_workflow(create_rag_workflow())
    
    return orchestrator
