"""
Hierarchical multi-agent orchestration for complex agentic RAG tasks.

Implements supervisor-worker pattern with dynamic task allocation,
specialization, and coordination for state-of-the-art performance.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from autorag_live.utils import get_logger

logger = get_logger(__name__)


class AgentRole(Enum):
    """Agent roles in hierarchy."""

    SUPERVISOR = "supervisor"  # Orchestrates other agents
    RETRIEVER = "retriever"  # Specialized in retrieval
    REASONER = "reasoner"  # Specialized in reasoning
    SYNTHESIZER = "synthesizer"  # Synthesizes final answer
    VALIDATOR = "validator"  # Validates outputs
    WORKER = "worker"  # General purpose worker


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentCapability:
    """Agent capability definition."""

    name: str
    description: str
    cost: float = 1.0  # Relative cost
    latency_ms: float = 100.0  # Expected latency


@dataclass
class Task:
    """Task for agent execution."""

    task_id: str
    task_type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    dependencies: Set[str] = field(default_factory=set)
    assigned_agent: Optional[str] = None
    status: str = "pending"
    result: Optional[Any] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentMetadata:
    """Metadata for agent in hierarchy."""

    agent_id: str
    role: AgentRole
    capabilities: List[AgentCapability]
    current_load: int = 0
    max_load: int = 5
    total_tasks_completed: int = 0
    avg_latency_ms: float = 0.0

    @property
    def is_available(self) -> bool:
        """Check if agent can accept more tasks."""
        return self.current_load < self.max_load

    @property
    def utilization(self) -> float:
        """Calculate utilization (0-1)."""
        return self.current_load / self.max_load if self.max_load > 0 else 0.0


class HierarchicalAgent:
    """
    Base agent in hierarchical system.

    Can act as supervisor or worker depending on role.
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        capabilities: Optional[List[AgentCapability]] = None,
    ):
        self.metadata = AgentMetadata(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities or [],
        )
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running = False

    async def execute_task(self, task: Task) -> Any:
        """
        Execute a task.

        Args:
            task: Task to execute

        Returns:
            Task result
        """
        logger.info(f"Agent {self.metadata.agent_id} executing task {task.task_id}")

        start_time = time.time()

        # Simulate task execution
        await asyncio.sleep(0.05)  # Mock work

        # Mock result based on role
        if self.metadata.role == AgentRole.RETRIEVER:
            result = {"documents": [f"Doc {i}" for i in range(5)]}
        elif self.metadata.role == AgentRole.REASONER:
            result = {"reasoning": "Analyzed query and identified key concepts"}
        elif self.metadata.role == AgentRole.SYNTHESIZER:
            result = {"answer": "Synthesized answer from retrieved documents"}
        else:
            result = {"status": "completed"}

        # Update metrics
        latency_ms = (time.time() - start_time) * 1000
        self.metadata.total_tasks_completed += 1
        self.metadata.avg_latency_ms = (
            self.metadata.avg_latency_ms * (self.metadata.total_tasks_completed - 1) + latency_ms
        ) / self.metadata.total_tasks_completed

        return result

    async def start(self) -> None:
        """Start agent task processing loop."""
        self.running = True
        logger.info(f"Agent {self.metadata.agent_id} started")

    async def stop(self) -> None:
        """Stop agent."""
        self.running = False
        logger.info(f"Agent {self.metadata.agent_id} stopped")

    def can_handle(self, task_type: str) -> bool:
        """Check if agent can handle task type."""
        return any(cap.name == task_type for cap in self.metadata.capabilities)


class SupervisorAgent(HierarchicalAgent):
    """
    Supervisor agent that orchestrates workers.

    Responsible for:
    - Task decomposition
    - Agent selection
    - Work distribution
    - Result aggregation
    """

    def __init__(self, agent_id: str = "supervisor"):
        super().__init__(agent_id, AgentRole.SUPERVISOR)
        self.workers: Dict[str, HierarchicalAgent] = {}
        self.task_graph: Dict[str, Task] = {}

    def register_worker(self, worker: HierarchicalAgent) -> None:
        """Register a worker agent."""
        self.workers[worker.metadata.agent_id] = worker
        logger.info(
            f"Registered worker {worker.metadata.agent_id} "
            f"with role {worker.metadata.role.value}"
        )

    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker agent."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            logger.info(f"Unregistered worker {worker_id}")

    async def orchestrate_query(
        self,
        query: str,
        workflow: Optional[str] = "standard",
    ) -> Dict[str, Any]:
        """
        Orchestrate query processing across agents.

        Args:
            query: User query
            workflow: Workflow type (standard, complex, fast)

        Returns:
            Final result
        """
        logger.info(f"Orchestrating query: {query}")

        # Decompose into tasks
        tasks = self._decompose_query(query, workflow)

        # Execute task graph
        results = await self._execute_task_graph(tasks)

        # Aggregate results
        final_result = self._aggregate_results(results)

        return final_result

    def _decompose_query(self, query: str, workflow: str) -> List[Task]:
        """Decompose query into tasks."""
        tasks = []

        if workflow == "standard":
            # Standard RAG workflow
            tasks.append(
                Task(
                    task_id="retrieve",
                    task_type="retrieve",
                    priority=TaskPriority.HIGH,
                    payload={"query": query, "top_k": 5},
                )
            )

            tasks.append(
                Task(
                    task_id="reason",
                    task_type="reason",
                    priority=TaskPriority.NORMAL,
                    payload={"query": query},
                    dependencies={"retrieve"},
                )
            )

            tasks.append(
                Task(
                    task_id="synthesize",
                    task_type="synthesize",
                    priority=TaskPriority.HIGH,
                    payload={"query": query},
                    dependencies={"retrieve", "reason"},
                )
            )

        elif workflow == "complex":
            # Complex multi-step workflow
            tasks.extend(
                [
                    Task(
                        task_id="retrieve_dense",
                        task_type="retrieve",
                        priority=TaskPriority.HIGH,
                        payload={"query": query, "method": "dense"},
                    ),
                    Task(
                        task_id="retrieve_sparse",
                        task_type="retrieve",
                        priority=TaskPriority.HIGH,
                        payload={"query": query, "method": "sparse"},
                    ),
                    Task(
                        task_id="reason",
                        task_type="reason",
                        priority=TaskPriority.NORMAL,
                        payload={"query": query},
                        dependencies={"retrieve_dense", "retrieve_sparse"},
                    ),
                    Task(
                        task_id="synthesize",
                        task_type="synthesize",
                        priority=TaskPriority.HIGH,
                        payload={"query": query},
                        dependencies={"reason"},
                    ),
                    Task(
                        task_id="validate",
                        task_type="validate",
                        priority=TaskPriority.NORMAL,
                        payload={"answer": None},
                        dependencies={"synthesize"},
                    ),
                ]
            )

        else:  # fast workflow
            tasks.append(
                Task(
                    task_id="retrieve",
                    task_type="retrieve",
                    priority=TaskPriority.CRITICAL,
                    payload={"query": query, "top_k": 3},
                )
            )
            tasks.append(
                Task(
                    task_id="synthesize",
                    task_type="synthesize",
                    priority=TaskPriority.CRITICAL,
                    payload={"query": query},
                    dependencies={"retrieve"},
                )
            )

        return tasks

    async def _execute_task_graph(self, tasks: List[Task]) -> Dict[str, Any]:
        """Execute tasks respecting dependencies."""
        results = {}
        completed = set()

        # Execute tasks in dependency order
        while len(completed) < len(tasks):
            # Find ready tasks (dependencies met)
            ready_tasks = [
                task
                for task in tasks
                if task.task_id not in completed and task.dependencies.issubset(completed)
            ]

            if not ready_tasks:
                logger.warning("No ready tasks, possible cyclic dependency")
                break

            # Execute ready tasks in parallel
            execute_coroutines = [self._execute_single_task(task) for task in ready_tasks]

            task_results = await asyncio.gather(*execute_coroutines)

            # Record results
            for task, result in zip(ready_tasks, task_results):
                results[task.task_id] = result
                completed.add(task.task_id)

        return results

    async def _execute_single_task(self, task: Task) -> Any:
        """Execute single task by selecting and delegating to worker."""
        # Select best worker for task
        worker = self._select_worker(task)

        if not worker:
            logger.error(f"No available worker for task {task.task_id}")
            return {"error": "No available worker"}

        # Assign and execute
        task.assigned_agent = worker.metadata.agent_id
        worker.metadata.current_load += 1

        try:
            result = await worker.execute_task(task)
            task.status = "completed"
            task.result = result
            return result

        finally:
            worker.metadata.current_load -= 1

    def _select_worker(self, task: Task) -> Optional[HierarchicalAgent]:
        """Select best worker for task using capability matching and load balancing."""
        # Filter workers by capability
        capable_workers = [
            worker
            for worker in self.workers.values()
            if worker.can_handle(task.task_type) and worker.metadata.is_available
        ]

        if not capable_workers:
            return None

        # Sort by utilization (load balance)
        capable_workers.sort(key=lambda w: w.metadata.utilization)

        return capable_workers[0]

    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate task results into final output."""
        # Extract final answer from synthesize task
        if "synthesize" in results:
            return {
                "answer": results["synthesize"].get("answer", ""),
                "retrieved_docs": results.get("retrieve", {}).get("documents", []),
                "reasoning": results.get("reason", {}).get("reasoning", ""),
                "task_count": len(results),
                "status": "success",
            }

        return {
            "status": "error",
            "message": "No synthesize result",
            "results": results,
        }

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics for all workers."""
        stats = {}
        for worker_id, worker in self.workers.items():
            stats[worker_id] = {
                "role": worker.metadata.role.value,
                "current_load": worker.metadata.current_load,
                "utilization": worker.metadata.utilization,
                "total_completed": worker.metadata.total_tasks_completed,
                "avg_latency_ms": worker.metadata.avg_latency_ms,
            }
        return stats


# Factory functions
def create_rag_hierarchy() -> SupervisorAgent:
    """
    Create standard RAG hierarchy.

    Returns:
        Configured supervisor with workers
    """
    supervisor = SupervisorAgent()

    # Create specialized workers
    retriever = HierarchicalAgent(
        "retriever_1",
        AgentRole.RETRIEVER,
        [AgentCapability("retrieve", "Retrieve documents")],
    )

    reasoner = HierarchicalAgent(
        "reasoner_1",
        AgentRole.REASONER,
        [AgentCapability("reason", "Analyze query")],
    )

    synthesizer = HierarchicalAgent(
        "synthesizer_1",
        AgentRole.SYNTHESIZER,
        [AgentCapability("synthesize", "Generate answer")],
    )

    validator = HierarchicalAgent(
        "validator_1",
        AgentRole.VALIDATOR,
        [AgentCapability("validate", "Validate answer")],
    )

    # Register workers
    supervisor.register_worker(retriever)
    supervisor.register_worker(reasoner)
    supervisor.register_worker(synthesizer)
    supervisor.register_worker(validator)

    return supervisor


# Example usage
async def example_hierarchical_rag():
    """Example of hierarchical RAG."""
    # Create hierarchy
    supervisor = create_rag_hierarchy()

    # Process queries
    queries = [
        "What is machine learning?",
        "Explain deep learning",
        "How does RAG work?",
    ]

    for query in queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        print("=" * 50)

        result = await supervisor.orchestrate_query(query, workflow="standard")

        print(f"\nAnswer: {result.get('answer', 'N/A')}")
        print(f"Tasks completed: {result.get('task_count', 0)}")

    # Get worker stats
    print("\n" + "=" * 50)
    print("Worker Statistics")
    print("=" * 50)

    stats = supervisor.get_worker_stats()
    for worker_id, worker_stats in stats.items():
        print(f"\n{worker_id}:")
        for key, value in worker_stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(example_hierarchical_rag())
