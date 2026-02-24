"""
Batched parallel retrieval for maximum throughput in agentic RAG.

Leverages concurrent execution to process multiple queries and
retrievers simultaneously, achieving 5-10x speedup over sequential.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

from autorag_live.utils import get_logger

logger = get_logger(__name__)


class RetrievalPriority(IntEnum):
    """Task priority levels for the priority-queue scheduler.

    Lower integer value = higher priority (min-heap ordering).
    """

    CRITICAL = 0  # Multi-hop reasoning / tool calls in the hot path
    HIGH = 1  # User-facing, interactive queries
    NORMAL = 2  # Standard retrieval
    LOW = 3  # Background pre-fetch / speculative retrieval


@dataclass
class RetrievalTask:
    """Single retrieval task."""

    task_id: str
    query: str
    retriever_name: str
    top_k: int = 5
    priority: RetrievalPriority = RetrievalPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Support ordering in a min-heap by (priority, task_id) so that tasks with
    # the same priority are served FIFO.
    def __lt__(self, other: "RetrievalTask") -> bool:
        return (self.priority, self.task_id) < (other.priority, other.task_id)


@dataclass
class RetrievalResult:
    """Result from retrieval task."""

    task_id: str
    query: str
    documents: List[Dict[str, Any]]
    retriever_name: str
    latency_ms: float
    success: bool = True
    error: Optional[str] = None


@dataclass
class BatchMetrics:
    """Metrics for batch retrieval."""

    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    total_time_ms: float
    avg_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    throughput_qps: float

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "total_time_ms": self.total_time_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "throughput_qps": self.throughput_qps,
        }


class ParallelRetriever:
    """
    Execute retrieval operations in parallel using asyncio.

    Supports multiple retrieval strategies executed concurrently
    for maximum throughput.
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        timeout_seconds: float = 30.0,
    ):
        """
        Initialize parallel retriever.

        Args:
            max_concurrent: Maximum concurrent retrievals
            timeout_seconds: Timeout for each retrieval
        """
        self.max_concurrent = max_concurrent
        self.timeout_seconds = timeout_seconds
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def retrieve_parallel(
        self,
        tasks: List[RetrievalTask],
        retriever_fn: Callable,
    ) -> Tuple[List[RetrievalResult], BatchMetrics]:
        """
        Execute retrieval tasks in parallel.

        Args:
            tasks: List of retrieval tasks
            retriever_fn: Retrieval function

        Returns:
            Tuple of (results, metrics)
        """
        start_time = time.time()

        # Create coroutines for each task
        coroutines = [self._retrieve_single(task, retriever_fn) for task in tasks]

        # Execute in parallel
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Process results
        valid_results = []
        latencies = []

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Retrieval failed: {result}")
                continue

            valid_results.append(result)
            if result.success:
                latencies.append(result.latency_ms)

        # Calculate metrics
        total_time_ms = (time.time() - start_time) * 1000

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
        else:
            avg_latency = max_latency = min_latency = 0.0

        successful = sum(1 for r in valid_results if r.success)
        failed = len(tasks) - successful

        throughput = len(tasks) / (total_time_ms / 1000) if total_time_ms > 0 else 0.0

        metrics = BatchMetrics(
            total_tasks=len(tasks),
            successful_tasks=successful,
            failed_tasks=failed,
            total_time_ms=total_time_ms,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            min_latency_ms=min_latency,
            throughput_qps=throughput,
        )

        return valid_results, metrics

    async def _retrieve_single(
        self,
        task: RetrievalTask,
        retriever_fn: Callable,
    ) -> RetrievalResult:
        """Execute single retrieval task."""
        async with self.semaphore:
            start_time = time.time()

            try:
                # Call retriever function
                if asyncio.iscoroutinefunction(retriever_fn):
                    documents = await asyncio.wait_for(
                        retriever_fn(task.query, task.top_k),
                        timeout=self.timeout_seconds,
                    )
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    documents = await asyncio.wait_for(
                        loop.run_in_executor(None, retriever_fn, task.query, task.top_k),
                        timeout=self.timeout_seconds,
                    )

                latency_ms = (time.time() - start_time) * 1000

                return RetrievalResult(
                    task_id=task.task_id,
                    query=task.query,
                    documents=documents,
                    retriever_name=task.retriever_name,
                    latency_ms=latency_ms,
                    success=True,
                )

            except asyncio.TimeoutError:
                latency_ms = (time.time() - start_time) * 1000
                return RetrievalResult(
                    task_id=task.task_id,
                    query=task.query,
                    documents=[],
                    retriever_name=task.retriever_name,
                    latency_ms=latency_ms,
                    success=False,
                    error="Timeout",
                )

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                return RetrievalResult(
                    task_id=task.task_id,
                    query=task.query,
                    documents=[],
                    retriever_name=task.retriever_name,
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e),
                )


class MultiRetrieverParallel:
    """
    Execute multiple retrievers in parallel for a single query.

    Combines results from BM25, dense, hybrid retrievers concurrently.
    """

    def __init__(self):
        self.parallel_retriever = ParallelRetriever()

    async def retrieve_multi(
        self,
        query: str,
        retrievers: Dict[str, Callable],
        top_k: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve using multiple retrievers in parallel.

        Args:
            query: Search query
            retrievers: Dictionary of retriever_name -> retriever_fn
            top_k: Number of results per retriever

        Returns:
            Dictionary of retriever_name -> documents
        """
        # Create tasks for each retriever
        tasks = [
            RetrievalTask(
                task_id=f"{query}_{name}",
                query=query,
                retriever_name=name,
                top_k=top_k,
            )
            for name in retrievers.keys()
        ]

        # Execute in parallel
        async def retrieve_with_name(task: RetrievalTask) -> RetrievalResult:
            retriever_fn = retrievers[task.retriever_name]
            return await self.parallel_retriever._retrieve_single(task, retriever_fn)

        results = await asyncio.gather(*[retrieve_with_name(task) for task in tasks])

        # Organize by retriever name
        results_by_retriever = {}
        for result in results:
            if result.success:
                results_by_retriever[result.retriever_name] = result.documents

        return results_by_retriever

    async def retrieve_multi_fusion(
        self,
        query: str,
        retrievers: Dict[str, Callable],
        top_k: int = 10,
        fusion_method: str = "reciprocal_rank",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and fuse results from multiple retrievers.

        Args:
            query: Search query
            retrievers: Dictionary of retriever_name -> retriever_fn
            top_k: Final number of results
            fusion_method: Fusion method (reciprocal_rank, weighted_sum)

        Returns:
            Fused and ranked documents
        """
        # Retrieve from all retrievers
        results_by_retriever = await self.retrieve_multi(query, retrievers, top_k * 2)

        # Fuse results
        if fusion_method == "reciprocal_rank":
            fused = self._reciprocal_rank_fusion(results_by_retriever)
        else:
            fused = self._weighted_sum_fusion(results_by_retriever)

        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self, results_by_retriever: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Fuse using reciprocal rank fusion."""
        # Score documents by reciprocal rank
        doc_scores: Dict[str, float] = {}

        for retriever_name, documents in results_by_retriever.items():
            for rank, doc in enumerate(documents, start=1):
                doc_id = doc.get("id", doc.get("text", ""))
                score = 1.0 / (60 + rank)  # RRF formula

                if doc_id in doc_scores:
                    doc_scores[doc_id] += score
                else:
                    doc_scores[doc_id] = score

        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Get documents
        doc_map = {}
        for documents in results_by_retriever.values():
            for doc in documents:
                doc_id = doc.get("id", doc.get("text", ""))
                doc_map[doc_id] = doc

        return [doc_map[doc_id] for doc_id, _ in sorted_docs if doc_id in doc_map]

    def _weighted_sum_fusion(
        self, results_by_retriever: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Fuse using weighted sum of scores."""
        # Equal weights for simplicity
        weights = {name: 1.0 / len(results_by_retriever) for name in results_by_retriever}

        doc_scores: Dict[str, float] = {}

        for retriever_name, documents in results_by_retriever.items():
            weight = weights[retriever_name]

            for doc in documents:
                doc_id = doc.get("id", doc.get("text", ""))
                score = doc.get("score", 0.0) * weight

                if doc_id in doc_scores:
                    doc_scores[doc_id] += score
                else:
                    doc_scores[doc_id] = score

        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Get documents
        doc_map = {}
        for documents in results_by_retriever.values():
            for doc in documents:
                doc_id = doc.get("id", doc.get("text", ""))
                doc_map[doc_id] = doc

        return [doc_map[doc_id] for doc_id, _ in sorted_docs if doc_id in doc_map]


class BatchQueryProcessor:
    """
    Process batches of queries efficiently.

    Optimizes for throughput when processing many queries.
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_concurrent_batches: int = 4,
    ):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.parallel_retriever = ParallelRetriever()

    async def process_batch(
        self,
        queries: List[str],
        retriever_fn: Callable,
        top_k: int = 5,
    ) -> Tuple[List[RetrievalResult], BatchMetrics]:
        """
        Process batch of queries.

        Args:
            queries: List of queries
            retriever_fn: Retrieval function
            top_k: Results per query

        Returns:
            Tuple of (results, metrics)
        """
        # Create tasks
        tasks = [
            RetrievalTask(
                task_id=f"batch_{idx}",
                query=query,
                retriever_name="batch_retriever",
                top_k=top_k,
            )
            for idx, query in enumerate(queries)
        ]

        # Process in parallel
        return await self.parallel_retriever.retrieve_parallel(tasks, retriever_fn)

    async def process_large_batch(
        self,
        queries: List[str],
        retriever_fn: Callable,
        top_k: int = 5,
    ) -> Tuple[List[RetrievalResult], BatchMetrics]:
        """
        Process large batch by splitting into smaller batches.

        Args:
            queries: List of queries
            retriever_fn: Retrieval function
            top_k: Results per query

        Returns:
            Tuple of (results, metrics)
        """
        # Split into batches
        batches = [
            queries[i : i + self.batch_size] for i in range(0, len(queries), self.batch_size)
        ]

        logger.info(f"Processing {len(queries)} queries in {len(batches)} batches")

        # Process batches
        all_results = []
        start_time = time.time()

        for batch_idx, batch in enumerate(batches):
            batch_results, batch_metrics = await self.process_batch(batch, retriever_fn, top_k)
            all_results.extend(batch_results)

            logger.debug(
                f"Batch {batch_idx + 1}/{len(batches)}: "
                f"{batch_metrics.successful_tasks}/{batch_metrics.total_tasks} succeeded, "
                f"{batch_metrics.throughput_qps:.1f} QPS"
            )

        # Aggregate metrics
        total_time_ms = (time.time() - start_time) * 1000
        successful = sum(1 for r in all_results if r.success)

        latencies = [r.latency_ms for r in all_results if r.success]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0
        min_latency = min(latencies) if latencies else 0.0

        throughput = len(queries) / (total_time_ms / 1000) if total_time_ms > 0 else 0.0

        metrics = BatchMetrics(
            total_tasks=len(queries),
            successful_tasks=successful,
            failed_tasks=len(queries) - successful,
            total_time_ms=total_time_ms,
            avg_latency_ms=avg_latency,
            max_latency_ms=max_latency,
            min_latency_ms=min_latency,
            throughput_qps=throughput,
        )

        return all_results, metrics


class PriorityParallelRetriever:
    """
    Priority-queue-based parallel retriever.

    Ensures that CRITICAL / HIGH priority tasks (e.g. multi-hop tool calls
    inside a ReAct loop) are dispatched before LOW priority background
    pre-fetches, reducing p99 latency for interactive queries.

    Design
    ------
    * Uses ``asyncio.PriorityQueue`` — a heap-based, coroutine-safe queue.
    * A configurable pool of *worker* coroutines drains the queue
      concurrently.  Workers are bounded by ``max_workers`` (analogous to
      ``ThreadPoolExecutor``).
    * Tasks are added via ``submit()`` and results are collected via
      ``collect()`` keyed on ``task_id``.

    Usage::

        retriever = PriorityParallelRetriever(max_workers=8)
        await retriever.submit(task_critical, retriever_fn)
        await retriever.submit(task_low, retriever_fn)
        results = await retriever.collect()
    """

    _SENTINEL = None  # Signals workers to stop

    def __init__(
        self,
        max_workers: int = 8,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._results: Dict[str, RetrievalResult] = {}
        self._pending: int = 0

    async def submit(
        self,
        task: RetrievalTask,
        retriever_fn: Callable,
    ) -> None:
        """Enqueue a retrieval task.

        Args:
            task: Task to enqueue (priority embedded in task).
            retriever_fn: Async or sync retrieval callable.
        """
        self._pending += 1
        await self._queue.put((task.priority, task, retriever_fn))

    async def collect(self) -> Dict[str, RetrievalResult]:
        """
        Drain the queue and return all results keyed by ``task_id``.

        Spawns ``max_workers`` worker coroutines that race to consume tasks
        in priority order.
        """
        if self._pending == 0:
            return {}

        # Add sentinels to stop workers
        for _ in range(self.max_workers):
            await self._queue.put((RetrievalPriority.LOW + 1, None, None))  # type: ignore[arg-type]

        workers = [asyncio.create_task(self._worker()) for _ in range(self.max_workers)]
        await asyncio.gather(*workers, return_exceptions=True)

        results = dict(self._results)
        self._results.clear()
        self._pending = 0
        return results

    async def _worker(self) -> None:
        """Worker coroutine that processes tasks from the priority queue."""
        while True:
            _, task, retriever_fn = await self._queue.get()
            if task is None:  # Sentinel — stop worker
                self._queue.task_done()
                return
            try:
                result = await self._run_task(task, retriever_fn)
                self._results[task.task_id] = result
            except Exception as exc:
                logger.error(f"Worker error on task {task.task_id}: {exc}")
                self._results[task.task_id] = RetrievalResult(
                    task_id=task.task_id,
                    query=task.query,
                    documents=[],
                    retriever_name=task.retriever_name,
                    latency_ms=0.0,
                    success=False,
                    error=str(exc),
                )
            finally:
                self._queue.task_done()

    async def _run_task(self, task: RetrievalTask, retriever_fn: Callable) -> RetrievalResult:
        start = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(retriever_fn):
                docs = await asyncio.wait_for(
                    retriever_fn(task.query, task.top_k),
                    timeout=self.timeout_seconds,
                )
            else:
                loop = asyncio.get_event_loop()
                docs = await asyncio.wait_for(
                    loop.run_in_executor(None, retriever_fn, task.query, task.top_k),
                    timeout=self.timeout_seconds,
                )
            return RetrievalResult(
                task_id=task.task_id,
                query=task.query,
                documents=docs,
                retriever_name=task.retriever_name,
                latency_ms=(time.perf_counter() - start) * 1000,
                success=True,
            )
        except asyncio.TimeoutError:
            return RetrievalResult(
                task_id=task.task_id,
                query=task.query,
                documents=[],
                retriever_name=task.retriever_name,
                latency_ms=(time.perf_counter() - start) * 1000,
                success=False,
                error="Timeout",
            )


# Mock retriever for testing
async def mock_retriever(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Mock retriever function."""
    await asyncio.sleep(0.05)  # Simulate I/O
    return [{"text": f"Document {i} for {query}", "score": 1.0 - i * 0.1} for i in range(top_k)]


# Example usage
async def example_parallel_retrieval():
    """Example of parallel retrieval."""
    # Single query, multiple retrievers
    print("=" * 50)
    print("Multi-retriever parallel execution")
    print("=" * 50)

    multi_retriever = MultiRetrieverParallel()

    retrievers = {
        "bm25": mock_retriever,
        "dense": mock_retriever,
        "hybrid": mock_retriever,
    }

    results = await multi_retriever.retrieve_multi("What is machine learning?", retrievers, top_k=5)

    for name, docs in results.items():
        print(f"\n{name}: {len(docs)} documents")

    # Batch processing
    print("\n" + "=" * 50)
    print("Batch query processing")
    print("=" * 50)

    processor = BatchQueryProcessor(batch_size=10)

    queries = [f"Query {i}" for i in range(50)]

    results, metrics = await processor.process_large_batch(queries, mock_retriever)

    print(f"\nProcessed {metrics.total_tasks} queries")
    print(f"Success rate: {metrics.successful_tasks}/{metrics.total_tasks}")
    print(f"Throughput: {metrics.throughput_qps:.1f} QPS")
    print(f"Avg latency: {metrics.avg_latency_ms:.1f} ms")


if __name__ == "__main__":
    asyncio.run(example_parallel_retrieval())
