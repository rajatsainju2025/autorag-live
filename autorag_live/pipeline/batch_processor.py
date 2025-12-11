"""
Batch Processing Engine for Agentic RAG Pipeline.

Enables efficient processing of multiple queries in parallel
with batching, concurrency control, and progress tracking.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class BatchStatus(str, Enum):
    """Status of a batch job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class ItemStatus(str, Enum):
    """Status of a batch item."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchItem(Generic[T, R]):
    """A single item in a batch."""

    item_id: str
    input_data: T
    status: ItemStatus = ItemStatus.PENDING
    result: Optional[R] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.item_id:
            self.item_id = str(uuid.uuid4())[:8]


@dataclass
class BatchResult(Generic[R]):
    """Result of batch processing."""

    batch_id: str
    status: BatchStatus
    total_items: int
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[BatchItem] = field(default_factory=list)
    total_time_ms: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_items == 0:
            return 0.0
        return self.successful / self.total_items

    @property
    def throughput(self) -> float:
        """Calculate items per second."""
        if self.total_time_ms == 0:
            return 0.0
        return self.total_items / (self.total_time_ms / 1000)


class BatchProcessor(ABC, Generic[T, R]):
    """Abstract base class for batch processors."""

    @abstractmethod
    def process_item(self, item: T) -> R:
        """Process a single item."""
        pass

    def process_batch(
        self,
        items: list[T],
        max_workers: int = 4,
        batch_size: Optional[int] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[R]:
        """Process a batch of items."""
        raise NotImplementedError


class ThreadedBatchProcessor(BatchProcessor[T, R]):
    """Batch processor using thread pool."""

    def __init__(
        self,
        process_fn: Callable[[T], R],
        max_workers: int = 4,
        timeout_per_item: Optional[float] = None,
    ):
        """Initialize threaded batch processor."""
        self.process_fn = process_fn
        self.max_workers = max_workers
        self.timeout_per_item = timeout_per_item

    def process_item(self, item: T) -> R:
        """Process a single item."""
        return self.process_fn(item)

    def process_batch(
        self,
        items: list[T],
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[R]:
        """Process items in parallel using thread pool."""
        batch_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        workers = max_workers or self.max_workers

        batch_items: list[BatchItem[T, R]] = [
            BatchItem(item_id=str(i), input_data=item) for i, item in enumerate(items)
        ]

        result = BatchResult(
            batch_id=batch_id,
            status=BatchStatus.PROCESSING,
            total_items=len(items),
            results=batch_items,
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {}
            for batch_item in batch_items:
                batch_item.status = ItemStatus.PROCESSING
                batch_item.started_at = datetime.now()
                future = executor.submit(self.process_item, batch_item.input_data)
                future_to_item[future] = batch_item

            completed = 0
            for future in as_completed(future_to_item):
                batch_item = future_to_item[future]
                item_start = time.time()

                try:
                    batch_item.result = future.result(timeout=self.timeout_per_item)
                    batch_item.status = ItemStatus.SUCCESS
                    result.successful += 1
                except TimeoutError:
                    batch_item.status = ItemStatus.FAILED
                    batch_item.error = "Timeout"
                    result.failed += 1
                except Exception as e:
                    batch_item.status = ItemStatus.FAILED
                    batch_item.error = str(e)
                    result.failed += 1

                batch_item.completed_at = datetime.now()
                batch_item.processing_time_ms = (time.time() - item_start) * 1000
                completed += 1

                if on_progress:
                    on_progress(completed, len(items))

        result.total_time_ms = (time.time() - start_time) * 1000
        result.completed_at = datetime.now()
        result.status = (
            BatchStatus.COMPLETED
            if result.failed == 0
            else BatchStatus.PARTIAL
            if result.successful > 0
            else BatchStatus.FAILED
        )

        return result


class AsyncBatchProcessor(BatchProcessor[T, R]):
    """Batch processor using async/await."""

    def __init__(
        self,
        process_fn: Callable[[T], R],
        max_concurrency: int = 10,
        timeout_per_item: Optional[float] = None,
    ):
        """Initialize async batch processor."""
        self.process_fn = process_fn
        self.max_concurrency = max_concurrency
        self.timeout_per_item = timeout_per_item
        self._semaphore: Optional[asyncio.Semaphore] = None

    def process_item(self, item: T) -> R:
        """Process a single item."""
        return self.process_fn(item)

    async def _process_item_async(self, batch_item: BatchItem[T, R]) -> None:
        """Process a single item asynchronously."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)

        async with self._semaphore:
            batch_item.status = ItemStatus.PROCESSING
            batch_item.started_at = datetime.now()
            item_start = time.time()

            try:
                if asyncio.iscoroutinefunction(self.process_fn):
                    if self.timeout_per_item:
                        batch_item.result = await asyncio.wait_for(
                            self.process_fn(batch_item.input_data),
                            timeout=self.timeout_per_item,
                        )
                    else:
                        batch_item.result = await self.process_fn(batch_item.input_data)
                else:
                    loop = asyncio.get_event_loop()
                    batch_item.result = await loop.run_in_executor(
                        None, self.process_fn, batch_item.input_data
                    )
                batch_item.status = ItemStatus.SUCCESS
            except asyncio.TimeoutError:
                batch_item.status = ItemStatus.FAILED
                batch_item.error = "Timeout"
            except Exception as e:
                batch_item.status = ItemStatus.FAILED
                batch_item.error = str(e)

            batch_item.completed_at = datetime.now()
            batch_item.processing_time_ms = (time.time() - item_start) * 1000

    async def process_batch_async(
        self,
        items: list[T],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[R]:
        """Process items asynchronously."""
        batch_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        batch_items: list[BatchItem[T, R]] = [
            BatchItem(item_id=str(i), input_data=item) for i, item in enumerate(items)
        ]

        self._semaphore = asyncio.Semaphore(self.max_concurrency)

        tasks = [self._process_item_async(item) for item in batch_items]
        await asyncio.gather(*tasks)

        successful = sum(1 for item in batch_items if item.status == ItemStatus.SUCCESS)
        failed = sum(1 for item in batch_items if item.status == ItemStatus.FAILED)

        return BatchResult(
            batch_id=batch_id,
            status=BatchStatus.COMPLETED if failed == 0 else BatchStatus.PARTIAL,
            total_items=len(items),
            successful=successful,
            failed=failed,
            results=batch_items,
            total_time_ms=(time.time() - start_time) * 1000,
            completed_at=datetime.now(),
        )

    def process_batch(
        self,
        items: list[T],
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[R]:
        """Synchronous wrapper for async batch processing."""
        return asyncio.run(self.process_batch_async(items, on_progress))


class ChunkedBatchProcessor(BatchProcessor[T, R]):
    """Batch processor that processes items in chunks."""

    def __init__(
        self,
        process_fn: Callable[[T], R],
        chunk_size: int = 10,
        inner_processor: Optional[BatchProcessor[T, R]] = None,
        delay_between_chunks: float = 0.0,
    ):
        """Initialize chunked batch processor."""
        self.process_fn = process_fn
        self.chunk_size = chunk_size
        self.inner_processor = inner_processor or ThreadedBatchProcessor(
            process_fn, max_workers=chunk_size
        )
        self.delay_between_chunks = delay_between_chunks

    def process_item(self, item: T) -> R:
        """Process a single item."""
        return self.process_fn(item)

    def process_batch(
        self,
        items: list[T],
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult[R]:
        """Process items in chunks."""
        batch_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        chunk_sz = batch_size or self.chunk_size

        all_results: list[BatchItem[T, R]] = []
        total_successful = 0
        total_failed = 0
        processed = 0

        for i in range(0, len(items), chunk_sz):
            chunk = items[i : i + chunk_sz]
            chunk_result = self.inner_processor.process_batch(
                chunk, max_workers=max_workers or self.chunk_size
            )

            for j, item_result in enumerate(chunk_result.results):
                item_result.item_id = str(i + j)
                all_results.append(item_result)

            total_successful += chunk_result.successful
            total_failed += chunk_result.failed
            processed += len(chunk)

            if on_progress:
                on_progress(processed, len(items))

            if self.delay_between_chunks > 0 and i + chunk_sz < len(items):
                time.sleep(self.delay_between_chunks)

        return BatchResult(
            batch_id=batch_id,
            status=BatchStatus.COMPLETED if total_failed == 0 else BatchStatus.PARTIAL,
            total_items=len(items),
            successful=total_successful,
            failed=total_failed,
            results=all_results,
            total_time_ms=(time.time() - start_time) * 1000,
            completed_at=datetime.now(),
        )


class QueryBatchProcessor:
    """Specialized batch processor for RAG queries."""

    def __init__(
        self,
        query_fn: Callable[[str], Any],
        max_workers: int = 4,
        max_concurrency: int = 10,
        use_async: bool = False,
        chunk_size: Optional[int] = None,
    ):
        """Initialize query batch processor."""
        self.query_fn = query_fn
        self.max_workers = max_workers
        self.max_concurrency = max_concurrency
        self.use_async = use_async
        self.chunk_size = chunk_size

        if use_async:
            self._processor: BatchProcessor = AsyncBatchProcessor(
                query_fn, max_concurrency=max_concurrency
            )
        elif chunk_size:
            self._processor = ChunkedBatchProcessor(
                query_fn, chunk_size=chunk_size
            )
        else:
            self._processor = ThreadedBatchProcessor(
                query_fn, max_workers=max_workers
            )

    def process_queries(
        self,
        queries: list[str],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """Process multiple queries."""
        return self._processor.process_batch(
            queries, on_progress=on_progress
        )

    def process_queries_with_context(
        self,
        queries: list[tuple[str, dict[str, Any]]],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """Process queries with context."""

        def process_with_context(item: tuple[str, dict[str, Any]]) -> Any:
            query, context = item
            return self.query_fn(query)

        processor = ThreadedBatchProcessor(
            process_with_context, max_workers=self.max_workers
        )
        return processor.process_batch(queries, on_progress=on_progress)


class BatchJobManager:
    """Manager for tracking and managing batch jobs."""

    def __init__(self):
        """Initialize job manager."""
        self._jobs: dict[str, BatchResult] = {}
        self._history: list[str] = []
        self._max_history = 100

    def submit(
        self,
        processor: BatchProcessor[T, R],
        items: list[T],
        job_id: Optional[str] = None,
    ) -> str:
        """Submit a batch job."""
        job_id = job_id or str(uuid.uuid4())[:8]

        result = processor.process_batch(items)
        result.batch_id = job_id

        self._jobs[job_id] = result
        self._history.append(job_id)

        if len(self._history) > self._max_history:
            old_id = self._history.pop(0)
            if old_id in self._jobs:
                del self._jobs[old_id]

        return job_id

    def get_job(self, job_id: str) -> Optional[BatchResult]:
        """Get job result by ID."""
        return self._jobs.get(job_id)

    def get_status(self, job_id: str) -> Optional[BatchStatus]:
        """Get job status by ID."""
        job = self._jobs.get(job_id)
        return job.status if job else None

    def list_jobs(self, status: Optional[BatchStatus] = None) -> list[str]:
        """List job IDs, optionally filtered by status."""
        if status is None:
            return list(self._jobs.keys())
        return [
            jid for jid, job in self._jobs.items() if job.status == status
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get overall statistics."""
        total_items = sum(j.total_items for j in self._jobs.values())
        total_successful = sum(j.successful for j in self._jobs.values())
        total_failed = sum(j.failed for j in self._jobs.values())

        return {
            "total_jobs": len(self._jobs),
            "total_items_processed": total_items,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "overall_success_rate": total_successful / max(total_items, 1),
            "jobs_by_status": {
                status.value: len(self.list_jobs(status))
                for status in BatchStatus
            },
        }


__all__ = [
    "BatchStatus",
    "ItemStatus",
    "BatchItem",
    "BatchResult",
    "BatchProcessor",
    "ThreadedBatchProcessor",
    "AsyncBatchProcessor",
    "ChunkedBatchProcessor",
    "QueryBatchProcessor",
    "BatchJobManager",
]
