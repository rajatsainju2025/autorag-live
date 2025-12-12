"""
Async task queue for AutoRAG-Live.

Provides priority-based async task execution with retries,
timeouts, and comprehensive error handling.

Features:
- Priority queues (HIGH, NORMAL, LOW)
- Automatic retries with exponential backoff
- Configurable timeouts
- Task dependencies
- Result caching
- Concurrent execution control

Example usage:
    >>> queue = TaskQueue(max_workers=4)
    >>> 
    >>> async def process_doc(doc_id):
    ...     return await fetch_and_process(doc_id)
    >>> 
    >>> task = await queue.submit(process_doc, "doc123", priority=Priority.HIGH)
    >>> result = await task.result()
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Priority(IntEnum):
    """Task priority levels."""
    
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(str):
    """Task status values."""
    
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


@dataclass
class TaskConfig:
    """Configuration for task execution."""
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    retry_on: Optional[List[type]] = None  # Exception types to retry
    
    # Timeout
    timeout: Optional[float] = 30.0
    
    # Priority
    priority: Priority = Priority.NORMAL
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Result handling
    cache_result: bool = False
    cache_ttl: int = 3600


@dataclass
class TaskResult(Generic[T]):
    """Result of task execution."""
    
    task_id: str
    status: str
    result: Optional[T] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    
    # Timing
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Retry info
    attempts: int = 0
    
    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def queue_time(self) -> Optional[float]:
        """Get time spent in queue."""
        if self.started_at:
            return self.started_at - self.created_at
        return None
    
    @property
    def success(self) -> bool:
        """Check if task succeeded."""
        return self.status == TaskStatus.COMPLETED


class Task(Generic[T]):
    """
    Represents an async task with lifecycle management.
    
    Example:
        >>> task = Task(my_async_func, "arg1", kwarg="value")
        >>> await task.run()
        >>> print(task.result)
    """
    
    def __init__(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        task_id: Optional[str] = None,
        config: Optional[TaskConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize task.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            task_id: Unique task identifier
            config: Task configuration
            **kwargs: Function keyword arguments
        """
        self.id = task_id or str(uuid.uuid4())
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.config = config or TaskConfig()
        
        self._status = TaskStatus.PENDING
        self._result: Optional[T] = None
        self._error: Optional[Exception] = None
        self._traceback: Optional[str] = None
        
        self._created_at = time.time()
        self._started_at: Optional[float] = None
        self._completed_at: Optional[float] = None
        self._attempts = 0
        
        self._event = asyncio.Event()
        self._cancelled = False
    
    @property
    def status(self) -> str:
        """Get task status."""
        return self._status
    
    @property
    def result_value(self) -> Optional[T]:
        """Get result value (may be None)."""
        return self._result
    
    async def result(self, timeout: Optional[float] = None) -> T:
        """
        Wait for and return the task result.
        
        Args:
            timeout: Maximum wait time
            
        Returns:
            Task result
            
        Raises:
            Exception: If task failed
            asyncio.TimeoutError: If timeout exceeded
        """
        if timeout:
            await asyncio.wait_for(self._event.wait(), timeout)
        else:
            await self._event.wait()
        
        if self._error:
            raise self._error
        
        return self._result
    
    async def run(self) -> TaskResult[T]:
        """
        Execute the task with retry logic.
        
        Returns:
            TaskResult with execution details
        """
        self._status = TaskStatus.RUNNING
        self._started_at = time.time()
        
        retry_delay = self.config.retry_delay
        
        for attempt in range(self.config.max_retries + 1):
            self._attempts = attempt + 1
            
            if self._cancelled:
                self._status = TaskStatus.CANCELLED
                break
            
            try:
                # Execute with timeout
                if self.config.timeout:
                    self._result = await asyncio.wait_for(
                        self.func(*self.args, **self.kwargs),
                        timeout=self.config.timeout,
                    )
                else:
                    self._result = await self.func(*self.args, **self.kwargs)
                
                self._status = TaskStatus.COMPLETED
                break
                
            except asyncio.TimeoutError:
                self._error = asyncio.TimeoutError(
                    f"Task {self.id} timed out after {self.config.timeout}s"
                )
                self._status = TaskStatus.TIMEOUT
                
                if attempt < self.config.max_retries:
                    self._status = TaskStatus.RETRYING
                    await asyncio.sleep(retry_delay)
                    retry_delay *= self.config.retry_backoff
                    
            except asyncio.CancelledError:
                self._status = TaskStatus.CANCELLED
                raise
                
            except Exception as e:
                self._error = e
                self._traceback = traceback.format_exc()
                
                # Check if should retry
                should_retry = (
                    attempt < self.config.max_retries
                    and (
                        self.config.retry_on is None
                        or any(isinstance(e, t) for t in self.config.retry_on)
                    )
                )
                
                if should_retry:
                    self._status = TaskStatus.RETRYING
                    logger.warning(
                        f"Task {self.id} failed (attempt {attempt + 1}), "
                        f"retrying in {retry_delay}s: {e}"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= self.config.retry_backoff
                else:
                    self._status = TaskStatus.FAILED
        
        self._completed_at = time.time()
        self._event.set()
        
        return TaskResult(
            task_id=self.id,
            status=self._status,
            result=self._result,
            error=str(self._error) if self._error else None,
            traceback=self._traceback,
            created_at=self._created_at,
            started_at=self._started_at,
            completed_at=self._completed_at,
            attempts=self._attempts,
        )
    
    def cancel(self) -> None:
        """Cancel the task."""
        self._cancelled = True
        self._status = TaskStatus.CANCELLED
        self._event.set()


class TaskQueue:
    """
    Async task queue with priority and concurrency control.
    
    Example:
        >>> queue = TaskQueue(max_workers=4)
        >>> await queue.start()
        >>> 
        >>> task = await queue.submit(my_async_func, "arg1")
        >>> result = await task.result()
        >>> 
        >>> await queue.stop()
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 1000,
    ):
        """
        Initialize task queue.
        
        Args:
            max_workers: Maximum concurrent tasks
            max_queue_size: Maximum pending tasks
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # Priority queues (one per priority level)
        self._queues: Dict[Priority, asyncio.Queue] = {
            p: asyncio.Queue(maxsize=max_queue_size // len(Priority))
            for p in Priority
        }
        
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._tasks: Dict[str, Task] = {}
        self._results: Dict[str, TaskResult] = {}
        
        # Statistics
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
    
    async def start(self) -> None:
        """Start the task queue workers."""
        if self._running:
            return
        
        self._running = True
        
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)
        
        logger.info(f"TaskQueue started with {self.max_workers} workers")
    
    async def stop(self, wait: bool = True) -> None:
        """
        Stop the task queue.
        
        Args:
            wait: Wait for pending tasks to complete
        """
        self._running = False
        
        if wait:
            # Wait for queues to empty
            for queue in self._queues.values():
                await queue.join()
        
        # Cancel workers
        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("TaskQueue stopped")
    
    async def submit(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        priority: Priority = Priority.NORMAL,
        config: Optional[TaskConfig] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Task[T]:
        """
        Submit a task to the queue.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            priority: Task priority
            config: Task configuration
            task_id: Optional task ID
            **kwargs: Function keyword arguments
            
        Returns:
            Task object for tracking
        """
        if config is None:
            config = TaskConfig(priority=priority)
        else:
            config.priority = priority
        
        task = Task(func, *args, task_id=task_id, config=config, **kwargs)
        self._tasks[task.id] = task
        self._total_submitted += 1
        
        # Check dependencies
        if config.depends_on:
            await self._wait_for_dependencies(config.depends_on)
        
        # Add to appropriate priority queue
        queue = self._queues[priority]
        await queue.put(task)
        
        logger.debug(f"Task {task.id} submitted with priority {priority.name}")
        
        return task
    
    async def _wait_for_dependencies(self, task_ids: List[str]) -> None:
        """Wait for dependent tasks to complete."""
        for task_id in task_ids:
            if task_id in self._tasks:
                await self._tasks[task_id].result()
    
    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes tasks."""
        logger.debug(f"Worker {worker_id} started")
        
        while self._running:
            task = await self._get_next_task()
            if task is None:
                await asyncio.sleep(0.1)
                continue
            
            try:
                result = await task.run()
                self._results[task.id] = result
                
                if result.success:
                    self._total_completed += 1
                else:
                    self._total_failed += 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self._total_failed += 1
            finally:
                # Mark task as done in its queue
                for queue in self._queues.values():
                    try:
                        queue.task_done()
                    except ValueError:
                        pass
        
        logger.debug(f"Worker {worker_id} stopped")
    
    async def _get_next_task(self) -> Optional[Task]:
        """Get next task by priority order."""
        for priority in Priority:
            queue = self._queues[priority]
            try:
                return queue.get_nowait()
            except asyncio.QueueEmpty:
                continue
        
        # If all queues empty, wait on highest priority
        try:
            return await asyncio.wait_for(
                self._queues[Priority.CRITICAL].get(),
                timeout=0.1,
            )
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
            return None
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Get a task result by ID."""
        return self._results.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self._tasks.get(task_id)
        if task:
            task.cancel()
            return True
        return False
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        queue_sizes = {
            p.name: self._queues[p].qsize() for p in Priority
        }
        
        return {
            "total_submitted": self._total_submitted,
            "total_completed": self._total_completed,
            "total_failed": self._total_failed,
            "pending": sum(queue_sizes.values()),
            "queue_sizes": queue_sizes,
            "active_workers": len([w for w in self._workers if not w.done()]),
        }
    
    @property
    def pending_count(self) -> int:
        """Get total pending tasks."""
        return sum(q.qsize() for q in self._queues.values())
    
    def __enter__(self) -> "TaskQueue":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        pass
    
    async def __aenter__(self) -> "TaskQueue":
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()


class BatchProcessor:
    """
    Process items in batches asynchronously.
    
    Example:
        >>> processor = BatchProcessor(batch_size=10)
        >>> 
        >>> async def process_batch(items):
        ...     return [item * 2 for item in items]
        >>> 
        >>> results = await processor.process(
        ...     items=[1, 2, 3, 4, 5],
        ...     func=process_batch
        ... )
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent: int = 4,
        timeout: Optional[float] = None,
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Items per batch
            max_concurrent: Maximum concurrent batches
            timeout: Timeout per batch
        """
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.timeout = timeout
    
    async def process(
        self,
        items: List[Any],
        func: Callable[[List[Any]], Awaitable[List[Any]]],
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            func: Batch processing function
            
        Returns:
            List of processed results
        """
        # Create batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Process with semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_batch(batch: List[Any], idx: int) -> tuple:
            async with semaphore:
                try:
                    if self.timeout:
                        result = await asyncio.wait_for(
                            func(batch), timeout=self.timeout
                        )
                    else:
                        result = await func(batch)
                    return idx, result
                except Exception as e:
                    logger.error(f"Batch {idx} failed: {e}")
                    return idx, []
        
        # Execute batches
        tasks = [
            process_batch(batch, i)
            for i, batch in enumerate(batches)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Combine results in order
        results.sort(key=lambda x: x[0])
        combined = []
        for _, batch_results in results:
            combined.extend(batch_results)
        
        return combined


class RetryExecutor:
    """
    Execute async functions with retry logic.
    
    Example:
        >>> executor = RetryExecutor(max_retries=3)
        >>> result = await executor.execute(flaky_function, arg1, arg2)
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        retry_on: Optional[List[type]] = None,
    ):
        """
        Initialize retry executor.
        
        Args:
            max_retries: Maximum retry attempts
            delay: Initial delay between retries
            backoff: Backoff multiplier
            retry_on: Exception types to retry on
        """
        self.max_retries = max_retries
        self.delay = delay
        self.backoff = backoff
        self.retry_on = retry_on or [Exception]
    
    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute function with retries.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        delay = self.delay
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except tuple(self.retry_on) as e:
                last_error = e
                
                if attempt < self.max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}, "
                        f"retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    delay *= self.backoff
        
        raise last_error


class ThrottledExecutor:
    """
    Execute functions with rate limiting.
    
    Example:
        >>> executor = ThrottledExecutor(rate=10)  # 10 per second
        >>> results = await executor.execute_many([
        ...     (func1, args1),
        ...     (func2, args2),
        ... ])
    """
    
    def __init__(
        self,
        rate: float = 10.0,
        burst: Optional[int] = None,
    ):
        """
        Initialize throttled executor.
        
        Args:
            rate: Maximum calls per second
            burst: Allow burst of this many calls
        """
        self.rate = rate
        self.burst = burst or int(rate)
        
        self._tokens = float(self.burst)
        self._last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def _acquire(self) -> None:
        """Acquire a rate limit token."""
        async with self._lock:
            # Refill tokens
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(
                self.burst,
                self._tokens + elapsed * self.rate,
            )
            self._last_update = now
            
            # Wait if needed
            if self._tokens < 1:
                wait = (1 - self._tokens) / self.rate
                await asyncio.sleep(wait)
                self._tokens = 1
            
            self._tokens -= 1
    
    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute function with rate limiting."""
        await self._acquire()
        return await func(*args, **kwargs)
    
    async def execute_many(
        self,
        calls: List[tuple],
    ) -> List[Any]:
        """
        Execute multiple calls with rate limiting.
        
        Args:
            calls: List of (func, args, kwargs) tuples
            
        Returns:
            List of results
        """
        results = []
        
        for call in calls:
            if len(call) == 2:
                func, args = call
                kwargs = {}
            else:
                func, args, kwargs = call
            
            result = await self.execute(func, *args, **kwargs)
            results.append(result)
        
        return results


# Global queue instance
_default_queue: Optional[TaskQueue] = None


async def get_task_queue(
    max_workers: int = 4,
) -> TaskQueue:
    """Get or create the default task queue."""
    global _default_queue
    if _default_queue is None:
        _default_queue = TaskQueue(max_workers=max_workers)
        await _default_queue.start()
    return _default_queue


async def submit_task(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    priority: Priority = Priority.NORMAL,
    **kwargs: Any,
) -> Task[T]:
    """
    Convenience function to submit a task.
    
    Args:
        func: Async function
        *args: Arguments
        priority: Task priority
        **kwargs: Keyword arguments
        
    Returns:
        Task object
    """
    queue = await get_task_queue()
    return await queue.submit(func, *args, priority=priority, **kwargs)
