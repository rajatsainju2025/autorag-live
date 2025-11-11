"""Batch processing utilities for efficient bulk operations.

This module provides utilities for efficient batch processing with:
- Configurable batch sizes for different hardware
- Progress tracking for long-running operations
- Memory-efficient streaming
- Parallelization support

Example:
    >>> batch_processor = BatchProcessor(batch_size=32)
    >>> results = batch_processor.process(
    ...     documents,
    ...     retriever.retrieve_batch,
    ...     progress=True
    ... )
"""

from typing import Any, Callable, Generic, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class BatchProcessor(Generic[T, R]):
    """Efficient batch processor with configurable batch sizes."""

    def __init__(
        self,
        batch_size: int = 32,
        verbose: bool = False,
    ):
        """Initialize batch processor.

        Args:
            batch_size: Size of each batch
            verbose: Whether to print progress
        """
        self.batch_size = batch_size
        self.verbose = verbose

    def process(
        self,
        items: List[T],
        processor_fn: Callable[[List[T]], List[R]],
        progress: bool = False,
    ) -> List[R]:
        """Process items in batches.

        Args:
            items: Items to process
            processor_fn: Function to process batch
            progress: Whether to show progress

        Returns:
            Processed results
        """
        if not items:
            return []

        results: List[R] = []
        n_batches = (len(items) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(items))

            batch = items[start_idx:end_idx]

            if self.verbose and progress:
                print(f"Processing batch {batch_idx + 1}/{n_batches} " f"({len(batch)} items)")

            batch_results = processor_fn(batch)
            results.extend(batch_results)

        return results

    def process_with_overhead(
        self,
        items: List[T],
        init_fn: Callable[[], Any],
        processor_fn: Callable[[Any, List[T]], List[R]],
        cleanup_fn: Optional[Callable[[Any], None]] = None,
    ) -> List[R]:
        """Process items with setup/teardown overhead.

        Args:
            items: Items to process
            init_fn: Initialization function
            processor_fn: Function that takes state and batch
            cleanup_fn: Optional cleanup function

        Returns:
            Processed results
        """
        state = init_fn()
        results: List[R] = []

        try:
            n_batches = (len(items) + self.batch_size - 1) // self.batch_size

            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(items))

                batch = items[start_idx:end_idx]

                if self.verbose:
                    print(f"Processing batch {batch_idx + 1}/{n_batches} " f"({len(batch)} items)")

                batch_results = processor_fn(state, batch)
                results.extend(batch_results)

            return results
        finally:
            if cleanup_fn is not None:
                cleanup_fn(state)


class ChunkIterator:
    """Memory-efficient iterator that yields chunks of data."""

    def __init__(
        self,
        items: List[T],
        chunk_size: int = 32,
    ):
        """Initialize chunk iterator.

        Args:
            items: Items to iterate over
            chunk_size: Size of each chunk
        """
        self.items = items
        self.chunk_size = chunk_size
        self.n_chunks = (len(items) + chunk_size - 1) // chunk_size

    def __iter__(self):
        """Iterate over chunks."""
        for chunk_idx in range(self.n_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(self.items))
            yield self.items[start_idx:end_idx]

    def __len__(self) -> int:
        """Return number of chunks."""
        return self.n_chunks


def estimate_optimal_batch_size(
    available_memory_mb: float,
    item_size_mb: float,
    overhead_factor: float = 2.0,
) -> int:
    """Estimate optimal batch size based on available memory.

    Args:
        available_memory_mb: Available memory in MB
        item_size_mb: Average size of each item in MB
        overhead_factor: Factor to account for processing overhead

    Returns:
        Recommended batch size"""
    if item_size_mb <= 0:
        return 32

    safe_memory = available_memory_mb / overhead_factor
    batch_size = int(safe_memory / item_size_mb)

    # Minimum batch size is 1, typical minimum is 32
    return max(1, min(batch_size, 1024))
