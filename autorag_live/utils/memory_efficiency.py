"""Memory efficiency improvements for large operations."""

import gc
from typing import Iterator, List, TypeVar

T = TypeVar("T")


def memory_efficient_iteration(
    items: List[T],
    batch_size: int = 1000,
) -> Iterator[List[T]]:
    """
    Iterate over items in batches with garbage collection.

    Args:
        items: List of items to iterate
        batch_size: Batch size for iteration

    Returns:
        Iterator yielding batches
    """
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        yield batch
        # Garbage collect between batches
        if i % (batch_size * 10) == 0:
            gc.collect()


def process_large_dataset(
    items: List[T],
    processor,
    batch_size: int = 1000,
):
    """
    Process large dataset with memory efficiency.

    Args:
        items: List of items to process
        processor: Function to process each item
        batch_size: Batch size

    Returns:
        Results
    """
    results = []
    for batch in memory_efficient_iteration(items, batch_size):
        batch_results = [processor(item) for item in batch]
        results.extend(batch_results)
    return results


class MemoryTracker:
    """Tracks memory usage during operations."""

    def __init__(self):
        """Initialize memory tracker."""
        self._peak_memory = 0
        self._current_memory = 0

    def __enter__(self):
        """Context manager entry."""
        import psutil

        self._process = psutil.Process()
        self._start_memory = self._process.memory_info().rss / 1024 / 1024  # MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        end_memory = self._process.memory_info().rss / 1024 / 1024  # MB
        self._current_memory = end_memory - self._start_memory
        self._peak_memory = max(self._peak_memory, self._current_memory)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "current_mb": self._current_memory,
            "peak_mb": self._peak_memory,
        }
