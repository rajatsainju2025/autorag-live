"""Adaptive batch sizing utilities for memory-efficient operations.

Provides intelligent batch size calculation and progress tracking for
large-scale batch operations like embedding generation.
"""

import logging
from typing import Callable, Iterator, List, Optional, TypeVar

import psutil

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def get_adaptive_batch_size(
    item_size_bytes: int,
    max_memory_fraction: float = 0.5,
    min_batch: int = 1,
    max_batch: int = 256,
) -> int:
    """Calculate optimal batch size from available memory.

    Args:
        item_size_bytes: Estimated memory per item in bytes
        max_memory_fraction: Fraction of available memory to use (0.0-1.0)
        min_batch: Minimum batch size
        max_batch: Maximum batch size

    Returns:
        Optimal batch size

    Example:
        >>> # Calculate for 100KB items
        >>> batch_size = get_adaptive_batch_size(100_000)
    """
    mem = psutil.virtual_memory()
    available_bytes = int(mem.available * max_memory_fraction)

    if item_size_bytes <= 0:
        return max_batch

    optimal = available_bytes // item_size_bytes
    return max(min_batch, min(optimal, max_batch))


def batch_iterator_with_progress(
    items: List[T],
    batch_size: int,
    callback: Optional[Callable[[int, int], None]] = None,
) -> Iterator[List[T]]:
    """Yield batches with progress tracking.

    Args:
        items: Items to batch
        batch_size: Batch size
        callback: Progress callback(current, total)

    Yields:
        Batches of items
    """
    total = len(items)
    for i in range(0, total, batch_size):
        batch = items[i : i + batch_size]
        yield batch
        if callback:
            callback(min(i + batch_size, total), total)


__all__ = ["get_adaptive_batch_size", "batch_iterator_with_progress"]
