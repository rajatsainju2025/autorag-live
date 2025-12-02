"""
High-performance heap-based top-k operations for streaming and batch processing.

This module provides memory-efficient heap data structures that significantly
outperform numpy.argpartition for scenarios where k << n (small k, large n).
"""

import heapq
import math
from typing import Generic, List, Optional, Tuple, TypeVar, Union

T = TypeVar("T")
S = TypeVar("S")  # Score type


class HeapTopK(Generic[T]):
    """
    Memory-efficient top-k selection using a min-heap.

    Maintains only k elements in memory at any time, making it ideal for
    streaming data or very large datasets where k << n.

    Example:
        >>> heap = HeapTopK(k=5)
        >>> for score, item in [(0.9, "doc1"), (0.8, "doc2"), (0.95, "doc3")]:
        ...     heap.add(score, item)
        >>> top_items = heap.get_sorted()  # [(0.95, "doc3"), (0.9, "doc1"), (0.8, "doc2")]
    """

    def __init__(self, k: int, reverse: bool = False):
        """
        Initialize the heap-based top-k selector.

        Args:
            k: Number of top items to maintain
            reverse: If False (default), keeps highest scores. If True, keeps lowest scores.
        """
        if k <= 0:
            raise ValueError("k must be positive")

        self.k = k
        self.reverse = reverse
        self._heap: List[Tuple[Union[float, int], T]] = []
        self._multiplier = -1 if reverse else 1

    def add(self, score: Union[float, int], item: T) -> None:
        """
        Add an item with its score to the top-k selection.

        Args:
            score: Numeric score for ranking
            item: Item to potentially include in top-k
        """
        # Multiply by -1 if we want highest scores (Python's heapq is a min-heap)
        heap_score = self._multiplier * score

        if len(self._heap) < self.k:
            # Heap not full, just add
            heapq.heappush(self._heap, (heap_score, item))
        elif heap_score > self._heap[0][0]:
            # New item is better than worst in heap, replace it
            heapq.heapreplace(self._heap, (heap_score, item))

    def add_batch(self, scores: List[Union[float, int]], items: List[T]) -> None:
        """
        Add multiple items efficiently.

        Args:
            scores: List of numeric scores
            items: List of corresponding items
        """
        if len(scores) != len(items):
            raise ValueError("scores and items must have the same length")

        for score, item in zip(scores, items):
            self.add(score, item)

    def get_sorted(
        self, reverse_output: Optional[bool] = None
    ) -> List[Tuple[Union[float, int], T]]:
        """
        Get top-k items sorted by score.

        Args:
            reverse_output: If None, uses natural order (best first).
                          If True, returns worst first. If False, returns best first.

        Returns:
            List of (score, item) tuples sorted by score
        """
        if not self._heap:
            return []

        # Extract all items with original scores
        items_with_scores = [(self._multiplier * score, item) for score, item in self._heap]

        # Sort based on reverse_output parameter
        if reverse_output is None:
            # Natural order: highest scores first (for normal mode), lowest first (for reverse mode)
            items_with_scores.sort(key=lambda x: x[0], reverse=not self.reverse)
        else:
            # Explicit reverse setting
            items_with_scores.sort(key=lambda x: x[0], reverse=not reverse_output)

        return items_with_scores

    def get_top_items(self, reverse_output: Optional[bool] = None) -> List[T]:
        """Get only the top-k items (without scores)."""
        return [item for _, item in self.get_sorted(reverse_output)]

    def get_top_scores(self, reverse_output: Optional[bool] = None) -> List[Union[float, int]]:
        """Get only the top-k scores."""
        return [score for score, _ in self.get_sorted(reverse_output)]

    def peek_worst(self) -> Optional[Tuple[Union[float, int], T]]:
        """
        Peek at the worst item currently in the top-k without removing it.

        Returns:
            (score, item) tuple of the worst item, or None if heap is empty
        """
        if not self._heap:
            return None
        score, item = self._heap[0]
        return (self._multiplier * score, item)

    def size(self) -> int:
        """Return current number of items in the heap."""
        return len(self._heap)

    def is_full(self) -> bool:
        """Check if heap has reached capacity k."""
        return len(self._heap) >= self.k

    def clear(self) -> None:
        """Remove all items from the heap."""
        self._heap.clear()

    def merge(self, other: "HeapTopK[T]") -> "HeapTopK[T]":
        """
        Merge another HeapTopK into a new instance.

        Args:
            other: Another HeapTopK instance

        Returns:
            New HeapTopK instance containing merged top-k items
        """
        if self.k != other.k or self.reverse != other.reverse:
            raise ValueError("Cannot merge heaps with different k or reverse settings")

        merged = HeapTopK(self.k, self.reverse)

        # Add all items from both heaps
        for score, item in self.get_sorted():
            merged.add(score, item)
        for score, item in other.get_sorted():
            merged.add(score, item)

        return merged


class StreamingTopK(HeapTopK[T]):
    """
    Streaming version of top-k that can handle continuous data efficiently.

    Includes additional features for streaming scenarios like statistics
    tracking and periodic cleanup.
    """

    def __init__(self, k: int, reverse: bool = False, track_stats: bool = True):
        super().__init__(k, reverse)
        self.track_stats = track_stats
        self.total_items_seen = 0
        self.items_accepted = 0
        self._last_accepted = False

    def add(self, score: Union[float, int], item: T) -> None:
        """
        Add item to the streaming top-k. Also tracks acceptance statistics.

        Use was_last_accepted() to check if the last item was accepted.
        """
        if self.track_stats:
            self.total_items_seen += 1

        # Check if item will be accepted before adding
        heap_score = self._multiplier * score
        will_be_accepted = len(self._heap) < self.k or (
            len(self._heap) >= self.k and heap_score > self._heap[0][0]
        )

        super().add(score, item)

        if self.track_stats and will_be_accepted:
            self.items_accepted += 1

        # Store for was_last_accepted() method
        self._last_accepted = will_be_accepted

    def was_last_accepted(self) -> bool:
        """Check if the last added item was accepted into top-k."""
        return self._last_accepted

    def get_acceptance_rate(self) -> float:
        """Get the rate of items accepted into top-k."""
        if self.total_items_seen == 0:
            return 0.0
        return self.items_accepted / self.total_items_seen

    def get_stats(self) -> dict:
        """Get statistics about the streaming process."""
        return {
            "total_items_seen": self.total_items_seen,
            "items_accepted": self.items_accepted,
            "acceptance_rate": self.get_acceptance_rate(),
            "current_size": self.size(),
            "is_full": self.is_full(),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.total_items_seen = 0
        self.items_accepted = 0


def batch_top_k(
    scores: List[Union[float, int]],
    items: Optional[List[T]] = None,
    k: int = 10,
    reverse: bool = False,
    method: str = "auto",
) -> Union[List[Tuple[Union[float, int], T]], List[Union[float, int]]]:
    """
    High-performance batch top-k selection with automatic method selection.

    Args:
        scores: List of numeric scores
        items: Optional list of items (if None, returns indices)
        k: Number of top items to return
        reverse: If False, returns highest scores. If True, returns lowest scores.
        method: Selection method ('heap', 'partition', 'auto')

    Returns:
        List of (score, item) tuples or just scores if items is None
    """
    if not scores:
        return []

    n = len(scores)
    k = min(k, n)

    # Handle k=0 case
    if k <= 0:
        return []

    if items is not None and len(items) != n:
        raise ValueError("scores and items must have the same length")

    # Auto method selection based on performance characteristics
    if method == "auto":
        # Use heap for small k relative to n, partition for larger k
        method = "heap" if k < 0.1 * n and k < 1000 else "partition"

    if method == "heap":
        return _batch_top_k_heap(scores, items, k, reverse)
    elif method == "partition":
        return _batch_top_k_partition(scores, items, k, reverse)
    else:
        raise ValueError("method must be 'heap', 'partition', or 'auto'")


def _batch_top_k_heap(
    scores: List[Union[float, int]],
    items: Optional[List[T]],
    k: int,
    reverse: bool,
) -> Union[List[Tuple[Union[float, int], T]], List[Union[float, int]]]:
    """Heap-based batch top-k implementation."""
    heap = HeapTopK(k, reverse)

    if items is not None:
        heap.add_batch(scores, items)
        return heap.get_sorted()
    else:
        # Use indices as items
        indices = list(range(len(scores)))
        heap.add_batch(scores, indices)
        result = heap.get_sorted()
        return [score for score, _ in result]


def _batch_top_k_partition(
    scores: List[Union[float, int]],
    items: Optional[List[T]],
    k: int,
    reverse: bool,
) -> Union[List[Tuple[Union[float, int], T]], List[Union[float, int]]]:
    """Partition-based batch top-k implementation using built-in heapq."""
    if items is not None:
        score_item_pairs = list(zip(scores, items))
        if reverse:
            # For lowest scores, use nsmallest
            top_pairs = heapq.nsmallest(k, score_item_pairs, key=lambda x: x[0])
        else:
            # For highest scores, use nlargest
            top_pairs = heapq.nlargest(k, score_item_pairs, key=lambda x: x[0])
        return top_pairs
    else:
        if reverse:
            top_scores = heapq.nsmallest(k, scores)
        else:
            top_scores = heapq.nlargest(k, scores)
        return top_scores


class AdaptiveTopK(Generic[T]):
    """
    Adaptive top-k that adjusts its capacity based on score distribution.

    Useful for scenarios where the optimal k is not known in advance or
    where the data distribution changes over time.
    """

    def __init__(
        self,
        initial_k: int,
        min_k: int = 1,
        max_k: int = 1000,
        adaptation_threshold: float = 0.1,
        reverse: bool = False,
    ):
        """
        Initialize adaptive top-k selector.

        Args:
            initial_k: Starting capacity
            min_k: Minimum allowed capacity
            max_k: Maximum allowed capacity
            adaptation_threshold: Threshold for score difference to trigger adaptation
            reverse: Score ordering preference
        """
        self.min_k = min_k
        self.max_k = max_k
        self.adaptation_threshold = adaptation_threshold
        self.reverse = reverse

        self.current_k = initial_k
        self._heap = HeapTopK(initial_k, reverse)
        self._recent_scores: List[float] = []
        self._adaptation_counter = 0

    def add(self, score: Union[float, int], item: T) -> None:
        """Add item and potentially adapt capacity."""
        self._heap.add(score, item)
        self._recent_scores.append(float(score))

        # Trigger adaptation check periodically
        self._adaptation_counter += 1
        if self._adaptation_counter % 100 == 0:
            self._maybe_adapt()

    def _maybe_adapt(self) -> None:
        """Check if capacity should be adapted based on recent scores."""
        if len(self._recent_scores) < 50:
            return

        recent_scores = self._recent_scores[-50:]
        score_std = self._calculate_std(recent_scores)
        score_mean = sum(recent_scores) / len(recent_scores)

        if score_mean == 0:
            coefficient_of_variation = 0
        else:
            coefficient_of_variation = score_std / abs(score_mean)

        # Adapt based on score variability
        if coefficient_of_variation > self.adaptation_threshold:
            # High variability, might need more capacity
            new_k = min(self.max_k, int(self.current_k * 1.2))
        else:
            # Low variability, might reduce capacity
            new_k = max(self.min_k, int(self.current_k * 0.9))

        if new_k != self.current_k:
            self._resize_heap(new_k)

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def _resize_heap(self, new_k: int) -> None:
        """Resize the heap to new capacity."""
        current_items = self._heap.get_sorted()
        self.current_k = new_k
        self._heap = HeapTopK(new_k, self.reverse)

        # Add back the items (heap will automatically maintain top-k)
        for score, item in current_items:
            self._heap.add(score, item)

    def get_sorted(
        self, reverse_output: Optional[bool] = None
    ) -> List[Tuple[Union[float, int], T]]:
        """Get current top items."""
        return self._heap.get_sorted(reverse_output)

    def get_capacity(self) -> int:
        """Get current capacity."""
        return self.current_k


def compare_top_k_methods(scores: List[Union[float, int]], k: int, iterations: int = 3) -> dict:
    """
    Compare performance of different top-k methods.

    Args:
        scores: Test scores
        k: Number of top items
        iterations: Number of timing iterations

    Returns:
        Dictionary with timing results and method recommendations
    """
    import time

    n = len(scores)
    results = {}

    # Test heap method
    heap_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        batch_top_k(scores, k=k, method="heap")
        heap_times.append(time.perf_counter() - start)

    # Test partition method
    partition_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        batch_top_k(scores, k=k, method="partition")
        partition_times.append(time.perf_counter() - start)

    heap_avg = sum(heap_times) / len(heap_times)
    partition_avg = sum(partition_times) / len(partition_times)

    results = {
        "n": n,
        "k": k,
        "k_ratio": k / n,
        "heap_time": heap_avg,
        "partition_time": partition_avg,
        "speedup": partition_avg / heap_avg if heap_avg > 0 else 1.0,
        "recommended": "heap" if heap_avg < partition_avg else "partition",
    }

    return results
