"""
High-performance ring buffer implementation for efficient cache eviction.

This module provides circular buffer data structures that offer O(1) insertion,
deletion, and fixed-size cache eviction with better performance characteristics
than OrderedDict for high-throughput scenarios.
"""

from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar, Union

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


class RingBufferNode(Generic[K, V]):
    """Node in the ring buffer for doubly-linked list operations."""

    def __init__(self, key: K, value: V):
        self.key = key
        self.value = value
        self.prev: Optional["RingBufferNode[Any, Any]"] = None
        self.next: Optional["RingBufferNode[Any, Any]"] = None


class RingBuffer(Generic[K, V]):
    """
    High-performance ring buffer with O(1) operations for cache implementations.

    Provides fixed-size circular buffer with efficient insertion, deletion, and
    LRU-style eviction without the overhead of OrderedDict operations.

    Example:
        >>> buffer = RingBuffer(capacity=3)
        >>> buffer.put("key1", "value1")
        >>> buffer.put("key2", "value2")
        >>> buffer.put("key3", "value3")
        >>> buffer.put("key4", "value4")  # Evicts key1
        >>> "key1" in buffer  # False
        >>> buffer.get("key2")  # "value2" (moves to head)
    """

    def __init__(self, capacity: int):
        """
        Initialize ring buffer with fixed capacity.

        Args:
            capacity: Maximum number of items to store
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self._size = 0
        self._map: Dict[K, RingBufferNode[K, V]] = {}

        # Create dummy head and tail nodes for easier manipulation
        self._head: RingBufferNode[Any, Any] = RingBufferNode(None, None)  # type: ignore
        self._tail: RingBufferNode[Any, Any] = RingBufferNode(None, None)  # type: ignore
        self._head.next = self._tail
        self._tail.prev = self._head

    def _add_to_head(self, node: RingBufferNode[K, V]) -> None:
        """Add node right after head (most recently used position)."""
        node.prev = self._head  # type: ignore
        node.next = self._head.next  # type: ignore

        if self._head.next:
            self._head.next.prev = node  # type: ignore
        self._head.next = node  # type: ignore

    def _remove_node(self, node: RingBufferNode[Any, Any]) -> None:
        """Remove node from its current position."""
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev

    def _remove_tail(self) -> Optional[RingBufferNode[K, V]]:
        """Remove and return the least recently used node (before tail)."""
        last_node = self._tail.prev
        if last_node and last_node != self._head:
            self._remove_node(last_node)
            return last_node  # type: ignore
        return None

    def _move_to_head(self, node: RingBufferNode[K, V]) -> None:
        """Move existing node to head (mark as most recently used)."""
        self._remove_node(node)
        self._add_to_head(node)

    def get(self, key: K) -> Optional[V]:
        """
        Get value by key and mark as most recently used.

        Args:
            key: Key to lookup

        Returns:
            Value if key exists, None otherwise
        """
        node = self._map.get(key)
        if node is None:
            return None

        # Move to head (most recently used)
        self._move_to_head(node)
        return node.value

    def put(self, key: K, value: V) -> Optional[Tuple[K, V]]:
        """
        Insert or update key-value pair.

        Args:
            key: Key to insert/update
            value: Value to store

        Returns:
            Evicted (key, value) pair if capacity exceeded, None otherwise
        """
        existing_node = self._map.get(key)
        evicted = None

        if existing_node:
            # Update existing key
            existing_node.value = value
            self._move_to_head(existing_node)
        else:
            # Add new key
            new_node = RingBufferNode(key, value)

            if self._size >= self.capacity:
                # Remove least recently used
                tail_node = self._remove_tail()
                if tail_node:
                    evicted = (tail_node.key, tail_node.value)
                    del self._map[tail_node.key]
                    self._size -= 1

            self._map[key] = new_node
            self._add_to_head(new_node)
            self._size += 1

        return evicted

    def peek(self, key: K) -> Optional[V]:
        """
        Get value without marking as recently used.

        Args:
            key: Key to lookup

        Returns:
            Value if key exists, None otherwise
        """
        node = self._map.get(key)
        return node.value if node else None

    def delete(self, key: K) -> bool:
        """
        Remove key from buffer.

        Args:
            key: Key to remove

        Returns:
            True if key was removed, False if not found
        """
        node = self._map.get(key)
        if node is None:
            return False

        self._remove_node(node)
        del self._map[key]
        self._size -= 1
        return True

    def clear(self) -> None:
        """Remove all items from the buffer."""
        self._map.clear()
        self._size = 0
        self._head.next = self._tail
        self._tail.prev = self._head

    def size(self) -> int:
        """Return current number of items."""
        return self._size

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self._size >= self.capacity

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._size == 0

    def keys(self) -> List[K]:
        """Return list of keys in access order (most recent first)."""
        keys = []
        current = self._head.next
        while current and current != self._tail:
            keys.append(current.key)
            current = current.next
        return keys

    def values(self) -> List[V]:
        """Return list of values in access order (most recent first)."""
        values = []
        current = self._head.next
        while current and current != self._tail:
            values.append(current.value)
            current = current.next
        return values

    def items(self) -> List[Tuple[K, V]]:
        """Return list of (key, value) pairs in access order."""
        items = []
        current = self._head.next
        while current and current != self._tail:
            items.append((current.key, current.value))
            current = current.next
        return items

    def get_oldest(self) -> Optional[Tuple[K, V]]:
        """Get the oldest (least recently used) item without removing it."""
        if self._tail.prev and self._tail.prev != self._head:
            node = self._tail.prev
            return (node.key, node.value)  # type: ignore
        return None

    def get_newest(self) -> Optional[Tuple[K, V]]:
        """Get the newest (most recently used) item without removing it."""
        if self._head.next and self._head.next != self._tail:
            node = self._head.next
            return (node.key, node.value)  # type: ignore
        return None

    def __contains__(self, key: K) -> bool:
        """Support 'in' operator without affecting access order."""
        return key in self._map

    def __len__(self) -> int:
        """Return number of items."""
        return self._size

    def __getitem__(self, key: K) -> V:
        """Support bracket notation (marks as accessed)."""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        """Support bracket assignment."""
        self.put(key, value)

    def __delitem__(self, key: K) -> None:
        """Support del operator."""
        if not self.delete(key):
            raise KeyError(key)

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys in access order."""
        return iter(self.keys())


class TTLRingBuffer(RingBuffer[K, V]):
    """
    Ring buffer with time-to-live (TTL) support for automatic expiration.

    Extends RingBuffer with TTL functionality where items automatically
    expire after a specified time period.
    """

    def __init__(self, capacity: int, ttl_seconds: float):
        """
        Initialize TTL ring buffer.

        Args:
            capacity: Maximum number of items
            ttl_seconds: Time-to-live in seconds
        """
        super().__init__(capacity)
        self.ttl_seconds = ttl_seconds
        self._timestamps: Dict[K, float] = {}

    def _current_time(self) -> float:
        """Get current timestamp."""
        import time

        return time.time()

    def _is_expired(self, key: K) -> bool:
        """Check if key has expired."""
        timestamp = self._timestamps.get(key)
        if timestamp is None:
            return True
        return self._current_time() - timestamp > self.ttl_seconds

    def _cleanup_expired(self) -> None:
        """Remove expired items."""
        current_time = self._current_time()
        expired_keys = []

        for key, timestamp in self._timestamps.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            self.delete(key)

    def get(self, key: K) -> Optional[V]:
        """Get value, checking for expiration first."""
        if self._is_expired(key):
            self.delete(key)
            return None

        value = super().get(key)
        if value is not None:
            # Update timestamp on access
            self._timestamps[key] = self._current_time()
        return value

    def put(self, key: K, value: V) -> Optional[Tuple[K, V]]:
        """Put value with current timestamp."""
        # Clean up expired items first
        self._cleanup_expired()

        # Update timestamp
        self._timestamps[key] = self._current_time()

        evicted = super().put(key, value)
        if evicted:
            # Clean up timestamp for evicted item
            self._timestamps.pop(evicted[0], None)

        return evicted

    def peek(self, key: K) -> Optional[V]:
        """Peek at value, checking expiration."""
        if self._is_expired(key):
            self.delete(key)
            return None
        return super().peek(key)

    def delete(self, key: K) -> bool:
        """Delete key and its timestamp."""
        self._timestamps.pop(key, None)
        return super().delete(key)

    def clear(self) -> None:
        """Clear all items and timestamps."""
        super().clear()
        self._timestamps.clear()

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get buffer statistics."""
        current_time = self._current_time()
        expired_count = sum(
            1
            for timestamp in self._timestamps.values()
            if current_time - timestamp > self.ttl_seconds
        )

        return {
            "size": self.size(),
            "capacity": self.capacity,
            "utilization": self.size() / self.capacity if self.capacity > 0 else 0,
            "expired_items": expired_count,
            "ttl_seconds": self.ttl_seconds,
        }


class CacheBuffer(Generic[K, V]):
    """
    High-level cache interface using ring buffer backend.

    Provides a simple cache interface with configurable capacity and TTL,
    designed as a drop-in replacement for dict-based caches with better
    performance characteristics.
    """

    def __init__(
        self,
        capacity: int,
        ttl_seconds: Optional[float] = None,
        on_evict: Optional[Callable[[K, V], None]] = None,
    ):
        """
        Initialize cache buffer.

        Args:
            capacity: Maximum cache size
            ttl_seconds: Time-to-live for items (None for no expiration)
            on_evict: Callback for when items are evicted (key, value) -> None
        """
        self.on_evict = on_evict

        if ttl_seconds is not None:
            self._buffer: Union[RingBuffer[K, V], TTLRingBuffer[K, V]] = TTLRingBuffer(
                capacity, ttl_seconds
            )
        else:
            self._buffer = RingBuffer(capacity)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get value with optional default."""
        value = self._buffer.get(key)
        return value if value is not None else default

    def put(self, key: K, value: V) -> None:
        """Put value, handling eviction callback."""
        evicted = self._buffer.put(key, value)
        if evicted and self.on_evict:
            self.on_evict(evicted[0], evicted[1])

    def delete(self, key: K) -> bool:
        """Delete key from cache."""
        return self._buffer.delete(key)

    def clear(self) -> None:
        """Clear all cached items."""
        self._buffer.clear()

    def size(self) -> int:
        """Get current cache size."""
        return self._buffer.size()

    def capacity(self) -> int:
        """Get cache capacity."""
        return self._buffer.capacity

    def hit_rate(self, hits: int, misses: int) -> float:
        """Calculate hit rate given hit and miss counts."""
        total = hits + misses
        return hits / total if total > 0 else 0.0

    def keys(self) -> List[K]:
        """Get all keys in access order."""
        return self._buffer.keys()

    def values(self) -> List[V]:
        """Get all values in access order."""
        return self._buffer.values()

    def items(self) -> List[Tuple[K, V]]:
        """Get all items in access order."""
        return self._buffer.items()

    def stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        if isinstance(self._buffer, TTLRingBuffer):
            return self._buffer.get_stats()
        else:
            return {
                "size": self.size(),
                "capacity": self.capacity(),
                "utilization": self.size() / self.capacity() if self.capacity() > 0 else 0,
            }

    def __contains__(self, key: K) -> bool:
        """Support 'in' operator."""
        return key in self._buffer

    def __len__(self) -> int:
        """Get number of items."""
        return len(self._buffer)

    def __getitem__(self, key: K) -> V:
        """Support bracket notation."""
        return self._buffer[key]

    def __setitem__(self, key: K, value: V) -> None:
        """Support bracket assignment."""
        self.put(key, value)

    def __delitem__(self, key: K) -> None:
        """Support del operator."""
        del self._buffer[key]


def benchmark_cache_performance(
    cache_implementations: List[Tuple[str, Any]], operations: int = 10000
) -> Dict[str, float]:
    """
    Benchmark different cache implementations.

    Args:
        cache_implementations: List of (name, cache_instance) tuples
        operations: Number of operations to perform

    Returns:
        Dictionary mapping implementation names to average operation times
    """
    import random
    import time

    results = {}

    for name, cache in cache_implementations:
        # Prepare test data
        keys = [f"key_{i}" for i in range(operations // 2)]
        values = [f"value_{i}" for i in range(operations // 2)]

        start_time = time.perf_counter()

        # Mixed workload: 70% gets, 30% puts
        for i in range(operations):
            if random.random() < 0.3:  # Put operation
                key = random.choice(keys)
                value = random.choice(values)
                if hasattr(cache, "put"):
                    cache.put(key, value)
                else:
                    cache[key] = value
            else:  # Get operation
                key = random.choice(keys)
                if hasattr(cache, "get"):
                    cache.get(key)
                else:
                    cache.get(key, None) if hasattr(cache, "get") else cache.get(key, None)

        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / operations * 1000  # ms per operation

        results[name] = avg_time

        # Clear cache for next test
        if hasattr(cache, "clear"):
            cache.clear()

    return results
