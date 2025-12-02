"""
Tests for ring buffer implementation.

This test suite validates the performance and correctness of the ring buffer
cache eviction system, including LRU behavior, TTL support, and performance
characteristics.
"""

import time
from collections import OrderedDict

import pytest

from autorag_live.utils.ring_buffer import (
    CacheBuffer,
    RingBuffer,
    RingBufferNode,
    TTLRingBuffer,
    benchmark_cache_performance,
)


class TestRingBufferNode:
    """Test cases for ring buffer node."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = RingBufferNode("key", "value")
        assert node.key == "key"
        assert node.value == "value"
        assert node.prev is None
        assert node.next is None

    def test_node_linking(self):
        """Test node linking operations."""
        node1 = RingBufferNode("key1", "value1")
        node2 = RingBufferNode("key2", "value2")

        node1.next = node2
        node2.prev = node1

        assert node1.next == node2
        assert node2.prev == node1


class TestRingBuffer:
    """Test cases for ring buffer."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = RingBuffer(capacity=3)
        assert buffer.capacity == 3
        assert buffer.size() == 0
        assert buffer.is_empty()
        assert not buffer.is_full()

    def test_initialization_invalid_capacity(self):
        """Test buffer initialization with invalid capacity."""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            RingBuffer(capacity=0)

        with pytest.raises(ValueError, match="Capacity must be positive"):
            RingBuffer(capacity=-1)

    def test_basic_operations(self):
        """Test basic put/get operations."""
        buffer = RingBuffer(capacity=3)

        # Test empty buffer
        assert buffer.get("missing") is None
        assert buffer.peek("missing") is None

        # Test single item
        evicted = buffer.put("key1", "value1")
        assert evicted is None
        assert buffer.size() == 1
        assert buffer.get("key1") == "value1"
        assert buffer.peek("key1") == "value1"

        # Test multiple items
        buffer.put("key2", "value2")
        buffer.put("key3", "value3")
        assert buffer.size() == 3
        assert buffer.is_full()

    def test_lru_eviction(self):
        """Test LRU eviction behavior."""
        buffer = RingBuffer(capacity=2)

        # Fill buffer
        buffer.put("key1", "value1")
        buffer.put("key2", "value2")

        # Access key1 to make it recently used
        buffer.get("key1")

        # Add key3 - should evict key2 (least recently used)
        evicted = buffer.put("key3", "value3")
        assert evicted == ("key2", "value2")
        assert buffer.get("key1") == "value1"
        assert buffer.get("key3") == "value3"
        assert buffer.get("key2") is None

    def test_update_existing_key(self):
        """Test updating existing key."""
        buffer = RingBuffer(capacity=2)

        buffer.put("key1", "value1")
        buffer.put("key2", "value2")

        # Update existing key - should not trigger eviction
        evicted = buffer.put("key1", "new_value1")
        assert evicted is None
        assert buffer.get("key1") == "new_value1"
        assert buffer.size() == 2

    def test_delete_operations(self):
        """Test delete operations."""
        buffer = RingBuffer(capacity=3)

        # Delete from empty buffer
        assert not buffer.delete("missing")

        # Add items and delete
        buffer.put("key1", "value1")
        buffer.put("key2", "value2")
        assert buffer.size() == 2

        # Delete existing key
        assert buffer.delete("key1")
        assert buffer.size() == 1
        assert buffer.get("key1") is None

        # Delete non-existing key
        assert not buffer.delete("missing")

    def test_clear_operations(self):
        """Test clear operations."""
        buffer = RingBuffer(capacity=3)

        buffer.put("key1", "value1")
        buffer.put("key2", "value2")
        buffer.clear()

        assert buffer.size() == 0
        assert buffer.is_empty()
        assert buffer.get("key1") is None

    def test_access_order_tracking(self):
        """Test that access order is correctly tracked."""
        buffer = RingBuffer(capacity=3)

        buffer.put("key1", "value1")
        buffer.put("key2", "value2")
        buffer.put("key3", "value3")

        # Access key1 to move it to front
        buffer.get("key1")

        # Keys should be in access order (most recent first)
        keys = buffer.keys()
        assert keys == ["key1", "key3", "key2"]

        # Peek should not affect order
        buffer.peek("key2")
        keys = buffer.keys()
        assert keys == ["key1", "key3", "key2"]

    def test_oldest_newest_access(self):
        """Test getting oldest and newest items."""
        buffer = RingBuffer(capacity=3)

        # Empty buffer
        assert buffer.get_oldest() is None
        assert buffer.get_newest() is None

        buffer.put("key1", "value1")
        assert buffer.get_oldest() == ("key1", "value1")
        assert buffer.get_newest() == ("key1", "value1")

        buffer.put("key2", "value2")
        buffer.put("key3", "value3")

        # key3 is newest, key1 is oldest
        assert buffer.get_newest() == ("key3", "value3")
        assert buffer.get_oldest() == ("key1", "value1")

        # Access key1 to make it newest
        buffer.get("key1")
        assert buffer.get_newest() == ("key1", "value1")
        assert buffer.get_oldest() == ("key2", "value2")

    def test_magic_methods(self):
        """Test magic method implementations."""
        buffer = RingBuffer(capacity=3)

        buffer.put("key1", "value1")
        buffer.put("key2", "value2")

        # Test __contains__
        assert "key1" in buffer
        assert "missing" not in buffer

        # Test __len__
        assert len(buffer) == 2

        # Test __getitem__
        assert buffer["key1"] == "value1"
        with pytest.raises(KeyError):
            _ = buffer["missing"]

        # Test __setitem__
        buffer["key3"] = "value3"
        assert buffer.get("key3") == "value3"

        # Test __delitem__
        del buffer["key1"]
        assert buffer.get("key1") is None
        with pytest.raises(KeyError):
            del buffer["missing"]

        # Test __iter__
        keys = list(buffer)
        assert "key3" in keys
        assert "key2" in keys

    def test_items_values_keys(self):
        """Test items, values, and keys methods."""
        buffer = RingBuffer(capacity=3)

        buffer.put("key1", "value1")
        buffer.put("key2", "value2")
        buffer.put("key3", "value3")

        # Access key1 to change order
        buffer.get("key1")

        keys = buffer.keys()
        values = buffer.values()
        items = buffer.items()

        assert keys == ["key1", "key3", "key2"]
        assert values == ["value1", "value3", "value2"]
        assert items == [("key1", "value1"), ("key3", "value3"), ("key2", "value2")]


class TestTTLRingBuffer:
    """Test cases for TTL ring buffer."""

    def test_ttl_basic_operations(self):
        """Test basic TTL operations."""
        buffer = TTLRingBuffer(capacity=3, ttl_seconds=0.1)

        # Add item
        buffer.put("key1", "value1")
        assert buffer.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.15)
        assert buffer.get("key1") is None

    def test_ttl_refresh_on_access(self):
        """Test that TTL refreshes on access."""
        buffer = TTLRingBuffer(capacity=3, ttl_seconds=0.1)

        buffer.put("key1", "value1")
        time.sleep(0.05)  # Half TTL

        # Access should refresh TTL
        assert buffer.get("key1") == "value1"
        time.sleep(0.05)  # Another half TTL

        # Should still be valid (total 0.1s, but refreshed at 0.05s)
        assert buffer.get("key1") == "value1"

    def test_ttl_peek_expiration(self):
        """Test that peek checks for expiration."""
        buffer = TTLRingBuffer(capacity=3, ttl_seconds=0.1)

        buffer.put("key1", "value1")
        time.sleep(0.15)

        # Peek should return None for expired item
        assert buffer.peek("key1") is None

    def test_ttl_cleanup_on_put(self):
        """Test that expired items are cleaned up on put."""
        buffer = TTLRingBuffer(capacity=3, ttl_seconds=0.1)

        buffer.put("key1", "value1")
        buffer.put("key2", "value2")
        assert buffer.size() == 2

        time.sleep(0.15)

        # New put should trigger cleanup
        buffer.put("key3", "value3")

        # Only key3 should remain
        assert buffer.size() == 1
        assert buffer.get("key3") == "value3"
        assert buffer.get("key1") is None
        assert buffer.get("key2") is None

    def test_ttl_stats(self):
        """Test TTL buffer statistics."""
        buffer = TTLRingBuffer(capacity=3, ttl_seconds=0.1)

        buffer.put("key1", "value1")
        buffer.put("key2", "value2")

        stats = buffer.get_stats()
        assert stats["size"] == 2
        assert stats["capacity"] == 3
        assert stats["utilization"] == 2 / 3
        assert stats["ttl_seconds"] == 0.1
        assert stats["expired_items"] == 0

        # Wait for expiration
        time.sleep(0.15)
        stats = buffer.get_stats()
        assert stats["expired_items"] == 2

    def test_ttl_delete_cleans_timestamp(self):
        """Test that delete removes timestamp."""
        buffer = TTLRingBuffer(capacity=3, ttl_seconds=1.0)

        buffer.put("key1", "value1")
        assert "key1" in buffer._timestamps

        buffer.delete("key1")
        assert "key1" not in buffer._timestamps

    def test_ttl_clear_cleans_timestamps(self):
        """Test that clear removes all timestamps."""
        buffer = TTLRingBuffer(capacity=3, ttl_seconds=1.0)

        buffer.put("key1", "value1")
        buffer.put("key2", "value2")
        assert len(buffer._timestamps) == 2

        buffer.clear()
        assert len(buffer._timestamps) == 0


class TestCacheBuffer:
    """Test cases for high-level cache buffer."""

    def test_cache_buffer_basic(self):
        """Test basic cache buffer operations."""
        cache = CacheBuffer(capacity=3)

        # Basic operations
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("missing", "default") == "default"

        assert cache.size() == 1
        assert cache.capacity() == 3

    def test_cache_buffer_with_ttl(self):
        """Test cache buffer with TTL."""
        cache = CacheBuffer(capacity=3, ttl_seconds=0.1)

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_cache_buffer_eviction_callback(self):
        """Test eviction callback functionality."""
        evicted_items = []

        def on_evict(key, value):
            evicted_items.append((key, value))

        cache = CacheBuffer(capacity=2, on_evict=on_evict)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should trigger eviction

        assert len(evicted_items) == 1
        assert evicted_items[0] == ("key1", "value1")

    def test_cache_buffer_stats(self):
        """Test cache buffer statistics."""
        cache = CacheBuffer(capacity=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        stats = cache.stats()
        assert stats["size"] == 2
        assert stats["capacity"] == 3
        assert stats["utilization"] == 2 / 3

        # Test with TTL cache
        ttl_cache = CacheBuffer(capacity=3, ttl_seconds=0.1)
        ttl_cache.put("key1", "value1")

        ttl_stats = ttl_cache.stats()
        assert "ttl_seconds" in ttl_stats

    def test_cache_buffer_hit_rate(self):
        """Test hit rate calculation."""
        cache = CacheBuffer(capacity=3)

        # Test hit rate calculation
        assert cache.hit_rate(0, 0) == 0.0
        assert cache.hit_rate(7, 3) == 0.7
        assert cache.hit_rate(10, 0) == 1.0

    def test_cache_buffer_magic_methods(self):
        """Test cache buffer magic methods."""
        cache = CacheBuffer(capacity=3)

        cache["key1"] = "value1"
        assert cache["key1"] == "value1"
        assert "key1" in cache
        assert len(cache) == 1

        del cache["key1"]
        assert "key1" not in cache

    def test_cache_buffer_collections(self):
        """Test cache buffer collection methods."""
        cache = CacheBuffer(capacity=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        keys = cache.keys()
        values = cache.values()
        items = cache.items()

        assert "key1" in keys
        assert "key2" in keys
        assert "value1" in values
        assert "value2" in values
        assert ("key1", "value1") in items
        assert ("key2", "value2") in items


class TestPerformance:
    """Performance tests for ring buffer."""

    def test_large_scale_operations(self):
        """Test performance with large number of operations."""
        buffer = RingBuffer(capacity=1000)

        # Add many items
        for i in range(1500):  # More than capacity
            buffer.put(f"key_{i}", f"value_{i}")

        # Should maintain capacity
        assert buffer.size() == 1000

        # Recent items should still be present
        assert buffer.get("key_1499") == "value_1499"
        assert buffer.get("key_1000") == "value_1000"

        # Older items should be evicted
        assert buffer.get("key_0") is None
        assert buffer.get("key_499") is None

    def test_access_pattern_performance(self):
        """Test performance with different access patterns."""
        buffer = RingBuffer(capacity=100)

        # Fill buffer
        for i in range(100):
            buffer.put(f"key_{i}", f"value_{i}")

        # Sequential access should be fast
        start_time = time.perf_counter()
        for i in range(100):
            buffer.get(f"key_{i}")
        sequential_time = time.perf_counter() - start_time

        # Random access should also be fast
        import random

        keys = [f"key_{i}" for i in range(100)]
        random.shuffle(keys)

        start_time = time.perf_counter()
        for key in keys:
            buffer.get(key)
        random_time = time.perf_counter() - start_time

        # Both should be reasonably fast (exact timing depends on system)
        assert sequential_time < 0.1  # Should complete in <100ms
        assert random_time < 0.1

    @pytest.mark.slow
    def test_memory_efficiency(self):
        """Test memory usage characteristics."""

        # Create buffer and measure memory usage patterns
        buffer = RingBuffer(capacity=1000)

        # Memory usage should not grow unbounded
        for i in range(10000):  # 10x capacity
            buffer.put(f"key_{i}", f"value_{i}" * 10)  # Larger values

        # Size should stay at capacity
        assert buffer.size() == 1000

        # Test that objects are properly released
        import gc

        gc.collect()  # Force garbage collection

        # Buffer should still function correctly
        buffer.put("test_key", "test_value")
        assert buffer.get("test_key") == "test_value"


def test_benchmark_functionality():
    """Test the benchmark function."""
    # Create different cache implementations for comparison
    ring_buffer = RingBuffer(capacity=100)
    ordered_dict = OrderedDict()
    regular_dict = {}

    def limited_ordered_dict_put(key, value):
        """Simulate LRU with OrderedDict."""
        if key in ordered_dict:
            del ordered_dict[key]
        elif len(ordered_dict) >= 100:
            ordered_dict.popitem(last=False)
        ordered_dict[key] = value

    def limited_dict_put(key, value):
        """Simulate simple dict cache."""
        if len(regular_dict) >= 100 and key not in regular_dict:
            # Remove arbitrary item
            regular_dict.pop(next(iter(regular_dict)))
        regular_dict[key] = value

    # Mock implementations for testing
    class MockOrderedDictCache:
        def __init__(self):
            self.data = OrderedDict()
            self.capacity = 100

        def put(self, key, value):
            limited_ordered_dict_put(key, value)

        def get(self, key):
            return self.data.get(key)

        def clear(self):
            self.data.clear()

    class MockDictCache:
        def __init__(self):
            self.data = {}
            self.capacity = 100

        def put(self, key, value):
            limited_dict_put(key, value)

        def get(self, key):
            return self.data.get(key)

        def clear(self):
            self.data.clear()

    # Test benchmark function
    implementations = [
        ("RingBuffer", ring_buffer),
        ("OrderedDict", MockOrderedDictCache()),
        ("Dict", MockDictCache()),
    ]

    results = benchmark_cache_performance(implementations, operations=1000)

    # Should return results for all implementations
    assert len(results) == 3
    assert "RingBuffer" in results
    assert "OrderedDict" in results
    assert "Dict" in results

    # All results should be positive numbers (ms per operation)
    for impl_name, time_per_op in results.items():
        assert time_per_op > 0
        print(f"{impl_name}: {time_per_op:.3f} ms/op")


def test_stress_testing():
    """Stress test the ring buffer with intensive operations."""
    buffer = RingBuffer(capacity=50)

    # Intensive mixed workload
    import random

    keys = [f"key_{i}" for i in range(200)]
    values = [f"value_{i}" for i in range(200)]

    operations = 0
    for _ in range(1000):
        operation = random.choice(["put", "get", "delete"])

        if operation == "put":
            key = random.choice(keys)
            value = random.choice(values)
            buffer.put(key, value)
            operations += 1

        elif operation == "get":
            key = random.choice(keys)
            buffer.get(key)
            operations += 1

        elif operation == "delete":
            key = random.choice(keys)
            buffer.delete(key)
            operations += 1

        # Invariants that should always hold
        assert buffer.size() <= buffer.capacity
        assert buffer.size() >= 0

        if buffer.size() == 0:
            assert buffer.is_empty()
            assert not buffer.is_full()
        elif buffer.size() == buffer.capacity:
            assert buffer.is_full()
            assert not buffer.is_empty()

    print(f"Completed {operations} mixed operations successfully")


if __name__ == "__main__":
    # Run some basic performance comparisons
    print("Ring Buffer Performance Test")
    print("=" * 40)

    ring_buffer = RingBuffer(capacity=1000)

    # Test different operation patterns
    patterns = {
        "Sequential Put": lambda: [ring_buffer.put(f"key_{i}", f"value_{i}") for i in range(1500)],
        "Sequential Get": lambda: [ring_buffer.get(f"key_{i}") for i in range(500, 1500)],
        "Random Access": lambda: [
            ring_buffer.get(f"key_{random.randint(500, 1499)}") for _ in range(1000)
        ],
    }

    import random

    for pattern_name, pattern_func in patterns.items():
        start_time = time.perf_counter()
        pattern_func()
        end_time = time.perf_counter()
        print(f"{pattern_name}: {(end_time - start_time) * 1000:.2f} ms")

    print(f"Buffer size: {ring_buffer.size()}/{ring_buffer.capacity}")
    print("Ring Buffer tests completed successfully!")
