"""Tests for optimized set operations with Bloom filters."""

import pytest

from autorag_live.utils.set_operations import (
    BloomFilter,
    OptimizedSetOperations,
    estimate_bloom_filter_params,
)


class TestBloomFilter:
    """Test Bloom filter implementation."""

    def test_basic_operations(self):
        """Test basic add and contains operations."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)

        # Add items
        bf.add("test1")
        bf.add("test2")
        bf.add("test3")

        # Check membership (no false negatives)
        assert "test1" in bf
        assert "test2" in bf
        assert "test3" in bf

        # Length tracking
        assert len(bf) == 3

    def test_false_positive_rate(self):
        """Test that false positive rate is within bounds."""
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.05)

        # Add 1000 items
        for i in range(1000):
            bf.add(f"item_{i}")

        # Test with items not in the filter
        false_positives = 0
        test_items = 1000
        for i in range(test_items):
            if f"not_in_filter_{i}" in bf:
                false_positives += 1

        actual_fp_rate = false_positives / test_items

        # Should be roughly within expected bounds (allowing some variance)
        assert actual_fp_rate < 0.1  # Well below 10%

    def test_bytes_input(self):
        """Test handling of bytes input."""
        bf = BloomFilter(expected_items=10, false_positive_rate=0.01)

        bf.add(b"bytes_test")
        assert b"bytes_test" in bf
        assert "bytes_test" in bf  # Should work with string version too

    def test_memory_usage(self):
        """Test memory usage reporting."""
        bf = BloomFilter(expected_items=100, false_positive_rate=0.01)
        memory_usage = bf.memory_usage()
        assert memory_usage > 0
        assert isinstance(memory_usage, int)

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            BloomFilter(expected_items=0, false_positive_rate=0.01)

        with pytest.raises(ValueError):
            BloomFilter(expected_items=100, false_positive_rate=0.0)

        with pytest.raises(ValueError):
            BloomFilter(expected_items=100, false_positive_rate=1.0)


class TestOptimizedSetOperations:
    """Test optimized set operations."""

    def test_fast_intersection_small_sets(self):
        """Test intersection with small sets (should use direct method)."""
        set_a = {"a", "b", "c"}
        set_b = {"b", "c", "d"}

        result = OptimizedSetOperations.fast_intersection(set_a, set_b)
        expected = {"b", "c"}

        assert result == expected

    def test_fast_intersection_large_sets(self):
        """Test intersection with large sets (should use Bloom filter)."""
        set_a = {f"item_{i}" for i in range(500)}
        set_b = {f"item_{i}" for i in range(250, 750)}

        result = OptimizedSetOperations.fast_intersection(set_a, set_b, use_bloom_filter=True)
        expected = {f"item_{i}" for i in range(250, 500)}

        assert result == expected

    def test_fast_intersection_no_bloom(self):
        """Test intersection without Bloom filter optimization."""
        set_a = {f"item_{i}" for i in range(500)}
        set_b = {f"item_{i}" for i in range(250, 750)}

        result = OptimizedSetOperations.fast_intersection(set_a, set_b, use_bloom_filter=False)
        expected = {f"item_{i}" for i in range(250, 500)}

        assert result == expected

    def test_fast_intersection_empty_sets(self):
        """Test intersection with empty sets."""
        assert OptimizedSetOperations.fast_intersection(set(), {"a"}) == set()
        assert OptimizedSetOperations.fast_intersection({"a"}, set()) == set()
        assert OptimizedSetOperations.fast_intersection(set(), set()) == set()

    def test_fast_membership_batch_small(self):
        """Test batch membership with small sets."""
        items = ["a", "b", "c", "d"]
        target_set = {"a", "c", "e"}

        result = OptimizedSetOperations.fast_membership_batch(items, target_set)
        expected = [True, False, True, False]

        assert result == expected

    def test_fast_membership_batch_large(self):
        """Test batch membership with large sets."""
        items = [f"item_{i}" for i in range(100)]
        target_set = {f"item_{i}" for i in range(0, 200, 2)}  # Even numbers

        result = OptimizedSetOperations.fast_membership_batch(
            items, target_set, use_bloom_filter=True
        )
        expected = [i % 2 == 0 for i in range(100)]

        assert result == expected

    def test_fast_membership_batch_empty(self):
        """Test batch membership with empty inputs."""
        assert OptimizedSetOperations.fast_membership_batch([], {"a"}) == []
        assert OptimizedSetOperations.fast_membership_batch(["a"], set()) == [False]

    def test_fast_deduplication_small(self):
        """Test deduplication with small lists."""
        items = ["a", "b", "a", "c", "b", "d"]

        result = OptimizedSetOperations.fast_deduplication(items, preserve_order=True)
        expected = ["a", "b", "c", "d"]

        assert result == expected

    def test_fast_deduplication_large(self):
        """Test deduplication with large lists."""
        items = [f"item_{i % 100}" for i in range(2000)]  # Many duplicates

        result = OptimizedSetOperations.fast_deduplication(items, preserve_order=True)

        # Should contain each unique item exactly once
        assert len(result) == 100
        assert len(set(result)) == 100  # All unique

    def test_fast_deduplication_no_order(self):
        """Test deduplication without preserving order."""
        items = ["a", "b", "a", "c", "b", "d"]

        result = OptimizedSetOperations.fast_deduplication(items, preserve_order=False)

        assert set(result) == {"a", "b", "c", "d"}
        assert len(result) == 4

    def test_fast_deduplication_empty(self):
        """Test deduplication with empty list."""
        assert OptimizedSetOperations.fast_deduplication([]) == []


class TestBloomFilterParams:
    """Test Bloom filter parameter estimation."""

    def test_estimate_params(self):
        """Test parameter estimation."""
        expected_items = 10000
        memory_budget_mb = 1.0

        bit_array_size, fp_rate = estimate_bloom_filter_params(expected_items, memory_budget_mb)

        assert bit_array_size > 0
        assert 0 < fp_rate < 1

        # Memory budget should be respected
        memory_used_mb = bit_array_size / (8 * 1024 * 1024)
        assert memory_used_mb <= memory_budget_mb


class TestPerformanceComparison:
    """Performance comparison tests (for manual verification)."""

    @pytest.mark.slow
    def test_intersection_performance(self):
        """Compare intersection performance (manual verification)."""
        import time

        # Create large sets
        set_a = {f"doc_{i}" for i in range(10000)}
        set_b = {f"doc_{i}" for i in range(5000, 15000)}

        # Time standard intersection
        start = time.perf_counter()
        result_standard = set_a & set_b
        time_standard = time.perf_counter() - start

        # Time optimized intersection
        start = time.perf_counter()
        result_optimized = OptimizedSetOperations.fast_intersection(
            set_a, set_b, use_bloom_filter=True
        )
        time_optimized = time.perf_counter() - start

        # Results should be the same
        assert result_standard == result_optimized

        # Print timing results for manual verification
        print("\nIntersection Performance:")
        print(f"Standard: {time_standard:.4f}s")
        print(f"Optimized: {time_optimized:.4f}s")
        print(f"Speedup: {time_standard/time_optimized:.2f}x")

    @pytest.mark.slow
    def test_membership_batch_performance(self):
        """Compare batch membership performance (manual verification)."""
        import time

        # Create test data
        target_set = {f"doc_{i}" for i in range(50000)}
        test_items = [f"doc_{i}" for i in range(0, 100000, 3)]  # Some matches, some don't

        # Time standard membership testing
        start = time.perf_counter()
        result_standard = [item in target_set for item in test_items]
        time_standard = time.perf_counter() - start

        # Time optimized membership testing
        start = time.perf_counter()
        result_optimized = OptimizedSetOperations.fast_membership_batch(
            test_items, target_set, use_bloom_filter=True
        )
        time_optimized = time.perf_counter() - start

        # Results should be the same
        assert result_standard == result_optimized

        # Print timing results for manual verification
        print("\nBatch Membership Performance:")
        print(f"Standard: {time_standard:.4f}s")
        print(f"Optimized: {time_optimized:.4f}s")
        print(f"Speedup: {time_standard/time_optimized:.2f}x")
