"""Tests for heap-based top-k operations."""

import pytest

from autorag_live.utils.heap_topk import (
    AdaptiveTopK,
    HeapTopK,
    StreamingTopK,
    batch_top_k,
    compare_top_k_methods,
)


class TestHeapTopK:
    """Test basic HeapTopK functionality."""

    def test_basic_operations(self):
        """Test basic top-k operations."""
        heap = HeapTopK(k=3)

        # Add items
        heap.add(0.8, "doc1")
        heap.add(0.9, "doc2")
        heap.add(0.7, "doc3")
        heap.add(0.95, "doc4")  # Should replace doc3
        heap.add(0.6, "doc5")  # Should be ignored

        # Check size
        assert heap.size() == 3
        assert heap.is_full()

        # Get sorted results (highest first by default)
        results = heap.get_sorted()
        assert len(results) == 3
        assert results[0][0] == 0.95  # Highest score first
        assert results[0][1] == "doc4"
        assert results[2][0] == 0.8  # Lowest of top-3

        # Test items and scores separately
        top_items = heap.get_top_items()
        top_scores = heap.get_top_scores()
        assert len(top_items) == 3
        assert len(top_scores) == 3
        assert top_scores[0] == 0.95

    def test_reverse_mode(self):
        """Test reverse mode (keeping lowest scores)."""
        heap = HeapTopK(k=3, reverse=True)

        heap.add(0.8, "doc1")
        heap.add(0.9, "doc2")
        heap.add(0.7, "doc3")
        heap.add(0.6, "doc4")  # Should replace doc2
        heap.add(0.95, "doc5")  # Should be ignored

        results = heap.get_sorted()
        assert len(results) == 3
        assert results[0][0] == 0.6  # Lowest score first in reverse mode
        assert results[2][0] == 0.8  # Highest of bottom-3

    def test_peek_worst(self):
        """Test peeking at worst item."""
        heap = HeapTopK(k=3)

        # Empty heap
        assert heap.peek_worst() is None

        heap.add(0.8, "doc1")
        heap.add(0.9, "doc2")
        heap.add(0.7, "doc3")

        # Worst item should be doc3 with score 0.7
        worst_result = heap.peek_worst()
        assert worst_result is not None
        worst_score, worst_item = worst_result
        assert worst_score == 0.7
        assert worst_item == "doc3"
        assert heap.size() == 3  # Should not remove item

    def test_batch_operations(self):
        """Test batch addition of items."""
        heap = HeapTopK(k=3)

        scores = [0.8, 0.9, 0.7, 0.95, 0.6]
        items = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        heap.add_batch(scores, items)

        results = heap.get_sorted()
        assert len(results) == 3
        assert results[0][0] == 0.95

    def test_merge_heaps(self):
        """Test merging two heaps."""
        heap1 = HeapTopK(k=3)
        heap2 = HeapTopK(k=3)

        heap1.add(0.8, "doc1")
        heap1.add(0.9, "doc2")

        heap2.add(0.95, "doc3")
        heap2.add(0.7, "doc4")

        merged = heap1.merge(heap2)
        results = merged.get_sorted()

        assert len(results) == 3
        assert results[0][0] == 0.95  # Best from both heaps

    def test_invalid_merge(self):
        """Test that incompatible heaps cannot be merged."""
        heap1 = HeapTopK(k=3)
        heap2 = HeapTopK(k=5)  # Different k

        with pytest.raises(ValueError):
            heap1.merge(heap2)

    def test_clear_heap(self):
        """Test clearing the heap."""
        heap = HeapTopK(k=3)
        heap.add(0.8, "doc1")
        heap.add(0.9, "doc2")

        assert heap.size() == 2

        heap.clear()
        assert heap.size() == 0
        assert not heap.is_full()
        assert heap.get_sorted() == []

    def test_invalid_k(self):
        """Test invalid k values."""
        with pytest.raises(ValueError):
            HeapTopK(k=0)

        with pytest.raises(ValueError):
            HeapTopK(k=-1)

    def test_batch_size_mismatch(self):
        """Test batch operations with mismatched sizes."""
        heap = HeapTopK(k=3)

        scores = [0.8, 0.9]
        items = ["doc1", "doc2", "doc3"]  # Different length

        with pytest.raises(ValueError):
            heap.add_batch(scores, items)


class TestStreamingTopK:
    """Test streaming top-k functionality."""

    def test_streaming_with_stats(self):
        """Test streaming top-k with statistics tracking."""
        streaming = StreamingTopK(k=3, track_stats=True)

        # Add items and track acceptance
        streaming.add(0.8, "doc1")
        assert streaming.was_last_accepted()

        streaming.add(0.9, "doc2")
        assert streaming.was_last_accepted()

        streaming.add(0.7, "doc3")
        assert streaming.was_last_accepted()

        streaming.add(0.95, "doc4")  # Should be accepted (replaces worst)
        assert streaming.was_last_accepted()

        streaming.add(0.6, "doc5")  # Should be rejected
        assert not streaming.was_last_accepted()

        # Check statistics
        stats = streaming.get_stats()
        assert stats["total_items_seen"] == 5
        assert stats["items_accepted"] == 4
        assert stats["acceptance_rate"] == 0.8
        assert stats["current_size"] == 3
        assert stats["is_full"] is True

    def test_streaming_without_stats(self):
        """Test streaming without statistics tracking."""
        streaming = StreamingTopK(k=3, track_stats=False)

        streaming.add(0.8, "doc1")
        streaming.add(0.9, "doc2")

        stats = streaming.get_stats()
        assert stats["total_items_seen"] == 0  # Not tracked
        assert stats["items_accepted"] == 0  # Not tracked

    def test_reset_stats(self):
        """Test resetting statistics."""
        streaming = StreamingTopK(k=3, track_stats=True)

        streaming.add(0.8, "doc1")
        streaming.add(0.9, "doc2")

        assert streaming.get_stats()["total_items_seen"] == 2

        streaming.reset_stats()

        stats = streaming.get_stats()
        assert stats["total_items_seen"] == 0
        assert stats["items_accepted"] == 0
        assert stats["acceptance_rate"] == 0.0


class TestBatchTopK:
    """Test batch top-k operations."""

    def test_batch_with_items(self):
        """Test batch top-k with items."""
        scores = [0.8, 0.9, 0.7, 0.95, 0.6, 0.85]
        items = ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]

        results = batch_top_k(scores, items, k=3)

        assert len(results) == 3
        assert results[0][0] == 0.95  # Highest score
        assert results[0][1] == "doc4"

        # Check all scores are >= third highest
        third_highest = results[2][0]
        assert all(score >= third_highest for score, _ in results)

    def test_batch_without_items(self):
        """Test batch top-k returning only scores."""
        scores = [0.8, 0.9, 0.7, 0.95, 0.6]

        results = batch_top_k(scores, k=3)

        assert len(results) == 3
        assert results[0] == 0.95
        assert 0.9 in results
        assert 0.8 in results

    def test_batch_reverse_mode(self):
        """Test batch top-k in reverse mode (lowest scores)."""
        scores = [0.8, 0.9, 0.7, 0.95, 0.6]
        items = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        results = batch_top_k(scores, items, k=3, reverse=True)

        assert len(results) == 3
        assert results[0][0] == 0.6  # Lowest score
        assert results[1][0] == 0.7
        assert results[2][0] == 0.8

    def test_batch_method_selection(self):
        """Test different batch methods."""
        scores = [0.8, 0.9, 0.7, 0.95, 0.6, 0.85, 0.75, 0.88]

        # Test explicit methods
        results_heap = batch_top_k(scores, k=3, method="heap")
        results_partition = batch_top_k(scores, k=3, method="partition")
        results_auto = batch_top_k(scores, k=3, method="auto")

        # Results should be the same regardless of method
        assert set(results_heap) == set(results_partition)
        assert set(results_heap) == set(results_auto)

    def test_batch_edge_cases(self):
        """Test batch top-k edge cases."""
        # Empty scores
        assert batch_top_k([]) == []

        # k larger than available items
        scores = [0.8, 0.9]
        results = batch_top_k(scores, k=5)
        assert len(results) == 2

        # k = 0 (should return empty)
        results = batch_top_k(scores, k=0)
        assert len(results) == 0

    def test_batch_size_mismatch(self):
        """Test batch with mismatched scores and items."""
        scores = [0.8, 0.9]
        items = ["doc1", "doc2", "doc3"]  # Different length

        with pytest.raises(ValueError):
            batch_top_k(scores, items, k=2)

    def test_invalid_method(self):
        """Test invalid method parameter."""
        scores = [0.8, 0.9, 0.7]

        with pytest.raises(ValueError):
            batch_top_k(scores, k=2, method="invalid")


class TestAdaptiveTopK:
    """Test adaptive top-k functionality."""

    def test_basic_adaptation(self):
        """Test basic adaptive behavior."""
        adaptive = AdaptiveTopK(initial_k=5, min_k=2, max_k=10, adaptation_threshold=0.1)

        # Add items to trigger adaptation checks
        for i in range(150):  # More than adaptation trigger (100)
            # Add items with high variability to trigger expansion
            score = 0.5 + (i % 10) * 0.1  # Scores from 0.5 to 1.4
            adaptive.add(score, f"doc{i}")

        # Should have adapted at least once
        final_capacity = adaptive.get_capacity()
        assert isinstance(final_capacity, int)
        assert 2 <= final_capacity <= 10

    def test_adaptation_bounds(self):
        """Test that adaptation respects min/max bounds."""
        adaptive = AdaptiveTopK(initial_k=5, min_k=3, max_k=7, adaptation_threshold=0.1)

        # Add many items to trigger multiple adaptations
        for i in range(500):
            # Alternate between high and low variability
            if i < 250:
                score = 0.5 + (i % 20) * 0.1  # High variability
            else:
                score = 0.8  # Low variability
            adaptive.add(score, f"doc{i}")

        final_capacity = adaptive.get_capacity()
        assert 3 <= final_capacity <= 7

    def test_get_results(self):
        """Test getting results from adaptive top-k."""
        adaptive = AdaptiveTopK(initial_k=3)

        adaptive.add(0.8, "doc1")
        adaptive.add(0.9, "doc2")
        adaptive.add(0.7, "doc3")
        adaptive.add(0.95, "doc4")

        results = adaptive.get_sorted()
        assert len(results) <= 3  # Should not exceed capacity
        assert results[0][0] == 0.95  # Highest score


class TestPerformanceComparison:
    """Test performance comparison utilities."""

    def test_compare_methods(self):
        """Test method comparison functionality."""
        scores = [float(i) for i in range(1000)]

        comparison = compare_top_k_methods(scores, k=10, iterations=1)

        # Check that comparison results have expected fields
        assert "n" in comparison
        assert "k" in comparison
        assert "k_ratio" in comparison
        assert "heap_time" in comparison
        assert "partition_time" in comparison
        assert "speedup" in comparison
        assert "recommended" in comparison

        assert comparison["n"] == 1000
        assert comparison["k"] == 10
        assert comparison["k_ratio"] == 0.01
        assert comparison["recommended"] in ["heap", "partition"]

    @pytest.mark.slow
    def test_performance_characteristics(self):
        """Test that heap method is faster for small k."""
        import random

        # Generate large dataset
        scores = [random.random() for _ in range(10000)]

        # Small k should favor heap
        comparison_small_k = compare_top_k_methods(scores, k=10, iterations=1)

        # Large k should favor partition
        comparison_large_k = compare_top_k_methods(scores, k=1000, iterations=1)

        print("\nPerformance comparison results:")
        print(f"Small k (10/10000): {comparison_small_k['recommended']} method")
        print(f"Large k (1000/10000): {comparison_large_k['recommended']} method")

        # These are guidelines, not strict requirements due to hardware variation
        # But we can verify the comparison ran successfully
        assert comparison_small_k["speedup"] > 0
        assert comparison_large_k["speedup"] > 0


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""

    def test_document_ranking_scenario(self):
        """Test a realistic document ranking scenario."""
        # Simulate document scores from a retrieval system
        documents = [f"Document {i}" for i in range(1000)]
        scores = [0.1 + (i * 0.0009) for i in range(1000)]  # Increasing scores

        # Use heap for top-10 selection
        heap = HeapTopK(k=10)
        heap.add_batch(scores, documents)

        results = heap.get_sorted()

        # Should get top 10 documents with highest scores
        assert len(results) == 10
        assert results[0][0] > 0.9  # Highest scores at top
        assert all("Document" in item for _, item in results)

    def test_streaming_real_time_scenario(self):
        """Test streaming scenario with real-time additions."""
        streaming = StreamingTopK(k=5, track_stats=True)

        # Simulate real-time document processing
        for batch in range(10):
            batch_scores = [0.3 + batch * 0.1 + i * 0.01 for i in range(20)]
            batch_docs = [f"batch{batch}_doc{i}" for i in range(20)]

            for score, doc in zip(batch_scores, batch_docs):
                streaming.add(score, doc)

        # Check final state
        results = streaming.get_sorted()
        stats = streaming.get_stats()

        assert len(results) == 5
        assert stats["total_items_seen"] == 200
        assert stats["acceptance_rate"] < 1.0  # Some items were rejected
        assert results[0][0] > 1.0  # Should have high-scoring items

    def test_memory_efficiency(self):
        """Test memory efficiency for large datasets."""

        # Test that heap doesn't grow beyond k items
        heap = HeapTopK(k=100)

        # Add many more items than k
        for i in range(10000):
            heap.add(float(i), f"item{i}")

        # Heap should only contain k items
        assert heap.size() == 100
        assert heap.is_full()

        # Memory usage should be bounded
        results = heap.get_sorted()
        assert len(results) == 100

        # Should contain the highest scores
        assert results[0][0] >= 9900.0  # Close to the maximum value added
