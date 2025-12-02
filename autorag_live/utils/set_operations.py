"""
Optimized set operations using Bloom filters for large-scale membership testing.

This module provides memory-efficient probabilistic data structures that can
dramatically improve performance for large set operations with acceptable
false positive rates.
"""

import hashlib
import math
from typing import List, Set, Union


class BloomFilter:
    """
    Memory-efficient Bloom filter for fast set membership testing.

    Provides O(1) membership testing with configurable false positive rate.
    Ideal for pre-filtering large datasets before expensive exact operations.

    Example:
        >>> bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)
        >>> bf.add("document_id_1")
        >>> bf.add("document_id_2")
        >>> "document_id_1" in bf  # True (might have false positives)
        >>> "nonexistent_id" in bf  # False (no false negatives)
    """

    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        """
        Initialize Bloom filter with optimal parameters.

        Args:
            expected_items: Expected number of items to store
            false_positive_rate: Desired false positive rate (0.0-1.0)
        """
        if not (0 < false_positive_rate < 1):
            raise ValueError("False positive rate must be between 0 and 1")
        if expected_items <= 0:
            raise ValueError("Expected items must be positive")

        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate

        # Calculate optimal bit array size and number of hash functions
        self.bit_array_size = self._optimal_bit_array_size(expected_items, false_positive_rate)
        self.num_hash_functions = self._optimal_num_hash_functions(
            self.bit_array_size, expected_items
        )

        # Initialize bit array
        self.bit_array = bytearray(math.ceil(self.bit_array_size / 8))
        self.num_items = 0

    @staticmethod
    def _optimal_bit_array_size(expected_items: int, false_positive_rate: float) -> int:
        """Calculate optimal bit array size for given parameters."""
        return int(-expected_items * math.log(false_positive_rate) / (math.log(2) ** 2))

    @staticmethod
    def _optimal_num_hash_functions(bit_array_size: int, expected_items: int) -> int:
        """Calculate optimal number of hash functions."""
        return max(1, int((bit_array_size / expected_items) * math.log(2)))

    def _hash(self, item: str) -> List[int]:
        """Generate multiple hash values for an item using double hashing."""
        # Use MD5 for primary hash (fast, good distribution)
        primary_hash = int(hashlib.md5(item.encode()).hexdigest(), 16)

        # Use SHA1 for secondary hash to avoid correlation
        secondary_hash = int(hashlib.sha1(item.encode()).hexdigest(), 16)

        hashes = []
        for i in range(self.num_hash_functions):
            # Double hashing: h1(x) + i * h2(x)
            hash_val = (primary_hash + i * secondary_hash) % self.bit_array_size
            hashes.append(hash_val)

        return hashes

    def add(self, item: Union[str, bytes]) -> None:
        """Add an item to the Bloom filter."""
        if isinstance(item, bytes):
            item = item.decode("utf-8", errors="ignore")

        for hash_val in self._hash(str(item)):
            byte_index = hash_val // 8
            bit_index = hash_val % 8
            self.bit_array[byte_index] |= 1 << bit_index

        self.num_items += 1

    def __contains__(self, item: Union[str, bytes]) -> bool:
        """Check if an item might be in the set (no false negatives)."""
        if isinstance(item, bytes):
            item = item.decode("utf-8", errors="ignore")

        for hash_val in self._hash(str(item)):
            byte_index = hash_val // 8
            bit_index = hash_val % 8
            if not (self.bit_array[byte_index] & (1 << bit_index)):
                return False
        return True

    def current_false_positive_rate(self) -> float:
        """Calculate current false positive rate based on actual usage."""
        if self.num_items == 0:
            return 0.0

        # Calculate actual false positive rate
        return (
            1 - math.exp(-self.num_hash_functions * self.num_items / self.bit_array_size)
        ) ** self.num_hash_functions

    def memory_usage(self) -> int:
        """Return memory usage in bytes."""
        return len(self.bit_array)

    def __len__(self) -> int:
        """Return the number of items added (not necessarily unique)."""
        return self.num_items


class OptimizedSetOperations:
    """
    High-performance set operations with Bloom filter pre-filtering.

    Uses Bloom filters to quickly eliminate impossible matches before
    performing expensive exact set operations.
    """

    @staticmethod
    def fast_intersection(
        set_a: Set[str], set_b: Set[str], use_bloom_filter: bool = True
    ) -> Set[str]:
        """
        Fast set intersection with optional Bloom filter pre-filtering.

        Args:
            set_a: First set
            set_b: Second set
            use_bloom_filter: Whether to use Bloom filter optimization

        Returns:
            Intersection of the two sets
        """
        if not set_a or not set_b:
            return set()

        # For small sets, direct intersection is faster
        if len(set_a) < 100 and len(set_b) < 100:
            return set_a & set_b

        # Ensure set_a is the smaller set for efficiency
        if len(set_a) > len(set_b):
            set_a, set_b = set_b, set_a

        if not use_bloom_filter:
            return set_a & set_b

        # Create Bloom filter for the larger set
        bf = BloomFilter(expected_items=len(set_b), false_positive_rate=0.01)
        for item in set_b:
            bf.add(item)

        # Pre-filter using Bloom filter, then exact check
        candidates = {item for item in set_a if item in bf}
        return candidates & set_b

    @staticmethod
    def fast_membership_batch(
        items: List[str], target_set: Set[str], use_bloom_filter: bool = True
    ) -> List[bool]:
        """
        Batch membership testing with Bloom filter optimization.

        Args:
            items: Items to test for membership
            target_set: Set to test against
            use_bloom_filter: Whether to use Bloom filter pre-filtering

        Returns:
            List of boolean results for each item
        """
        if not items or not target_set:
            return [False] * len(items)

        # For small sets, direct membership is faster
        if len(target_set) < 100:
            return [item in target_set for item in items]

        if not use_bloom_filter:
            return [item in target_set for item in items]

        # Create Bloom filter for the target set
        bf = BloomFilter(expected_items=len(target_set), false_positive_rate=0.01)
        for item in target_set:
            bf.add(item)

        results = []
        for item in items:
            # First check Bloom filter (fast pre-filter)
            if item not in bf:
                # Definitely not in set (no false negatives)
                results.append(False)
            else:
                # Might be in set, check exactly
                results.append(item in target_set)

        return results

    @staticmethod
    def fast_deduplication(items: List[str], preserve_order: bool = True) -> List[str]:
        """
        Fast deduplication using Bloom filter for large datasets.

        Args:
            items: Items to deduplicate
            preserve_order: Whether to preserve original order

        Returns:
            Deduplicated list of items
        """
        if not items:
            return []

        # For small lists, use set directly
        if len(items) < 1000:
            if preserve_order:
                seen = set()
                return [item for item in items if not (item in seen or seen.add(item))]
            else:
                return list(set(items))

        # Use Bloom filter for large lists to reduce memory pressure
        bf = BloomFilter(expected_items=len(items), false_positive_rate=0.01)
        exact_seen = set()
        result = []

        for item in items:
            # Quick check with Bloom filter
            if item not in bf:
                # Definitely new item
                bf.add(item)
                exact_seen.add(item)
                result.append(item)
            else:
                # Might be duplicate, check exactly
                if item not in exact_seen:
                    exact_seen.add(item)
                    result.append(item)

        return result


def estimate_bloom_filter_params(expected_items: int, memory_budget_mb: float) -> tuple[int, float]:
    """
    Estimate optimal Bloom filter parameters given memory budget.

    Args:
        expected_items: Expected number of items
        memory_budget_mb: Available memory in MB

    Returns:
        Tuple of (optimal_bit_array_size, achievable_false_positive_rate)
    """
    memory_budget_bits = int(memory_budget_mb * 8 * 1024 * 1024)

    # Calculate achievable false positive rate with given memory
    bits_per_item = memory_budget_bits / expected_items

    # Optimal number of hash functions for this bit density
    optimal_k = max(1, int(bits_per_item * math.log(2)))

    # Achievable false positive rate
    false_positive_rate = (1 - math.exp(-optimal_k / bits_per_item)) ** optimal_k

    return memory_budget_bits, false_positive_rate
