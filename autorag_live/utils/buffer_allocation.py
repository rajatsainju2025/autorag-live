"""Buffer pre-allocation for efficient numpy operations."""

from typing import Tuple

import numpy as np


class BufferAllocator:
    """Pre-allocates buffers for efficient numpy operations."""

    def __init__(self, max_buffer_size: int = 100000):
        """Initialize buffer allocator."""
        self._max_size = max_buffer_size
        self._buffers = {}

    def allocate_float_buffer(self, size: int, shape: Tuple[int, ...]) -> np.ndarray:
        """Allocate float buffer with given shape."""
        key = f"float_{size}_{shape}"
        if key not in self._buffers:
            if size <= self._max_size:
                self._buffers[key] = np.zeros(shape, dtype=np.float32)
            else:
                return np.zeros(shape, dtype=np.float32)
        return self._buffers[key]

    def allocate_int_buffer(self, size: int, shape: Tuple[int, ...]) -> np.ndarray:
        """Allocate int buffer with given shape."""
        key = f"int_{size}_{shape}"
        if key not in self._buffers:
            if size <= self._max_size:
                self._buffers[key] = np.zeros(shape, dtype=np.int32)
            else:
                return np.zeros(shape, dtype=np.int32)
        return self._buffers[key]

    def clear(self):
        """Clear all allocated buffers."""
        self._buffers.clear()


# Global allocator
_allocator = BufferAllocator()


def get_buffer_allocator() -> BufferAllocator:
    """Get global buffer allocator."""
    return _allocator


def preallocate_similarity_matrix(n_queries: int, n_docs: int) -> np.ndarray:
    """Pre-allocate similarity matrix for batch operations."""
    return np.zeros((n_queries, n_docs), dtype=np.float32)


def preallocate_scores_array(n_items: int) -> np.ndarray:
    """Pre-allocate scores array."""
    return np.zeros(n_items, dtype=np.float32)
