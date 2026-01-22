"""
Mixed precision embedding support for memory and compute efficiency.

Automatically converts embeddings to FP16 when beneficial, reducing
memory footprint by 50% with minimal accuracy loss.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np


class MixedPrecisionEmbedding:
    """
    Wrapper for embeddings with automatic mixed precision.

    Stores embeddings in FP16 when precision loss is acceptable,
    falling back to FP32 for critical operations.
    """

    def __init__(
        self,
        embedding: Union[List[float], np.ndarray],
        use_fp16: bool = True,
        precision_threshold: float = 1e-4,
    ):
        """
        Initialize mixed precision embedding.

        Args:
            embedding: Original embedding vector
            use_fp16: Enable FP16 storage
            precision_threshold: Max acceptable precision loss
        """
        self.original_dtype = np.float32
        self.use_fp16 = use_fp16
        self.precision_threshold = precision_threshold

        # Convert to numpy
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        else:
            embedding = embedding.astype(np.float32)

        # Determine if FP16 is safe
        if use_fp16 and self._is_fp16_safe(embedding):
            self._data = embedding.astype(np.float16)
            self._stored_dtype = np.float16
        else:
            self._data = embedding
            self._stored_dtype = np.float32

    def _is_fp16_safe(self, embedding: np.ndarray) -> bool:
        """Check if FP16 conversion is safe."""
        # Convert to FP16 and back
        fp16_version = embedding.astype(np.float16).astype(np.float32)

        # Check precision loss
        max_error = np.max(np.abs(embedding - fp16_version))
        relative_error = max_error / (np.max(np.abs(embedding)) + 1e-10)

        return relative_error < self.precision_threshold

    @property
    def vector(self) -> np.ndarray:
        """Get embedding as FP32 for computation."""
        if self._stored_dtype == np.float16:
            return self._data.astype(np.float32)
        return self._data

    @property
    def storage_vector(self) -> np.ndarray:
        """Get embedding in storage format."""
        return self._data

    @property
    def dtype(self) -> np.dtype:
        """Get storage dtype."""
        return self._stored_dtype

    @property
    def memory_saved(self) -> float:
        """Memory savings as percentage."""
        if self._stored_dtype == np.float16:
            return 50.0
        return 0.0

    def to_list(self) -> List[float]:
        """Convert to Python list."""
        return self.vector.tolist()

    def similarity(self, other: "MixedPrecisionEmbedding") -> float:
        """Compute cosine similarity with another embedding."""
        a = self.vector
        b = other.vector
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class EmbeddingQuantizer:
    """
    Quantizer for batch embedding compression.

    Applies mixed precision strategies to batches of embeddings
    for optimal memory/accuracy tradeoff.
    """

    @staticmethod
    def quantize_batch(
        embeddings: List[Union[List[float], np.ndarray]],
        use_fp16: bool = True,
        precision_threshold: float = 1e-4,
    ) -> List[MixedPrecisionEmbedding]:
        """
        Quantize a batch of embeddings.

        Args:
            embeddings: List of embedding vectors
            use_fp16: Enable FP16 storage
            precision_threshold: Max acceptable precision loss

        Returns:
            List of MixedPrecisionEmbedding objects
        """
        return [MixedPrecisionEmbedding(emb, use_fp16, precision_threshold) for emb in embeddings]

    @staticmethod
    def estimate_memory_savings(
        num_embeddings: int,
        dimensions: int,
        fp16_ratio: float = 1.0,
    ) -> dict:
        """
        Estimate memory savings from mixed precision.

        Args:
            num_embeddings: Number of embeddings
            dimensions: Embedding dimensions
            fp16_ratio: Fraction stored in FP16

        Returns:
            Dictionary with memory statistics
        """
        fp32_bytes = num_embeddings * dimensions * 4
        fp16_bytes = num_embeddings * dimensions * 2 * fp16_ratio
        fp32_remaining = num_embeddings * dimensions * 4 * (1 - fp16_ratio)
        total_bytes = fp16_bytes + fp32_remaining

        return {
            "original_mb": fp32_bytes / (1024**2),
            "compressed_mb": total_bytes / (1024**2),
            "saved_mb": (fp32_bytes - total_bytes) / (1024**2),
            "saved_percent": ((fp32_bytes - total_bytes) / fp32_bytes) * 100,
        }


def convert_embeddings_to_mixed_precision(
    embeddings: List[List[float]],
    use_fp16: bool = True,
) -> List[MixedPrecisionEmbedding]:
    """
    Convert embeddings to mixed precision format.

    Args:
        embeddings: List of embedding vectors
        use_fp16: Enable FP16 storage

    Returns:
        List of MixedPrecisionEmbedding objects
    """
    return EmbeddingQuantizer.quantize_batch(embeddings, use_fp16)
