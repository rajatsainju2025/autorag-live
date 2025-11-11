"""Optimized numpy operations for retrieval pipelines.

This module provides vectorized and batched numpy operations for:
- Efficient similarity computation
- Memory-optimized array operations
- Vectorized ranking operations
- Batch processing with reduced allocations

Example:
    >>> from autorag_live.utils.numpy_ops import top_k_similarity
    >>> scores = np.random.random((1000, 100))
    >>> indices = top_k_similarity(scores, k=10)
"""

from typing import Tuple

import numpy as np


def top_k_similarity(
    similarity_matrix: np.ndarray,
    k: int,
    axis: int = 1,
) -> np.ndarray:
    """Get top-k indices for each row without full sorting.

    Uses argpartition for O(n) instead of O(n log n) complexity.

    Args:
        similarity_matrix: Shape (n_queries, n_docs) similarity scores
        k: Number of top items to retrieve
        axis: Axis to partition along

    Returns:
        Array of shape (n_queries, k) with top-k indices
    """
    if k >= similarity_matrix.shape[axis]:
        # All items
        return np.argsort(similarity_matrix, axis=axis)[:, -k:]

    # Partial sort for efficiency
    indices = np.argpartition(similarity_matrix, -k, axis=axis)
    return np.take_along_axis(
        indices,
        np.argsort(np.take_along_axis(similarity_matrix, indices[:, -k:], axis=axis), axis=axis)[
            :, -k:
        ],
        axis=axis,
    )


def batched_dot_product(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute dot products in batches to reduce memory usage.

    Args:
        query_embeddings: Shape (n_queries, d) query embeddings
        corpus_embeddings: Shape (n_docs, d) corpus embeddings
        batch_size: Batch size for memory efficiency

    Returns:
        Shape (n_queries, n_docs) similarity matrix
    """
    n_queries = query_embeddings.shape[0]
    n_docs = corpus_embeddings.shape[0]

    similarities = np.empty((n_queries, n_docs), dtype=np.float32)

    for i in range(0, n_queries, batch_size):
        end_idx = min(i + batch_size, n_queries)
        batch_queries = query_embeddings[i:end_idx]
        # Use @ operator for efficient matrix multiplication
        similarities[i:end_idx] = batch_queries @ corpus_embeddings.T

    return similarities


def cosine_similarity_fast(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
) -> np.ndarray:
    """Fast cosine similarity for normalized embeddings.

    When embeddings are L2-normalized, cosine similarity = dot product.

    Args:
        query_embedding: Shape (d,) or (n, d) query embedding(s)
        corpus_embeddings: Shape (m, d) corpus embeddings

    Returns:
        Shape (m,) or (n, m) similarity scores
    """
    # For normalized vectors, cosine similarity is dot product
    # This avoids expensive normalization
    if query_embedding.ndim == 1:
        return corpus_embeddings @ query_embedding
    else:
        return query_embedding @ corpus_embeddings.T


def reduce_memory_copies(
    array: np.ndarray,
) -> np.ndarray:
    """Ensure array uses C-contiguous memory layout.

    Improves performance of subsequent operations by ensuring
    optimal memory access patterns.

    Args:
        array: Input array

    Returns:
        C-contiguous array
    """
    if not array.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(array)
    return array


def ranked_top_k(
    scores: np.ndarray,
    indices: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get top-k scores and corresponding indices, sorted by scores.

    Args:
        scores: Array of scores
        indices: Corresponding indices
        k: Number of top items

    Returns:
        Tuple of (top_scores, top_indices) both sorted descending
    """
    # Use argpartition for efficiency
    k_min = min(k, len(scores))
    top_idx = np.argpartition(scores, -k_min)[-k_min:]

    # Sort the top k by score
    sorted_idx = np.argsort(scores[top_idx])[::-1]

    return scores[top_idx[sorted_idx]], indices[top_idx[sorted_idx]]


def batch_cosine_similarity(
    queries: np.ndarray,
    corpus: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity matrix efficiently.

    Handles numerical stability for normalized vectors.

    Args:
        queries: Shape (n_queries, d) query vectors
        corpus: Shape (n_docs, d) corpus vectors

    Returns:
        Shape (n_queries, n_docs) similarity matrix
    """
    # Ensure C-contiguous for optimal performance
    queries = reduce_memory_copies(queries)
    corpus = reduce_memory_copies(corpus)

    # Use @ operator which is optimized in NumPy
    similarities = queries @ corpus.T

    # Clip to [-1, 1] to handle numerical errors
    return np.clip(similarities, -1.0, 1.0)


def chunk_array(
    array: np.ndarray,
    chunk_size: int,
) -> list[np.ndarray]:
    """Split array into chunks for batch processing.

    Args:
        array: Array to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunked arrays
    """
    n_chunks = (len(array) + chunk_size - 1) // chunk_size
    return [array[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]
