"""
Compressed Sparse Row (CSR) matrix operations for efficient sparse matrix handling.

This module provides CSR matrix implementations optimized for sparse data scenarios
common in NLP and information retrieval, offering significant memory savings and
faster operations compared to dense matrices.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import norm


class CSRMatrix:
    """
    High-performance Compressed Sparse Row matrix wrapper.

    Provides optimized operations for sparse matrices with memory-efficient
    storage and fast arithmetic operations. Ideal for document-term matrices,
    embedding matrices with many zero values, and similarity computations.

    Example:
        >>> # Create from dense data
        >>> dense = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 6]])
        >>> csr = CSRMatrix.from_dense(dense)
        >>> print(csr.density())  # 0.67 (6/9 non-zero elements)

        >>> # Create from sparse data directly
        >>> rows = [0, 0, 1, 2, 2, 2]
        >>> cols = [0, 2, 2, 0, 1, 2]
        >>> data = [1, 2, 3, 4, 5, 6]
        >>> csr = CSRMatrix.from_coo(rows, cols, data, shape=(3, 3))
    """

    def __init__(self, matrix: csr_matrix):
        """
        Initialize CSR matrix wrapper.

        Args:
            matrix: Scipy CSR matrix
        """
        if not isinstance(matrix, csr_matrix):
            raise TypeError("Matrix must be scipy csr_matrix")
        self.matrix = matrix
        self._shape = matrix.shape

    @classmethod
    def from_dense(cls, dense_array: np.ndarray, threshold: float = 1e-10) -> "CSRMatrix":
        """
        Create CSR matrix from dense array.

        Args:
            dense_array: Dense numpy array
            threshold: Values below this threshold are considered zero

        Returns:
            CSRMatrix instance
        """
        # Zero out small values for better compression
        dense_copy = dense_array.copy()
        dense_copy[np.abs(dense_copy) < threshold] = 0
        return cls(csr_matrix(dense_copy))

    @classmethod
    def from_coo(
        cls,
        rows: List[int],
        cols: List[int],
        data: List[float],
        shape: Optional[Tuple[int, int]] = None,
    ) -> "CSRMatrix":
        """
        Create CSR matrix from coordinate (COO) format.

        Args:
            rows: Row indices
            cols: Column indices
            data: Data values
            shape: Matrix shape (auto-detected if None)

        Returns:
            CSRMatrix instance
        """
        from scipy.sparse import coo_matrix

        if shape is None:
            shape = (max(rows) + 1, max(cols) + 1)

        coo = coo_matrix((data, (rows, cols)), shape=shape)
        return cls(coo.tocsr())

    @classmethod
    def from_dict(
        cls, data_dict: Dict[Tuple[int, int], float], shape: Tuple[int, int]
    ) -> "CSRMatrix":
        """
        Create CSR matrix from dictionary mapping (row, col) -> value.

        Args:
            data_dict: Dictionary with (row, col) keys and float values
            shape: Matrix dimensions

        Returns:
            CSRMatrix instance
        """
        rows, cols, data = [], [], []
        for (r, c), value in data_dict.items():
            rows.append(r)
            cols.append(c)
            data.append(value)

        return cls.from_coo(rows, cols, data, shape)

    @classmethod
    def eye(cls, n: int, dtype: np.dtype = np.float64) -> "CSRMatrix":
        """Create identity CSR matrix."""
        from scipy.sparse import eye

        return cls(eye(n, dtype=dtype, format="csr"))

    @classmethod
    def zeros(cls, shape: Tuple[int, int], dtype: np.dtype = np.float64) -> "CSRMatrix":
        """Create zero CSR matrix."""
        return cls(csr_matrix(shape, dtype=dtype))

    @classmethod
    def random_sparse(
        cls,
        shape: Tuple[int, int],
        density: float = 0.1,
        random_state: Optional[int] = None,
    ) -> "CSRMatrix":
        """
        Create random sparse matrix.

        Args:
            shape: Matrix dimensions
            density: Proportion of non-zero elements
            random_state: Random seed for reproducibility

        Returns:
            Random CSRMatrix
        """
        from scipy.sparse import random

        sparse_matrix = random(
            shape[0], shape[1], density=density, random_state=random_state, format="csr"
        )
        return cls(sparse_matrix)

    def shape(self) -> Tuple[int, int]:
        """Return matrix dimensions."""
        return self._shape

    def nnz(self) -> int:
        """Return number of non-zero elements."""
        return self.matrix.nnz

    def density(self) -> float:
        """Return density (proportion of non-zero elements)."""
        total_elements = self._shape[0] * self._shape[1]
        return self.nnz() / total_elements if total_elements > 0 else 0.0

    def memory_usage(self) -> Dict[str, int]:
        """
        Return memory usage information.

        Returns:
            Dictionary with memory usage details in bytes
        """
        data_bytes = self.matrix.data.nbytes
        indices_bytes = self.matrix.indices.nbytes
        indptr_bytes = self.matrix.indptr.nbytes
        total_bytes = data_bytes + indices_bytes + indptr_bytes

        # Compare with dense equivalent
        dense_bytes = np.prod(self._shape) * 8  # float64 size

        return {
            "total_bytes": total_bytes,
            "data_bytes": data_bytes,
            "indices_bytes": indices_bytes,
            "indptr_bytes": indptr_bytes,
            "dense_equivalent_bytes": dense_bytes,
            "memory_savings": dense_bytes - total_bytes,
            "compression_ratio": dense_bytes / total_bytes if total_bytes > 0 else float("inf"),
        }

    def to_dense(self) -> np.ndarray:
        """Convert to dense numpy array."""
        return self.matrix.toarray()

    def copy(self) -> "CSRMatrix":
        """Create a copy of the matrix."""
        return CSRMatrix(self.matrix.copy())

    def transpose(self) -> "CSRMatrix":
        """Return transposed matrix."""
        return CSRMatrix(self.matrix.transpose().tocsr())  # type: ignore

    def __getitem__(self, key) -> Union["CSRMatrix", float]:
        """Support indexing operations."""
        result = self.matrix[key]

        # Check if it's a matrix result
        if hasattr(result, "shape") and len(getattr(result, "shape", [])) > 0:
            # Convert to CSR matrix
            try:
                if hasattr(result, "tocsr"):
                    return CSRMatrix(result.tocsr())  # type: ignore
                elif isinstance(result, csr_matrix):
                    return CSRMatrix(result)
                else:
                    return CSRMatrix(csr_matrix(result))
            except Exception:
                # Fallback for any conversion issues
                return CSRMatrix(csr_matrix(result))

        # It's a scalar - handle different numpy types
        try:
            if hasattr(result, "item"):
                return float(result.item())  # type: ignore
            else:
                return float(result)  # type: ignore
        except (TypeError, ValueError):
            # For complex types that can't be converted to float
            return 0.0

    def __setitem__(self, key, value):
        """Support item assignment."""
        self.matrix[key] = value

    def get_row(self, row_idx: int) -> "CSRMatrix":
        """Get specific row as CSR matrix."""
        return CSRMatrix(self.matrix.getrow(row_idx))

    def get_col(self, col_idx: int) -> "CSRMatrix":
        """Get specific column as CSR matrix."""
        return CSRMatrix(self.matrix.getcol(col_idx))

    def dot(self, other: Union["CSRMatrix", np.ndarray]) -> Union["CSRMatrix", np.ndarray]:
        """
        Matrix multiplication.

        Args:
            other: Matrix or array to multiply with

        Returns:
            Result of multiplication
        """
        if isinstance(other, CSRMatrix):
            result = self.matrix.dot(other.matrix)
            if hasattr(result, "shape") and len(result.shape) == 2:
                return CSRMatrix(result.tocsr())
            return result
        else:
            return self.matrix.dot(other)

    def multiply(self, other: Union["CSRMatrix", float, np.ndarray]) -> "CSRMatrix":
        """
        Element-wise multiplication.

        Args:
            other: Value or matrix to multiply element-wise

        Returns:
            Result of element-wise multiplication
        """
        if isinstance(other, CSRMatrix):
            return CSRMatrix(self.matrix.multiply(other.matrix))
        else:
            return CSRMatrix(self.matrix.multiply(other))

    def add(self, other: "CSRMatrix") -> "CSRMatrix":
        """Element-wise addition."""
        return CSRMatrix(self.matrix + other.matrix)

    def subtract(self, other: "CSRMatrix") -> "CSRMatrix":
        """Element-wise subtraction."""
        return CSRMatrix(self.matrix - other.matrix)

    def sum(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Sum along specified axis."""
        result = self.matrix.sum(axis=axis)
        if isinstance(result, np.matrix):
            return np.asarray(result).flatten()
        return result

    def mean(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Mean along specified axis."""
        result = self.matrix.mean(axis=axis)
        if isinstance(result, np.matrix):
            return np.asarray(result).flatten()
        return result

    def max(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Maximum along specified axis."""
        result = self.matrix.max(axis=axis)
        if isinstance(result, np.matrix):
            return np.asarray(result).flatten()
        return result

    def min(self, axis: Optional[int] = None) -> Union[float, np.ndarray]:
        """Minimum along specified axis."""
        result = self.matrix.min(axis=axis)
        if isinstance(result, np.matrix):
            return np.asarray(result).flatten()
        return result

    def norm(self, ord: Union[str, int] = "fro") -> float:
        """Calculate matrix norm."""
        return float(norm(self.matrix, ord=ord))

    def eliminate_zeros(self) -> "CSRMatrix":
        """Remove explicit zero entries."""
        matrix_copy = self.matrix.copy()
        matrix_copy.eliminate_zeros()
        return CSRMatrix(matrix_copy)

    def __add__(self, other: "CSRMatrix") -> "CSRMatrix":
        """Addition operator."""
        return self.add(other)

    def __sub__(self, other: "CSRMatrix") -> "CSRMatrix":
        """Subtraction operator."""
        return self.subtract(other)

    def __mul__(self, other: Union["CSRMatrix", float]) -> "CSRMatrix":
        """Multiplication operator (element-wise for matrices, scalar for numbers)."""
        if isinstance(other, (int, float)):
            return CSRMatrix(self.matrix * other)
        return self.multiply(other)

    def __rmul__(self, other: Union[float, int]) -> "CSRMatrix":
        """Right multiplication by scalar."""
        return CSRMatrix(other * self.matrix)

    def __truediv__(self, other: Union[float, int]) -> "CSRMatrix":
        """Division by scalar."""
        return CSRMatrix(self.matrix / other)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CSRMatrix(shape={self.shape()}, nnz={self.nnz()}, " f"density={self.density():.3f})"
        )


class BatchCSROperations:
    """
    Batch operations for CSR matrices to improve performance.

    Provides optimized batch operations for common CSR matrix operations
    like similarity computation, nearest neighbors, and matrix concatenation.
    """

    @staticmethod
    def vstack(matrices: List[CSRMatrix]) -> CSRMatrix:
        """
        Vertically stack CSR matrices.

        Args:
            matrices: List of CSR matrices to stack

        Returns:
            Vertically stacked matrix
        """
        if not matrices:
            raise ValueError("Cannot stack empty list of matrices")

        scipy_matrices = [m.matrix for m in matrices]
        stacked = vstack(scipy_matrices, format="csr")
        return CSRMatrix(stacked.tocsr())  # type: ignore

    @staticmethod
    def hstack(matrices: List[CSRMatrix]) -> CSRMatrix:
        """
        Horizontally stack CSR matrices.

        Args:
            matrices: List of CSR matrices to stack

        Returns:
            Horizontally stacked matrix
        """
        if not matrices:
            raise ValueError("Cannot stack empty list of matrices")

        from scipy.sparse import hstack

        scipy_matrices = [m.matrix for m in matrices]
        stacked = hstack(scipy_matrices, format="csr")
        return CSRMatrix(stacked.tocsr())  # type: ignore

    @staticmethod
    def cosine_similarity(matrix1: CSRMatrix, matrix2: Optional[CSRMatrix] = None) -> CSRMatrix:
        """
        Compute cosine similarity between rows of matrices.

        Args:
            matrix1: First matrix
            matrix2: Second matrix (if None, compute pairwise for matrix1)

        Returns:
            Cosine similarity matrix
        """
        from sklearn.metrics.pairwise import cosine_similarity

        if matrix2 is None:
            sim_matrix = cosine_similarity(matrix1.matrix)
        else:
            sim_matrix = cosine_similarity(matrix1.matrix, matrix2.matrix)

        return CSRMatrix.from_dense(sim_matrix)

    @staticmethod
    def euclidean_distances(matrix1: CSRMatrix, matrix2: Optional[CSRMatrix] = None) -> np.ndarray:
        """
        Compute Euclidean distances between rows of matrices.

        Args:
            matrix1: First matrix
            matrix2: Second matrix (if None, compute pairwise for matrix1)

        Returns:
            Distance matrix
        """
        from sklearn.metrics.pairwise import euclidean_distances

        if matrix2 is None:
            return euclidean_distances(matrix1.matrix)
        else:
            return euclidean_distances(matrix1.matrix, matrix2.matrix)

    @staticmethod
    def top_k_similarity(
        query_matrix: CSRMatrix, corpus_matrix: CSRMatrix, k: int = 10, metric: str = "cosine"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find top-k most similar items for each query.

        Args:
            query_matrix: Query vectors
            corpus_matrix: Corpus vectors to search in
            k: Number of top results to return
            metric: Similarity metric ('cosine' or 'euclidean')

        Returns:
            Tuple of (similarities, indices) arrays
        """
        if metric == "cosine":
            from sklearn.metrics.pairwise import cosine_similarity

            similarities = cosine_similarity(query_matrix.matrix, corpus_matrix.matrix)
        elif metric == "euclidean":
            from sklearn.metrics.pairwise import euclidean_distances

            # Convert distances to similarities (inverse relationship)
            distances = euclidean_distances(query_matrix.matrix, corpus_matrix.matrix)
            similarities = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Get top-k for each query
        top_indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
        top_similarities = np.take_along_axis(similarities, top_indices, axis=1)

        # Sort within top-k results
        sorted_indices = np.argsort(-top_similarities, axis=1)
        final_indices = np.take_along_axis(top_indices, sorted_indices, axis=1)
        final_similarities = np.take_along_axis(top_similarities, sorted_indices, axis=1)

        return final_similarities, final_indices


class SparseEmbeddingMatrix:
    """
    Specialized CSR matrix for embedding storage and operations.

    Optimized for NLP embeddings that often have many zero or near-zero
    values, providing memory-efficient storage and fast similarity operations.
    """

    def __init__(
        self,
        embeddings: Union[np.ndarray, List[List[float]], CSRMatrix],
        vocab: Optional[List[str]] = None,
        sparsity_threshold: float = 1e-6,
    ):
        """
        Initialize sparse embedding matrix.

        Args:
            embeddings: Dense or sparse embeddings
            vocab: Vocabulary list (optional)
            sparsity_threshold: Threshold below which values are zeroed
        """
        if isinstance(embeddings, CSRMatrix):
            self.matrix = embeddings
        elif isinstance(embeddings, np.ndarray):
            self.matrix = CSRMatrix.from_dense(embeddings, threshold=sparsity_threshold)
        else:
            dense_embeddings = np.array(embeddings)
            self.matrix = CSRMatrix.from_dense(dense_embeddings, threshold=sparsity_threshold)

        self.vocab = vocab or [f"item_{i}" for i in range(self.matrix.shape()[0])]

        if len(self.vocab) != self.matrix.shape()[0]:
            raise ValueError("Vocabulary size must match number of embeddings")

    def get_embedding(self, item: Union[str, int]) -> np.ndarray:
        """
        Get embedding for specific item.

        Args:
            item: Item name (str) or index (int)

        Returns:
            Embedding vector
        """
        if isinstance(item, str):
            if item not in self.vocab:
                raise KeyError(f"Item '{item}' not found in vocabulary")
            idx = self.vocab.index(item)
        else:
            idx = item

        return self.matrix.get_row(idx).to_dense().flatten()

    def find_similar(
        self, query: Union[str, int, np.ndarray], k: int = 10, metric: str = "cosine"
    ) -> List[Tuple[str, float]]:
        """
        Find most similar items to query.

        Args:
            query: Query item name, index, or embedding vector
            k: Number of results to return
            metric: Similarity metric

        Returns:
            List of (item_name, similarity_score) tuples
        """
        if isinstance(query, (str, int)):
            query_vector = self.get_embedding(query).reshape(1, -1)
            query_matrix = CSRMatrix.from_dense(query_vector)
        else:
            query_matrix = CSRMatrix.from_dense(query.reshape(1, -1))

        similarities, indices = BatchCSROperations.top_k_similarity(
            query_matrix, self.matrix, k=k, metric=metric
        )

        results = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            item_name = self.vocab[idx]
            results.append((item_name, float(sim)))

        return results

    def add_embedding(self, item_name: str, embedding: np.ndarray) -> None:
        """
        Add new embedding to the matrix.

        Args:
            item_name: Name of the new item
            embedding: Embedding vector
        """
        if item_name in self.vocab:
            raise ValueError(f"Item '{item_name}' already exists")

        new_row = CSRMatrix.from_dense(embedding.reshape(1, -1))
        self.matrix = BatchCSROperations.vstack([self.matrix, new_row])
        self.vocab.append(item_name)

    def save(self, filepath: str) -> None:
        """Save sparse embedding matrix to file."""
        from scipy.sparse import save_npz

        save_npz(f"{filepath}_matrix.npz", self.matrix.matrix)

        # Save vocabulary
        import json

        with open(f"{filepath}_vocab.json", "w") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, filepath: str) -> "SparseEmbeddingMatrix":
        """Load sparse embedding matrix from file."""
        import json

        from scipy.sparse import load_npz

        matrix = CSRMatrix(load_npz(f"{filepath}_matrix.npz"))

        with open(f"{filepath}_vocab.json", "r") as f:
            vocab = json.load(f)

        return cls(matrix, vocab)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding matrix."""
        memory_info = self.matrix.memory_usage()

        return {
            "shape": self.matrix.shape(),
            "vocabulary_size": len(self.vocab),
            "embedding_dimension": self.matrix.shape()[1],
            "density": self.matrix.density(),
            "nnz": self.matrix.nnz(),
            "memory_usage_mb": memory_info["total_bytes"] / (1024 * 1024),
            "memory_savings_mb": memory_info["memory_savings"] / (1024 * 1024),
            "compression_ratio": memory_info["compression_ratio"],
        }

    def __len__(self) -> int:
        """Return number of embeddings."""
        return len(self.vocab)

    def __contains__(self, item: str) -> bool:
        """Check if item exists in vocabulary."""
        return item in self.vocab

    def __getitem__(self, item: Union[str, int]) -> np.ndarray:
        """Get embedding by item name or index."""
        return self.get_embedding(item)


def optimize_sparse_matrix(matrix: CSRMatrix, target_density: float = 0.1) -> CSRMatrix:
    """
    Optimize CSR matrix by removing small values to achieve target density.

    Args:
        matrix: Input CSR matrix
        target_density: Target density ratio

    Returns:
        Optimized CSR matrix
    """
    current_density = matrix.density()

    if current_density <= target_density:
        return matrix.copy()

    # Find threshold to achieve target density
    data = np.abs(matrix.matrix.data)
    sorted_data = np.sort(data)

    # Calculate how many elements to remove
    total_elements = len(data)
    target_nnz = int(target_density * np.prod(matrix.shape()))
    elements_to_remove = max(0, total_elements - target_nnz)

    if elements_to_remove > 0:
        threshold = sorted_data[elements_to_remove - 1]

        # Create new matrix with values below threshold zeroed
        new_matrix = matrix.matrix.copy()
        new_matrix.data[np.abs(new_matrix.data) <= threshold] = 0
        new_matrix.eliminate_zeros()

        return CSRMatrix(new_matrix)

    return matrix.copy()


def benchmark_sparse_vs_dense(
    shapes: List[Tuple[int, int]], densities: List[float]
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark sparse vs dense matrix operations.

    Args:
        shapes: List of matrix dimensions to test
        densities: List of density ratios to test

    Returns:
        Benchmark results dictionary
    """
    import time

    results = {}

    for shape in shapes:
        for density in densities:
            test_name = f"shape_{shape[0]}x{shape[1]}_density_{density:.2f}"
            results[test_name] = {}

            # Create test matrices
            sparse_matrix = CSRMatrix.random_sparse(shape, density=density, random_state=42)
            dense_matrix = sparse_matrix.to_dense()

            # Benchmark matrix multiplication
            test_vector = np.random.random(shape[1])

            # Sparse multiplication
            start = time.perf_counter()
            _ = sparse_matrix.dot(test_vector)
            sparse_time = time.perf_counter() - start

            # Dense multiplication
            start = time.perf_counter()
            _ = dense_matrix.dot(test_vector)
            dense_time = time.perf_counter() - start

            # Memory usage
            sparse_memory = sparse_matrix.memory_usage()

            results[test_name] = {
                "sparse_mult_time": sparse_time * 1000,  # ms
                "dense_mult_time": dense_time * 1000,  # ms
                "speedup_ratio": dense_time / sparse_time if sparse_time > 0 else float("inf"),
                "memory_savings_ratio": sparse_memory["compression_ratio"],
                "density": sparse_matrix.density(),
            }

    return results
