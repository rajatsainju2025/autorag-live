"""
Tests for Compressed Sparse Row (CSR) matrix operations.

This test suite validates the correctness and performance of sparse matrix
implementations for efficient memory usage and fast computations.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from autorag_live.utils.csr_matrix import (
    BatchCSROperations,
    CSRMatrix,
    SparseEmbeddingMatrix,
    benchmark_sparse_vs_dense,
    optimize_sparse_matrix,
)


class TestCSRMatrix:
    """Test cases for CSRMatrix class."""

    def test_from_dense_creation(self):
        """Test creating CSR matrix from dense array."""
        dense = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
        csr = CSRMatrix.from_dense(dense)

        assert csr.shape() == (3, 3)
        assert csr.nnz() == 5  # 5 non-zero elements
        assert csr.density() == 5 / 9

        # Test threshold filtering
        dense_with_small = np.array([[1e-12, 1, 0], [2, 0, 1e-11]])
        csr_filtered = CSRMatrix.from_dense(dense_with_small, threshold=1e-10)
        assert csr_filtered.nnz() == 2  # Small values filtered out

    def test_from_coo_creation(self):
        """Test creating CSR matrix from coordinate format."""
        rows = [0, 0, 1, 2, 2]
        cols = [0, 2, 1, 0, 2]
        data = [1, 2, 3, 4, 5]

        csr = CSRMatrix.from_coo(rows, cols, data, shape=(3, 3))
        assert csr.shape() == (3, 3)
        assert csr.nnz() == 5

        # Test auto shape detection
        csr_auto = CSRMatrix.from_coo(rows, cols, data)
        assert csr_auto.shape() == (3, 3)

    def test_from_dict_creation(self):
        """Test creating CSR matrix from dictionary."""
        data_dict = {
            (0, 0): 1.0,
            (0, 2): 2.0,
            (1, 1): 3.0,
            (2, 0): 4.0,
            (2, 2): 5.0,
        }

        csr = CSRMatrix.from_dict(data_dict, shape=(3, 3))
        assert csr.shape() == (3, 3)
        assert csr.nnz() == 5

        # Verify values
        dense = csr.to_dense()
        assert dense[0, 0] == 1.0
        assert dense[0, 2] == 2.0
        assert dense[1, 1] == 3.0

    def test_special_matrices(self):
        """Test creating special matrices."""
        # Identity matrix
        eye = CSRMatrix.eye(3)
        assert eye.shape() == (3, 3)
        assert eye.nnz() == 3
        assert np.allclose(eye.to_dense(), np.eye(3))

        # Zero matrix
        zeros = CSRMatrix.zeros((2, 3))
        assert zeros.shape() == (2, 3)
        assert zeros.nnz() == 0

        # Random sparse matrix
        random_sparse = CSRMatrix.random_sparse((10, 10), density=0.2, random_state=42)
        assert random_sparse.shape() == (10, 10)
        assert 0.15 < random_sparse.density() < 0.25  # Approximately 20% density

    def test_matrix_properties(self):
        """Test matrix property methods."""
        dense = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
        csr = CSRMatrix.from_dense(dense)

        assert csr.shape() == (3, 3)
        assert csr.nnz() == 5
        assert abs(csr.density() - 5 / 9) < 1e-10

        # Test memory usage
        memory_info = csr.memory_usage()
        assert memory_info["total_bytes"] > 0
        assert memory_info["compression_ratio"] > 0
        # Memory savings can be negative for very small/dense matrices
        # assert memory_info["memory_savings"] > 0

    def test_matrix_operations(self):
        """Test basic matrix operations."""
        dense1 = np.array([[1, 0, 2], [0, 3, 0]])
        dense2 = np.array([[2, 1], [0, 0], [1, 2]])

        csr1 = CSRMatrix.from_dense(dense1)
        csr2 = CSRMatrix.from_dense(dense2)

        # Test matrix multiplication
        result = csr1.dot(csr2)
        expected = dense1.dot(dense2)
        if isinstance(result, CSRMatrix):
            assert np.allclose(result.to_dense(), expected)
        else:
            assert np.allclose(result, expected)

        # Element-wise operations
        csr1_copy = CSRMatrix.from_dense(dense1)
        sum_result = csr1.add(csr1_copy)
        assert np.allclose(sum_result.to_dense(), dense1 + dense1)

        # Scalar multiplication
        scaled = csr1 * 2.0
        assert np.allclose(scaled.to_dense(), dense1 * 2.0)

    def test_indexing_and_slicing(self):
        """Test matrix indexing and slicing."""
        dense = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        csr = CSRMatrix.from_dense(dense)

        # Get row
        row1 = csr.get_row(1)
        assert np.allclose(row1.to_dense(), [[4, 5, 6]])

        # Get column
        col2 = csr.get_col(2)
        assert np.allclose(col2.to_dense(), [[3], [6], [9]])

        # Indexing
        assert csr[1, 2] == 6.0
        submatrix = csr[0:2, 1:3]
        if isinstance(submatrix, CSRMatrix):
            assert submatrix.shape() == (2, 2)

    def test_statistical_operations(self):
        """Test statistical operations."""
        dense = np.array([[1, 0, 3], [2, 4, 0], [0, 1, 2]])
        csr = CSRMatrix.from_dense(dense)

        # Sum operations
        assert csr.sum() == 13.0
        row_sums = csr.sum(axis=1)
        assert np.allclose(row_sums, [4, 6, 3])

        col_sums = csr.sum(axis=0)
        assert np.allclose(col_sums, [3, 5, 5])

        # Mean operations
        assert abs(csr.mean() - 13.0 / 9) < 1e-10

        # Max/min operations
        assert csr.max() == 4.0
        assert csr.min() == 0.0

    def test_matrix_norms(self):
        """Test matrix norm calculations."""
        dense = np.array([[3, 0, 4], [0, 5, 0]])
        csr = CSRMatrix.from_dense(dense)

        # Frobenius norm
        fro_norm = csr.norm("fro")
        expected_fro = np.linalg.norm(dense, "fro")
        assert abs(fro_norm - expected_fro) < 1e-10

    def test_matrix_utilities(self):
        """Test utility methods."""
        dense = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 0]])
        csr = CSRMatrix.from_dense(dense)

        # Copy
        csr_copy = csr.copy()
        assert np.allclose(csr.to_dense(), csr_copy.to_dense())
        assert csr.matrix is not csr_copy.matrix  # Different objects

        # Transpose
        csr_t = csr.transpose()
        assert np.allclose(csr_t.to_dense(), dense.T)

        # Eliminate zeros (add explicit zeros first)
        csr.matrix[0, 1] = 0.0  # Add explicit zero
        cleaned = csr.eliminate_zeros()
        assert cleaned.nnz() == 5  # Should remain same

    def test_magic_methods(self):
        """Test magic methods implementation."""
        dense1 = np.array([[1, 2], [3, 4]])
        dense2 = np.array([[2, 0], [1, 1]])

        csr1 = CSRMatrix.from_dense(dense1)
        csr2 = CSRMatrix.from_dense(dense2)

        # Addition
        result_add = csr1 + csr2
        assert np.allclose(result_add.to_dense(), dense1 + dense2)

        # Subtraction
        result_sub = csr1 - csr2
        assert np.allclose(result_sub.to_dense(), dense1 - dense2)

        # Scalar multiplication
        result_mul = csr1 * 3
        assert np.allclose(result_mul.to_dense(), dense1 * 3)

        # Right scalar multiplication
        result_rmul = 3 * csr1
        assert np.allclose(result_rmul.to_dense(), 3 * dense1)

        # Division
        result_div = csr1 / 2
        assert np.allclose(result_div.to_dense(), dense1 / 2)

        # String representation
        repr_str = repr(csr1)
        assert "CSRMatrix" in repr_str
        assert "shape=(2, 2)" in repr_str

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        # Invalid matrix type
        with pytest.raises(TypeError):
            CSRMatrix("not_a_matrix")  # type: ignore

        # Invalid shapes for operations
        csr1 = CSRMatrix.from_dense(np.array([[1, 2]]))
        csr2 = CSRMatrix.from_dense(np.array([[1], [2], [3]]))

        with pytest.raises((ValueError, Exception)):  # Scipy may raise different errors
            csr1.dot(csr2)


class TestBatchCSROperations:
    """Test cases for batch CSR operations."""

    def test_matrix_stacking(self):
        """Test vertical and horizontal matrix stacking."""
        mat1 = CSRMatrix.from_dense(np.array([[1, 2], [3, 4]]))
        mat2 = CSRMatrix.from_dense(np.array([[5, 6], [7, 8]]))

        # Vertical stacking
        vstacked = BatchCSROperations.vstack([mat1, mat2])
        expected_v = np.vstack([mat1.to_dense(), mat2.to_dense()])
        assert np.allclose(vstacked.to_dense(), expected_v)

        # Horizontal stacking
        hstacked = BatchCSROperations.hstack([mat1, mat2])
        expected_h = np.hstack([mat1.to_dense(), mat2.to_dense()])
        assert np.allclose(hstacked.to_dense(), expected_h)

        # Empty list error
        with pytest.raises(ValueError):
            BatchCSROperations.vstack([])

    def test_similarity_operations(self):
        """Test similarity computation operations."""
        # Create test matrices
        mat1 = CSRMatrix.from_dense(np.array([[1, 0, 1], [0, 1, 1]]))
        mat2 = CSRMatrix.from_dense(np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]]))

        # Cosine similarity
        cos_sim = BatchCSROperations.cosine_similarity(mat1, mat2)
        assert cos_sim.shape() == (2, 3)

        # Self similarity
        self_sim = BatchCSROperations.cosine_similarity(mat1)
        assert self_sim.shape() == (2, 2)

        # Euclidean distances
        euc_dist = BatchCSROperations.euclidean_distances(mat1, mat2)
        assert euc_dist.shape == (2, 3)

    def test_top_k_similarity(self):
        """Test top-k similarity search."""
        # Create test matrices
        query = CSRMatrix.from_dense(np.array([[1, 0, 1]]))
        corpus = CSRMatrix.from_dense(
            np.array(
                [
                    [1, 0, 1],  # Identical
                    [0, 1, 0],  # Different
                    [1, 0, 0],  # Partial match
                    [0, 0, 1],  # Partial match
                ]
            )
        )

        # Test cosine similarity
        similarities, indices = BatchCSROperations.top_k_similarity(
            query, corpus, k=2, metric="cosine"
        )

        assert similarities.shape == (1, 2)
        assert indices.shape == (1, 2)
        assert indices[0, 0] == 0  # Most similar should be identical vector

        # Test euclidean similarity
        similarities_euc, indices_euc = BatchCSROperations.top_k_similarity(
            query, corpus, k=3, metric="euclidean"
        )

        assert similarities_euc.shape == (1, 3)
        assert indices_euc.shape == (1, 3)

        # Test invalid metric
        with pytest.raises(ValueError):
            BatchCSROperations.top_k_similarity(query, corpus, k=1, metric="invalid")


class TestSparseEmbeddingMatrix:
    """Test cases for SparseEmbeddingMatrix."""

    def test_initialization(self):
        """Test initialization with different input types."""
        # From numpy array
        embeddings = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
        vocab = ["word1", "word2", "word3"]

        sem = SparseEmbeddingMatrix(embeddings, vocab=vocab)
        assert len(sem) == 3
        assert sem.matrix.shape() == (3, 3)
        assert "word1" in sem

        # From list of lists
        embeddings_list = [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]]
        sem_list = SparseEmbeddingMatrix(embeddings_list, vocab=vocab)
        assert np.allclose(sem.matrix.to_dense(), sem_list.matrix.to_dense())

        # Auto-generated vocabulary
        sem_auto = SparseEmbeddingMatrix(embeddings)
        assert len(sem_auto.vocab) == 3
        assert sem_auto.vocab[0] == "item_0"

    def test_embedding_access(self):
        """Test accessing embeddings by name and index."""
        embeddings = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
        vocab = ["apple", "banana", "cherry"]
        sem = SparseEmbeddingMatrix(embeddings, vocab=vocab)

        # Access by name
        apple_embedding = sem.get_embedding("apple")
        assert np.allclose(apple_embedding, [1, 0, 2])

        # Access by index
        banana_embedding = sem.get_embedding(1)
        assert np.allclose(banana_embedding, [0, 3, 0])

        # Access via indexing
        cherry_embedding = sem["cherry"]
        assert np.allclose(cherry_embedding, [4, 0, 5])

        # Test KeyError for missing item
        with pytest.raises(KeyError):
            sem.get_embedding("missing")

    def test_similarity_search(self):
        """Test finding similar items."""
        # Create test embeddings with clear similarity patterns
        embeddings = np.array(
            [
                [1, 0, 0],  # fruit1
                [1, 0, 0],  # fruit2 (identical to fruit1)
                [0, 1, 0],  # vegetable1
                [0, 0, 1],  # meat1
            ]
        )
        vocab = ["apple", "pear", "carrot", "beef"]
        sem = SparseEmbeddingMatrix(embeddings, vocab=vocab)

        # Find similar to apple
        similar = sem.find_similar("apple", k=2)
        assert len(similar) == 2
        # The query item itself might be returned first, then the most similar
        similar_names = [item[0] for item in similar]
        assert "pear" in similar_names  # Most similar (identical embedding)
        # Check for high similarity (should be 1.0 for identical embeddings)
        assert any(score >= 0.99 for _, score in similar)

        # Find similar with external query vector
        query_vector = np.array([1, 0, 0])
        similar_external = sem.find_similar(query_vector, k=2)
        assert len(similar_external) == 2

    def test_dynamic_operations(self):
        """Test adding new embeddings."""
        embeddings = np.array([[1, 0], [0, 1]])
        vocab = ["item1", "item2"]
        sem = SparseEmbeddingMatrix(embeddings, vocab=vocab)

        # Add new embedding
        new_embedding = np.array([1, 1])
        sem.add_embedding("item3", new_embedding)

        assert len(sem) == 3
        assert "item3" in sem
        assert np.allclose(sem["item3"], [1, 1])

        # Test error for duplicate item
        with pytest.raises(ValueError):
            sem.add_embedding("item1", new_embedding)

    def test_save_load_functionality(self):
        """Test saving and loading sparse embedding matrices."""
        embeddings = np.array([[1, 0, 2], [0, 3, 0]])
        vocab = ["word1", "word2"]
        sem = SparseEmbeddingMatrix(embeddings, vocab=vocab)

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "test_embeddings"

            # Save
            sem.save(str(filepath))

            # Verify files exist
            assert (Path(temp_dir) / "test_embeddings_matrix.npz").exists()
            assert (Path(temp_dir) / "test_embeddings_vocab.json").exists()

            # Load
            sem_loaded = SparseEmbeddingMatrix.load(str(filepath))

            # Verify loaded data
            assert len(sem_loaded) == len(sem)
            assert sem_loaded.vocab == sem.vocab
            assert np.allclose(sem_loaded.matrix.to_dense(), sem.matrix.to_dense())

    def test_statistics(self):
        """Test getting embedding matrix statistics."""
        embeddings = np.array([[1, 0, 2, 0], [0, 3, 0, 0], [4, 0, 5, 0]])
        vocab = ["a", "b", "c"]
        sem = SparseEmbeddingMatrix(embeddings, vocab=vocab, sparsity_threshold=0.1)

        stats = sem.get_stats()

        assert stats["shape"] == (3, 4)
        assert stats["vocabulary_size"] == 3
        assert stats["embedding_dimension"] == 4
        assert stats["nnz"] > 0
        assert stats["density"] > 0
        assert stats["memory_usage_mb"] >= 0
        assert stats["compression_ratio"] >= 1.0

    def test_error_handling(self):
        """Test error handling in SparseEmbeddingMatrix."""
        embeddings = np.array([[1, 2], [3, 4]])

        # Vocabulary size mismatch
        with pytest.raises(ValueError):
            SparseEmbeddingMatrix(embeddings, vocab=["too", "few", "many"])


class TestMatrixOptimization:
    """Test cases for matrix optimization functions."""

    def test_optimize_sparse_matrix(self):
        """Test sparse matrix optimization."""
        # Create matrix with various value magnitudes
        dense = np.array([[1.0, 0.1, 0.01], [0.001, 2.0, 0.0001], [0.5, 0.0, 3.0]])
        csr = CSRMatrix.from_dense(dense)

        # Optimize to lower density
        optimized = optimize_sparse_matrix(csr, target_density=0.5)

        assert optimized.density() <= 0.5
        assert optimized.nnz() <= csr.nnz()

        # Test case where current density is already lower
        low_density = optimize_sparse_matrix(csr, target_density=1.0)
        assert low_density.density() == csr.density()

    def test_benchmark_sparse_vs_dense(self):
        """Test benchmarking functionality."""
        shapes = [(50, 50), (100, 100)]
        densities = [0.1, 0.3]

        results = benchmark_sparse_vs_dense(shapes, densities)

        # Check that all combinations were tested
        expected_tests = len(shapes) * len(densities)
        assert len(results) == expected_tests

        for test_name, metrics in results.items():
            # Each test should have timing and memory metrics
            assert "sparse_mult_time" in metrics
            assert "dense_mult_time" in metrics
            assert "speedup_ratio" in metrics
            assert "memory_savings_ratio" in metrics
            assert "density" in metrics

            # Sanity checks
            assert metrics["sparse_mult_time"] > 0
            assert metrics["dense_mult_time"] > 0
            assert metrics["speedup_ratio"] > 0
            assert metrics["memory_savings_ratio"] >= 1.0


class TestPerformanceCharacteristics:
    """Test performance characteristics of sparse matrices."""

    def test_memory_efficiency(self):
        """Test memory efficiency of sparse matrices."""
        # Create a large sparse matrix
        shape = (1000, 1000)
        density = 0.01  # 1% density

        sparse_matrix = CSRMatrix.random_sparse(shape, density=density, random_state=42)
        memory_info = sparse_matrix.memory_usage()

        # Sparse should use significantly less memory than dense
        assert memory_info["compression_ratio"] > 10  # At least 10x compression
        assert memory_info["memory_savings"] > 0

        # Density should be close to target
        assert 0.005 < sparse_matrix.density() < 0.015

    def test_operation_correctness(self):
        """Test that sparse operations produce same results as dense."""
        # Create test matrices
        np.random.seed(42)
        dense1 = np.random.random((50, 50))
        dense1[dense1 < 0.8] = 0  # Make sparse

        dense2 = np.random.random((50, 30))
        dense2[dense2 < 0.7] = 0  # Make sparse

        sparse1 = CSRMatrix.from_dense(dense1)
        sparse2 = CSRMatrix.from_dense(dense2)

        # Test matrix multiplication
        sparse_result = sparse1.dot(sparse2)
        dense_result = dense1.dot(dense2)
        if isinstance(sparse_result, CSRMatrix):
            assert np.allclose(sparse_result.to_dense(), dense_result, atol=1e-10)
        else:
            assert np.allclose(sparse_result, dense_result, atol=1e-10)

        # Test addition
        sparse1_copy = CSRMatrix.from_dense(dense1)
        sparse_sum = sparse1.add(sparse1_copy)
        dense_sum = dense1 + dense1
        assert np.allclose(sparse_sum.to_dense(), dense_sum, atol=1e-10)

        # Test scalar operations
        sparse_scaled = sparse1 * 2.5
        dense_scaled = dense1 * 2.5
        assert np.allclose(sparse_scaled.to_dense(), dense_scaled, atol=1e-10)

    @pytest.mark.slow
    def test_large_scale_operations(self):
        """Test operations on larger matrices."""
        # Create larger test matrices
        shape = (500, 500)
        density = 0.05

        matrix1 = CSRMatrix.random_sparse(shape, density=density, random_state=42)
        matrix2 = CSRMatrix.random_sparse(shape, density=density, random_state=43)

        # Test that operations complete without errors
        result_add = matrix1 + matrix2
        assert result_add.shape() == shape

        test_vector = np.random.random(shape[1])
        result_mult = matrix1.dot(test_vector)
        assert result_mult.shape == (shape[0],)

        # Test statistical operations
        matrix_sum = matrix1.sum()
        assert isinstance(matrix_sum, (int, float))

        matrix_mean = matrix1.mean()
        assert isinstance(matrix_mean, (int, float))


def test_integration_with_existing_code():
    """Test that CSR matrices integrate well with existing numpy/scipy code."""
    # Create CSR matrix
    dense = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 5]])
    csr = CSRMatrix.from_dense(dense)

    # Should work with numpy functions that accept sparse matrices
    # (Most numpy functions will convert to dense automatically)
    from scipy.sparse.linalg import norm as sparse_norm

    # Test with scipy sparse functions
    norm_result = sparse_norm(csr.matrix)
    expected_norm = np.linalg.norm(dense)
    assert abs(norm_result - expected_norm) < 1e-10


if __name__ == "__main__":
    # Run some basic performance tests
    print("CSR Matrix Performance Tests")
    print("=" * 40)

    # Test different scenarios
    shapes_densities = [
        ((100, 100), 0.05),
        ((500, 500), 0.02),
        ((1000, 1000), 0.01),
    ]

    for (rows, cols), density in shapes_densities:
        # Create test matrix
        sparse_matrix = CSRMatrix.random_sparse((rows, cols), density=density, random_state=42)
        memory_info = sparse_matrix.memory_usage()

        print(f"\nMatrix {rows}x{cols}, density {density:.3f}:")
        print(f"  Memory savings: {memory_info['compression_ratio']:.1f}x")
        print(f"  Sparse memory: {memory_info['total_bytes'] / (1024**2):.1f} MB")
        print(f"  Dense equivalent: {memory_info['dense_equivalent_bytes'] / (1024**2):.1f} MB")

        # Test multiplication performance
        import time

        test_vector = np.random.random(cols)

        start_time = time.perf_counter()
        result = sparse_matrix.dot(test_vector)
        sparse_time = time.perf_counter() - start_time

        print(f"  Multiplication time: {sparse_time * 1000:.2f} ms")

    print("\nCSR Matrix tests completed successfully!")
