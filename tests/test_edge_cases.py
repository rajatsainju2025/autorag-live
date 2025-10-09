"""Edge case and error handling tests."""
import os
import tempfile
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from autorag_live.evals.small import exact_match, run_small_suite, simple_qa_answer, token_f1
from autorag_live.retrievers import bm25, dense, hybrid
from autorag_live.utils.validation import ConfigurationError, validate_config


class TestRetrieverEdgeCases:
    """Test retrievers with edge cases and error conditions."""

    def test_bm25_empty_query(self):
        """Test BM25 with empty query."""
        corpus = ["test document"]
        with pytest.raises(ValueError, match="Query cannot be empty"):
            bm25.bm25_retrieve("", corpus, 1)

    def test_bm25_whitespace_query(self):
        """Test BM25 with whitespace-only query."""
        corpus = ["test document"]
        with pytest.raises(ValueError, match="Query cannot be empty"):
            bm25.bm25_retrieve("   ", corpus, 1)

    def test_bm25_invalid_k(self):
        """Test BM25 with invalid k values."""
        corpus = ["test document"]
        with pytest.raises(ValueError, match="k must be positive"):
            bm25.bm25_retrieve("test", corpus, 0)

        with pytest.raises(ValueError, match="k must be positive"):
            bm25.bm25_retrieve("test", corpus, -1)

    def test_bm25_empty_corpus(self):
        """Test BM25 with empty corpus."""
        result = bm25.bm25_retrieve("test query", [], 5)
        assert result == []

    def test_dense_empty_query(self):
        """Test dense retrieval with empty query."""
        corpus = ["test document"]
        with pytest.raises(ValueError, match="Query cannot be empty"):
            dense.dense_retrieve("", corpus, 1)

    def test_dense_invalid_k(self):
        """Test dense retrieval with invalid k values."""
        corpus = ["test document"]
        with pytest.raises(ValueError, match="k must be positive"):
            dense.dense_retrieve("test", corpus, 0)

    def test_dense_empty_corpus(self):
        """Test dense retrieval with empty corpus."""
        result = dense.dense_retrieve("test query", [], 5)
        assert result == []

    def test_hybrid_empty_query(self):
        """Test hybrid retrieval with empty query."""
        corpus = ["test document"]
        with pytest.raises(ValueError, match="Query cannot be empty"):
            hybrid.hybrid_retrieve("", corpus, 1)

    def test_hybrid_invalid_k(self):
        """Test hybrid retrieval with invalid k values."""
        corpus = ["test document"]
        with pytest.raises(ValueError, match="k must be positive"):
            hybrid.hybrid_retrieve("test", corpus, 0)

    def test_hybrid_invalid_weights(self):
        """Test hybrid retrieval with invalid weights."""
        corpus = ["test document"]
        with pytest.raises(ValueError, match="bm25_weight must be between 0.0 and 1.0"):
            hybrid.hybrid_retrieve("test", corpus, 1, bm25_weight=-0.1)

        with pytest.raises(ValueError, match="bm25_weight must be between 0.0 and 1.0"):
            hybrid.hybrid_retrieve("test", corpus, 1, bm25_weight=1.5)

    def test_hybrid_empty_corpus(self):
        """Test hybrid retrieval with empty corpus."""
        result = hybrid.hybrid_retrieve("test query", [], 5)
        assert result == []


class TestEvaluationEdgeCases:
    """Test evaluation functions with edge cases."""

    def test_exact_match_invalid_types(self):
        """Test exact_match with invalid input types."""
        with pytest.raises(TypeError, match="Both pred and gold must be strings"):
            exact_match(123, "gold")  # type: ignore

        with pytest.raises(TypeError, match="Both pred and gold must be strings"):
            exact_match("pred", None)  # type: ignore

    def test_token_f1_invalid_types(self):
        """Test token_f1 with invalid input types."""
        with pytest.raises(TypeError, match="Both pred and gold must be strings"):
            token_f1([], "gold")  # type: ignore

        with pytest.raises(TypeError, match="Both pred and gold must be strings"):
            token_f1("pred", 42)  # type: ignore

    def test_simple_qa_answer_invalid_types(self):
        """Test simple_qa_answer with invalid input types."""
        with pytest.raises(TypeError, match="Query must be a string"):
            simple_qa_answer(123, ["doc"])  # type: ignore

        with pytest.raises(TypeError, match="Docs must be a list of strings"):
            simple_qa_answer("query", "not a list")  # type: ignore

        with pytest.raises(TypeError, match="Docs must be a list of strings"):
            simple_qa_answer("query", [123, 456])  # type: ignore

    def test_run_small_suite_invalid_inputs(self):
        """Test run_small_suite with invalid inputs."""
        with pytest.raises(ValueError, match="runs_dir must be a non-empty string"):
            run_small_suite("")

        with pytest.raises(ValueError, match="judge_type must be a non-empty string"):
            run_small_suite("runs", "")

    def test_evaluation_with_empty_lists(self):
        """Test evaluation functions with empty inputs."""
        # These should not crash
        assert exact_match("", "") == 1.0
        assert token_f1("", "") == 1.0
        assert simple_qa_answer("test", []) == "unknown"


class TestConfigurationEdgeCases:
    """Test configuration validation with edge cases."""

    def test_validate_config_invalid_dataclass(self):
        """Test validation with non-dataclass."""
        config = OmegaConf.create({"test": "value"})
        with pytest.raises(ConfigurationError, match="is not a dataclass"):
            validate_config(config, str)  # str is not a dataclass

    def test_validate_config_missing_required(self):
        """Test validation with missing required fields."""
        from autorag_live.evals.small import QAItem

        config = OmegaConf.create(
            {
                "id": "test",
                "question": "What?",
                # Missing required fields: context_docs, answer
            }
        )

        with pytest.raises(ConfigurationError, match="Missing required configuration fields"):
            validate_config(config, QAItem)


class TestFileOperationEdgeCases:
    """Test file operations with edge cases."""

    def test_run_small_suite_invalid_directory(self):
        """Test run_small_suite with invalid directory paths."""
        # This should handle permission errors gracefully
        with patch("os.makedirs", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                run_small_suite("/root/forbidden", "deterministic")

    def test_run_small_suite_file_as_directory(self):
        """Test run_small_suite when runs_dir is a file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name

        try:
            # This should handle the case where runs_dir exists but is a file
            with pytest.raises((OSError, PermissionError)):
                run_small_suite(temp_file, "deterministic")
        finally:
            os.unlink(temp_file)


class TestErrorPropagation:
    """Test that errors are properly propagated and handled."""

    def test_retriever_dependency_errors(self):
        """Test that missing dependencies raise appropriate errors."""
        corpus = ["test"]

        # Mock BM25 as unavailable
        with patch("autorag_live.retrievers.bm25.BM25_AVAILABLE", False):
            with pytest.raises(ImportError, match="rank_bm25 is required"):
                bm25.bm25_retrieve("test", corpus, 1)

    def test_dense_fallback_behavior(self):
        """Test dense retrieval fallback when transformers unavailable."""
        corpus = ["test document"]

        # Mock as if SentenceTransformer is not available
        with patch("autorag_live.retrievers.dense.SentenceTransformer", None):
            # Should still work with fallback implementation
            result = dense.dense_retrieve("test query", corpus, 1)
            assert isinstance(result, list)
            assert len(result) <= 1
