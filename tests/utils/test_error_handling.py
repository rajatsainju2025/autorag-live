"""Tests for error handling utilities."""

import logging
from unittest.mock import patch

import pytest

from autorag_live.types import (
    AutoRAGError,
    DataError,
    EvaluationError,
    ModelError,
    PipelineError,
    RetrieverError,
    ValidationError,
)
from autorag_live.utils.error_handling import (
    ErrorContext,
    configure_error_logging,
    data_error,
    evaluation_error,
    get_logger,
    handle_errors,
    model_error,
    pipeline_error,
    retrieval_error,
    safe_execute,
    validation_error,
    with_retry,
)


class TestAutoRAGError:
    """Test the base AutoRAGError class."""

    def test_basic_error_creation(self):
        """Test creating a basic AutoRAGError."""
        error = AutoRAGError("Test error")
        # Now __str__ includes suggestion and context
        assert "Test error" in str(error)
        assert "AutoRAGError" in str(error)
        assert error.message == "Test error"
        assert error.error_code == "AutoRAGError"
        assert error.context == {}
        assert error.cause is None
        assert error.suggestion is not None  # Has default suggestion

    def test_error_with_code_and_context(self):
        """Test error with custom code and context."""
        context = {"user_id": 123, "operation": "test"}
        error = AutoRAGError("Test error", error_code="TEST_001", context=context)

        assert error.error_code == "TEST_001"
        assert error.context == context

    def test_error_with_cause(self):
        """Test error with underlying cause."""
        cause = ValueError("Original error")
        error = AutoRAGError("Wrapped error", cause=cause)

        assert error.cause == cause
        assert isinstance(error.cause, ValueError)

    def test_to_dict(self):
        """Test converting error to dictionary."""
        context = {"key": "value"}
        cause = RuntimeError("cause")
        error = AutoRAGError("Test", error_code="TEST", context=context, cause=cause)

        error_dict = error.to_dict()
        assert error_dict["error_type"] == "AutoRAGError"
        assert error_dict["error_code"] == "TEST"
        assert error_dict["message"] == "Test"
        assert error_dict["context"] == context
        assert error_dict["cause"] == "cause"
        assert "timestamp" in error_dict


class TestSpecificErrorTypes:
    """Test specific error type classes."""

    def test_retrieval_error(self):
        """Test RetrieverError class."""
        error = RetrieverError("Retrieval failed")
        assert isinstance(error, AutoRAGError)
        assert error.error_code == "RetrieverError"

    def test_evaluation_error(self):
        """Test EvaluationError class."""
        error = EvaluationError("Evaluation failed")
        assert isinstance(error, AutoRAGError)
        assert error.error_code == "EvaluationError"

    def test_pipeline_error(self):
        """Test PipelineError class."""
        error = PipelineError("Pipeline failed")
        assert isinstance(error, AutoRAGError)
        assert error.error_code == "PipelineError"

    def test_model_error(self):
        """Test ModelError class."""
        error = ModelError("Model failed")
        assert isinstance(error, AutoRAGError)
        assert error.error_code == "ModelError"

    def test_data_error(self):
        """Test DataError class."""
        error = DataError("Data error")
        assert isinstance(error, AutoRAGError)
        assert error.error_code == "DataError"

    def test_validation_error(self):
        """Test ValidationError class."""
        error = ValidationError("Validation failed")
        assert isinstance(error, AutoRAGError)
        assert error.error_code == "ValidationError"


class TestConvenienceFunctions:
    """Test convenience functions for creating errors."""

    def test_retrieval_error_function(self):
        """Test retrieval_error convenience function."""
        error = retrieval_error("Test retrieval error", error_code="RET_001")
        assert isinstance(error, RetrieverError)
        assert error.message == "Test retrieval error"
        assert error.error_code == "RET_001"

    def test_evaluation_error_function(self):
        """Test evaluation_error convenience function."""
        error = evaluation_error("Test evaluation error")
        assert isinstance(error, EvaluationError)

    def test_pipeline_error_function(self):
        """Test pipeline_error convenience function."""
        error = pipeline_error("Test pipeline error")
        assert isinstance(error, PipelineError)

    def test_model_error_function(self):
        """Test model_error convenience function."""
        error = model_error("Test model error")
        assert isinstance(error, ModelError)

    def test_data_error_function(self):
        """Test data_error convenience function."""
        error = data_error("Test data error")
        assert isinstance(error, DataError)

    def test_validation_error_function(self):
        """Test validation_error convenience function."""
        error = validation_error("Test validation error")
        assert isinstance(error, ValidationError)


class TestGetLogger:
    """Test logger creation functionality."""

    def test_get_logger_creates_logger(self):
        """Test that get_logger returns a configured logger."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"
        assert len(logger.handlers) > 0

    def test_get_logger_reuses_handlers(self):
        """Test that get_logger doesn't add duplicate handlers."""
        logger1 = get_logger("test.reuse")
        handler_count1 = len(logger1.handlers)

        logger2 = get_logger("test.reuse")
        handler_count2 = len(logger2.handlers)

        assert handler_count1 == handler_count2


class TestHandleErrors:
    """Test the handle_errors decorator."""

    def test_decorator_success(self):
        """Test decorator allows successful function execution."""

        @handle_errors()
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_decorator_wraps_exceptions(self):
        """Test decorator wraps exceptions in AutoRAGError."""

        @handle_errors()
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(AutoRAGError) as exc_info:
            failing_func()

        assert "Test error" in str(exc_info.value)
        assert exc_info.value.cause.__class__ == ValueError

    def test_decorator_preserves_autorag_errors(self):
        """Test decorator preserves AutoRAGError instances."""

        @handle_errors()
        def failing_func():
            raise RetrieverError("Retrieval failed")

        with pytest.raises(RetrieverError) as exc_info:
            failing_func()

        # Now includes suggestion in string representation
        assert "Retrieval failed" in str(exc_info.value)
        assert exc_info.value.error_code == "RetrieverError"

    def test_decorator_with_return_value(self):
        """Test decorator returns default value instead of raising."""

        @handle_errors(reraise=False, return_value="default")
        def failing_func():
            raise ValueError("Test error")

        result = failing_func()
        assert result == "default"

    def test_decorator_custom_error_type(self):
        """Test decorator with custom error type."""

        @handle_errors(error_type=RetrieverError)
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(RetrieverError) as exc_info:
            failing_func()

        assert isinstance(exc_info.value, RetrieverError)


class TestWithRetry:
    """Test the with_retry decorator."""

    def test_retry_success_on_first_attempt(self):
        """Test that successful calls don't retry."""

        @with_retry(max_attempts=3)
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_retry_eventually_succeeds(self):
        """Test retry until success."""
        call_count = 0

        @with_retry(max_attempts=3, delay=0.01)
        def eventually_successful_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = eventually_successful_func()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausts_attempts(self):
        """Test retry exhausts all attempts and raises error."""

        @with_retry(max_attempts=2, delay=0.01)
        def always_failing_func():
            raise ValueError("Always fails")

        with pytest.raises(AutoRAGError) as exc_info:
            always_failing_func()

        assert "failed after 2 attempts" in str(exc_info.value)

    def test_retry_preserves_autorag_errors(self):
        """Test that AutoRAGError instances are preserved."""

        @with_retry(max_attempts=2)
        def failing_func():
            raise RetrieverError("Retrieval failed")

        with pytest.raises(RetrieverError):
            failing_func()

    def test_retry_custom_exceptions(self):
        """Test retry with custom exception types."""

        @with_retry(max_attempts=2, exceptions=(KeyError,))
        def failing_func():
            raise KeyError("Key not found")

        with pytest.raises(AutoRAGError):
            failing_func()


class TestSafeExecute:
    """Test the safe_execute function."""

    def test_safe_execute_success(self):
        """Test successful execution."""
        result = safe_execute(lambda: "success")
        assert result == "success"

    def test_safe_execute_with_args(self):
        """Test execution with arguments."""

        def add(a, b):
            return a + b

        result = safe_execute(add, 2, 3)
        assert result == 5

    def test_safe_execute_failure_returns_default(self):
        """Test failure returns default value."""
        result = safe_execute(lambda: 1 / 0, default_value="default")
        assert result == "default"

    def test_safe_execute_failure_reraises_autorag_error(self):
        """Test that AutoRAGError instances are re-raised."""

        def failing_func():
            raise RetrieverError("test")

        with pytest.raises(RetrieverError):
            safe_execute(failing_func, default_value="default")

    def test_safe_execute_failure_raises_wrapped_error(self):
        """Test that other exceptions are wrapped."""
        with pytest.raises(AutoRAGError):
            safe_execute(lambda: 1 / 0)


class TestErrorContext:
    """Test the ErrorContext context manager."""

    def test_context_success(self):
        """Test successful context execution."""
        with ErrorContext("test_operation") as ctx:
            assert ctx.operation == "test_operation"
        # Should not raise

    def test_context_autorag_error_reraises(self):
        """Test that AutoRAGError is re-raised."""
        with pytest.raises(RetrieverError):
            with ErrorContext("test_operation"):
                raise RetrieverError("Test error")

    def test_context_wraps_other_errors(self):
        """Test that other errors are wrapped."""
        with pytest.raises(AutoRAGError):
            with ErrorContext("test_operation"):
                raise ValueError("Test error")

    def test_context_no_reraise(self):
        """Test context suppresses errors when reraise=False."""
        with ErrorContext("test_operation", reraise=False):
            raise ValueError("Test error")
        # Should not raise

    def test_context_custom_error_type(self):
        """Test context with custom error type."""
        with pytest.raises(RetrieverError):
            with ErrorContext("test_operation", error_type=RetrieverError):
                raise ValueError("Test error")


class TestConfigureErrorLogging:
    """Test error logging configuration."""

    def test_configure_basic_logging(self):
        """Test basic logging configuration."""
        with patch("logging.basicConfig"):
            configure_error_logging(log_level="DEBUG")

        # Test that it doesn't raise errors
        assert True

    def test_configure_with_file_logging(self, tmp_path):
        """Test logging configuration with file output."""
        log_file = tmp_path / "test.log"

        configure_error_logging(log_file=str(log_file))

        # Check that file was created
        assert log_file.exists()

    def test_configure_log_size_parsing(self, tmp_path):
        """Test log size parsing."""
        log_file = tmp_path / "test.log"

        # Test MB parsing
        configure_error_logging(log_file=str(log_file), max_log_size="5MB")
        assert log_file.exists()

        # Test GB parsing
        configure_error_logging(log_file=str(log_file), max_log_size="1GB")
        assert log_file.exists()
