"""
Standardized error handling and logging for AutoRAG-Live.

This module provides consistent error patterns, logging utilities, and recovery
strategies across the entire codebase.
"""
import functools
import logging
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Type, TypeVar

from autorag_live.types import (
    AutoRAGError,
    DataError,
    EvaluationError,
    ModelError,
    PipelineError,
    RetrieverError,
    ValidationError,
)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])
# Type variable for function return types
T = TypeVar("T")

# Thread-safe logger cache to avoid repeated logger creation
_LOGGER_CACHE: Dict[str, logging.Logger] = {}
_LOGGER_CACHE_LOCK = threading.Lock()


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance with thread-safe caching.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger
    """
    # Fast path: check if logger already exists without lock
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    # Slow path: create logger with thread safety
    with _LOGGER_CACHE_LOCK:
        # Double-check after acquiring lock
        if name in _LOGGER_CACHE:
            return _LOGGER_CACHE[name]

        logger = logging.getLogger(name)

        # Avoid adding handlers multiple times
        if not logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)

            # Create formatter
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)

            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)

        _LOGGER_CACHE[name] = logger
        return logger


def handle_errors(
    error_type: Type[AutoRAGError] = AutoRAGError,
    reraise: bool = True,
    log_errors: bool = True,
    return_value: Optional[Any] = None,
) -> Callable[[F], F]:
    """
    Decorator for standardized error handling.

    Args:
        error_type: Exception type to catch and wrap
        reraise: Whether to reraise the wrapped exception
        log_errors: Whether to log errors
        return_value: Value to return on error (if not reraising)

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        logger = get_logger(func.__module__)  # Cache logger at decoration time
        func_name = func.__name__  # Cache function name at decoration time

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except AutoRAGError:
                # Re-raise AutoRAG errors as-is
                raise

            except Exception as e:
                # Wrap other exceptions with minimal overhead
                context = {
                    "function": func_name,
                }
                # Only include args/kwargs if they're small
                if args:
                    args_str = str(args)[:100]
                    if len(args_str) < 100:
                        context["args"] = args_str
                if kwargs:
                    kwargs_str = str(kwargs)[:100]
                    if len(kwargs_str) < 100:
                        context["kwargs"] = kwargs_str

                wrapped_error = error_type(
                    message=f"Error in {func_name}: {str(e)}",
                    context=context,
                    cause=e,
                )

                if log_errors:
                    error_dict = wrapped_error.to_dict()
                    # Remove 'message' key to avoid LogRecord conflict
                    error_dict.pop("message", None)
                    logger.error(f"Error in {func_name}: {str(e)}", extra=error_dict, exc_info=True)

                if reraise:
                    raise wrapped_error
                else:
                    return return_value

        return wrapper  # type: ignore

    return decorator


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    log_attempts: bool = True,
) -> Callable[[F], F]:
    """
    Decorator for automatic retry with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        backoff_factor: Multiplier for delay after each failure
        exceptions: Exceptions to retry on
        log_attempts: Whether to log retry attempts

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        logger = get_logger(func.__module__)  # Cache logger
        func_name = func.__name__  # Cache function name

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        # Last attempt failed
                        break

                    if log_attempts:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func_name}: {str(e)}"
                        )

                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            # All attempts failed
            if isinstance(last_exception, AutoRAGError):
                raise last_exception
            else:
                raise AutoRAGError(
                    message=f"Function {func_name} failed after {max_attempts} attempts",
                    context={"last_error": str(last_exception)},
                    cause=last_exception,
                )

        return wrapper  # type: ignore

    return decorator


def safe_execute(
    func: Callable[..., T],
    *args,
    error_type: Type[AutoRAGError] = AutoRAGError,
    default_value: Optional[T] = None,
    log_errors: bool = True,
    **kwargs,
) -> T:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Positional arguments for function
        error_type: Exception type to wrap errors in
        default_value: Value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for function

    Returns:
        Function result or default_value on error
    """
    logger = get_logger(func.__module__)

    try:
        return func(*args, **kwargs)

    except AutoRAGError:
        # Re-raise AutoRAG errors
        raise

    except Exception as e:
        if log_errors:
            logger.error(f"Error executing {func.__name__}: {str(e)}", exc_info=True)

        if default_value is not None:
            return default_value

        raise error_type(message=f"Error executing {func.__name__}: {str(e)}", cause=e)


class ErrorContext:
    """Context manager for error handling and logging."""

    def __init__(
        self,
        operation: str,
        error_type: Type[AutoRAGError] = AutoRAGError,
        reraise: bool = True,
        log_errors: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.operation = operation
        self.error_type = error_type
        self.reraise = reraise
        self.log_errors = log_errors
        self.context = context or {}
        self.logger = get_logger(__name__)

    def __enter__(self):
        self.logger.debug(f"Starting operation: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.debug(f"Completed operation: {self.operation}")
            return False

        if issubclass(exc_type, AutoRAGError):
            # AutoRAG error - just log and reraise
            if self.log_errors:
                self.logger.error(f"AutoRAG error in {self.operation}: {str(exc_val)}")
            return not self.reraise

        # Wrap other exceptions
        wrapped_error = self.error_type(
            message=f"Error in {self.operation}: {str(exc_val)}",
            context=self.context,
            cause=exc_val,
        )

        if self.log_errors:
            error_dict = wrapped_error.to_dict()
            error_dict.pop("message", None)  # Avoid LogRecord conflict
            self.logger.error(
                f"Error in {self.operation}: {str(exc_val)}", extra=error_dict, exc_info=True
            )

        if self.reraise:
            raise wrapped_error

        return True  # Suppress the original exception


def configure_error_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_log_size: str = "10MB",
    backup_count: int = 5,
) -> None:
    """
    Configure error logging for the application.

    Args:
        log_level: Logging level
        log_file: Log file path (optional)
        max_log_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """
    # Convert log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        from logging.handlers import RotatingFileHandler

        # Parse max size
        if max_log_size.endswith("MB"):
            max_bytes = int(float(max_log_size[:-2]) * 1024 * 1024)
        elif max_log_size.endswith("GB"):
            max_bytes = int(float(max_log_size[:-2]) * 1024 * 1024 * 1024)
        else:
            max_bytes = int(max_log_size)

        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


# Optimized error factory using dictionary dispatch
_ERROR_FACTORIES = {
    "retriever": RetrieverError,
    "evaluation": EvaluationError,
    "pipeline": PipelineError,
    "model": ModelError,
    "data": DataError,
    "validation": ValidationError,
}


def create_error(error_type: str, message: str, **kwargs) -> AutoRAGError:
    """
    Create an error of the specified type with standard formatting.

    Args:
        error_type: Type of error (retriever, evaluation, pipeline, model, data, validation)
        message: Error message
        **kwargs: Additional error context

    Returns:
        Appropriate AutoRAGError subclass

    Raises:
        ValueError: If error_type is unknown
    """
    error_class = _ERROR_FACTORIES.get(error_type)
    if error_class is None:
        raise ValueError(
            f"Unknown error type: {error_type}. Available types: {list(_ERROR_FACTORIES.keys())}"
        )
    return error_class(message, **kwargs)


# Convenience functions for backward compatibility
def retrieval_error(message: str, **kwargs) -> RetrieverError:
    """Create a RetrieverError with standard formatting."""
    return RetrieverError(message, **kwargs)


def evaluation_error(message: str, **kwargs) -> EvaluationError:
    """Create an EvaluationError with standard formatting."""
    return EvaluationError(message, **kwargs)


def pipeline_error(message: str, **kwargs) -> PipelineError:
    """Create a PipelineError with standard formatting."""
    return PipelineError(message, **kwargs)


def model_error(message: str, **kwargs) -> ModelError:
    """Create a ModelError with standard formatting."""
    return ModelError(message, **kwargs)


def data_error(message: str, **kwargs) -> DataError:
    """Create a DataError with standard formatting."""
    return DataError(message, **kwargs)


def validation_error(message: str, **kwargs) -> ValidationError:
    """Create a ValidationError with standard formatting."""
    return ValidationError(message, **kwargs)
