"""Consolidated error patterns and decorators."""

from functools import wraps
from typing import Any, Callable

from autorag_live.utils import get_logger

logger = get_logger(__name__)


def safe_operation(
    operation_name: str,
    default_return: Any = None,
    log_error: bool = True,
) -> Callable:
    """
    Decorator for safe operation with error handling.

    Args:
        operation_name: Name of operation for logging
        default_return: Default value to return on error
        log_error: Whether to log errors

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {operation_name}: {str(e)}")
                return default_return

        return wrapper

    return decorator


def with_fallback(fallback_fn: Callable) -> Callable:
    """
    Decorator to provide fallback function on error.

    Args:
        fallback_fn: Function to call if main function fails

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed: {str(e)}. Using fallback.")
                return fallback_fn(*args, **kwargs)

        return wrapper

    return decorator


class ErrorContext:
    """Context manager for error handling."""

    def __init__(self, operation_name: str, default_return: Any = None):
        """Initialize error context."""
        self.operation_name = operation_name
        self.default_return = default_return
        self.error = None

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context with error handling."""
        if exc_type is not None:
            self.error = exc_val
            logger.error(f"Error in {self.operation_name}: {str(exc_val)}")
            return True  # Suppress exception
        return False
