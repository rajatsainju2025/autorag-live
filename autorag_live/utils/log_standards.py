"""Logging standards and guidelines for AutoRAG-Live.

This module defines standard logging levels and practices to maintain consistency
across the codebase.
"""

# Standard log levels and when to use them
LOGGING_STANDARDS = {
    "DEBUG": {
        "usage": "Detailed diagnostic information for debugging",
        "examples": [
            "Cache hit/miss details",
            "Intermediate computation results",
            "Performance profiling data",
        ],
    },
    "INFO": {
        "usage": "Normal operational messages and progress updates",
        "examples": [
            "Starting/completing major operations",
            "Configuration loaded successfully",
            "Retrieved N documents for query",
        ],
    },
    "WARNING": {
        "usage": "Potentially problematic situations that don't prevent operation",
        "examples": [
            "Using fallback method",
            "Deprecated feature usage",
            "Suboptimal configuration detected",
        ],
    },
    "ERROR": {
        "usage": "Error events that may still allow continued operation",
        "examples": [
            "Failed to load optional module",
            "API call failed but fallback succeeded",
            "Invalid input handled gracefully",
        ],
    },
    "CRITICAL": {
        "usage": "Severe errors that prevent core functionality",
        "examples": [
            "Cannot initialize required components",
            "Data corruption detected",
            "Out of memory errors",
        ],
    },
}


def get_standard_log_level(operation_type: str) -> str:
    """Get recommended log level for operation type.

    Args:
        operation_type: Type of operation (e.g., 'retrieval', 'config', 'optimization')

    Returns:
        Recommended log level string
    """
    standards = {
        "retrieval": "INFO",  # Document retrieval operations
        "config": "INFO",  # Configuration operations
        "optimization": "INFO",  # Optimization runs
        "evaluation": "INFO",  # Evaluation metrics
        "cache": "DEBUG",  # Cache operations
        "validation": "WARNING",  # Input validation issues
        "fallback": "WARNING",  # Using fallback methods
        "error": "ERROR",  # Error handling
        "startup": "INFO",  # Application startup
        "shutdown": "INFO",  # Application shutdown
    }
    return standards.get(operation_type, "INFO")


# Standard log message formats
LOG_FORMATS = {
    "operation_start": "{operation} started for query: '{query}'",
    "operation_complete": "{operation} completed in {duration:.3f}s",
    "retrieval_result": "Retrieved {count} documents in {duration:.3f}s",
    "cache_hit": "Cache hit for key: {key}",
    "cache_miss": "Cache miss for key: {key}",
    "fallback": "Using fallback {fallback_method} due to: {reason}",
    "config_loaded": "Configuration loaded from {path}",
    "validation_error": "Validation failed for {field}: {error}",
    "api_error": "API call failed: {error}. Retrying ({attempt}/{max_attempts})",
}


def format_log_message(template_key: str, **kwargs) -> str:
    """Format a log message using standard template.

    Args:
        template_key: Key from LOG_FORMATS
        **kwargs: Values to format into template

    Returns:
        Formatted log message

    Example:
        >>> msg = format_log_message("operation_complete",
        ...                          operation="retrieval", duration=0.123)
        >>> print(msg)
        retrieval completed in 0.123s
    """
    template = LOG_FORMATS.get(template_key, "{}")
    return template.format(**kwargs)


__all__ = [
    "LOGGING_STANDARDS",
    "get_standard_log_level",
    "LOG_FORMATS",
    "format_log_message",
]
