# Error Handling Guide

AutoRAG-Live provides comprehensive error handling with consistent patterns across the codebase.

## Error Hierarchy

All errors inherit from `AutoRAGError`:

```python
from autorag_live.types import (
    AutoRAGError,           # Base error class
    RetrieverError,         # Retrieval-related errors
    EvaluationError,        # Evaluation-related errors
    PipelineError,          # Pipeline optimization errors
    ModelError,             # Model loading/inference errors
    DataError,              # Data processing errors
    ValidationError,        # Configuration validation errors
)
```

## Using Error Handling Decorators

### @handle_errors Decorator

Automatically wraps exceptions and logs them:

```python
from autorag_live.utils.error_handling import handle_errors

@handle_errors(RetrieverError, reraise=True, log_errors=True)
def my_retrieval_function():
    # Any exception here will be wrapped in RetrieverError
    pass
```

Parameters:
- `error_type`: Exception type to wrap other exceptions in
- `reraise`: If True, raises the wrapped exception (default: True)
- `log_errors`: If True, logs the error (default: True)
- `return_value`: Value to return if error occurs and reraise=False

### @with_retry Decorator

Implements automatic retry with exponential backoff:

```python
from autorag_live.utils.error_handling import with_retry

@with_retry(max_attempts=3, delay=1.0, backoff_factor=2.0)
def flaky_operation():
    # Will retry up to 3 times with exponential backoff
    pass
```

Parameters:
- `max_attempts`: Number of retry attempts
- `delay`: Initial delay between attempts (seconds)
- `backoff_factor`: Multiplier for delay after each failure
- `exceptions`: Tuple of exceptions to retry on
- `log_attempts`: Whether to log retry attempts

### Combining Decorators

```python
@handle_errors(RetrieverError)
@with_retry(max_attempts=3, delay=1.0)
def robust_retrieval():
    # Retry logic + error handling
    pass
```

## Error Context

Errors include context information for debugging:

```python
from autorag_live.types import RetrieverError

error = RetrieverError(
    message="Failed to load model",
    context={
        "model_name": "all-MiniLM-L6-v2",
        "attempt": 3,
    },
    cause=original_exception,
)

# Access error details
error_dict = error.to_dict()
print(error_dict["context"])  # {'model_name': '...', 'attempt': 3}
print(error_dict["cause"])    # Original exception details
```

## Best Practices

1. **Use specific error types**: Choose the appropriate error type for your context
2. **Provide context**: Include relevant information in the context dict
3. **Chain exceptions**: Pass the original exception as `cause` for traceability
4. **Log strategically**: Enable logging for debugging but disable for performance-critical code
5. **Handle retryable errors**: Use @with_retry for flaky operations

## Convenience Functions

Quick error creation:

```python
from autorag_live.utils.error_handling import (
    retrieval_error,
    evaluation_error,
    pipeline_error,
    model_error,
    data_error,
    validation_error,
)

# These automatically set the error type
error = retrieval_error("Failed to retrieve documents")
raise retrieval_error("Query is empty")
```

## Configuring Error Logging

```python
from autorag_live.utils.error_handling import configure_error_logging

configure_error_logging(
    log_level="DEBUG",
    log_file="/var/log/autorag.log",
    max_log_size="10MB",
    backup_count=5,
)
```
