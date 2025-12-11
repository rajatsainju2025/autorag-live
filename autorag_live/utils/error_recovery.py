"""
Error Recovery System for Agentic RAG Pipeline.

Provides intelligent error recovery strategies, fallback mechanisms,
and graceful degradation for production reliability.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(str, Enum):
    """Types of recovery actions."""

    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    DEGRADE = "degrade"
    ESCALATE = "escalate"
    ABORT = "abort"


@dataclass
class ErrorEvent:
    """Represents an error occurrence."""

    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class RecoveryResult(Generic[T]):
    """Result of a recovery attempt."""

    success: bool
    result: Optional[T] = None
    action_taken: RecoveryAction = RecoveryAction.RETRY
    attempts: int = 1
    total_time_ms: float = 0.0
    error: Optional[str] = None
    fallback_used: bool = False


class RecoveryStrategy(ABC):
    """Base class for recovery strategies."""

    @abstractmethod
    def can_recover(self, error: Exception, context: dict[str, Any]) -> bool:
        """Check if this strategy can handle the error."""
        pass

    @abstractmethod
    def recover(
        self,
        func: Callable[..., T],
        error: Exception,
        args: tuple,
        kwargs: dict[str, Any],
        context: dict[str, Any],
    ) -> RecoveryResult[T]:
        """Attempt recovery."""
        pass


class RetryStrategy(RecoveryStrategy):
    """Retry with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_multiplier: float = 2.0,
        retryable_errors: Optional[tuple[type[Exception], ...]] = None,
    ):
        """Initialize retry strategy."""
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.retryable_errors = retryable_errors or (
            ConnectionError,
            TimeoutError,
            IOError,
        )

    def can_recover(self, error: Exception, context: dict[str, Any]) -> bool:
        """Check if error is retryable."""
        return isinstance(error, self.retryable_errors)

    def recover(
        self,
        func: Callable[..., T],
        error: Exception,
        args: tuple,
        kwargs: dict[str, Any],
        context: dict[str, Any],
    ) -> RecoveryResult[T]:
        """Retry the function with backoff."""
        start_time = time.time()
        delay = self.initial_delay
        last_error = error

        for attempt in range(self.max_retries):
            time.sleep(delay)
            try:
                result = func(*args, **kwargs)
                return RecoveryResult(
                    success=True,
                    result=result,
                    action_taken=RecoveryAction.RETRY,
                    attempts=attempt + 1,
                    total_time_ms=(time.time() - start_time) * 1000,
                )
            except Exception as e:
                last_error = e
                delay = min(delay * self.backoff_multiplier, self.max_delay)

        return RecoveryResult(
            success=False,
            action_taken=RecoveryAction.RETRY,
            attempts=self.max_retries,
            total_time_ms=(time.time() - start_time) * 1000,
            error=str(last_error),
        )


class FallbackStrategy(RecoveryStrategy):
    """Use fallback function on error."""

    def __init__(
        self,
        fallback_fn: Callable[..., Any],
        error_types: Optional[tuple[type[Exception], ...]] = None,
    ):
        """Initialize fallback strategy."""
        self.fallback_fn = fallback_fn
        self.error_types = error_types or (Exception,)

    def can_recover(self, error: Exception, context: dict[str, Any]) -> bool:
        """Check if error can be handled by fallback."""
        return isinstance(error, self.error_types)

    def recover(
        self,
        func: Callable[..., T],
        error: Exception,
        args: tuple,
        kwargs: dict[str, Any],
        context: dict[str, Any],
    ) -> RecoveryResult[T]:
        """Execute fallback function."""
        start_time = time.time()
        try:
            result = self.fallback_fn(*args, **kwargs)
            return RecoveryResult(
                success=True,
                result=result,
                action_taken=RecoveryAction.FALLBACK,
                attempts=1,
                total_time_ms=(time.time() - start_time) * 1000,
                fallback_used=True,
            )
        except Exception as e:
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.FALLBACK,
                attempts=1,
                total_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
                fallback_used=True,
            )


class GracefulDegradationStrategy(RecoveryStrategy):
    """Return degraded response instead of failing."""

    def __init__(
        self,
        degraded_response_fn: Callable[[Exception, dict[str, Any]], Any],
        error_types: Optional[tuple[type[Exception], ...]] = None,
    ):
        """Initialize degradation strategy."""
        self.degraded_response_fn = degraded_response_fn
        self.error_types = error_types or (Exception,)

    def can_recover(self, error: Exception, context: dict[str, Any]) -> bool:
        """Check if error can be degraded."""
        return isinstance(error, self.error_types)

    def recover(
        self,
        func: Callable[..., T],
        error: Exception,
        args: tuple,
        kwargs: dict[str, Any],
        context: dict[str, Any],
    ) -> RecoveryResult[T]:
        """Return degraded response."""
        start_time = time.time()
        try:
            result = self.degraded_response_fn(error, context)
            return RecoveryResult(
                success=True,
                result=result,
                action_taken=RecoveryAction.DEGRADE,
                attempts=1,
                total_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.DEGRADE,
                attempts=1,
                total_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )


class ErrorRecoveryManager:
    """Manages error recovery with multiple strategies."""

    def __init__(self):
        """Initialize recovery manager."""
        self._strategies: list[RecoveryStrategy] = []
        self._error_history: list[ErrorEvent] = []
        self._max_history = 1000
        self._recovery_stats: dict[str, int] = {
            "total_errors": 0,
            "recovered": 0,
            "failed": 0,
        }

    def add_strategy(self, strategy: RecoveryStrategy, priority: int = 0) -> None:
        """Add a recovery strategy."""
        self._strategies.insert(priority, strategy)

    def execute_with_recovery(
        self,
        func: Callable[..., T],
        *args,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> RecoveryResult[T]:
        """Execute function with automatic recovery."""
        context = context or {}
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            return RecoveryResult(
                success=True,
                result=result,
                action_taken=RecoveryAction.RETRY,
                attempts=1,
                total_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            self._record_error(e, context)

            for strategy in self._strategies:
                if strategy.can_recover(e, context):
                    recovery_result = strategy.recover(func, e, args, kwargs, context)

                    if recovery_result.success:
                        self._recovery_stats["recovered"] += 1
                    else:
                        self._recovery_stats["failed"] += 1

                    return recovery_result

            self._recovery_stats["failed"] += 1
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.ABORT,
                attempts=1,
                total_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def _record_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Record an error event."""
        import traceback

        event = ErrorEvent(
            error_type=type(error).__name__,
            message=str(error),
            severity=self._classify_severity(error),
            context=context,
            stack_trace=traceback.format_exc(),
        )

        self._error_history.append(event)
        if len(self._error_history) > self._max_history:
            self._error_history = self._error_history[-self._max_history :]

        self._recovery_stats["total_errors"] += 1

    def _classify_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity."""
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (MemoryError, RecursionError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (TimeoutError, ConnectionError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def get_stats(self) -> dict[str, Any]:
        """Get recovery statistics."""
        total = self._recovery_stats["total_errors"]
        recovered = self._recovery_stats["recovered"]

        return {
            **self._recovery_stats,
            "recovery_rate": recovered / max(total, 1),
            "recent_errors": len(self._error_history),
        }

    def get_recent_errors(self, limit: int = 10) -> list[ErrorEvent]:
        """Get recent error events."""
        return self._error_history[-limit:]


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker pattern for failure isolation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitBreakerState:
        """Get current state, checking for automatic transitions."""
        if self._state == CircuitBreakerState.OPEN:
            if self._last_failure_time:
                time_since_failure = (datetime.now() - self._last_failure_time).total_seconds()
                if time_since_failure >= self.recovery_timeout:
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_calls = 0
        return self._state

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function through circuit breaker."""
        current_state = self.state

        if current_state == CircuitBreakerState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open. Retry after {self.recovery_timeout}s"
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self) -> None:
        """Handle successful call."""
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
        else:
            self._success_count += 1

    def _on_failure(self) -> None:
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitBreakerState.OPEN
        elif self._state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.OPEN

    def reset(self) -> None:
        """Reset circuit breaker."""
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class BulkheadIsolator:
    """Bulkhead pattern for resource isolation."""

    def __init__(self, max_concurrent: int = 10, max_queue: int = 100):
        """Initialize bulkhead."""
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_count = 0

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with bulkhead isolation."""
        if self._queue_count >= self.max_queue:
            raise BulkheadFullError("Bulkhead queue is full")

        self._queue_count += 1
        try:
            async with self._semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        finally:
            self._queue_count -= 1


class BulkheadFullError(Exception):
    """Exception raised when bulkhead is full."""

    pass


def with_recovery(
    strategies: Optional[list[RecoveryStrategy]] = None,
    context: Optional[dict[str, Any]] = None,
) -> Callable:
    """Decorator for automatic error recovery."""
    import functools

    manager = ErrorRecoveryManager()
    if strategies:
        for i, strategy in enumerate(strategies):
            manager.add_strategy(strategy, i)
    else:
        manager.add_strategy(RetryStrategy())

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            result = manager.execute_with_recovery(func, *args, context=context, **kwargs)
            if result.success:
                return result.result  # type: ignore
            else:
                raise RuntimeError(f"Recovery failed: {result.error}")

        return wrapper

    return decorator


__all__ = [
    "ErrorSeverity",
    "RecoveryAction",
    "ErrorEvent",
    "RecoveryResult",
    "RecoveryStrategy",
    "RetryStrategy",
    "FallbackStrategy",
    "GracefulDegradationStrategy",
    "ErrorRecoveryManager",
    "CircuitBreakerState",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "BulkheadIsolator",
    "BulkheadFullError",
    "with_recovery",
]
