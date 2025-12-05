"""Circuit breaker pattern for resilient external API calls.

This module implements the circuit breaker pattern to prevent cascading failures
when calling external services. The circuit breaker monitors failures and can
automatically open to prevent further calls when a failure threshold is reached.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests fail fast
- HALF_OPEN: Testing if service recovered, limited requests pass

Example:
    >>> breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    >>> @breaker.protected
    ... def call_external_api():
    ...     return requests.get("https://api.example.com")
"""

import logging
import threading
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

from ..types import AutoRAGError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(AutoRAGError):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """Circuit breaker for external API calls.

    Monitors failures and opens circuit when threshold is exceeded.
    Automatically attempts recovery after timeout period.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exceptions: Tuple of exception types to catch
        success_threshold: Successes needed in half-open to close circuit

    Example:
        >>> breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        >>> @breaker.protected
        ... def risky_call():
        ...     return external_api.call()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exceptions: tuple = (Exception,),
        success_threshold: int = 2,
    ):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        self.success_threshold = success_threshold

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

        # Statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN and self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("Circuit breaker entering HALF_OPEN state (testing recovery)")

            return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return False
        return time.time() - self._last_failure_time >= self.recovery_timeout

    def _record_success(self) -> None:
        """Record successful call."""
        with self._lock:
            self._total_calls += 1
            self._total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(
                        f"Circuit breaker CLOSED (recovered after {self.success_threshold} successes)"
                    )

    def _record_failure(self) -> None:
        """Record failed call."""
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Immediately reopen on failure during recovery test
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker REOPENED (recovery test failed)")

            elif (
                self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.error(
                    f"Circuit breaker OPENED (failure threshold {self.failure_threshold} exceeded)"
                )

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call function through circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception if call fails
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitBreakerError(
                "Circuit breaker is OPEN - rejecting request",
                context={
                    "failure_count": self._failure_count,
                    "last_failure_time": self._last_failure_time,
                    "recovery_timeout": self.recovery_timeout,
                },
            )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except self.expected_exceptions as e:
            self._record_failure()
            logger.warning(f"Circuit breaker call failed: {e}")
            raise

    def protected(self, func: F) -> F:
        """Decorator to protect function with circuit breaker.

        Args:
            func: Function to protect

        Returns:
            Wrapped function

        Example:
            >>> @breaker.protected
            ... def api_call():
            ...     return requests.get("https://api.example.com")
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(func, *args, **kwargs)

        return wrapper  # type: ignore

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            logger.info("Circuit breaker manually RESET to CLOSED state")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with statistics

        Example:
            >>> stats = breaker.get_stats()
            >>> print(f"State: {stats['state']}, Failures: {stats['failure_count']}")
        """
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "failure_threshold": self.failure_threshold,
                "success_threshold": self.success_threshold,
                "recovery_timeout": self.recovery_timeout,
                "last_failure_time": self._last_failure_time,
            }


__all__ = ["CircuitBreaker", "CircuitBreakerError", "CircuitState"]
