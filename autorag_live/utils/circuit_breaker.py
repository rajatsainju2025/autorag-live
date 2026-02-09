"""Circuit breaker v2 — adaptive resilience for agentic RAG pipelines.

Improvements over v1
---------------------
* **Sliding-window failure tracking** — only failures within the last
  ``window_duration`` seconds count toward the threshold, so a handful of
  intermittent errors hours apart never trip the breaker.
* **Exponential-backoff recovery** — each consecutive trip doubles the
  recovery timeout (capped at ``max_recovery_timeout``), giving genuinely
  degraded services time to recover.
* **Health score** — a rolling success-rate ∈ [0.0, 1.0] derived from the
  sliding window, useful for load-balancing or UI dashboards.
* **Per-endpoint registry** — ``CircuitBreakerRegistry`` hands out (or
  lazily creates) named breakers, letting callers isolate e.g.
  ``/v1/embeddings`` from ``/v1/chat/completions``.
* **Bug fix** — ``async_protected`` referenced a non-existent
  ``_check_recovery_timeout``; both sync and async paths now share the
  same state machine through ``_gate()`` / ``_record_outcome()``.

States:
    CLOSED → normal operation, requests pass through.
    OPEN   → failure threshold exceeded, requests fail-fast.
    HALF_OPEN → testing recovery, limited requests pass.

Example::

    registry = CircuitBreakerRegistry()
    breaker  = registry.get("openai-embeddings",
                            failure_threshold=5, window_duration=120.0)

    @breaker.protected
    def call_embeddings(texts):
        return openai.embeddings.create(input=texts, model="text-embedding-3-small")
"""

from __future__ import annotations

import collections
import logging
import math
import threading
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from ..types import AutoRAGError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(AutoRAGError):
    """Raised when circuit breaker is open."""

    pass


# ---------------------------------------------------------------------------
# Sliding-window event log
# ---------------------------------------------------------------------------

_Event = collections.namedtuple("_Event", ["timestamp", "success"])


class _SlidingWindow:
    """Fixed-duration sliding window that tracks success/failure events.

    All public methods are **not** thread-safe — the caller (``CircuitBreaker``)
    is expected to hold a lock.
    """

    __slots__ = ("_events", "_duration")

    def __init__(self, duration: float = 120.0) -> None:
        self._events: collections.deque[_Event] = collections.deque()
        self._duration = max(duration, 1.0)

    # -- mutators -----------------------------------------------------------

    def record(self, success: bool) -> None:
        """Append an event and evict stale entries."""
        now = time.monotonic()
        self._events.append(_Event(now, success))
        self._evict(now)

    def clear(self) -> None:
        self._events.clear()

    # -- queries ------------------------------------------------------------

    @property
    def failure_count(self) -> int:
        self._evict(time.monotonic())
        return sum(1 for e in self._events if not e.success)

    @property
    def success_count(self) -> int:
        self._evict(time.monotonic())
        return sum(1 for e in self._events if e.success)

    @property
    def total(self) -> int:
        self._evict(time.monotonic())
        return len(self._events)

    @property
    def health_score(self) -> float:
        """Rolling success rate ∈ [0.0, 1.0].  Returns 1.0 when empty."""
        self._evict(time.monotonic())
        n = len(self._events)
        if n == 0:
            return 1.0
        return sum(1 for e in self._events if e.success) / n

    # -- internal -----------------------------------------------------------

    def _evict(self, now: float) -> None:
        cutoff = now - self._duration
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()


# ---------------------------------------------------------------------------
# Circuit breaker v2
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Adaptive circuit breaker with sliding-window tracking.

    Args:
        failure_threshold: Failures within *window* to trip the breaker.
        recovery_timeout: Initial seconds to wait before half-open probe.
        expected_exceptions: Exception types that count as failures.
        success_threshold: Successes in half-open needed to close.
        window_duration: Sliding-window width in seconds (default 120 s).
        max_recovery_timeout: Upper cap on exponential backoff (default 600 s).
        backoff_multiplier: Factor by which recovery timeout grows per trip.

    Example::

        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

        @breaker.protected
        def risky_call():
            return external_api.call()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exceptions: tuple = (Exception,),
        success_threshold: int = 2,
        window_duration: float = 120.0,
        max_recovery_timeout: float = 600.0,
        backoff_multiplier: float = 2.0,
    ) -> None:
        # Configuration
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        self.success_threshold = success_threshold
        self.max_recovery_timeout = max_recovery_timeout
        self.backoff_multiplier = backoff_multiplier

        # Sliding window replaces static counter
        self._window = _SlidingWindow(duration=window_duration)

        # State machine
        self._state = CircuitState.CLOSED
        self._half_open_successes = 0
        self._last_failure_time: Optional[float] = None
        self._consecutive_trips = 0  # drives exponential backoff
        self._current_recovery_timeout = recovery_timeout
        self._lock = threading.Lock()

        # Lifetime statistics (never reset by window eviction)
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0

    # -- public properties --------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current state (performs OPEN→HALF_OPEN transition if due)."""
        with self._lock:
            self._maybe_half_open()
            return self._state

    @property
    def health_score(self) -> float:
        """Rolling success rate from the sliding window ∈ [0.0, 1.0]."""
        with self._lock:
            return self._window.health_score

    # -- gate / outcome (shared by sync + async paths) ----------------------

    def _gate(self) -> None:
        """Check state; raise if OPEN.  Must be called *with* lock held."""
        self._maybe_half_open()
        if self._state == CircuitState.OPEN:
            raise CircuitBreakerError(
                "Circuit breaker is OPEN — rejecting request",
                context={
                    "window_failures": self._window.failure_count,
                    "last_failure_time": self._last_failure_time,
                    "recovery_timeout": self._current_recovery_timeout,
                    "health_score": self._window.health_score,
                    "consecutive_trips": self._consecutive_trips,
                },
            )

    def _record_outcome(self, success: bool) -> None:
        """Update state machine after a call completes.  Acquires lock."""
        with self._lock:
            self._total_calls += 1
            self._window.record(success)

            if success:
                self._total_successes += 1
                if self._state == CircuitState.HALF_OPEN:
                    self._half_open_successes += 1
                    if self._half_open_successes >= self.success_threshold:
                        self._state = CircuitState.CLOSED
                        self._consecutive_trips = 0
                        self._current_recovery_timeout = self.recovery_timeout
                        logger.info(
                            "Circuit breaker CLOSED "
                            f"(recovered after {self.success_threshold} successes)"
                        )
            else:
                self._total_failures += 1
                self._last_failure_time = time.monotonic()

                if self._state == CircuitState.HALF_OPEN:
                    self._trip("recovery probe failed")

                elif (
                    self._state == CircuitState.CLOSED
                    and self._window.failure_count >= self.failure_threshold
                ):
                    self._trip(
                        f"failure threshold {self.failure_threshold} "
                        f"exceeded within sliding window"
                    )

    # -- internal state transitions -----------------------------------------

    def _trip(self, reason: str) -> None:
        """Open the circuit (must be called with lock held)."""
        self._state = CircuitState.OPEN
        self._consecutive_trips += 1
        self._current_recovery_timeout = min(
            self.recovery_timeout * math.pow(self.backoff_multiplier, self._consecutive_trips - 1),
            self.max_recovery_timeout,
        )
        logger.error(
            f"Circuit breaker OPENED ({reason}). "
            f"Recovery in {self._current_recovery_timeout:.1f}s "
            f"(trip #{self._consecutive_trips})"
        )

    def _maybe_half_open(self) -> None:
        """Transition OPEN→HALF_OPEN if recovery timeout elapsed.  Lock held."""
        if self._state != CircuitState.OPEN:
            return
        if self._last_failure_time is None:
            return
        if time.monotonic() - self._last_failure_time >= self._current_recovery_timeout:
            self._state = CircuitState.HALF_OPEN
            self._half_open_successes = 0
            logger.info(
                "Circuit breaker entering HALF_OPEN "
                f"(after {self._current_recovery_timeout:.1f}s backoff)"
            )

    # -- sync call path -----------------------------------------------------

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *func* through the circuit breaker.

        Raises:
            CircuitBreakerError: If circuit is OPEN.
        """
        with self._lock:
            self._gate()

        try:
            result = func(*args, **kwargs)
        except self.expected_exceptions as exc:
            self._record_outcome(success=False)
            logger.warning(f"Circuit breaker call failed: {exc}")
            raise
        else:
            self._record_outcome(success=True)
            return result

    # -- decorators ---------------------------------------------------------

    def protected(self, func: F) -> F:
        """Decorator for synchronous functions."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(func, *args, **kwargs)

        return wrapper  # type: ignore

    def async_protected(self, func: F) -> F:
        """Decorator for ``async`` functions.

        Example::

            @breaker.async_protected
            async def embed(texts):
                return await client.embeddings.create(input=texts)
        """

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                self._gate()

            try:
                result = await func(*args, **kwargs)
            except self.expected_exceptions as exc:
                self._record_outcome(success=False)
                logger.warning(f"Circuit breaker async call failed: {exc}")
                raise
            else:
                self._record_outcome(success=True)
                return result

        return wrapper  # type: ignore

    # -- management ---------------------------------------------------------

    def reset(self) -> None:
        """Manually reset to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._half_open_successes = 0
            self._last_failure_time = None
            self._consecutive_trips = 0
            self._current_recovery_timeout = self.recovery_timeout
            self._window.clear()
            logger.info("Circuit breaker manually RESET to CLOSED")

    def get_stats(self) -> Dict[str, Any]:
        """Snapshot of breaker statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "health_score": self._window.health_score,
                "window_failures": self._window.failure_count,
                "window_successes": self._window.success_count,
                "window_total": self._window.total,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "failure_threshold": self.failure_threshold,
                "success_threshold": self.success_threshold,
                "current_recovery_timeout": self._current_recovery_timeout,
                "consecutive_trips": self._consecutive_trips,
                "last_failure_time": self._last_failure_time,
            }


# ---------------------------------------------------------------------------
# Per-endpoint circuit breaker registry
# ---------------------------------------------------------------------------


class CircuitBreakerRegistry:
    """Thread-safe registry of named circuit breakers.

    Provides per-endpoint isolation so that e.g. a failing embeddings API
    does not trip the breaker for chat completions.

    Example::

        registry = CircuitBreakerRegistry()
        embed_breaker = registry.get("openai-embeddings", failure_threshold=3)
        chat_breaker  = registry.get("openai-chat", failure_threshold=5)
    """

    def __init__(self) -> None:
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str, **kwargs: Any) -> CircuitBreaker:
        """Get or create a named circuit breaker.

        On the *first* call for a given *name*, a ``CircuitBreaker`` is
        created with the supplied ``**kwargs``.  Subsequent calls return
        the same instance (``**kwargs`` are ignored after creation).
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(**kwargs)
                logger.debug(f"Created circuit breaker '{name}'")
            return self._breakers[name]

    def reset_all(self) -> None:
        """Reset every registered breaker."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return ``{name: stats}`` for every registered breaker."""
        with self._lock:
            return {name: cb.get_stats() for name, cb in self._breakers.items()}

    @property
    def names(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._breakers.keys())


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitBreakerRegistry",
    "CircuitState",
]
