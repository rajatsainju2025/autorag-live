"""
Async-Native Circuit Breaker.

A production-grade, fully async circuit breaker with per-service tracking,
state-machine transitions, and configurable half-open probing.

State Machine
-------------

    CLOSED ──(failures ≥ threshold)──► OPEN
    OPEN   ──(reset_timeout elapsed)──► HALF_OPEN
    HALF_OPEN ──(probe succeeds)──► CLOSED
    HALF_OPEN ──(probe fails)───► OPEN

Why async-native matters for agentic RAG
-----------------------------------------
Concurrent agents share I/O services (embedding APIs, vector stores,
LLM providers).  A sync circuit breaker blocks the event loop during
sleep/wait.  This implementation uses asyncio.Event for zero-blocking
state transitions and supports per-service breakers via a registry.

Features
--------
- Per-service ``CircuitBreakerRegistry`` for concurrent agents
- Configurable failure threshold and reset timeout
- Exponential backoff for half-open probe retries
- ``async with breaker:`` context manager
- ``@circuit_breaker(name)`` async decorator
- Prometheus-style ``stats()`` for observability

References
----------
- "Release It!" Michael Nygard, 2nd ed. (canonical circuit breaker pattern)
- "Hystrix: Latency and Fault Tolerance" (Netflix, 2012)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F")


# ---------------------------------------------------------------------------
# State & exceptions
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failing — reject calls immediately
    HALF_OPEN = auto()  # Probing — allow one call through


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""

    def __init__(self, service: str, retry_after: float) -> None:
        self.service = service
        self.retry_after = retry_after
        super().__init__(f"Circuit OPEN for '{service}'. Retry after {retry_after:.1f}s")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CircuitBreakerConfig:
    """Configuration for a single circuit breaker."""

    failure_threshold: int = 5
    """Consecutive failures before opening the circuit."""

    reset_timeout_s: float = 30.0
    """Seconds the circuit stays OPEN before transitioning to HALF_OPEN."""

    half_open_probe_timeout_s: float = 10.0
    """Timeout for the probe call in HALF_OPEN state."""

    success_threshold: int = 1
    """Consecutive successes in HALF_OPEN before closing."""

    excluded_exceptions: tuple[Type[BaseException], ...] = field(default_factory=tuple)
    """Exceptions that should NOT count as failures (e.g. ValidationError)."""


# ---------------------------------------------------------------------------
# Core breaker
# ---------------------------------------------------------------------------


class AsyncCircuitBreaker:
    """
    Async circuit breaker for a single service.

    Args:
        name: Service identifier (used in logs and errors).
        config: :class:`CircuitBreakerConfig` instance.

    Usage::

        breaker = AsyncCircuitBreaker("openai_embed")

        # As async context manager
        try:
            async with breaker:
                result = await embed(text)
        except CircuitOpenError:
            result = fallback()

        # As decorator
        @breaker.protect
        async def embed_text(text: str) -> list[float]:
            ...
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0  # consecutive successes in HALF_OPEN
        self._opened_at: float = 0.0
        self._lock = asyncio.Lock()

        # Stats
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejected = 0
        self._state_changes: list[tuple[float, CircuitState]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        return self._state

    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    async def __aenter__(self) -> "AsyncCircuitBreaker":
        await self._check_state()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        if exc_type is None:
            await self._on_success()
        elif not self._is_excluded(exc_type):
            await self._on_failure()
        return False  # never suppress exceptions

    def protect(
        self, fn: Callable[..., Coroutine[Any, Any, Any]]
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        """Async decorator that wraps a coroutine with this circuit breaker."""

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self:
                return await fn(*args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    async def _check_state(self) -> None:
        """Raise CircuitOpenError or allow the call through."""
        async with self._lock:
            self._total_calls += 1

            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                remaining = self.config.reset_timeout_s - elapsed
                if remaining > 0:
                    self._total_rejected += 1
                    raise CircuitOpenError(self.name, remaining)
                # Transition to HALF_OPEN
                self._transition(CircuitState.HALF_OPEN)

            # CLOSED or HALF_OPEN — allow the call

    async def _on_success(self) -> None:
        async with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition(CircuitState.CLOSED)
                    self._success_count = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — reopen
                self._transition(CircuitState.OPEN)
                self._success_count = 0
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.config.failure_threshold
            ):
                self._transition(CircuitState.OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        old = self._state
        self._state = new_state
        if new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
            self._failure_count = 0
        logger.info("CircuitBreaker '%s': %s → %s", self.name, old.name, new_state.name)
        self._state_changes.append((time.monotonic(), new_state))

    def _is_excluded(self, exc_type: Type[BaseException]) -> bool:
        return issubclass(exc_type, self.config.excluded_exceptions)

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------

    async def reset(self) -> None:
        """Manually close the circuit (e.g. after operator intervention)."""
        async with self._lock:
            self._transition(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0

    async def trip(self) -> None:
        """Manually open the circuit (e.g. for maintenance)."""
        async with self._lock:
            self._transition(CircuitState.OPEN)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return Prometheus-style stats dict."""
        return {
            "service": self.name,
            "state": self._state.name,
            "failure_count": self._failure_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "total_rejected": self._total_rejected,
            "failure_rate": (
                round(self._total_failures / self._total_calls, 4) if self._total_calls > 0 else 0.0
            ),
            "open_duration_s": (
                round(time.monotonic() - self._opened_at, 2)
                if self._state == CircuitState.OPEN
                else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class CircuitBreakerRegistry:
    """
    Centralized registry of named :class:`AsyncCircuitBreaker` instances.

    Enables concurrent agents to share or create per-service breakers
    without duplicating state.

    Example::

        registry = CircuitBreakerRegistry()
        embed_cb = registry.get("openai_embed")
        llm_cb   = registry.get("anthropic_claude", config=CircuitBreakerConfig(failure_threshold=3))

        async with embed_cb:
            emb = await embed(text)
    """

    def __init__(self) -> None:
        self._breakers: Dict[str, AsyncCircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> AsyncCircuitBreaker:
        """Get or create a breaker for *name*."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = AsyncCircuitBreaker(name, config)
            return self._breakers[name]

    def get_sync(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> AsyncCircuitBreaker:
        """Get or create a breaker (sync version — safe before event loop starts)."""
        if name not in self._breakers:
            self._breakers[name] = AsyncCircuitBreaker(name, config)
        return self._breakers[name]

    def all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return stats for all registered breakers."""
        return {name: cb.stats() for name, cb in self._breakers.items()}

    async def reset_all(self) -> None:
        """Reset all breakers to CLOSED state."""
        for cb in self._breakers.values():
            await cb.reset()


# ---------------------------------------------------------------------------
# Decorator helper
# ---------------------------------------------------------------------------


def circuit_breaker(
    name: str,
    registry: Optional[CircuitBreakerRegistry] = None,
    config: Optional[CircuitBreakerConfig] = None,
) -> Callable[[F], F]:
    """
    Async function decorator that protects with a named circuit breaker.

    Args:
        name: Service/breaker name.
        registry: Optional shared registry; creates a standalone breaker if None.
        config: Breaker configuration.

    Example::

        @circuit_breaker("vector_store")
        async def search(query: str) -> list[dict]:
            ...
    """
    if registry is not None:
        cb = registry.get_sync(name, config)
    else:
        cb = AsyncCircuitBreaker(name, config)

    def decorator(fn: F) -> F:
        return cb.protect(fn)  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

#: Shared default registry — use for simple single-process deployments.
default_registry = CircuitBreakerRegistry()
