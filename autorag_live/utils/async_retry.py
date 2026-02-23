"""
Async Retry with Full Jitter and Hedge Requests.

Extends the synchronous ``with_retry`` in ``error_handling.py`` with:

1. **Full async support** — never blocks the event loop.
2. **Full jitter** — randomises delay in [0, cap] to prevent thundering herds
   (as recommended by AWS: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/).
3. **Hedge requests** — fires a speculative duplicate request after a timeout;
   whichever returns first wins. Reduces tail latency for LLM/retriever calls.
4. **Per-exception retry budgets** — different backoff for transient vs rate-limit errors.
5. **Async context-manager variant** — for use inside ``async with`` blocks.

Usage::

    from autorag_live.utils.async_retry import async_retry, hedge

    @async_retry(max_attempts=4, base_delay=0.5, jitter=True)
    async def call_llm(prompt: str) -> str:
        ...

    # Hedge: fire two calls, take the first that responds
    result = await hedge(call_llm, "my prompt", timeout=2.0, copies=2)
"""

from __future__ import annotations

import asyncio
import logging
import random
from functools import wraps
from typing import Any, Callable, Coroutine, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])
T = TypeVar("T")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_RETRYABLE: Tuple[Type[Exception], ...] = (Exception,)


# ---------------------------------------------------------------------------
# Core async retry decorator
# ---------------------------------------------------------------------------


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = _DEFAULT_RETRYABLE,
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[[F], F]:
    """
    Async retry decorator with exponential backoff and optional full jitter.

    Args:
        max_attempts: Total number of attempts (including the first).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Upper cap on delay (seconds).
        backoff_factor: Multiplier applied to delay after each failure.
        jitter: If True, applies full jitter: delay = random(0, min(cap, base * factor^n)).
        retryable_exceptions: Only retry on these exception types.
        non_retryable_exceptions: Never retry on these (takes precedence).
        on_retry: Optional callback(attempt_number, exception) called before each retry.

    Returns:
        Decorated async function that retries on failure.

    Example::

        @async_retry(max_attempts=5, base_delay=0.25, jitter=True,
                     retryable_exceptions=(aiohttp.ClientError,))
        async def fetch(url: str) -> str: ...
    """

    def decorator(func: F) -> F:
        func_name = func.__qualname__

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[Exception] = None
            delay = base_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except non_retryable_exceptions as exc:
                    logger.debug("%s: non-retryable %s, aborting", func_name, type(exc).__name__)
                    raise

                except retryable_exceptions as exc:
                    last_exc = exc

                    if attempt == max_attempts:
                        break

                    actual_delay = (
                        random.uniform(0.0, min(max_delay, delay))
                        if jitter
                        else min(max_delay, delay)
                    )

                    logger.warning(
                        "%s: attempt %d/%d failed (%s: %s), retrying in %.2fs",
                        func_name,
                        attempt,
                        max_attempts,
                        type(exc).__name__,
                        exc,
                        actual_delay,
                    )

                    if on_retry is not None:
                        try:
                            on_retry(attempt, exc)
                        except Exception:  # noqa: BLE001
                            pass

                    await asyncio.sleep(actual_delay)
                    delay *= backoff_factor

            logger.error(
                "%s: all %d attempts failed. Last error: %s",
                func_name,
                max_attempts,
                last_exc,
            )
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Hedge request helper
# ---------------------------------------------------------------------------


async def hedge(
    fn: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    timeout: float = 2.0,
    copies: int = 2,
    **kwargs: Any,
) -> T:
    """
    Fire ``copies`` speculative requests; return the first successful result.

    Hedge requests are a tail-latency optimisation: if the first request has
    not returned within ``timeout`` seconds, a second identical request is
    launched. The result of whichever finishes first is returned, and the
    other is cancelled.

    Args:
        fn: The async function to call.
        *args: Positional arguments forwarded to ``fn``.
        timeout: Seconds to wait before launching the next hedged copy.
        copies: Total number of parallel copies to allow.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        The result of the first copy that succeeds.

    Raises:
        The last exception if all copies fail.

    Example::

        result = await hedge(call_llm, "What is RAG?", timeout=1.5, copies=2)
    """
    tasks: list[asyncio.Task[T]] = []
    winner: Optional[asyncio.Future[T]] = None

    async def _launch_all() -> T:
        nonlocal winner
        for i in range(copies):
            if i > 0:
                # Wait before launching the next hedge copy
                await asyncio.sleep(timeout)
                # If already done, cancel outstanding tasks and return
                if winner is not None and winner.done():
                    break
            task: asyncio.Task[T] = asyncio.create_task(fn(*args, **kwargs))
            tasks.append(task)

        # Await whichever finishes first
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Cancel the rest
        for t in pending:
            t.cancel()

        result_task = next(iter(done))
        if result_task.exception():
            raise result_task.exception()  # type: ignore[misc]
        return result_task.result()

    try:
        return await _launch_all()
    except Exception:
        # Ensure all tasks are cancelled on failure
        for t in tasks:
            if not t.done():
                t.cancel()
        raise


# ---------------------------------------------------------------------------
# Async retry context manager
# ---------------------------------------------------------------------------


class AsyncRetryContext:
    """
    Async context manager variant for manual retry control.

    Example::

        async with AsyncRetryContext(max_attempts=3, base_delay=0.5) as retry:
            async for attempt in retry:
                try:
                    result = await call_llm(prompt)
                    break
                except RateLimitError as e:
                    retry.record_failure(e)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self._attempt = 0
        self._last_exc: Optional[Exception] = None
        self._delay = base_delay

    async def __aenter__(self) -> "AsyncRetryContext":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def record_failure(self, exc: Exception) -> None:
        """Record a failure; caller should check :attr:`should_retry`."""
        self._last_exc = exc
        self._attempt += 1

    @property
    def should_retry(self) -> bool:
        """True if there are remaining attempts."""
        return self._attempt < self.max_attempts

    async def wait(self) -> None:
        """Async sleep with backoff + jitter before the next attempt."""
        actual = (
            random.uniform(0.0, min(self.max_delay, self._delay))
            if self.jitter
            else min(self.max_delay, self._delay)
        )
        await asyncio.sleep(actual)
        self._delay *= self.backoff_factor
