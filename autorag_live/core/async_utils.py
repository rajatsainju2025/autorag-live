"""Async helpers: timeout, retry, and executor-safe wrappers."""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("autorag_live.async_utils")


class AsyncTimeoutError(Exception):
    """Raised when an async operation times out."""


class AsyncRetryError(Exception):
    """Raised when retries are exhausted."""


def timeout(seconds: float):
    """Decorator to apply asyncio.wait_for timeout to an async function.

    Usage:
        @timeout(5.0)
        async def fetch(...):
            ...
    """

    def _decorator(func: Callable[..., Any]):
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("@timeout can only be applied to async functions")

        @functools.wraps(func)
        async def _wrapped(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError as e:
                logger.warning("Operation timed out (%s): %s", seconds, func.__name__)
                raise AsyncTimeoutError(str(e)) from e

        return _wrapped

    return _decorator


def retry(
    attempts: int = 3,
    initial_delay: float = 0.1,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Async retry decorator with exponential backoff.

    Retries the decorated async function on exceptions listed in `exceptions`.
    """

    def _decorator(func: Callable[..., Any]):
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("@retry can only be applied to async functions")

        @functools.wraps(func)
        async def _wrapped(*args, **kwargs):
            delay = initial_delay
            last_exc: Optional[BaseException] = None
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt,
                        attempts,
                        func.__name__,
                        e,
                    )
                    if attempt == attempts:
                        break
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
            raise AsyncRetryError(f"Retries exhausted: {last_exc}") from last_exc

        return _wrapped

    return _decorator


def run_sync_in_executor(
    func: Callable[..., Any], *args, executor: Optional[asyncio.AbstractEventLoop] = None, **kwargs
):
    """Run a synchronous function in the default thread executor safely.

    Returns a coroutine that can be awaited.
    """

    async def _runner():
        loop = asyncio.get_event_loop()
        pfunc = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, pfunc)

    return _runner()
