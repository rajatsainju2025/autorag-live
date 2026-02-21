"""Async singleflight request coalescing.

Coalesces concurrent calls sharing the same key so only one producer executes
while all followers await the same result. This reduces duplicate LLM,
retriever, and embedding calls under bursty traffic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _InFlight(Generic[T]):
    future: asyncio.Future[T]
    started_at: float
    followers: int = 0


class AsyncSingleFlight:
    """Deduplicate concurrent calls by key.

    Example:
        sf = AsyncSingleFlight()
        result = await sf.do("q:what is rag", lambda: expensive_call())
    """

    def __init__(self, max_keys: int = 10_000) -> None:
        self._lock = asyncio.Lock()
        self._inflight: Dict[str, _InFlight[Any]] = {}
        self._max_keys = max_keys

        # counters
        self._primary_executions = 0
        self._coalesced_followers = 0
        self._errors = 0

    async def do(self, key: str, fn: Callable[[], Coroutine[Any, Any, T]]) -> T:
        """Execute `fn` once per key for concurrent callers."""
        async with self._lock:
            item = self._inflight.get(key)
            if item is not None:
                item.followers += 1
                self._coalesced_followers += 1
                fut: asyncio.Future[T] = item.future
                leader = False
            else:
                if len(self._inflight) >= self._max_keys:
                    # Soft pressure relief: evict oldest key.
                    oldest_key = min(
                        self._inflight,
                        key=lambda k: self._inflight[k].started_at,
                    )
                    self._inflight.pop(oldest_key, None)
                fut = asyncio.get_running_loop().create_future()
                self._inflight[key] = _InFlight(
                    future=fut,
                    started_at=time.monotonic(),
                )
                self._primary_executions += 1
                leader = True

        if not leader:
            return await fut

        try:
            result = await fn()
            if not fut.done():
                fut.set_result(result)
            return result
        except Exception as exc:  # noqa: BLE001
            self._errors += 1
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._inflight.pop(key, None)

    async def forget(self, key: str) -> None:
        """Remove a key from in-flight tracking if present."""
        async with self._lock:
            self._inflight.pop(key, None)

    def stats(self) -> Dict[str, Any]:
        active = len(self._inflight)
        saved_calls = self._coalesced_followers
        total_requested = self._primary_executions + self._coalesced_followers
        coalesce_rate = saved_calls / total_requested if total_requested else 0.0
        return {
            "active_keys": active,
            "primary_executions": self._primary_executions,
            "coalesced_followers": self._coalesced_followers,
            "saved_calls": saved_calls,
            "coalesce_rate": round(coalesce_rate, 4),
            "errors": self._errors,
        }


def create_singleflight(max_keys: int = 10_000) -> AsyncSingleFlight:
    """Factory for `AsyncSingleFlight`."""
    return AsyncSingleFlight(max_keys=max_keys)
