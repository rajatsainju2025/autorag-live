"""Async stale-while-revalidate cache.

Serves stale values immediately after soft TTL expiry while refreshing in the
background. Reduces tail latency and avoids cache stampedes in RAG pipelines.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    value: T
    fresh_until: float
    stale_until: float


class AsyncSWRCache(Generic[T]):
    """In-memory stale-while-revalidate cache."""

    def __init__(self, max_size: int = 10_000) -> None:
        self._data: Dict[str, _Entry[T]] = {}
        self._max_size = max_size
        self._refreshing: set[str] = set()
        self._lock = asyncio.Lock()

        self._hits = 0
        self._stale_hits = 0
        self._misses = 0

    async def get_or_set(
        self,
        key: str,
        loader: Callable[[], Awaitable[T]],
        ttl_s: float,
        stale_ttl_s: float = 30.0,
    ) -> T:
        now = time.monotonic()
        entry = self._data.get(key)
        if entry is not None:
            if now <= entry.fresh_until:
                self._hits += 1
                return entry.value
            if now <= entry.stale_until:
                self._stale_hits += 1
                await self._trigger_refresh(key, loader, ttl_s, stale_ttl_s)
                return entry.value

        self._misses += 1
        value = await loader()
        await self._set(key, value, ttl_s, stale_ttl_s)
        return value

    async def _set(self, key: str, value: T, ttl_s: float, stale_ttl_s: float) -> None:
        async with self._lock:
            if len(self._data) >= self._max_size and key not in self._data:
                # simple oldest-ish eviction by first key insertion order
                first_key = next(iter(self._data))
                self._data.pop(first_key, None)
            now = time.monotonic()
            self._data[key] = _Entry(
                value=value,
                fresh_until=now + max(0.0, ttl_s),
                stale_until=now + max(0.0, ttl_s + stale_ttl_s),
            )

    async def _trigger_refresh(
        self,
        key: str,
        loader: Callable[[], Awaitable[T]],
        ttl_s: float,
        stale_ttl_s: float,
    ) -> None:
        async with self._lock:
            if key in self._refreshing:
                return
            self._refreshing.add(key)

        async def _run() -> None:
            try:
                value = await loader()
                await self._set(key, value, ttl_s, stale_ttl_s)
            except Exception:
                # keep stale value on refresh failure
                pass
            finally:
                async with self._lock:
                    self._refreshing.discard(key)

        asyncio.create_task(_run())

    async def invalidate(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._data.clear()
            self._refreshing.clear()

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._stale_hits + self._misses
        hit_rate = (self._hits + self._stale_hits) / total if total else 0.0
        return {
            "size": len(self._data),
            "hits": self._hits,
            "stale_hits": self._stale_hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "refreshing": len(self._refreshing),
        }


def create_swr_cache(max_size: int = 10_000) -> AsyncSWRCache[Any]:
    """Factory for `AsyncSWRCache`."""
    return AsyncSWRCache(max_size=max_size)
