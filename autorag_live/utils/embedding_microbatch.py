"""Async embedding micro-batcher.

Aggregates small embedding requests into micro-batches to improve throughput,
reduce p95 latency under load, and lower provider overhead.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List

BatchEmbedFn = Callable[[List[str]], Coroutine[Any, Any, List[List[float]]]]


@dataclass
class _QueueItem:
    text: str
    fut: asyncio.Future[List[float]]


class AsyncEmbeddingMicroBatcher:
    """Queue-based embedding micro-batcher.

    Requests are flushed when:
    - batch size reaches `max_batch_size`, or
    - `max_wait_ms` elapses.
    """

    def __init__(
        self,
        batch_embed_fn: BatchEmbedFn,
        max_batch_size: int = 64,
        max_wait_ms: int = 8,
    ) -> None:
        self._embed = batch_embed_fn
        self._max_batch = max_batch_size
        self._max_wait_s = max_wait_ms / 1000.0

        self._queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._closed = False

        self._total_requests = 0
        self._total_batches = 0
        self._total_wait_s = 0.0

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def close(self) -> None:
        self._closed = True
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def embed(self, text: str) -> List[float]:
        """Enqueue one text and await its embedding."""
        if self._closed:
            raise RuntimeError("Batcher is closed")
        if self._worker_task is None:
            await self.start()

        fut: asyncio.Future[List[float]] = asyncio.get_running_loop().create_future()
        await self._queue.put(_QueueItem(text=text, fut=fut))
        self._total_requests += 1
        return await fut

    async def _worker(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]
            t0 = time.monotonic()

            while len(batch) < self._max_batch:
                remaining = self._max_wait_s - (time.monotonic() - t0)
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except TimeoutError:
                    break

            texts = [i.text for i in batch]
            try:
                vectors = await self._embed(texts)
                if len(vectors) != len(batch):
                    raise ValueError(
                        f"Embedding provider returned {len(vectors)} vectors for "
                        f"{len(batch)} inputs"
                    )
                for item, vec in zip(batch, vectors):
                    if not item.fut.done():
                        item.fut.set_result(vec)
            except Exception as exc:  # noqa: BLE001
                for item in batch:
                    if not item.fut.done():
                        item.fut.set_exception(exc)
            finally:
                self._total_batches += 1
                self._total_wait_s += max(0.0, time.monotonic() - t0)

    def stats(self) -> dict[str, float]:
        avg_batch = self._total_requests / self._total_batches if self._total_batches else 0.0
        avg_wait_ms = (
            (self._total_wait_s / self._total_batches) * 1000 if self._total_batches else 0.0
        )
        return {
            "queued": float(self._queue.qsize()),
            "total_requests": float(self._total_requests),
            "total_batches": float(self._total_batches),
            "avg_batch_size": round(avg_batch, 2),
            "avg_batch_wait_ms": round(avg_wait_ms, 2),
        }


def create_embedding_microbatcher(
    batch_embed_fn: BatchEmbedFn,
    max_batch_size: int = 64,
    max_wait_ms: int = 8,
) -> AsyncEmbeddingMicroBatcher:
    """Factory for `AsyncEmbeddingMicroBatcher`."""
    return AsyncEmbeddingMicroBatcher(
        batch_embed_fn=batch_embed_fn,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
    )
