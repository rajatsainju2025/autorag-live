import asyncio
from typing import Any, Awaitable, Callable, Iterable, List


async def gather_in_batches(
    callables: Iterable[Callable[[], Awaitable[Any]]], batch_size: int = 10
) -> List[Any]:
    """Run callables (zero-arg async callables) in batches to limit concurrency.

    Each callable is expected to be a function returning an awaitable (coroutine).
    This helper lets higher-level code avoid unbounded concurrency when processing
    many documents or LLM calls.
    """
    results: List[Any] = []
    batch = []
    for c in callables:
        batch.append(asyncio.create_task(c()))
        if len(batch) >= batch_size:
            results.extend(await asyncio.gather(*batch))
            batch = []

    if batch:
        results.extend(await asyncio.gather(*batch))

    return results


__all__ = ["gather_in_batches"]
