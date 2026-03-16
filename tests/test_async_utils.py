import asyncio

import pytest

from autorag_live.core.async_utils import (
    AsyncRetryError,
    AsyncTimeoutError,
    retry,
    run_sync_in_executor,
    timeout,
)


@pytest.mark.asyncio
async def test_timeout_applies():
    @timeout(0.05)
    async def slow():
        await asyncio.sleep(0.2)
        return "ok"

    with pytest.raises(AsyncTimeoutError):
        await slow()


@pytest.mark.asyncio
async def test_retry_success_after_failures():
    calls = {"n": 0}

    @retry(attempts=3, initial_delay=0.01, backoff_factor=1.5, exceptions=(RuntimeError,))
    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return "ok"

    val = await flaky()
    assert val == "ok"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_retry_exhausted_raises():
    @retry(attempts=2, initial_delay=0.01, exceptions=(ValueError,))
    async def always_fail():
        raise ValueError("fail")

    with pytest.raises(AsyncRetryError):
        await always_fail()


def test_run_sync_in_executor():
    def sync_add(x, y):
        return x + y

    res = asyncio.get_event_loop().run_until_complete(run_sync_in_executor(sync_add, 2, 3))
    assert res == 5
