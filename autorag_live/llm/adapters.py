from __future__ import annotations

import asyncio
from typing import Any, List

from autorag_live.core.interfaces import LLMProvider


class AsyncLLMAdapter(LLMProvider):
    """Adapter that wraps a synchronous `LLMProvider`-like object and exposes
    async methods by delegating sync calls to a thread executor.

    Usage:
        sync_llm = SomeSyncLLM(...)
        async_adapter = AsyncLLMAdapter(sync_llm)
        await async_adapter.ainvoke("hi")
    """

    def __init__(
        self, sync_provider: LLMProvider | Any, loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        # We accept duck-typed providers that implement `invoke`/`agenerate`.
        self._provider = sync_provider
        self._loop = loop or asyncio.get_event_loop()

    async def agenerate(self, prompts: List[str]) -> Any:
        # If the provider already implements async generation, use it.
        if hasattr(self._provider, "agenerate") and asyncio.iscoroutinefunction(
            getattr(self._provider, "agenerate")
        ):
            return await self._provider.agenerate(prompts)

        # Otherwise run a sync method in a thread.
        def sync_generate():
            # prefer `agenerate` if available (sync returning list), else fallback to repeated invoke
            if hasattr(self._provider, "agenerate"):
                try:
                    return self._provider.agenerate(prompts)
                except Exception:
                    pass
            if hasattr(self._provider, "invoke"):
                return [self._provider.invoke(p) for p in prompts]
            raise AttributeError("Underlying provider has no generation method")

        return await self._loop.run_in_executor(None, sync_generate)

    async def ainvoke(self, prompt: str) -> Any:
        if hasattr(self._provider, "ainvoke") and asyncio.iscoroutinefunction(
            getattr(self._provider, "ainvoke")
        ):
            return await self._provider.ainvoke(prompt)

        if hasattr(self._provider, "invoke"):
            return await self._loop.run_in_executor(None, self._provider.invoke, prompt)

        raise AttributeError("Underlying provider has no invoke/ainvoke method")

    def invoke(self, prompt: str) -> Any:
        # Provide a direct sync passthrough for callers that need it.
        if hasattr(self._provider, "invoke"):
            return self._provider.invoke(prompt)
        # If only async exists, run it to completion (blocking)
        if hasattr(self._provider, "ainvoke") and asyncio.iscoroutinefunction(
            getattr(self._provider, "ainvoke")
        ):
            return asyncio.get_event_loop().run_until_complete(self._provider.ainvoke(prompt))
        raise AttributeError("Underlying provider has no invoke/ainvoke method")
