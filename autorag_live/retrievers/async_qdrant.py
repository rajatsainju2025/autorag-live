from __future__ import annotations

import asyncio
from typing import Any, Dict, List


class AsyncQdrantWrapper:
    """Simple async wrapper around a sync Qdrant retriever instance.

    It delegates `retrieve` to a thread executor to avoid blocking the event loop.
    """

    def __init__(self, sync_retriever: Any, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._retriever = sync_retriever
        self._loop = loop or asyncio.get_event_loop()

    async def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        return await self._loop.run_in_executor(None, self._retriever.retrieve, query, k)

    async def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        return await self._loop.run_in_executor(None, self._retriever.add_documents, docs)
