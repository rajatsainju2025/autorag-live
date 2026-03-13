from __future__ import annotations

import asyncio
from typing import Any, Dict, List


class AsyncElasticsearchWrapper:
    """Async wrapper for synchronous Elasticsearch retriever.

    Delegates blocking `search` and `add_documents` calls to a thread executor.
    """

    def __init__(self, sync_retriever: Any, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._retriever = sync_retriever
        self._loop = loop or asyncio.get_event_loop()

    async def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents asynchronously from Elasticsearch."""
        return await self._loop.run_in_executor(None, self._retriever.retrieve, query, k)

    async def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Add documents to index asynchronously."""
        return await self._loop.run_in_executor(None, self._retriever.add_documents, docs)

    async def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents from index asynchronously."""
        if hasattr(self._retriever, "delete_documents"):
            return await self._loop.run_in_executor(
                None, self._retriever.delete_documents, doc_ids
            )
