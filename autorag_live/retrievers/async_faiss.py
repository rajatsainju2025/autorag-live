from __future__ import annotations

import asyncio
from typing import Any, Dict, List


class AsyncFAISSRetriever:
    """Async wrapper for FAISS-based retriever with executor-based similarity computation.

    Runs CPU-intensive similarity searches in a thread executor to avoid blocking.
    """

    def __init__(self, sync_retriever: Any, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._retriever = sync_retriever
        self._loop = loop or asyncio.get_event_loop()

    async def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents asynchronously using FAISS similarity search."""
        return await self._loop.run_in_executor(None, self._retriever.retrieve, query, k)

    async def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Add and index documents asynchronously."""
        return await self._loop.run_in_executor(None, self._retriever.add_documents, docs)

    async def build_index(self) -> None:
        """Build FAISS index asynchronously."""
        if hasattr(self._retriever, "build_index"):
            return await self._loop.run_in_executor(None, self._retriever.build_index)
