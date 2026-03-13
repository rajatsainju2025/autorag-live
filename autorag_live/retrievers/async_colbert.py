"""Async wrapper for ColBERT retriever."""

import asyncio
from typing import Any, Dict, List


class AsyncColBERTRetriever:
    """Async wrapper for ColBERT retriever with executor-based delegation."""

    def __init__(self, sync_retriever: Any, loop: asyncio.AbstractEventLoop | None = None):
        """Initialize with a sync ColBERT retriever.

        Args:
            sync_retriever: Synchronous ColBERT retriever instance
            loop: Optional event loop
        """
        self._retriever = sync_retriever
        self._loop = loop or asyncio.get_event_loop()

    async def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve documents asynchronously.

        Args:
            query: Query string
            k: Number of results to retrieve

        Returns:
            List of retrieved documents
        """
        return await self._loop.run_in_executor(
            None, self._retriever.retrieve, query, k
        )

    async def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Add documents asynchronously.

        Args:
            docs: List of documents to add
        """
        await self._loop.run_in_executor(None, self._retriever.add_documents, docs)

    async def encode_queries(self, queries: List[str]) -> List[List[float]]:
        """Encode queries asynchronously.

        Args:
            queries: List of query strings

        Returns:
            List of query embeddings
        """
        return await self._loop.run_in_executor(
            None, self._retriever.encode_queries, queries
        )
