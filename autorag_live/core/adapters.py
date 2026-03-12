import asyncio
from typing import Any, Dict, List

from autorag_live.retrievers.base import BaseRetriever


class AsyncRetrieverAdapter:
    """Wrap a sync BaseRetriever and expose an async `retrieve` method."""

    def __init__(self, retriever: BaseRetriever):
        self._retriever = retriever

    async def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._retriever.retrieve, query, k)

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        return self._retriever.add_documents(docs)


__all__ = ["AsyncRetrieverAdapter"]
