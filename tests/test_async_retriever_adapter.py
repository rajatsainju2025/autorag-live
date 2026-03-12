import pytest

from autorag_live.core.adapters import AsyncRetrieverAdapter
from autorag_live.retrievers.base import BaseRetriever


class DummySyncRetriever(BaseRetriever):
    def __init__(self):
        super().__init__()

    def retrieve(self, query, k=5):
        return [("doc1", 0.9)]

    def add_documents(self, documents):
        self._docs = documents

    def load(self, path: str) -> None:
        pass

    def save(self, path: str) -> None:
        pass


@pytest.mark.asyncio
async def test_async_adapter_retrieves():
    sync = DummySyncRetriever()
    adapter = AsyncRetrieverAdapter(sync)
    res = await adapter.retrieve("q")
    assert isinstance(res, list)
