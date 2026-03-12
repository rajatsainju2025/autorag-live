import asyncio

from autorag_live.retrievers.async_qdrant import AsyncQdrantWrapper


class DummySyncRetriever:
    def retrieve(self, query: str, k: int = 10):
        return [{"id": i, "text": f"doc-{i}"} for i in range(min(k, 3))]

    def add_documents(self, docs):
        return None


async def test_async_qdrant_retrieve():
    sync = DummySyncRetriever()
    w = AsyncQdrantWrapper(sync)
    out = await w.retrieve("hello", k=2)
    assert isinstance(out, list)
    assert len(out) <= 2
