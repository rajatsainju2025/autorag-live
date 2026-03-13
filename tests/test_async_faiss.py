from autorag_live.retrievers.async_faiss import AsyncFAISSRetriever


class DummySyncFAISSRetriever:
    def retrieve(self, query: str, k: int = 10):
        return [{"id": f"faiss-{i}", "distance": i * 0.05} for i in range(min(k, 4))]

    def add_documents(self, docs):
        pass

    def build_index(self):
        pass


async def test_async_faiss_retrieve():
    sync = DummySyncFAISSRetriever()
    retriever = AsyncFAISSRetriever(sync)
    results = await retriever.retrieve("embedding query", k=3)
    assert isinstance(results, list)
    assert len(results) <= 3


async def test_async_faiss_build_index():
    sync = DummySyncFAISSRetriever()
    retriever = AsyncFAISSRetriever(sync)
    await retriever.build_index()
