from autorag_live.retrievers.async_elasticsearch import AsyncElasticsearchWrapper


class DummySyncESRetriever:
    def retrieve(self, query: str, k: int = 10):
        return [{"id": f"es-{i}", "score": 0.9 - i * 0.1} for i in range(min(k, 3))]

    def add_documents(self, docs):
        pass

    def delete_documents(self, doc_ids):
        pass


async def test_async_elasticsearch_retrieve():
    sync = DummySyncESRetriever()
    wrapper = AsyncElasticsearchWrapper(sync)
    results = await wrapper.retrieve("test query", k=2)
    assert isinstance(results, list)
    assert len(results) <= 2


async def test_async_elasticsearch_add_documents():
    sync = DummySyncESRetriever()
    wrapper = AsyncElasticsearchWrapper(sync)
    docs = [{"id": "1", "text": "hello"}]
    await wrapper.add_documents(docs)
