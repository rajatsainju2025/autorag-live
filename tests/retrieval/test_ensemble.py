import pytest

from autorag_live.retrieval.ensemble import (
    EnsembleRetriever,
    FusionConfig,
    FusionMethod,
    RetrievedDocument,
)


class StaticRetriever:
    def __init__(self, name: str, results: list[RetrievedDocument]) -> None:
        self._name = name
        self._results = results

    @property
    def name(self) -> str:
        return self._name

    async def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedDocument]:
        del query
        return self._results[:top_k]


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text)), float(len(text)) / 2.0]


@pytest.mark.asyncio
async def test_ensemble_merges_documents_without_doc_ids() -> None:
    docs_a = [RetrievedDocument(content="shared", score=0.9)]
    docs_b = [RetrievedDocument(content="shared", score=0.7)]

    retriever = EnsembleRetriever(
        retrievers=[StaticRetriever("sparse", docs_a), StaticRetriever("dense", docs_b)],
        config=FusionConfig(method=FusionMethod.RRF),
    )

    result = await retriever.retrieve("shared", top_k=5)

    assert len(result.documents) == 1
    assert result.documents[0].doc_id.startswith("content:")
    assert list(result.fusion_scores) == [result.documents[0].doc_id]
