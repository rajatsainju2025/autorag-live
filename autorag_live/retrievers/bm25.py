from typing import List
from rank_bm25 import BM25Okapi

from .base import BaseRetriever
from ..types.types import QueryText, DocumentText, RetrievalResult, RetrieverError
from ..utils import monitor_performance, get_logger

logger = get_logger(__name__)


def bm25_retrieve(query: str, corpus: List[str], k: int) -> List[str]:
    """
    Retrieves top-k documents from the corpus using BM25.
    """
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)

    top_k_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]

    return [corpus[i] for i in top_k_indices]


class BM25Retriever(BaseRetriever):
    """BM25 retriever implementation."""

    def __init__(self):
        super().__init__()
        self.corpus: List[str] = []
        self.bm25 = None

    def add_documents(self, documents: List[DocumentText]) -> None:
        """Add documents to the retriever's index."""
        with monitor_performance("BM25Retriever.add_documents", {"num_docs": len(documents)}):
            self.corpus = documents
            tokenized_corpus = [doc.split(" ") for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._is_initialized = True

    def retrieve(self, query: QueryText, k: int = 5) -> RetrievalResult:
        """Retrieve documents for a query."""
        if not self.is_initialized:
            raise RetrieverError("Retriever not initialized. Call add_documents() first.")

        with monitor_performance("BM25Retriever.retrieve", {"query_length": len(query), "k": k}):
            if self.bm25 is None:
                return []

            tokenized_query = query.split(" ")
            doc_scores = self.bm25.get_scores(tokenized_query)

            # Get top k results with scores
            top_k_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]

            results = []
            for idx in top_k_indices:
                results.append((self.corpus[idx], float(doc_scores[idx])))

            return results

    def load(self, path: str) -> None:
        """Load retriever state from disk."""
        raise NotImplementedError("BM25 retriever persistence not implemented")

    def save(self, path: str) -> None:
        """Save retriever state to disk."""
        raise NotImplementedError("BM25 retriever persistence not implemented")
