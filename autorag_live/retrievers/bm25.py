import hashlib
import re
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency guard
    from rank_bm25 import BM25Okapi  # type: ignore

    BM25_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when rank_bm25 missing
    BM25_AVAILABLE = False
    BM25Okapi = None  # type: ignore

from ..types.types import DocumentText, QueryText, RetrievalResult, RetrieverError
from ..utils import get_logger, monitor_performance
from .base import BaseRetriever

logger = get_logger(__name__)

# Pre-compiled token pattern for fast tokenization (alphanumeric word tokens)
_TOKEN_PATTERN = re.compile(r"\w+")

# Small LRU cache for BM25 instances keyed by corpus signature
_BM25_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_BM25_CACHE_MAXSIZE = 2


def _corpus_signature(corpus: List[str]) -> str:
    """Compute a stable signature for a corpus for caching."""
    md5 = hashlib.md5()
    md5.update(str(len(corpus)).encode())
    for doc in corpus:
        md5.update(hashlib.md5(doc.encode()).digest())
    return md5.hexdigest()


def _get_bm25_for_corpus(corpus: List[str]) -> Any:
    """Get or build a BM25Okapi instance for the given corpus with LRU caching."""
    if not BM25_AVAILABLE or BM25Okapi is None:
        raise ImportError("rank_bm25 is required for bm25_retrieve but is not installed")

    sig = _corpus_signature(corpus)
    cached = _BM25_CACHE.get(sig)
    if cached is not None:
        # LRU: move to end
        _BM25_CACHE.move_to_end(sig)
        return cached

    tokenized_corpus = [BM25Retriever._tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)  # type: ignore[call-arg]

    # Insert into cache and enforce size
    _BM25_CACHE[sig] = bm25
    while len(_BM25_CACHE) > _BM25_CACHE_MAXSIZE:
        _BM25_CACHE.popitem(last=False)

    return bm25


def bm25_retrieve(query: str, corpus: List[str], k: int) -> List[str]:
    """
    Retrieves top-k documents from the corpus using BM25.
    """
    if not corpus:
        return []

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if k <= 0:
        raise ValueError("k must be positive")

    bm25 = _get_bm25_for_corpus(corpus)
    tokenized_query = BM25Retriever._tokenize(query)
    doc_scores = np.asarray(bm25.get_scores(tokenized_query), dtype=np.float32)

    if doc_scores.size == 0:
        return []

    effective_k = min(k, len(doc_scores))
    top_indices = np.argpartition(doc_scores, -effective_k)[-effective_k:]
    top_sorted = top_indices[np.argsort(doc_scores[top_indices])[::-1]]

    return [corpus[int(i)] for i in top_sorted]


class BM25Retriever(BaseRetriever):
    """BM25 retriever implementation."""

    def __init__(self):
        super().__init__()
        self.corpus: List[str] = []
        self.bm25: Any = None
        self._tokenized_corpus: List[List[str]] = []
        self._scores_cache: Dict[Tuple[str, ...], np.ndarray] = {}

    def add_documents(self, documents: List[DocumentText]) -> None:
        """Add documents to the retriever's index."""
        with monitor_performance("BM25Retriever.add_documents", {"num_docs": len(documents)}):
            self.corpus = documents
            self._tokenized_corpus = [self._tokenize(doc) for doc in documents]
            if not BM25_AVAILABLE:
                raise RetrieverError("rank_bm25 is required for BM25Retriever but is not installed")

            if not self._tokenized_corpus:
                self.bm25 = None
            else:
                self.bm25 = BM25Okapi(self._tokenized_corpus)  # type: ignore[call-arg]
            self._scores_cache.clear()
            self._is_initialized = True

    def retrieve(self, query: QueryText, k: int = 5) -> RetrievalResult:
        """Retrieve documents for a query."""
        if not self.is_initialized:
            raise RetrieverError("Retriever not initialized. Call add_documents() first.")

        with monitor_performance("BM25Retriever.retrieve", {"query_length": len(query), "k": k}):
            if self.bm25 is None:
                return []

            tokenized_query = self._tokenize(query)
            cache_key = tuple(tokenized_query)
            cached_scores = self._scores_cache.get(cache_key)

            if cached_scores is None:
                scores = np.asarray(self.bm25.get_scores(tokenized_query), dtype=np.float32)
                self._scores_cache[cache_key] = scores
            else:
                scores = cached_scores

            if scores.size == 0:
                return []

            effective_k = min(k, scores.size)
            if effective_k <= 0:
                return []

            # Use argpartition for efficient top-k selection
            top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
            top_sorted = top_indices[np.argsort(scores[top_indices])[::-1]]

            results: RetrievalResult = []
            for idx in top_sorted:
                results.append((self.corpus[int(idx)], float(scores[int(idx)])))

            return results

    def load(self, path: str) -> None:
        """Load retriever state from disk."""
        raise NotImplementedError("BM25 retriever persistence not implemented")

    def save(self, path: str) -> None:
        """Save retriever state to disk."""
        raise NotImplementedError("BM25 retriever persistence not implemented")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize incoming text consistently."""
        # Use regex to avoid costly split() on large texts and strip punctuation
        return [t for t in _TOKEN_PATTERN.findall(text.lower()) if t]
