"""Incremental TF-IDF for BM25 optimization."""

from typing import Dict, List

import numpy as np


class IncrementalTFIDF:
    """Incrementally maintains TF-IDF statistics without recomputation."""

    def __init__(self):
        """Initialize incremental TF-IDF."""
        self._doc_frequencies: Dict[str, int] = {}
        self._doc_count = 0
        self._term_doc_freq: Dict[str, int] = {}

    def add_document(self, tokens: List[str]) -> None:
        """Add document and update statistics."""
        self._doc_count += 1
        unique_tokens = set(tokens)

        for token in unique_tokens:
            self._term_doc_freq[token] = self._term_doc_freq.get(token, 0) + 1

    def get_idf(self, term: str) -> float:
        """Get IDF for a term."""
        if self._doc_count == 0:
            return 0.0

        doc_freq = self._term_doc_freq.get(term, 0)
        if doc_freq == 0:
            return 0.0

        return np.log(self._doc_count / (1 + doc_freq))

    def get_idfs_batch(self, terms: List[str]) -> List[float]:
        """Get IDFs for multiple terms efficiently."""
        return [self.get_idf(term) for term in terms]

    def clear(self):
        """Clear statistics."""
        self._doc_frequencies.clear()
        self._term_doc_freq.clear()
        self._doc_count = 0

    def get_stats(self) -> Dict:
        """Get statistics."""
        return {
            "doc_count": self._doc_count,
            "unique_terms": len(self._term_doc_freq),
        }
