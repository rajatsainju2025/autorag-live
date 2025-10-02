from typing import List

from . import bm25, dense


def hybrid_retrieve(query: str, corpus: List[str], k: int, bm25_weight: float = 0.5) -> List[str]:
    """
    Retrieves top-k documents from the corpus using a hybrid of BM25 and dense retrieval.
    """
    # Placeholder implementation
    bm25_results = bm25.bm25_retrieve(query, corpus, k)
    dense_results = dense.dense_retrieve(query, corpus, k)

    # Simple interleaving for now
    results = []
    for i in range(k):
        if i % 2 == 0 and bm25_results:
            results.append(bm25_results.pop(0))
        elif dense_results:
            results.append(dense_results.pop(0))
    return results
