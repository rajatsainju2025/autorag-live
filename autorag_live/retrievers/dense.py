from typing import List

import numpy as np

# Try to import heavy deps; fall back to simple embedding if unavailable
try:  # pragma: no cover - import guard
    from sentence_transformers import SentenceTransformer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
except Exception:  # pragma: no cover - offline fallback
    SentenceTransformer = None  # type: ignore
    cosine_similarity = None  # type: ignore


def dense_retrieve(
    query: str, corpus: List[str], k: int, model_name: str = "all-MiniLM-L6-v2"
) -> List[str]:
    """
    Retrieves top-k documents from the corpus using a dense retriever.
    """
    if not corpus:
        return []

    if SentenceTransformer is not None and cosine_similarity is not None:
        model = SentenceTransformer(model_name)
        query_embedding = model.encode([query])
        corpus_embeddings = model.encode(corpus)
        sims = cosine_similarity(query_embedding, corpus_embeddings)[0]
    else:
        # Deterministic lightweight fallback: Jaccard on tokens as a proxy
        q_set = set(query.lower().split())
        sims = []
        for doc in corpus:
            d_set = set(doc.lower().split())
            inter = len(q_set & d_set)
            union = len(q_set | d_set) or 1
            sims.append(inter / union)
        sims = np.array(sims, dtype=float)

    top_k_indices = np.argsort(sims)[-k:][::-1]
    return [corpus[i] for i in top_k_indices]
