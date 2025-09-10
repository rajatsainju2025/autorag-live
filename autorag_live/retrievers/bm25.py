from typing import List
from rank_bm25 import BM25Okapi

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
