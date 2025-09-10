from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def dense_retrieve(query: str, corpus: List[str], k: int, model_name: str = 'all-MiniLM-L6-v2') -> List[str]:
    """
    Retrieves top-k documents from the corpus using a dense retriever.
    """
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    corpus_embeddings = model.encode(corpus)
    
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    return [corpus[i] for i in top_k_indices]
