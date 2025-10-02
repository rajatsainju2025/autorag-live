"""
Retrievers module for AutoRAG-Live.

This module provides various retrieval implementations including:
- BM25: Traditional keyword-based retrieval
- Dense: Embedding-based semantic retrieval
- Hybrid: Combination of BM25 and dense retrieval
- FAISS: Vector database adapter for dense retrieval
- Qdrant: Cloud-native vector database adapter
- Elasticsearch: Search engine adapter
"""


from .bm25 import BM25Retriever, bm25_retrieve
from .dense import dense_retrieve
from .elasticsearch_adapter import ElasticsearchRetriever
from .faiss_adapter import (
    DenseRetriever,
    SentenceTransformerRetriever,
    create_dense_retriever,
    load_retriever_index,
    save_retriever_index,
)
from .hybrid import hybrid_retrieve
from .qdrant_adapter import QdrantRetriever

__all__ = [
    "bm25_retrieve",
    "BM25Retriever",
    "dense_retrieve",
    "hybrid_retrieve",
    "DenseRetriever",
    "SentenceTransformerRetriever",
    "create_dense_retriever",
    "save_retriever_index",
    "load_retriever_index",
    "QdrantRetriever",
    "ElasticsearchRetriever",
]
