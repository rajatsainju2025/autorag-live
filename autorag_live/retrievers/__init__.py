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

from typing import List, Tuple, Optional, Dict, Any

from ..types.types import (
    QueryText,
    DocumentText,
    RetrievalResult,
    Retriever,
    RetrieverError
)

from .bm25 import bm25_retrieve
from .dense import dense_retrieve
from .hybrid import hybrid_retrieve
from .faiss_adapter import (
    DenseRetriever,
    SentenceTransformerRetriever,
    create_dense_retriever,
    save_retriever_index,
    load_retriever_index
)
from .qdrant_adapter import QdrantRetriever
from .elasticsearch_adapter import ElasticsearchRetriever

__all__ = [
    "bm25_retrieve",
    "dense_retrieve",
    "hybrid_retrieve",
    "DenseRetriever",
    "SentenceTransformerRetriever",
    "create_dense_retriever",
    "save_retriever_index",
    "load_retriever_index",
    "QdrantRetriever",
    "ElasticsearchRetriever"
]