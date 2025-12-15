"""
Semantic search utilities for AutoRAG-Live.

Provides embedding-based semantic search with support for
multiple embedding models and efficient similarity computation.

Features:
- Embedding generation with multiple providers
- Vector similarity search
- Approximate nearest neighbor search
- Hybrid search (semantic + lexical)
- Query-document matching

Example usage:
    >>> searcher = SemanticSearcher(embedding_model="default")
    >>> results = searcher.search(
    ...     query="What is machine learning?",
    ...     documents=["ML is a subset of AI...", "Deep learning..."]
    ... )
    >>> for doc, score in results:
    ...     print(f"Score: {score:.3f} - {doc[:50]}...")
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class SimilarityMetric(str, Enum):
    """Supported similarity metrics."""
    
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class EmbeddingProvider(str, Enum):
    """Embedding model providers."""
    
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    CUSTOM = "custom"


@dataclass
class SearchResult:
    """Result from semantic search."""
    
    document: str
    score: float
    index: int
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: SearchResult) -> bool:
        return self.score < other.score


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    
    provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True
    
    # Provider-specific options
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Cache settings
    cache_embeddings: bool = True


class BaseEmbedder(ABC):
    """Base class for embedding models."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of shape (len(texts), dimension)
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Array of shape (dimension,)
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing using TF-IDF like approach."""
    
    def __init__(self, dimension: int = 384):
        """Initialize mock embedder."""
        self._dimension = dimension
        self._vocab: Dict[str, int] = {}
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings."""
        embeddings = []
        for text in texts:
            embedding = self._text_to_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate mock query embedding."""
        return self._text_to_embedding(query)
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to pseudo-embedding."""
        # Create deterministic embedding from text
        words = text.lower().split()
        embedding = np.zeros(self._dimension)
        
        for word in words:
            # Hash word to get index
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = word_hash % self._dimension
            embedding[idx] += 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using sentence-transformers library."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize sentence transformer embedder.
        
        Args:
            model_name: Model name from HuggingFace
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                logger.warning("sentence-transformers not installed, using mock embedder")
                self._model = MockEmbedder()
                self._dimension = 384
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        self._load_model()
        
        if isinstance(self._model, MockEmbedder):
            return self._model.embed(texts)
        
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query."""
        embeddings = self.embed([query])
        return embeddings[0]
    
    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI API."""
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100,
    ):
        """
        Initialize OpenAI embedder.
        
        Args:
            model: OpenAI embedding model
            api_key: API key
            batch_size: Batch size for API calls
        """
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self._dimension = self._get_dimension()
    
    def _get_dimension(self) -> int:
        """Get dimension for model."""
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model, 1536)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                response = client.embeddings.create(model=self.model, input=batch)
                embeddings = [e.embedding for e in response.data]
                all_embeddings.extend(embeddings)
            
            return np.array(all_embeddings)
        except ImportError:
            logger.warning("openai not installed, using mock embedder")
            mock = MockEmbedder(self._dimension)
            return mock.embed(texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query."""
        embeddings = self.embed([query])
        return embeddings[0]
    
    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingCache:
    """Cache for embeddings."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum cache size
        """
        self.max_size = max_size
        self._cache: Dict[str, np.ndarray] = {}
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = self._hash(text)
        return self._cache.get(key)
    
    def set(self, text: str, embedding: np.ndarray) -> None:
        """Cache embedding."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        key = self._hash(text)
        self._cache[key] = embedding
    
    def _hash(self, text: str) -> str:
        """Create cache key."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()


class VectorSimilarity:
    """Compute vector similarity using various metrics."""
    
    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    @staticmethod
    def dot_product(a: np.ndarray, b: np.ndarray) -> float:
        """Compute dot product similarity."""
        return float(np.dot(a, b))
    
    @staticmethod
    def euclidean(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance (as similarity)."""
        distance = np.linalg.norm(a - b)
        return 1.0 / (1.0 + distance)
    
    @staticmethod
    def manhattan(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Manhattan distance (as similarity)."""
        distance = np.sum(np.abs(a - b))
        return 1.0 / (1.0 + distance)
    
    @classmethod
    def compute(
        cls,
        a: np.ndarray,
        b: np.ndarray,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> float:
        """
        Compute similarity using specified metric.
        
        Args:
            a: First vector
            b: Second vector
            metric: Similarity metric
            
        Returns:
            Similarity score
        """
        methods = {
            SimilarityMetric.COSINE: cls.cosine,
            SimilarityMetric.DOT_PRODUCT: cls.dot_product,
            SimilarityMetric.EUCLIDEAN: cls.euclidean,
            SimilarityMetric.MANHATTAN: cls.manhattan,
        }
        return methods[metric](a, b)
    
    @classmethod
    def batch_similarity(
        cls,
        query: np.ndarray,
        documents: np.ndarray,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> np.ndarray:
        """
        Compute similarity between query and multiple documents.
        
        Args:
            query: Query vector (dimension,)
            documents: Document vectors (n_docs, dimension)
            metric: Similarity metric
            
        Returns:
            Array of similarity scores
        """
        if metric == SimilarityMetric.COSINE:
            # Efficient batch cosine similarity
            query_norm = np.linalg.norm(query)
            doc_norms = np.linalg.norm(documents, axis=1)
            
            # Avoid division by zero
            query_norm = max(query_norm, 1e-10)
            doc_norms = np.maximum(doc_norms, 1e-10)
            
            dots = np.dot(documents, query)
            return dots / (doc_norms * query_norm)
        
        elif metric == SimilarityMetric.DOT_PRODUCT:
            return np.dot(documents, query)
        
        elif metric == SimilarityMetric.EUCLIDEAN:
            distances = np.linalg.norm(documents - query, axis=1)
            return 1.0 / (1.0 + distances)
        
        else:  # Manhattan
            distances = np.sum(np.abs(documents - query), axis=1)
            return 1.0 / (1.0 + distances)


class SemanticSearcher:
    """
    Main semantic search interface.
    
    Example:
        >>> searcher = SemanticSearcher()
        >>> 
        >>> # Index documents
        >>> documents = [
        ...     "Machine learning is a subset of artificial intelligence.",
        ...     "Deep learning uses neural networks with many layers.",
        ...     "Natural language processing handles text data.",
        ... ]
        >>> searcher.index(documents)
        >>> 
        >>> # Search
        >>> results = searcher.search("What is deep learning?", top_k=2)
        >>> for result in results:
        ...     print(f"{result.score:.3f}: {result.document[:50]}...")
    """
    
    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        cache_embeddings: bool = True,
    ):
        """
        Initialize semantic searcher.
        
        Args:
            embedder: Embedding model (default: SentenceTransformerEmbedder)
            metric: Similarity metric
            cache_embeddings: Whether to cache embeddings
        """
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.metric = metric
        
        self.cache = EmbeddingCache() if cache_embeddings else None
        
        # Index storage
        self._documents: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._metadata: List[Dict[str, Any]] = []
    
    def index(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Index documents for search.
        
        Args:
            documents: Documents to index
            metadata: Optional metadata for each document
        """
        self._documents = documents
        self._embeddings = self.embedder.embed(documents)
        self._metadata = metadata or [{} for _ in documents]
        
        # Cache embeddings
        if self.cache:
            for doc, emb in zip(documents, self._embeddings):
                self.cache.set(doc, emb)
        
        logger.info(f"Indexed {len(documents)} documents")
    
    def search(
        self,
        query: str,
        documents: Optional[List[str]] = None,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            documents: Documents to search (uses indexed if None)
            top_k: Number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of SearchResult
        """
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        
        # Get document embeddings
        if documents is not None:
            doc_embeddings = self._get_document_embeddings(documents)
            doc_list = documents
            metadata_list = [{} for _ in documents]
        else:
            if self._embeddings is None:
                raise ValueError("No documents indexed. Call index() first or provide documents.")
            doc_embeddings = self._embeddings
            doc_list = self._documents
            metadata_list = self._metadata
        
        # Compute similarities
        scores = VectorSimilarity.batch_similarity(
            query_embedding, doc_embeddings, self.metric
        )
        
        # Create results
        results = []
        for idx, (doc, score) in enumerate(zip(doc_list, scores)):
            if score >= threshold:
                results.append(SearchResult(
                    document=doc,
                    score=float(score),
                    index=idx,
                    embedding=doc_embeddings[idx],
                    metadata=metadata_list[idx],
                ))
        
        # Sort by score and return top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[List[SearchResult]]:
        """
        Search for multiple queries.
        
        Args:
            queries: Search queries
            top_k: Number of results per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of result lists
        """
        return [self.search(q, top_k=top_k, threshold=threshold) for q in queries]
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for query, using cache if available."""
        if self.cache:
            cached = self.cache.get(query)
            if cached is not None:
                return cached
        
        embedding = self.embedder.embed_query(query)
        
        if self.cache:
            self.cache.set(query, embedding)
        
        return embedding
    
    def _get_document_embeddings(self, documents: List[str]) -> np.ndarray:
        """Get embeddings for documents, using cache when available."""
        if not self.cache:
            return self.embedder.embed(documents)
        
        embeddings = []
        uncached_docs = []
        uncached_indices = []
        
        for i, doc in enumerate(documents):
            cached = self.cache.get(doc)
            if cached is not None:
                embeddings.append((i, cached))
            else:
                uncached_docs.append(doc)
                uncached_indices.append(i)
        
        # Embed uncached documents
        if uncached_docs:
            new_embeddings = self.embedder.embed(uncached_docs)
            for idx, emb in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, emb))
                self.cache.set(documents[idx], emb)
        
        # Sort by original index
        embeddings.sort(key=lambda x: x[0])
        return np.array([e[1] for e in embeddings])


class HybridSearcher:
    """
    Hybrid search combining semantic and lexical search.
    
    Example:
        >>> searcher = HybridSearcher()
        >>> results = searcher.search(
        ...     query="machine learning algorithms",
        ...     documents=["ML uses algorithms...", "Deep learning..."],
        ...     semantic_weight=0.7,
        ... )
    """
    
    def __init__(
        self,
        semantic_searcher: Optional[SemanticSearcher] = None,
        lexical_scorer: Optional[Callable[[str, str], float]] = None,
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            semantic_searcher: Semantic search component
            lexical_scorer: Function for lexical scoring
        """
        self.semantic = semantic_searcher or SemanticSearcher()
        self.lexical_scorer = lexical_scorer or self._bm25_score
    
    def search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        semantic_weight: float = 0.7,
    ) -> List[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            documents: Documents to search
            top_k: Number of results
            semantic_weight: Weight for semantic scores (0-1)
            
        Returns:
            List of SearchResult
        """
        # Get semantic scores
        semantic_results = self.semantic.search(
            query, documents, top_k=len(documents)
        )
        semantic_scores = {r.index: r.score for r in semantic_results}
        
        # Get lexical scores
        lexical_scores = {
            i: self.lexical_scorer(query, doc)
            for i, doc in enumerate(documents)
        }
        
        # Normalize scores
        semantic_scores = self._normalize_scores(semantic_scores)
        lexical_scores = self._normalize_scores(lexical_scores)
        
        # Combine scores
        lexical_weight = 1 - semantic_weight
        combined_scores = {}
        
        for idx in range(len(documents)):
            sem_score = semantic_scores.get(idx, 0)
            lex_score = lexical_scores.get(idx, 0)
            combined_scores[idx] = semantic_weight * sem_score + lexical_weight * lex_score
        
        # Create results
        results = [
            SearchResult(
                document=documents[idx],
                score=score,
                index=idx,
            )
            for idx, score in combined_scores.items()
        ]
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return scores
        
        min_score = min(scores.values())
        max_score = max(scores.values())
        
        if max_score == min_score:
            return {k: 1.0 for k in scores}
        
        return {
            k: (v - min_score) / (max_score - min_score)
            for k, v in scores.items()
        }
    
    def _bm25_score(
        self,
        query: str,
        document: str,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """
        Simple BM25-like scoring.
        
        Args:
            query: Query text
            document: Document text
            k1: Term frequency saturation
            b: Length normalization
            
        Returns:
            BM25 score
        """
        query_terms = set(query.lower().split())
        doc_terms = document.lower().split()
        doc_len = len(doc_terms)
        avg_doc_len = 100  # Assumed average
        
        score = 0.0
        for term in query_terms:
            tf = doc_terms.count(term)
            if tf > 0:
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
                score += numerator / denominator
        
        return score


class MaximalMarginalRelevance:
    """
    Maximal Marginal Relevance for diverse retrieval.
    
    Example:
        >>> mmr = MaximalMarginalRelevance(lambda_param=0.5)
        >>> diverse_results = mmr.select(
        ...     query_embedding=query_emb,
        ...     document_embeddings=doc_embs,
        ...     documents=docs,
        ...     top_k=5,
        ... )
    """
    
    def __init__(self, lambda_param: float = 0.5):
        """
        Initialize MMR.
        
        Args:
            lambda_param: Trade-off between relevance and diversity
        """
        self.lambda_param = lambda_param
    
    def select(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        documents: List[str],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Select diverse documents using MMR.
        
        Args:
            query_embedding: Query embedding
            document_embeddings: Document embeddings
            documents: Document texts
            top_k: Number of results
            
        Returns:
            Diverse search results
        """
        # Compute query-document similarities
        query_sims = VectorSimilarity.batch_similarity(
            query_embedding, document_embeddings, SimilarityMetric.COSINE
        )
        
        # Greedy selection
        selected = []
        selected_indices = set()
        
        for _ in range(min(top_k, len(documents))):
            best_idx = -1
            best_score = float('-inf')
            
            for idx in range(len(documents)):
                if idx in selected_indices:
                    continue
                
                # Relevance to query
                relevance = query_sims[idx]
                
                # Similarity to selected documents
                if selected_indices:
                    max_sim_to_selected = max(
                        VectorSimilarity.cosine(
                            document_embeddings[idx],
                            document_embeddings[sel_idx]
                        )
                        for sel_idx in selected_indices
                    )
                else:
                    max_sim_to_selected = 0
                
                # MMR score
                mmr_score = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * max_sim_to_selected
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx >= 0:
                selected_indices.add(best_idx)
                selected.append(SearchResult(
                    document=documents[best_idx],
                    score=float(query_sims[best_idx]),
                    index=best_idx,
                    embedding=document_embeddings[best_idx],
                ))
        
        return selected


class DocumentReranker:
    """
    Rerank documents based on relevance to query.
    
    Example:
        >>> reranker = DocumentReranker()
        >>> reranked = reranker.rerank(
        ...     query="machine learning",
        ...     documents=["doc1", "doc2", "doc3"],
        ...     initial_scores=[0.5, 0.8, 0.3],
        ... )
    """
    
    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        rerank_weight: float = 0.6,
    ):
        """
        Initialize reranker.
        
        Args:
            embedder: Embedding model for reranking
            rerank_weight: Weight for reranking score
        """
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.rerank_weight = rerank_weight
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        initial_scores: Optional[List[float]] = None,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Rerank documents.
        
        Args:
            query: Query text
            documents: Documents to rerank
            initial_scores: Initial retrieval scores
            top_k: Return top K results
            
        Returns:
            Reranked results
        """
        if initial_scores is None:
            initial_scores = [1.0] * len(documents)
        
        # Compute semantic similarity
        query_emb = self.embedder.embed_query(query)
        doc_embs = self.embedder.embed(documents)
        semantic_scores = VectorSimilarity.batch_similarity(
            query_emb, doc_embs, SimilarityMetric.COSINE
        )
        
        # Normalize initial scores
        if initial_scores:
            max_init = max(initial_scores)
            if max_init > 0:
                initial_scores = [s / max_init for s in initial_scores]
        
        # Combine scores
        initial_weight = 1 - self.rerank_weight
        final_scores = [
            self.rerank_weight * semantic_scores[i] + initial_weight * initial_scores[i]
            for i in range(len(documents))
        ]
        
        # Create results
        results = [
            SearchResult(
                document=doc,
                score=score,
                index=idx,
                embedding=doc_embs[idx],
            )
            for idx, (doc, score) in enumerate(zip(documents, final_scores))
        ]
        
        results.sort(key=lambda r: r.score, reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results


# Convenience functions

def semantic_search(
    query: str,
    documents: List[str],
    top_k: int = 10,
) -> List[SearchResult]:
    """
    Perform semantic search.
    
    Args:
        query: Search query
        documents: Documents to search
        top_k: Number of results
        
    Returns:
        Search results
    """
    searcher = SemanticSearcher()
    return searcher.search(query, documents, top_k=top_k)


def hybrid_search(
    query: str,
    documents: List[str],
    top_k: int = 10,
    semantic_weight: float = 0.7,
) -> List[SearchResult]:
    """
    Perform hybrid search.
    
    Args:
        query: Search query
        documents: Documents to search
        top_k: Number of results
        semantic_weight: Weight for semantic scores
        
    Returns:
        Search results
    """
    searcher = HybridSearcher()
    return searcher.search(query, documents, top_k=top_k, semantic_weight=semantic_weight)


def compute_similarity(
    text1: str,
    text2: str,
    embedder: Optional[BaseEmbedder] = None,
) -> float:
    """
    Compute semantic similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        embedder: Embedding model
        
    Returns:
        Similarity score
    """
    embedder = embedder or SentenceTransformerEmbedder()
    embeddings = embedder.embed([text1, text2])
    return float(VectorSimilarity.cosine(embeddings[0], embeddings[1]))
