"""
Document scoring and ranking utilities for AutoRAG-Live.

Provides comprehensive document scoring and ranking capabilities
for retrieval and reranking stages.

Features:
- Multiple scoring algorithms
- Score combination and fusion
- Rank-based and score-based reranking
- Cross-encoder scoring
- Learning-to-rank utilities

Example usage:
    >>> scorer = DocumentScorer()
    >>> scores = scorer.score(
    ...     query="What is machine learning?",
    ...     documents=["ML is a subset of AI...", "Deep learning..."]
    ... )
    >>> ranked = scorer.rank(query, documents, top_k=5)
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ScoringMethod(str, Enum):
    """Available scoring methods."""
    
    BM25 = "bm25"
    TF_IDF = "tf_idf"
    JACCARD = "jaccard"
    OVERLAP = "overlap"
    SEMANTIC = "semantic"
    CROSS_ENCODER = "cross_encoder"
    COMBINED = "combined"


class FusionMethod(str, Enum):
    """Score fusion methods."""
    
    RRF = "rrf"  # Reciprocal Rank Fusion
    COMBSUM = "combsum"
    COMBMNZ = "combmnz"
    WEIGHTED = "weighted"
    MAX = "max"
    MIN = "min"


@dataclass
class ScoredDocument:
    """A document with its score."""
    
    document: str
    score: float
    index: int
    
    # Score breakdown
    score_components: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: ScoredDocument) -> bool:
        return self.score < other.score


@dataclass
class RankingResult:
    """Result of document ranking."""
    
    query: str
    ranked_documents: List[ScoredDocument]
    
    # Statistics
    num_documents: int = 0
    scoring_method: ScoringMethod = ScoringMethod.BM25
    
    @property
    def top_document(self) -> Optional[ScoredDocument]:
        """Get top-ranked document."""
        return self.ranked_documents[0] if self.ranked_documents else None
    
    def get_scores(self) -> List[float]:
        """Get list of scores."""
        return [d.score for d in self.ranked_documents]
    
    def get_documents(self) -> List[str]:
        """Get list of documents."""
        return [d.document for d in self.ranked_documents]


class BaseScorer(ABC):
    """Base class for document scorers."""
    
    @abstractmethod
    def score(
        self,
        query: str,
        document: str,
    ) -> float:
        """Score a single document."""
        pass
    
    def score_batch(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Score multiple documents."""
        return [self.score(query, doc) for doc in documents]


class BM25Scorer(BaseScorer):
    """
    BM25 scoring for document ranking.
    
    Example:
        >>> scorer = BM25Scorer()
        >>> score = scorer.score("machine learning", "ML is a subset of AI")
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25 scorer.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            epsilon: Floor for IDF
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # Corpus statistics (updated when scoring batch)
        self._corpus_size = 0
        self._avg_doc_len = 0
        self._doc_freqs: Dict[str, int] = {}
    
    def score(
        self,
        query: str,
        document: str,
    ) -> float:
        """Score using BM25."""
        query_terms = self._tokenize(query)
        doc_terms = self._tokenize(document)
        doc_len = len(doc_terms)
        
        # Use defaults if corpus not set
        avg_len = self._avg_doc_len if self._avg_doc_len > 0 else 100
        n_docs = max(self._corpus_size, 1)
        
        score = 0.0
        term_counts = Counter(doc_terms)
        
        for term in query_terms:
            if term not in term_counts:
                continue
            
            tf = term_counts[term]
            df = self._doc_freqs.get(term, 1)
            
            # IDF
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            idf = max(idf, self.epsilon)
            
            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
            
            score += idf * numerator / denominator
        
        return score
    
    def fit(self, documents: List[str]) -> None:
        """
        Fit scorer to corpus for IDF calculation.
        
        Args:
            documents: Corpus documents
        """
        self._corpus_size = len(documents)
        total_len = 0
        self._doc_freqs.clear()
        
        for doc in documents:
            terms = self._tokenize(doc)
            total_len += len(terms)
            
            unique_terms = set(terms)
            for term in unique_terms:
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1
        
        self._avg_doc_len = total_len / max(len(documents), 1)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return re.findall(r'\b\w+\b', text.lower())


class TFIDFScorer(BaseScorer):
    """TF-IDF scoring."""
    
    def __init__(self, smooth: bool = True):
        """
        Initialize TF-IDF scorer.
        
        Args:
            smooth: Whether to use smoothed IDF
        """
        self.smooth = smooth
        self._idf: Dict[str, float] = {}
        self._corpus_size = 0
    
    def score(
        self,
        query: str,
        document: str,
    ) -> float:
        """Score using TF-IDF."""
        query_terms = self._tokenize(query)
        doc_terms = self._tokenize(document)
        
        term_counts = Counter(doc_terms)
        doc_len = len(doc_terms)
        
        score = 0.0
        for term in query_terms:
            if term not in term_counts:
                continue
            
            # TF (normalized)
            tf = term_counts[term] / doc_len if doc_len > 0 else 0
            
            # IDF
            idf = self._idf.get(term, 1.0)
            
            score += tf * idf
        
        return score
    
    def fit(self, documents: List[str]) -> None:
        """Fit to corpus."""
        self._corpus_size = len(documents)
        doc_freqs: Dict[str, int] = {}
        
        for doc in documents:
            unique_terms = set(self._tokenize(doc))
            for term in unique_terms:
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
        
        # Calculate IDF
        n = self._corpus_size
        for term, df in doc_freqs.items():
            if self.smooth:
                self._idf[term] = math.log((n + 1) / (df + 1)) + 1
            else:
                self._idf[term] = math.log(n / df)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return re.findall(r'\b\w+\b', text.lower())


class JaccardScorer(BaseScorer):
    """Jaccard similarity scoring."""
    
    def score(
        self,
        query: str,
        document: str,
    ) -> float:
        """Score using Jaccard similarity."""
        query_terms = set(self._tokenize(query))
        doc_terms = set(self._tokenize(document))
        
        if not query_terms or not doc_terms:
            return 0.0
        
        intersection = len(query_terms & doc_terms)
        union = len(query_terms | doc_terms)
        
        return intersection / union if union > 0 else 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return re.findall(r'\b\w+\b', text.lower())


class OverlapScorer(BaseScorer):
    """Word overlap scoring."""
    
    def __init__(self, normalize: bool = True):
        """
        Initialize overlap scorer.
        
        Args:
            normalize: Normalize by query length
        """
        self.normalize = normalize
    
    def score(
        self,
        query: str,
        document: str,
    ) -> float:
        """Score using word overlap."""
        query_terms = set(self._tokenize(query))
        doc_terms = set(self._tokenize(document))
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms & doc_terms)
        
        if self.normalize:
            return overlap / len(query_terms)
        return float(overlap)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return re.findall(r'\b\w+\b', text.lower())


class SemanticScorer(BaseScorer):
    """Semantic similarity scoring using embeddings."""
    
    def __init__(
        self,
        embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        """
        Initialize semantic scorer.
        
        Args:
            embedding_func: Function to generate embeddings
        """
        self.embedding_func = embedding_func
    
    def score(
        self,
        query: str,
        document: str,
    ) -> float:
        """Score using semantic similarity."""
        if not self.embedding_func:
            # Fallback to overlap
            return OverlapScorer().score(query, document)
        
        embeddings = self.embedding_func([query, document])
        return self._cosine_similarity(embeddings[0], embeddings[1])
    
    def score_batch(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Batch semantic scoring."""
        if not self.embedding_func:
            return [OverlapScorer().score(query, doc) for doc in documents]
        
        texts = [query] + documents
        embeddings = self.embedding_func(texts)
        
        query_emb = embeddings[0]
        return [
            self._cosine_similarity(query_emb, doc_emb)
            for doc_emb in embeddings[1:]
        ]
    
    def _cosine_similarity(
        self,
        a: List[float],
        b: List[float],
    ) -> float:
        """Compute cosine similarity."""
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))


class CrossEncoderScorer(BaseScorer):
    """Cross-encoder scoring for precise relevance."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Initialize cross-encoder scorer.
        
        Args:
            model_name: Cross-encoder model name
        """
        self.model_name = model_name
        self._model = None
    
    def _load_model(self):
        """Lazy load model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not available")
                self._model = "fallback"
    
    def score(
        self,
        query: str,
        document: str,
    ) -> float:
        """Score using cross-encoder."""
        self._load_model()
        
        if self._model == "fallback":
            return OverlapScorer().score(query, document)
        
        return float(self._model.predict([(query, document)])[0])
    
    def score_batch(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Batch cross-encoder scoring."""
        self._load_model()
        
        if self._model == "fallback":
            return [OverlapScorer().score(query, doc) for doc in documents]
        
        pairs = [(query, doc) for doc in documents]
        return [float(s) for s in self._model.predict(pairs)]


class ScoreFusion:
    """
    Fuse scores from multiple rankers.
    
    Example:
        >>> fusion = ScoreFusion(method=FusionMethod.RRF)
        >>> combined = fusion.fuse([
        ...     [("doc1", 0.9), ("doc2", 0.8)],
        ...     [("doc2", 0.95), ("doc1", 0.7)]
        ... ])
    """
    
    def __init__(
        self,
        method: FusionMethod = FusionMethod.RRF,
        weights: Optional[List[float]] = None,
        k: int = 60,
    ):
        """
        Initialize score fusion.
        
        Args:
            method: Fusion method
            weights: Weights for each ranker
            k: Parameter for RRF
        """
        self.method = method
        self.weights = weights
        self.k = k
    
    def fuse(
        self,
        rankings: List[List[Tuple[str, float]]],
    ) -> List[Tuple[str, float]]:
        """
        Fuse multiple rankings.
        
        Args:
            rankings: List of (doc_id, score) lists
            
        Returns:
            Fused ranking
        """
        if self.method == FusionMethod.RRF:
            return self._rrf(rankings)
        elif self.method == FusionMethod.COMBSUM:
            return self._combsum(rankings)
        elif self.method == FusionMethod.COMBMNZ:
            return self._combmnz(rankings)
        elif self.method == FusionMethod.WEIGHTED:
            return self._weighted(rankings)
        elif self.method == FusionMethod.MAX:
            return self._max_fusion(rankings)
        else:
            return self._combsum(rankings)
    
    def _rrf(
        self,
        rankings: List[List[Tuple[str, float]]],
    ) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion."""
        doc_scores: Dict[str, float] = {}
        
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking, 1):
                rrf_score = 1 / (self.k + rank)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
        
        result = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return result
    
    def _combsum(
        self,
        rankings: List[List[Tuple[str, float]]],
    ) -> List[Tuple[str, float]]:
        """CombSUM fusion."""
        doc_scores: Dict[str, float] = {}
        
        # Normalize scores within each ranking
        for ranking in rankings:
            if not ranking:
                continue
            
            max_score = max(s for _, s in ranking)
            min_score = min(s for _, s in ranking)
            score_range = max_score - min_score
            
            for doc_id, score in ranking:
                if score_range > 0:
                    norm_score = (score - min_score) / score_range
                else:
                    norm_score = 1.0
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + norm_score
        
        result = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return result
    
    def _combmnz(
        self,
        rankings: List[List[Tuple[str, float]]],
    ) -> List[Tuple[str, float]]:
        """CombMNZ fusion (CombSUM * number of non-zero scores)."""
        doc_scores: Dict[str, float] = {}
        doc_counts: Dict[str, int] = {}
        
        for ranking in rankings:
            if not ranking:
                continue
            
            max_score = max(s for _, s in ranking)
            min_score = min(s for _, s in ranking)
            score_range = max_score - min_score
            
            for doc_id, score in ranking:
                if score_range > 0:
                    norm_score = (score - min_score) / score_range
                else:
                    norm_score = 1.0
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + norm_score
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        # Multiply by count
        final_scores = {
            doc_id: score * doc_counts[doc_id]
            for doc_id, score in doc_scores.items()
        }
        
        result = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return result
    
    def _weighted(
        self,
        rankings: List[List[Tuple[str, float]]],
    ) -> List[Tuple[str, float]]:
        """Weighted combination."""
        weights = self.weights or [1.0] * len(rankings)
        doc_scores: Dict[str, float] = {}
        
        for weight, ranking in zip(weights, rankings):
            if not ranking:
                continue
            
            max_score = max(s for _, s in ranking)
            min_score = min(s for _, s in ranking)
            score_range = max_score - min_score
            
            for doc_id, score in ranking:
                if score_range > 0:
                    norm_score = (score - min_score) / score_range
                else:
                    norm_score = 1.0
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + weight * norm_score
        
        result = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return result
    
    def _max_fusion(
        self,
        rankings: List[List[Tuple[str, float]]],
    ) -> List[Tuple[str, float]]:
        """Max score fusion."""
        doc_scores: Dict[str, float] = {}
        
        for ranking in rankings:
            if not ranking:
                continue
            
            max_score = max(s for _, s in ranking)
            min_score = min(s for _, s in ranking)
            score_range = max_score - min_score
            
            for doc_id, score in ranking:
                if score_range > 0:
                    norm_score = (score - min_score) / score_range
                else:
                    norm_score = 1.0
                doc_scores[doc_id] = max(doc_scores.get(doc_id, 0), norm_score)
        
        result = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return result


class DocumentScorer:
    """
    Main document scoring interface.
    
    Example:
        >>> scorer = DocumentScorer(method=ScoringMethod.BM25)
        >>> 
        >>> # Score documents
        >>> result = scorer.score_and_rank(
        ...     query="machine learning algorithms",
        ...     documents=["ML uses algorithms...", "Deep learning..."],
        ...     top_k=5
        ... )
        >>> 
        >>> for doc in result.ranked_documents:
        ...     print(f"{doc.score:.3f}: {doc.document[:50]}...")
    """
    
    def __init__(
        self,
        method: ScoringMethod = ScoringMethod.BM25,
        scorer: Optional[BaseScorer] = None,
    ):
        """
        Initialize document scorer.
        
        Args:
            method: Scoring method
            scorer: Custom scorer instance
        """
        self.method = method
        
        if scorer:
            self.scorer = scorer
        else:
            self.scorer = self._create_scorer(method)
    
    def _create_scorer(self, method: ScoringMethod) -> BaseScorer:
        """Create scorer for method."""
        scorers = {
            ScoringMethod.BM25: BM25Scorer,
            ScoringMethod.TF_IDF: TFIDFScorer,
            ScoringMethod.JACCARD: JaccardScorer,
            ScoringMethod.OVERLAP: OverlapScorer,
            ScoringMethod.SEMANTIC: SemanticScorer,
            ScoringMethod.CROSS_ENCODER: CrossEncoderScorer,
        }
        return scorers.get(method, BM25Scorer)()
    
    def score(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """
        Score documents.
        
        Args:
            query: Query string
            documents: Documents to score
            
        Returns:
            List of scores
        """
        return self.scorer.score_batch(query, documents)
    
    def score_and_rank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> RankingResult:
        """
        Score and rank documents.
        
        Args:
            query: Query string
            documents: Documents to rank
            top_k: Return top K documents
            metadata: Optional metadata per document
            
        Returns:
            RankingResult
        """
        # Fit corpus for BM25/TF-IDF if applicable
        if isinstance(self.scorer, (BM25Scorer, TFIDFScorer)):
            self.scorer.fit(documents)
        
        # Score all documents
        scores = self.scorer.score_batch(query, documents)
        
        # Create scored documents
        scored_docs = [
            ScoredDocument(
                document=doc,
                score=score,
                index=i,
                metadata=metadata[i] if metadata and i < len(metadata) else {},
            )
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        
        # Sort by score
        scored_docs.sort(key=lambda d: d.score, reverse=True)
        
        # Apply top_k
        if top_k:
            scored_docs = scored_docs[:top_k]
        
        return RankingResult(
            query=query,
            ranked_documents=scored_docs,
            num_documents=len(documents),
            scoring_method=self.method,
        )
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        initial_scores: Optional[List[float]] = None,
        alpha: float = 0.6,
        top_k: Optional[int] = None,
    ) -> RankingResult:
        """
        Rerank documents with optional score combination.
        
        Args:
            query: Query string
            documents: Documents to rerank
            initial_scores: Initial retrieval scores
            alpha: Weight for new scores (vs initial)
            top_k: Return top K documents
            
        Returns:
            RankingResult
        """
        # Get new scores
        new_scores = self.score(query, documents)
        
        # Combine with initial scores if provided
        if initial_scores:
            # Normalize both
            new_max = max(new_scores) if new_scores else 1
            init_max = max(initial_scores) if initial_scores else 1
            
            final_scores = [
                alpha * (n / max(new_max, 1e-10)) + (1 - alpha) * (i / max(init_max, 1e-10))
                for n, i in zip(new_scores, initial_scores)
            ]
        else:
            final_scores = new_scores
        
        # Create scored documents
        scored_docs = [
            ScoredDocument(
                document=doc,
                score=score,
                index=i,
                score_components={
                    "rerank_score": new_scores[i],
                    "initial_score": initial_scores[i] if initial_scores else 0,
                },
            )
            for i, (doc, score) in enumerate(zip(documents, final_scores))
        ]
        
        scored_docs.sort(key=lambda d: d.score, reverse=True)
        
        if top_k:
            scored_docs = scored_docs[:top_k]
        
        return RankingResult(
            query=query,
            ranked_documents=scored_docs,
            num_documents=len(documents),
            scoring_method=self.method,
        )


class MultiMethodScorer:
    """
    Score using multiple methods and fuse results.
    
    Example:
        >>> scorer = MultiMethodScorer(
        ...     methods=[ScoringMethod.BM25, ScoringMethod.SEMANTIC],
        ...     fusion=FusionMethod.RRF
        ... )
        >>> result = scorer.score_and_rank(query, documents)
    """
    
    def __init__(
        self,
        methods: List[ScoringMethod],
        fusion: FusionMethod = FusionMethod.RRF,
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize multi-method scorer.
        
        Args:
            methods: Scoring methods to use
            fusion: How to fuse scores
            weights: Weights per method
        """
        self.methods = methods
        self.scorers = [
            DocumentScorer(method=m)
            for m in methods
        ]
        self.fusion = ScoreFusion(method=fusion, weights=weights)
    
    def score_and_rank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> RankingResult:
        """Score and rank using multiple methods."""
        # Get rankings from each method
        rankings = []
        for scorer in self.scorers:
            result = scorer.score_and_rank(query, documents)
            ranking = [
                (str(d.index), d.score)
                for d in result.ranked_documents
            ]
            rankings.append(ranking)
        
        # Fuse rankings
        fused = self.fusion.fuse(rankings)
        
        # Create result
        scored_docs = []
        for doc_idx, score in fused:
            idx = int(doc_idx)
            scored_docs.append(ScoredDocument(
                document=documents[idx],
                score=score,
                index=idx,
            ))
        
        if top_k:
            scored_docs = scored_docs[:top_k]
        
        return RankingResult(
            query=query,
            ranked_documents=scored_docs,
            num_documents=len(documents),
            scoring_method=ScoringMethod.COMBINED,
        )


# Convenience functions

def score_documents(
    query: str,
    documents: List[str],
    method: ScoringMethod = ScoringMethod.BM25,
) -> List[float]:
    """
    Score documents.
    
    Args:
        query: Query string
        documents: Documents to score
        method: Scoring method
        
    Returns:
        List of scores
    """
    scorer = DocumentScorer(method=method)
    return scorer.score(query, documents)


def rank_documents(
    query: str,
    documents: List[str],
    top_k: int = 10,
    method: ScoringMethod = ScoringMethod.BM25,
) -> List[Tuple[str, float]]:
    """
    Rank documents.
    
    Args:
        query: Query string
        documents: Documents to rank
        top_k: Number of results
        method: Scoring method
        
    Returns:
        List of (document, score) tuples
    """
    scorer = DocumentScorer(method=method)
    result = scorer.score_and_rank(query, documents, top_k=top_k)
    return [(d.document, d.score) for d in result.ranked_documents]


def fuse_rankings(
    rankings: List[List[Tuple[str, float]]],
    method: FusionMethod = FusionMethod.RRF,
) -> List[Tuple[str, float]]:
    """
    Fuse multiple rankings.
    
    Args:
        rankings: List of rankings
        method: Fusion method
        
    Returns:
        Fused ranking
    """
    fusion = ScoreFusion(method=method)
    return fusion.fuse(rankings)
