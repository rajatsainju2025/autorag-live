"""
Result aggregation utilities for AutoRAG-Live.

Provides intelligent aggregation of results from multiple retrievers,
with support for various fusion strategies and score normalization.

Features:
- Multi-retriever result fusion
- Score normalization strategies
- Rank-based fusion (RRF, Borda)
- Score-based fusion (weighted, max, mean)
- Duplicate detection and merging
- Result diversity optimization

Example usage:
    >>> aggregator = ResultAggregator(strategy="rrf")
    >>> combined = aggregator.aggregate([
    ...     sparse_results,
    ...     dense_results,
    ...     hybrid_results
    ... ])
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """Result fusion strategies."""
    
    # Rank-based
    RRF = "rrf"                    # Reciprocal Rank Fusion
    BORDA = "borda"                # Borda Count
    CONDORCET = "condorcet"        # Condorcet method
    
    # Score-based
    LINEAR = "linear"              # Weighted linear combination
    MAX = "max"                    # Maximum score
    MIN = "min"                    # Minimum score
    MEAN = "mean"                  # Average score
    GEOMETRIC = "geometric"        # Geometric mean
    HARMONIC = "harmonic"          # Harmonic mean
    
    # Advanced
    LEARNED = "learned"            # Learned weights
    ADAPTIVE = "adaptive"          # Query-adaptive fusion


class NormalizationMethod(str, Enum):
    """Score normalization methods."""
    
    NONE = "none"
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    PERCENTILE = "percentile"
    SOFTMAX = "softmax"


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""
    
    id: str
    content: str
    score: float = 0.0
    
    # Source information
    source: str = ""
    retriever: str = ""
    
    # Ranking information
    rank: int = 0
    original_rank: int = 0
    
    # Additional scores
    scores: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, RetrievedDocument):
            return self.id == other.id
        return False


@dataclass
class RetrievalResult:
    """Result from a single retriever."""
    
    documents: List[RetrievedDocument]
    retriever_name: str
    
    # Metadata
    query: str = ""
    latency: float = 0.0
    total_found: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def count(self) -> int:
        """Get document count."""
        return len(self.documents)
    
    def get_ids(self) -> Set[str]:
        """Get set of document IDs."""
        return {doc.id for doc in self.documents}


@dataclass
class AggregatedResult:
    """Aggregated result from multiple retrievers."""
    
    documents: List[RetrievedDocument]
    
    # Aggregation info
    strategy: FusionStrategy = FusionStrategy.RRF
    source_counts: Dict[str, int] = field(default_factory=dict)
    
    # Statistics
    total_unique: int = 0
    total_duplicates: int = 0
    overlap_ratio: float = 0.0
    
    # Component results
    component_results: List[RetrievalResult] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def count(self) -> int:
        """Get document count."""
        return len(self.documents)
    
    def top_k(self, k: int) -> List[RetrievedDocument]:
        """Get top k documents."""
        return self.documents[:k]


class ScoreNormalizer:
    """Normalize scores across retrievers."""
    
    def __init__(
        self,
        method: NormalizationMethod = NormalizationMethod.MIN_MAX,
    ):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method
        """
        self.method = method
    
    def normalize(
        self,
        scores: List[float],
        reference_scores: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Normalize scores.
        
        Args:
            scores: Scores to normalize
            reference_scores: Optional reference for normalization
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        ref = reference_scores if reference_scores else scores
        
        if self.method == NormalizationMethod.NONE:
            return scores
        elif self.method == NormalizationMethod.MIN_MAX:
            return self._min_max(scores, ref)
        elif self.method == NormalizationMethod.Z_SCORE:
            return self._z_score(scores, ref)
        elif self.method == NormalizationMethod.PERCENTILE:
            return self._percentile(scores, ref)
        elif self.method == NormalizationMethod.SOFTMAX:
            return self._softmax(scores)
        
        return scores
    
    def _min_max(
        self,
        scores: List[float],
        ref: List[float],
    ) -> List[float]:
        """Min-max normalization."""
        min_val = min(ref)
        max_val = max(ref)
        
        if max_val == min_val:
            return [0.5] * len(scores)
        
        return [(s - min_val) / (max_val - min_val) for s in scores]
    
    def _z_score(
        self,
        scores: List[float],
        ref: List[float],
    ) -> List[float]:
        """Z-score normalization."""
        mean = sum(ref) / len(ref)
        variance = sum((s - mean) ** 2 for s in ref) / len(ref)
        std = variance ** 0.5
        
        if std == 0:
            return [0.0] * len(scores)
        
        return [(s - mean) / std for s in scores]
    
    def _percentile(
        self,
        scores: List[float],
        ref: List[float],
    ) -> List[float]:
        """Percentile normalization."""
        sorted_ref = sorted(ref)
        n = len(sorted_ref)
        
        result = []
        for s in scores:
            # Find percentile
            count = sum(1 for r in sorted_ref if r <= s)
            result.append(count / n)
        
        return result
    
    def _softmax(self, scores: List[float]) -> List[float]:
        """Softmax normalization."""
        import math
        
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        
        return [e / total for e in exp_scores]


class DuplicateDetector:
    """Detect duplicate documents."""
    
    def __init__(
        self,
        similarity_threshold: float = 0.9,
        use_content_hash: bool = True,
    ):
        """
        Initialize detector.
        
        Args:
            similarity_threshold: Threshold for considering duplicates
            use_content_hash: Use content hash for detection
        """
        self.similarity_threshold = similarity_threshold
        self.use_content_hash = use_content_hash
    
    def find_duplicates(
        self,
        documents: List[RetrievedDocument],
    ) -> Dict[str, List[RetrievedDocument]]:
        """
        Find duplicate documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Dict mapping canonical ID to duplicate docs
        """
        duplicates = defaultdict(list)
        seen_hashes = {}
        
        for doc in documents:
            # Get content hash
            content_hash = self._hash_content(doc.content)
            
            if content_hash in seen_hashes:
                # Found duplicate
                canonical_id = seen_hashes[content_hash]
                duplicates[canonical_id].append(doc)
            else:
                seen_hashes[content_hash] = doc.id
                duplicates[doc.id] = []
        
        return dict(duplicates)
    
    def _hash_content(self, content: str) -> str:
        """Hash document content."""
        # Normalize content
        normalized = ' '.join(content.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()


class DocumentMerger:
    """Merge duplicate documents."""
    
    def merge(
        self,
        documents: List[RetrievedDocument],
        score_aggregation: str = "max",
    ) -> RetrievedDocument:
        """
        Merge multiple documents into one.
        
        Args:
            documents: Documents to merge
            score_aggregation: How to aggregate scores
            
        Returns:
            Merged document
        """
        if not documents:
            raise ValueError("No documents to merge")
        
        if len(documents) == 1:
            return documents[0]
        
        # Use first document as base
        base = documents[0]
        
        # Aggregate scores
        scores = [d.score for d in documents]
        if score_aggregation == "max":
            final_score = max(scores)
        elif score_aggregation == "min":
            final_score = min(scores)
        elif score_aggregation == "mean":
            final_score = sum(scores) / len(scores)
        else:
            final_score = max(scores)
        
        # Best rank
        best_rank = min(d.rank for d in documents if d.rank > 0) or 1
        
        # Merge metadata
        merged_metadata = {}
        for doc in documents:
            merged_metadata.update(doc.metadata)
        
        # Track sources
        sources = list(set(d.retriever for d in documents))
        
        return RetrievedDocument(
            id=base.id,
            content=base.content,
            score=final_score,
            source=base.source,
            retriever=','.join(sources),
            rank=best_rank,
            scores={d.retriever: d.score for d in documents},
            metadata={
                **merged_metadata,
                'merged_from': len(documents),
                'sources': sources,
            },
        )


class BaseFusionStrategy(ABC):
    """Base class for fusion strategies."""
    
    @abstractmethod
    def fuse(
        self,
        results: List[RetrievalResult],
        weights: Optional[List[float]] = None,
    ) -> List[RetrievedDocument]:
        """Fuse results from multiple retrievers."""
        pass
    
    @property
    def name(self) -> str:
        return self.__class__.__name__


class RRFFusion(BaseFusionStrategy):
    """Reciprocal Rank Fusion."""
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion.
        
        Args:
            k: RRF parameter (typically 60)
        """
        self.k = k
    
    def fuse(
        self,
        results: List[RetrievalResult],
        weights: Optional[List[float]] = None,
    ) -> List[RetrievedDocument]:
        """Apply RRF fusion."""
        if weights is None:
            weights = [1.0] * len(results)
        
        # Calculate RRF scores
        doc_scores = defaultdict(float)
        doc_map = {}
        
        for result, weight in zip(results, weights):
            for rank, doc in enumerate(result.documents, 1):
                rrf_score = weight / (self.k + rank)
                doc_scores[doc.id] += rrf_score
                
                if doc.id not in doc_map:
                    doc_map[doc.id] = doc
        
        # Sort by RRF score
        sorted_ids = sorted(
            doc_scores.keys(),
            key=lambda x: doc_scores[x],
            reverse=True
        )
        
        # Create result documents
        fused = []
        for rank, doc_id in enumerate(sorted_ids, 1):
            doc = doc_map[doc_id]
            fused.append(RetrievedDocument(
                id=doc.id,
                content=doc.content,
                score=doc_scores[doc_id],
                source=doc.source,
                retriever=doc.retriever,
                rank=rank,
                original_rank=doc.rank,
                metadata=doc.metadata,
            ))
        
        return fused


class BordaFusion(BaseFusionStrategy):
    """Borda Count fusion."""
    
    def fuse(
        self,
        results: List[RetrievalResult],
        weights: Optional[List[float]] = None,
    ) -> List[RetrievedDocument]:
        """Apply Borda Count fusion."""
        if weights is None:
            weights = [1.0] * len(results)
        
        # Calculate Borda scores
        doc_scores = defaultdict(float)
        doc_map = {}
        
        for result, weight in zip(results, weights):
            n = len(result.documents)
            for rank, doc in enumerate(result.documents, 1):
                # Borda score: n - rank + 1
                borda_score = weight * (n - rank + 1)
                doc_scores[doc.id] += borda_score
                
                if doc.id not in doc_map:
                    doc_map[doc.id] = doc
        
        # Sort by Borda score
        sorted_ids = sorted(
            doc_scores.keys(),
            key=lambda x: doc_scores[x],
            reverse=True
        )
        
        # Create result documents
        fused = []
        for rank, doc_id in enumerate(sorted_ids, 1):
            doc = doc_map[doc_id]
            fused.append(RetrievedDocument(
                id=doc.id,
                content=doc.content,
                score=doc_scores[doc_id],
                source=doc.source,
                retriever=doc.retriever,
                rank=rank,
                original_rank=doc.rank,
                metadata=doc.metadata,
            ))
        
        return fused


class LinearFusion(BaseFusionStrategy):
    """Linear weighted score fusion."""
    
    def __init__(
        self,
        normalizer: Optional[ScoreNormalizer] = None,
    ):
        """
        Initialize linear fusion.
        
        Args:
            normalizer: Score normalizer
        """
        self.normalizer = normalizer or ScoreNormalizer(
            NormalizationMethod.MIN_MAX
        )
    
    def fuse(
        self,
        results: List[RetrievalResult],
        weights: Optional[List[float]] = None,
    ) -> List[RetrievedDocument]:
        """Apply linear fusion."""
        if weights is None:
            weights = [1.0] * len(results)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Normalize scores and combine
        doc_scores = defaultdict(float)
        doc_map = {}
        
        for result, weight in zip(results, weights):
            # Get all scores for normalization
            scores = [doc.score for doc in result.documents]
            normalized = self.normalizer.normalize(scores)
            
            for doc, norm_score in zip(result.documents, normalized):
                doc_scores[doc.id] += weight * norm_score
                
                if doc.id not in doc_map:
                    doc_map[doc.id] = doc
        
        # Sort by combined score
        sorted_ids = sorted(
            doc_scores.keys(),
            key=lambda x: doc_scores[x],
            reverse=True
        )
        
        # Create result documents
        fused = []
        for rank, doc_id in enumerate(sorted_ids, 1):
            doc = doc_map[doc_id]
            fused.append(RetrievedDocument(
                id=doc.id,
                content=doc.content,
                score=doc_scores[doc_id],
                source=doc.source,
                retriever=doc.retriever,
                rank=rank,
                original_rank=doc.rank,
                metadata=doc.metadata,
            ))
        
        return fused


class MaxFusion(BaseFusionStrategy):
    """Maximum score fusion."""
    
    def fuse(
        self,
        results: List[RetrievalResult],
        weights: Optional[List[float]] = None,
    ) -> List[RetrievedDocument]:
        """Apply max fusion."""
        doc_scores = {}
        doc_map = {}
        
        for result in results:
            for doc in result.documents:
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = doc.score
                    doc_map[doc.id] = doc
                else:
                    doc_scores[doc.id] = max(
                        doc_scores[doc.id], doc.score
                    )
        
        # Sort by max score
        sorted_ids = sorted(
            doc_scores.keys(),
            key=lambda x: doc_scores[x],
            reverse=True
        )
        
        # Create result documents
        fused = []
        for rank, doc_id in enumerate(sorted_ids, 1):
            doc = doc_map[doc_id]
            fused.append(RetrievedDocument(
                id=doc.id,
                content=doc.content,
                score=doc_scores[doc_id],
                source=doc.source,
                retriever=doc.retriever,
                rank=rank,
                original_rank=doc.rank,
                metadata=doc.metadata,
            ))
        
        return fused


class DiversityOptimizer:
    """Optimize result diversity."""
    
    def __init__(
        self,
        lambda_param: float = 0.5,
    ):
        """
        Initialize diversity optimizer.
        
        Args:
            lambda_param: Balance between relevance and diversity
        """
        self.lambda_param = lambda_param
    
    def optimize(
        self,
        documents: List[RetrievedDocument],
        k: int,
    ) -> List[RetrievedDocument]:
        """
        Optimize for diversity using MMR-like approach.
        
        Args:
            documents: Input documents
            k: Number of documents to return
            
        Returns:
            Diverse subset of documents
        """
        if len(documents) <= k:
            return documents
        
        selected = []
        remaining = list(documents)
        
        # Select first document (highest score)
        selected.append(remaining.pop(0))
        
        # Iteratively select diverse documents
        while len(selected) < k and remaining:
            best_idx = 0
            best_score = float('-inf')
            
            for i, doc in enumerate(remaining):
                # Relevance score
                rel_score = doc.score
                
                # Diversity score (max similarity to selected)
                div_score = max(
                    self._similarity(doc, s)
                    for s in selected
                )
                
                # MMR score
                mmr = (
                    self.lambda_param * rel_score -
                    (1 - self.lambda_param) * div_score
                )
                
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        # Update ranks
        for i, doc in enumerate(selected, 1):
            doc.rank = i
        
        return selected
    
    def _similarity(
        self,
        doc1: RetrievedDocument,
        doc2: RetrievedDocument,
    ) -> float:
        """Calculate document similarity."""
        # Simple word overlap similarity
        words1 = set(doc1.content.lower().split())
        words2 = set(doc2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        
        return overlap / union if union > 0 else 0.0


class ResultAggregator:
    """
    Main result aggregation interface.
    
    Example:
        >>> aggregator = ResultAggregator(strategy="rrf")
        >>> 
        >>> # Aggregate results
        >>> combined = aggregator.aggregate([
        ...     sparse_results,
        ...     dense_results,
        ... ])
        >>> 
        >>> # With weights
        >>> combined = aggregator.aggregate(
        ...     [sparse_results, dense_results],
        ...     weights=[0.3, 0.7]
        ... )
        >>> 
        >>> # With diversity optimization
        >>> combined = aggregator.aggregate(
        ...     results,
        ...     optimize_diversity=True,
        ...     diversity_k=10
        ... )
    """
    
    def __init__(
        self,
        strategy: Union[str, FusionStrategy] = FusionStrategy.RRF,
        normalization: NormalizationMethod = NormalizationMethod.MIN_MAX,
        merge_duplicates: bool = True,
        optimize_diversity: bool = False,
        diversity_lambda: float = 0.5,
    ):
        """
        Initialize aggregator.
        
        Args:
            strategy: Fusion strategy
            normalization: Score normalization method
            merge_duplicates: Merge duplicate documents
            optimize_diversity: Optimize for diversity
            diversity_lambda: Diversity parameter
        """
        if isinstance(strategy, str):
            strategy = FusionStrategy(strategy)
        
        self.strategy = strategy
        self.normalization = normalization
        self.merge_duplicates = merge_duplicates
        self.optimize_diversity = optimize_diversity
        self.diversity_lambda = diversity_lambda
        
        # Components
        self._fusion = self._get_fusion_strategy(strategy)
        self._normalizer = ScoreNormalizer(normalization)
        self._duplicate_detector = DuplicateDetector()
        self._merger = DocumentMerger()
        self._diversity_optimizer = DiversityOptimizer(diversity_lambda)
    
    def _get_fusion_strategy(
        self,
        strategy: FusionStrategy,
    ) -> BaseFusionStrategy:
        """Get fusion strategy implementation."""
        if strategy == FusionStrategy.RRF:
            return RRFFusion()
        elif strategy == FusionStrategy.BORDA:
            return BordaFusion()
        elif strategy in {FusionStrategy.LINEAR, FusionStrategy.MEAN}:
            return LinearFusion(self._normalizer)
        elif strategy == FusionStrategy.MAX:
            return MaxFusion()
        else:
            return RRFFusion()  # Default
    
    def aggregate(
        self,
        results: List[RetrievalResult],
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        optimize_diversity: Optional[bool] = None,
        diversity_k: Optional[int] = None,
    ) -> AggregatedResult:
        """
        Aggregate results from multiple retrievers.
        
        Args:
            results: List of retrieval results
            weights: Optional weights for each retriever
            top_k: Maximum documents to return
            optimize_diversity: Override diversity optimization
            diversity_k: Number of diverse documents
            
        Returns:
            AggregatedResult
        """
        if not results:
            return AggregatedResult(
                documents=[],
                strategy=self.strategy,
            )
        
        # Apply fusion
        fused = self._fusion.fuse(results, weights)
        
        # Merge duplicates
        if self.merge_duplicates:
            fused = self._merge_duplicates(fused)
        
        # Apply diversity optimization
        should_optimize = (
            optimize_diversity if optimize_diversity is not None
            else self.optimize_diversity
        )
        if should_optimize:
            k = diversity_k or top_k or len(fused)
            fused = self._diversity_optimizer.optimize(fused, k)
        
        # Apply top_k
        if top_k:
            fused = fused[:top_k]
        
        # Calculate statistics
        source_counts = self._count_sources(results)
        total_unique = len(fused)
        total_input = sum(r.count for r in results)
        overlap = self._calculate_overlap(results)
        
        return AggregatedResult(
            documents=fused,
            strategy=self.strategy,
            source_counts=source_counts,
            total_unique=total_unique,
            total_duplicates=total_input - total_unique,
            overlap_ratio=overlap,
            component_results=results,
        )
    
    def _merge_duplicates(
        self,
        documents: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """Merge duplicate documents."""
        # Group by ID
        doc_groups = defaultdict(list)
        for doc in documents:
            doc_groups[doc.id].append(doc)
        
        # Merge each group
        merged = []
        for doc_id, group in doc_groups.items():
            merged_doc = self._merger.merge(group)
            merged.append(merged_doc)
        
        # Re-sort by score
        merged.sort(key=lambda d: d.score, reverse=True)
        
        # Update ranks
        for i, doc in enumerate(merged, 1):
            doc.rank = i
        
        return merged
    
    def _count_sources(
        self,
        results: List[RetrievalResult],
    ) -> Dict[str, int]:
        """Count documents per source."""
        counts = {}
        for result in results:
            counts[result.retriever_name] = result.count
        return counts
    
    def _calculate_overlap(
        self,
        results: List[RetrievalResult],
    ) -> float:
        """Calculate overlap ratio between results."""
        if len(results) < 2:
            return 0.0
        
        id_sets = [r.get_ids() for r in results]
        
        # Calculate pairwise overlaps
        overlaps = []
        for i in range(len(id_sets)):
            for j in range(i + 1, len(id_sets)):
                if id_sets[i] and id_sets[j]:
                    overlap = len(id_sets[i] & id_sets[j])
                    min_size = min(len(id_sets[i]), len(id_sets[j]))
                    overlaps.append(overlap / min_size)
        
        return sum(overlaps) / len(overlaps) if overlaps else 0.0


# Convenience functions

def aggregate_results(
    results: List[RetrievalResult],
    strategy: str = "rrf",
    **kwargs,
) -> AggregatedResult:
    """
    Quick result aggregation.
    
    Args:
        results: List of retrieval results
        strategy: Fusion strategy
        **kwargs: Additional parameters
        
    Returns:
        AggregatedResult
    """
    aggregator = ResultAggregator(strategy=strategy)
    return aggregator.aggregate(results, **kwargs)


def rrf_fusion(
    results: List[RetrievalResult],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[RetrievedDocument]:
    """
    Apply RRF fusion.
    
    Args:
        results: Retrieval results
        k: RRF parameter
        weights: Optional weights
        
    Returns:
        Fused documents
    """
    fusion = RRFFusion(k=k)
    return fusion.fuse(results, weights)


def create_retrieval_result(
    documents: List[Dict[str, Any]],
    retriever_name: str,
    query: str = "",
) -> RetrievalResult:
    """
    Create a retrieval result from documents.
    
    Args:
        documents: List of document dicts
        retriever_name: Name of retriever
        query: Original query
        
    Returns:
        RetrievalResult
    """
    docs = []
    for i, doc in enumerate(documents, 1):
        docs.append(RetrievedDocument(
            id=doc.get('id', str(i)),
            content=doc.get('content', doc.get('text', '')),
            score=doc.get('score', 0.0),
            source=doc.get('source', ''),
            retriever=retriever_name,
            rank=i,
            metadata=doc.get('metadata', {}),
        ))
    
    return RetrievalResult(
        documents=docs,
        retriever_name=retriever_name,
        query=query,
    )
