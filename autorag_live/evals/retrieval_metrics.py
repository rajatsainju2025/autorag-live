"""
Retrieval evaluation metrics for AutoRAG-Live.

Provides comprehensive metrics for evaluating retrieval quality
including precision, recall, MRR, NDCG, and MAP.

Features:
- Standard retrieval metrics (P@K, R@K, F1)
- Ranking metrics (MRR, NDCG, MAP)
- Relevance judgment support
- Batch evaluation
- Statistical significance testing

Example usage:
    >>> evaluator = RetrievalMetrics()
    >>> scores = evaluator.evaluate(
    ...     retrieved=["doc1", "doc2", "doc3"],
    ...     relevant=["doc1", "doc3", "doc5"]
    ... )
    >>> print(scores.precision_at_k(3))
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class RelevanceLevel(int, Enum):
    """Relevance judgment levels."""
    
    NOT_RELEVANT = 0
    MARGINALLY_RELEVANT = 1
    RELEVANT = 2
    HIGHLY_RELEVANT = 3


@dataclass
class RelevanceJudgment:
    """A relevance judgment for a document."""
    
    doc_id: str
    level: RelevanceLevel = RelevanceLevel.RELEVANT
    
    # Optional graded relevance (0-1)
    score: float = 1.0
    
    # Metadata
    query_id: Optional[str] = None
    annotator: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_relevant(self) -> bool:
        """Check if judged relevant."""
        return self.level >= RelevanceLevel.RELEVANT


@dataclass
class RetrievalResult:
    """Result of a retrieval for evaluation."""
    
    query_id: str
    retrieved_ids: List[str]
    
    # Optional scores
    scores: Optional[List[float]] = None
    
    # Relevance judgments
    judgments: Dict[str, RelevanceJudgment] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_relevance(self, doc_id: str) -> float:
        """Get relevance score for document."""
        if doc_id in self.judgments:
            return self.judgments[doc_id].score
        return 0.0
    
    def is_relevant(self, doc_id: str) -> bool:
        """Check if document is relevant."""
        if doc_id in self.judgments:
            return self.judgments[doc_id].is_relevant
        return False


@dataclass
class MetricScores:
    """Collection of retrieval metric scores."""
    
    # Set-based metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    # At K metrics
    precision_at: Dict[int, float] = field(default_factory=dict)
    recall_at: Dict[int, float] = field(default_factory=dict)
    
    # Ranking metrics
    mrr: float = 0.0           # Mean Reciprocal Rank
    map_score: float = 0.0     # Mean Average Precision
    ndcg: float = 0.0          # Normalized Discounted Cumulative Gain
    ndcg_at: Dict[int, float] = field(default_factory=dict)
    
    # Success metrics
    hit_rate: float = 0.0
    hit_rate_at: Dict[int, float] = field(default_factory=dict)
    
    # Additional
    num_retrieved: int = 0
    num_relevant: int = 0
    num_relevant_retrieved: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def precision_at_k(self, k: int) -> float:
        """Get precision at K."""
        return self.precision_at.get(k, 0.0)
    
    def recall_at_k(self, k: int) -> float:
        """Get recall at K."""
        return self.recall_at.get(k, 0.0)
    
    def ndcg_at_k(self, k: int) -> float:
        """Get NDCG at K."""
        return self.ndcg_at.get(k, 0.0)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary."""
        result = {
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'mrr': self.mrr,
            'map': self.map_score,
            'ndcg': self.ndcg,
            'hit_rate': self.hit_rate,
        }
        
        for k, v in self.precision_at.items():
            result[f'precision@{k}'] = v
        
        for k, v in self.recall_at.items():
            result[f'recall@{k}'] = v
        
        for k, v in self.ndcg_at.items():
            result[f'ndcg@{k}'] = v
        
        for k, v in self.hit_rate_at.items():
            result[f'hit_rate@{k}'] = v
        
        return result


class BaseMetric(ABC):
    """Base class for retrieval metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name."""
        pass
    
    @abstractmethod
    def compute(
        self,
        retrieved: List[str],
        relevant: Set[str],
        **kwargs,
    ) -> float:
        """Compute metric value."""
        pass


class PrecisionMetric(BaseMetric):
    """Precision metric."""
    
    def __init__(self, k: Optional[int] = None):
        """
        Initialize precision metric.
        
        Args:
            k: Optional K for P@K
        """
        self.k = k
    
    @property
    def name(self) -> str:
        return f"precision@{self.k}" if self.k else "precision"
    
    def compute(
        self,
        retrieved: List[str],
        relevant: Set[str],
        **kwargs,
    ) -> float:
        """Compute precision."""
        if self.k:
            retrieved = retrieved[:self.k]
        
        if not retrieved:
            return 0.0
        
        relevant_retrieved = sum(1 for doc in retrieved if doc in relevant)
        return relevant_retrieved / len(retrieved)


class RecallMetric(BaseMetric):
    """Recall metric."""
    
    def __init__(self, k: Optional[int] = None):
        """
        Initialize recall metric.
        
        Args:
            k: Optional K for R@K
        """
        self.k = k
    
    @property
    def name(self) -> str:
        return f"recall@{self.k}" if self.k else "recall"
    
    def compute(
        self,
        retrieved: List[str],
        relevant: Set[str],
        **kwargs,
    ) -> float:
        """Compute recall."""
        if self.k:
            retrieved = retrieved[:self.k]
        
        if not relevant:
            return 0.0
        
        relevant_retrieved = sum(1 for doc in retrieved if doc in relevant)
        return relevant_retrieved / len(relevant)


class F1Metric(BaseMetric):
    """F1 score metric."""
    
    def __init__(self, k: Optional[int] = None):
        """
        Initialize F1 metric.
        
        Args:
            k: Optional K for F1@K
        """
        self.k = k
    
    @property
    def name(self) -> str:
        return f"f1@{self.k}" if self.k else "f1"
    
    def compute(
        self,
        retrieved: List[str],
        relevant: Set[str],
        **kwargs,
    ) -> float:
        """Compute F1 score."""
        precision = PrecisionMetric(self.k).compute(retrieved, relevant)
        recall = RecallMetric(self.k).compute(retrieved, relevant)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)


class MRRMetric(BaseMetric):
    """Mean Reciprocal Rank metric."""
    
    @property
    def name(self) -> str:
        return "mrr"
    
    def compute(
        self,
        retrieved: List[str],
        relevant: Set[str],
        **kwargs,
    ) -> float:
        """Compute Mean Reciprocal Rank."""
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0


class MAPMetric(BaseMetric):
    """Mean Average Precision metric."""
    
    def __init__(self, k: Optional[int] = None):
        """
        Initialize MAP metric.
        
        Args:
            k: Optional cutoff K
        """
        self.k = k
    
    @property
    def name(self) -> str:
        return f"map@{self.k}" if self.k else "map"
    
    def compute(
        self,
        retrieved: List[str],
        relevant: Set[str],
        **kwargs,
    ) -> float:
        """Compute Average Precision."""
        if not relevant:
            return 0.0
        
        if self.k:
            retrieved = retrieved[:self.k]
        
        precisions = []
        relevant_count = 0
        
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant)


class NDCGMetric(BaseMetric):
    """Normalized Discounted Cumulative Gain metric."""
    
    def __init__(self, k: Optional[int] = None):
        """
        Initialize NDCG metric.
        
        Args:
            k: Optional cutoff K
        """
        self.k = k
    
    @property
    def name(self) -> str:
        return f"ndcg@{self.k}" if self.k else "ndcg"
    
    def compute(
        self,
        retrieved: List[str],
        relevant: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> float:
        """Compute NDCG."""
        if self.k:
            retrieved = retrieved[:self.k]
        
        if not retrieved or not relevant:
            return 0.0
        
        # Get relevance scores
        if relevance_scores is None:
            relevance_scores = {doc: 1.0 for doc in relevant}
        
        # Compute DCG
        dcg = self._dcg(retrieved, relevance_scores)
        
        # Compute ideal DCG
        ideal_order = sorted(
            relevant,
            key=lambda d: relevance_scores.get(d, 0),
            reverse=True
        )
        if self.k:
            ideal_order = ideal_order[:self.k]
        
        idcg = self._dcg(ideal_order, relevance_scores)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _dcg(
        self,
        docs: List[str],
        scores: Dict[str, float],
    ) -> float:
        """Compute Discounted Cumulative Gain."""
        dcg = 0.0
        for i, doc in enumerate(docs):
            rel = scores.get(doc, 0)
            # Using log2(i+2) to handle i=0
            dcg += (2 ** rel - 1) / math.log2(i + 2)
        return dcg


class HitRateMetric(BaseMetric):
    """Hit rate (success) metric."""
    
    def __init__(self, k: Optional[int] = None):
        """
        Initialize hit rate metric.
        
        Args:
            k: Optional cutoff K
        """
        self.k = k
    
    @property
    def name(self) -> str:
        return f"hit_rate@{self.k}" if self.k else "hit_rate"
    
    def compute(
        self,
        retrieved: List[str],
        relevant: Set[str],
        **kwargs,
    ) -> float:
        """Compute hit rate (1 if any relevant in top k, else 0)."""
        if self.k:
            retrieved = retrieved[:self.k]
        
        for doc in retrieved:
            if doc in relevant:
                return 1.0
        return 0.0


class RetrievalMetrics:
    """
    Main retrieval metrics interface.
    
    Example:
        >>> metrics = RetrievalMetrics()
        >>> 
        >>> # Evaluate single query
        >>> scores = metrics.evaluate(
        ...     retrieved=["doc1", "doc2", "doc3"],
        ...     relevant=["doc1", "doc3", "doc5"]
        ... )
        >>> print(f"Precision: {scores.precision:.3f}")
        >>> print(f"MRR: {scores.mrr:.3f}")
        >>> 
        >>> # Evaluate batch
        >>> results = [
        ...     RetrievalResult(query_id="q1", retrieved_ids=[...]),
        ...     RetrievalResult(query_id="q2", retrieved_ids=[...]),
        ... ]
        >>> avg_scores = metrics.evaluate_batch(results, qrels)
    """
    
    def __init__(
        self,
        k_values: List[int] = [1, 3, 5, 10, 20],
        include_ndcg: bool = True,
        include_map: bool = True,
    ):
        """
        Initialize retrieval metrics.
        
        Args:
            k_values: K values for @K metrics
            include_ndcg: Include NDCG computation
            include_map: Include MAP computation
        """
        self.k_values = k_values
        self.include_ndcg = include_ndcg
        self.include_map = include_map
        
        # Build metrics
        self._metrics = self._build_metrics()
    
    def _build_metrics(self) -> Dict[str, BaseMetric]:
        """Build metric instances."""
        metrics = {}
        
        # Base metrics
        metrics['precision'] = PrecisionMetric()
        metrics['recall'] = RecallMetric()
        metrics['f1'] = F1Metric()
        metrics['mrr'] = MRRMetric()
        metrics['hit_rate'] = HitRateMetric()
        
        if self.include_map:
            metrics['map'] = MAPMetric()
        
        if self.include_ndcg:
            metrics['ndcg'] = NDCGMetric()
        
        # @K metrics
        for k in self.k_values:
            metrics[f'precision@{k}'] = PrecisionMetric(k)
            metrics[f'recall@{k}'] = RecallMetric(k)
            metrics[f'hit_rate@{k}'] = HitRateMetric(k)
            
            if self.include_ndcg:
                metrics[f'ndcg@{k}'] = NDCGMetric(k)
        
        return metrics
    
    def evaluate(
        self,
        retrieved: List[str],
        relevant: Union[List[str], Set[str]],
        relevance_scores: Optional[Dict[str, float]] = None,
    ) -> MetricScores:
        """
        Evaluate retrieval for a single query.
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: Set of relevant document IDs
            relevance_scores: Optional graded relevance scores
            
        Returns:
            MetricScores
        """
        if isinstance(relevant, list):
            relevant = set(relevant)
        
        scores = MetricScores()
        
        # Compute all metrics
        for name, metric in self._metrics.items():
            value = metric.compute(
                retrieved=retrieved,
                relevant=relevant,
                relevance_scores=relevance_scores,
            )
            
            # Store in appropriate field
            if name == 'precision':
                scores.precision = value
            elif name == 'recall':
                scores.recall = value
            elif name == 'f1':
                scores.f1 = value
            elif name == 'mrr':
                scores.mrr = value
            elif name == 'map':
                scores.map_score = value
            elif name == 'ndcg':
                scores.ndcg = value
            elif name == 'hit_rate':
                scores.hit_rate = value
            elif name.startswith('precision@'):
                k = int(name.split('@')[1])
                scores.precision_at[k] = value
            elif name.startswith('recall@'):
                k = int(name.split('@')[1])
                scores.recall_at[k] = value
            elif name.startswith('ndcg@'):
                k = int(name.split('@')[1])
                scores.ndcg_at[k] = value
            elif name.startswith('hit_rate@'):
                k = int(name.split('@')[1])
                scores.hit_rate_at[k] = value
        
        # Counts
        scores.num_retrieved = len(retrieved)
        scores.num_relevant = len(relevant)
        scores.num_relevant_retrieved = len(
            set(retrieved) & relevant
        )
        
        return scores
    
    def evaluate_batch(
        self,
        results: List[RetrievalResult],
        qrels: Optional[Dict[str, Set[str]]] = None,
    ) -> MetricScores:
        """
        Evaluate batch of retrieval results.
        
        Args:
            results: List of retrieval results
            qrels: Query relevance judgments (query_id -> relevant doc_ids)
            
        Returns:
            Averaged MetricScores
        """
        if not results:
            return MetricScores()
        
        all_scores = []
        
        for result in results:
            # Get relevant docs for this query
            if qrels and result.query_id in qrels:
                relevant = qrels[result.query_id]
            elif result.judgments:
                relevant = {
                    doc_id for doc_id, j in result.judgments.items()
                    if j.is_relevant
                }
            else:
                continue
            
            # Get relevance scores
            relevance_scores = None
            if result.judgments:
                relevance_scores = {
                    doc_id: j.score
                    for doc_id, j in result.judgments.items()
                }
            
            scores = self.evaluate(
                retrieved=result.retrieved_ids,
                relevant=relevant,
                relevance_scores=relevance_scores,
            )
            all_scores.append(scores)
        
        # Average scores
        return self._average_scores(all_scores)
    
    def _average_scores(
        self,
        scores_list: List[MetricScores],
    ) -> MetricScores:
        """Average multiple MetricScores."""
        if not scores_list:
            return MetricScores()
        
        n = len(scores_list)
        
        avg = MetricScores()
        avg.precision = sum(s.precision for s in scores_list) / n
        avg.recall = sum(s.recall for s in scores_list) / n
        avg.f1 = sum(s.f1 for s in scores_list) / n
        avg.mrr = sum(s.mrr for s in scores_list) / n
        avg.map_score = sum(s.map_score for s in scores_list) / n
        avg.ndcg = sum(s.ndcg for s in scores_list) / n
        avg.hit_rate = sum(s.hit_rate for s in scores_list) / n
        
        # @K metrics
        for k in self.k_values:
            avg.precision_at[k] = sum(
                s.precision_at.get(k, 0) for s in scores_list
            ) / n
            avg.recall_at[k] = sum(
                s.recall_at.get(k, 0) for s in scores_list
            ) / n
            avg.ndcg_at[k] = sum(
                s.ndcg_at.get(k, 0) for s in scores_list
            ) / n
            avg.hit_rate_at[k] = sum(
                s.hit_rate_at.get(k, 0) for s in scores_list
            ) / n
        
        # Totals
        avg.num_retrieved = sum(s.num_retrieved for s in scores_list)
        avg.num_relevant = sum(s.num_relevant for s in scores_list)
        avg.num_relevant_retrieved = sum(
            s.num_relevant_retrieved for s in scores_list
        )
        
        avg.metadata['num_queries'] = n
        
        return avg
    
    def compare(
        self,
        baseline_scores: MetricScores,
        comparison_scores: MetricScores,
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Compare two sets of scores.
        
        Args:
            baseline_scores: Baseline metric scores
            comparison_scores: Comparison metric scores
            
        Returns:
            Dict mapping metric name to (baseline, comparison, diff)
        """
        comparison = {}
        
        base_dict = baseline_scores.to_dict()
        comp_dict = comparison_scores.to_dict()
        
        for name in base_dict:
            if name in comp_dict:
                base_val = base_dict[name]
                comp_val = comp_dict[name]
                diff = comp_val - base_val
                comparison[name] = (base_val, comp_val, diff)
        
        return comparison


class StatisticalTester:
    """Statistical significance testing for retrieval metrics."""
    
    @staticmethod
    def paired_t_test(
        scores1: List[float],
        scores2: List[float],
        alpha: float = 0.05,
    ) -> Tuple[float, bool]:
        """
        Perform paired t-test.
        
        Args:
            scores1: First set of scores
            scores2: Second set of scores
            alpha: Significance level
            
        Returns:
            Tuple of (p-value, is_significant)
        """
        if len(scores1) != len(scores2):
            raise ValueError("Score lists must have same length")
        
        n = len(scores1)
        if n < 2:
            return 1.0, False
        
        # Calculate differences
        diffs = [s1 - s2 for s1, s2 in zip(scores1, scores2)]
        
        # Mean and std of differences
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        std_diff = var_diff ** 0.5
        
        if std_diff == 0:
            return 1.0 if mean_diff == 0 else 0.0, mean_diff != 0
        
        # t-statistic
        t_stat = mean_diff / (std_diff / (n ** 0.5))
        
        # Approximate p-value (using normal distribution for large n)
        from math import erf
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / (2 ** 0.5))))
        
        return p_value, p_value < alpha
    
    @staticmethod
    def bootstrap_confidence_interval(
        scores: List[float],
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval.
        
        Args:
            scores: Score values
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        import random
        
        n = len(scores)
        if n == 0:
            return 0.0, 0.0, 0.0
        
        # Bootstrap samples
        means = []
        for _ in range(n_bootstrap):
            sample = [random.choice(scores) for _ in range(n)]
            means.append(sum(sample) / n)
        
        means.sort()
        
        # Confidence interval
        alpha = 1 - confidence
        lower_idx = int(alpha / 2 * n_bootstrap)
        upper_idx = int((1 - alpha / 2) * n_bootstrap)
        
        mean = sum(scores) / n
        lower = means[lower_idx]
        upper = means[upper_idx]
        
        return mean, lower, upper


# Convenience functions

def compute_precision(
    retrieved: List[str],
    relevant: List[str],
    k: Optional[int] = None,
) -> float:
    """
    Compute precision.
    
    Args:
        retrieved: Retrieved document IDs
        relevant: Relevant document IDs
        k: Optional cutoff
        
    Returns:
        Precision score
    """
    metric = PrecisionMetric(k)
    return metric.compute(retrieved, set(relevant))


def compute_recall(
    retrieved: List[str],
    relevant: List[str],
    k: Optional[int] = None,
) -> float:
    """
    Compute recall.
    
    Args:
        retrieved: Retrieved document IDs
        relevant: Relevant document IDs
        k: Optional cutoff
        
    Returns:
        Recall score
    """
    metric = RecallMetric(k)
    return metric.compute(retrieved, set(relevant))


def compute_mrr(
    retrieved: List[str],
    relevant: List[str],
) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        retrieved: Retrieved document IDs
        relevant: Relevant document IDs
        
    Returns:
        MRR score
    """
    metric = MRRMetric()
    return metric.compute(retrieved, set(relevant))


def compute_ndcg(
    retrieved: List[str],
    relevant: List[str],
    k: Optional[int] = None,
    relevance_scores: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute NDCG.
    
    Args:
        retrieved: Retrieved document IDs
        relevant: Relevant document IDs
        k: Optional cutoff
        relevance_scores: Optional graded relevance
        
    Returns:
        NDCG score
    """
    metric = NDCGMetric(k)
    return metric.compute(
        retrieved,
        set(relevant),
        relevance_scores=relevance_scores,
    )


def evaluate_retrieval(
    retrieved: List[str],
    relevant: List[str],
) -> Dict[str, float]:
    """
    Quick retrieval evaluation.
    
    Args:
        retrieved: Retrieved document IDs
        relevant: Relevant document IDs
        
    Returns:
        Dict of metric scores
    """
    evaluator = RetrievalMetrics()
    scores = evaluator.evaluate(retrieved, relevant)
    return scores.to_dict()
