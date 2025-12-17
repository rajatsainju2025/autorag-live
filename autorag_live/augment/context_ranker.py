"""
Context ranking utilities for AutoRAG-Live.

Provides intelligent context passage ranking based on relevance,
quality, and position for optimal LLM context selection.

Features:
- Multi-signal passage ranking
- Relevance scoring
- Quality assessment
- Position-aware ranking
- Diversity-aware selection
- Context window optimization

Example usage:
    >>> ranker = ContextRanker()
    >>> ranked = ranker.rank(passages, query, max_tokens=4000)
    >>> selected = ranker.select_context(passages, query, budget=2000)
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class RankingSignal(str, Enum):
    """Ranking signal types."""
    
    RELEVANCE = "relevance"
    QUALITY = "quality"
    POSITION = "position"
    RECENCY = "recency"
    AUTHORITY = "authority"
    DIVERSITY = "diversity"
    DENSITY = "density"


class SelectionStrategy(str, Enum):
    """Context selection strategies."""
    
    TOP_K = "top_k"                # Select top k passages
    TOKEN_BUDGET = "token_budget"  # Select within token budget
    DIVERSITY = "diversity"        # Diversity-aware selection
    COVERAGE = "coverage"          # Query coverage-based
    GREEDY = "greedy"              # Greedy marginal utility


@dataclass
class Passage:
    """A context passage for ranking."""
    
    id: str
    content: str
    
    # Scores
    relevance_score: float = 0.0
    quality_score: float = 0.0
    position_score: float = 0.0
    
    # Combined
    final_score: float = 0.0
    rank: int = 0
    
    # Metadata
    source: str = ""
    position: int = 0
    token_count: int = 0
    
    # Additional scores
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = len(self.content.split())


@dataclass
class RankingResult:
    """Result of context ranking."""
    
    passages: List[Passage]
    
    # Selection info
    total_passages: int = 0
    selected_passages: int = 0
    total_tokens: int = 0
    
    # Statistics
    relevance_stats: Dict[str, float] = field(default_factory=dict)
    quality_stats: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    strategy: SelectionStrategy = SelectionStrategy.TOP_K
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def count(self) -> int:
        """Get passage count."""
        return len(self.passages)
    
    def get_content(self, separator: str = "\n\n") -> str:
        """Get combined content."""
        return separator.join(p.content for p in self.passages)
    
    def top_k(self, k: int) -> List[Passage]:
        """Get top k passages."""
        return self.passages[:k]


class RelevanceScorer:
    """Score passage relevance to query."""
    
    def __init__(
        self,
        use_embedding: bool = False,
        embedding_func: Optional[Callable] = None,
    ):
        """
        Initialize relevance scorer.
        
        Args:
            use_embedding: Use embeddings for scoring
            embedding_func: Optional embedding function
        """
        self.use_embedding = use_embedding
        self.embedding_func = embedding_func
    
    def score(
        self,
        passage: str,
        query: str,
    ) -> float:
        """
        Score passage relevance.
        
        Args:
            passage: Passage text
            query: Query text
            
        Returns:
            Relevance score (0-1)
        """
        if self.use_embedding and self.embedding_func:
            return self._embedding_score(passage, query)
        return self._lexical_score(passage, query)
    
    def _lexical_score(self, passage: str, query: str) -> float:
        """Compute lexical relevance score."""
        query_terms = set(query.lower().split())
        passage_terms = set(passage.lower().split())
        
        if not query_terms:
            return 0.0
        
        # Term overlap
        overlap = len(query_terms & passage_terms)
        coverage = overlap / len(query_terms)
        
        # Term frequency
        passage_lower = passage.lower()
        tf_score = 0.0
        for term in query_terms:
            count = passage_lower.count(term)
            tf_score += min(count / 3, 1.0)  # Cap at 3 occurrences
        tf_score /= len(query_terms)
        
        # Combine
        return 0.6 * coverage + 0.4 * tf_score
    
    def _embedding_score(self, passage: str, query: str) -> float:
        """Compute embedding-based relevance score."""
        try:
            query_emb = self.embedding_func(query)
            passage_emb = self.embedding_func(passage)
            
            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_emb, passage_emb))
            norm_q = sum(a * a for a in query_emb) ** 0.5
            norm_p = sum(a * a for a in passage_emb) ** 0.5
            
            if norm_q > 0 and norm_p > 0:
                return dot / (norm_q * norm_p)
        except Exception as e:
            logger.warning(f"Embedding scoring failed: {e}")
        
        return self._lexical_score(passage, query)


class QualityScorer:
    """Score passage quality."""
    
    def __init__(
        self,
        min_length: int = 50,
        ideal_length: int = 500,
        max_length: int = 2000,
    ):
        """
        Initialize quality scorer.
        
        Args:
            min_length: Minimum passage length
            ideal_length: Ideal passage length
            max_length: Maximum passage length
        """
        self.min_length = min_length
        self.ideal_length = ideal_length
        self.max_length = max_length
    
    def score(self, passage: str) -> Tuple[float, Dict[str, float]]:
        """
        Score passage quality.
        
        Args:
            passage: Passage text
            
        Returns:
            Tuple of (overall_score, component_scores)
        """
        components = {}
        
        # Length score
        components['length'] = self._score_length(passage)
        
        # Readability score
        components['readability'] = self._score_readability(passage)
        
        # Information density
        components['density'] = self._score_density(passage)
        
        # Structure score
        components['structure'] = self._score_structure(passage)
        
        # Combine scores
        weights = {
            'length': 0.25,
            'readability': 0.25,
            'density': 0.25,
            'structure': 0.25,
        }
        
        overall = sum(
            components[k] * weights[k]
            for k in components
        )
        
        return overall, components
    
    def _score_length(self, passage: str) -> float:
        """Score based on length."""
        length = len(passage)
        
        if length < self.min_length:
            return length / self.min_length * 0.5
        elif length <= self.ideal_length:
            return 0.5 + (length - self.min_length) / (self.ideal_length - self.min_length) * 0.5
        elif length <= self.max_length:
            return 1.0 - (length - self.ideal_length) / (self.max_length - self.ideal_length) * 0.3
        else:
            return 0.7
    
    def _score_readability(self, passage: str) -> float:
        """Score readability."""
        # Simple readability metrics
        words = passage.split()
        if not words:
            return 0.0
        
        sentences = re.split(r'[.!?]+', passage)
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return 0.5
        
        # Average sentence length
        avg_sentence_len = len(words) / len(sentences)
        
        # Penalize very short or very long sentences
        if 10 <= avg_sentence_len <= 25:
            return 1.0
        elif 5 <= avg_sentence_len < 10 or 25 < avg_sentence_len <= 35:
            return 0.8
        else:
            return 0.6
    
    def _score_density(self, passage: str) -> float:
        """Score information density."""
        words = passage.lower().split()
        if not words:
            return 0.0
        
        # Stop words ratio
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'and',
            'but', 'or', 'if', 'of', 'to', 'in', 'for', 'on', 'with',
        }
        
        content_words = [w for w in words if w not in stop_words]
        density = len(content_words) / len(words)
        
        return density
    
    def _score_structure(self, passage: str) -> float:
        """Score structural quality."""
        score = 1.0
        
        # Check for proper capitalization
        sentences = re.split(r'[.!?]+', passage)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[0].isupper():
                score *= 0.95
        
        # Check for proper punctuation
        if not passage.strip()[-1:] in '.!?':
            score *= 0.95
        
        # Check for balanced quotes/parentheses
        if passage.count('(') != passage.count(')'):
            score *= 0.9
        if passage.count('"') % 2 != 0:
            score *= 0.9
        
        return score


class PositionScorer:
    """Score based on passage position."""
    
    def __init__(
        self,
        decay_factor: float = 0.1,
        position_weight: str = "linear",
    ):
        """
        Initialize position scorer.
        
        Args:
            decay_factor: Position decay factor
            position_weight: Weighting method (linear, exponential, log)
        """
        self.decay_factor = decay_factor
        self.position_weight = position_weight
    
    def score(
        self,
        position: int,
        total: int,
    ) -> float:
        """
        Score based on position.
        
        Args:
            position: Passage position (0-indexed)
            total: Total passages
            
        Returns:
            Position score (0-1)
        """
        if total <= 0:
            return 1.0
        
        if position < 0:
            position = 0
        
        if self.position_weight == "linear":
            return 1.0 - (position / total) * self.decay_factor
        elif self.position_weight == "exponential":
            import math
            return math.exp(-self.decay_factor * position)
        elif self.position_weight == "log":
            import math
            return 1.0 / (1.0 + self.decay_factor * math.log(position + 1))
        
        return 1.0


class DiversityScorer:
    """Score passage diversity."""
    
    def __init__(
        self,
        method: str = "jaccard",
    ):
        """
        Initialize diversity scorer.
        
        Args:
            method: Similarity method (jaccard, overlap)
        """
        self.method = method
    
    def score_diversity(
        self,
        passage: str,
        selected: List[str],
    ) -> float:
        """
        Score diversity relative to selected passages.
        
        Args:
            passage: New passage
            selected: Already selected passages
            
        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        if not selected:
            return 1.0
        
        # Calculate similarity to each selected passage
        similarities = [
            self._similarity(passage, s)
            for s in selected
        ]
        
        # Max similarity (lower = more diverse)
        max_sim = max(similarities)
        
        # Convert to diversity score
        return 1.0 - max_sim
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        if self.method == "jaccard":
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0
        else:  # overlap
            intersection = len(words1 & words2)
            return intersection / min(len(words1), len(words2))


class QueryCoverageScorer:
    """Score query term coverage."""
    
    def score_coverage(
        self,
        passage: str,
        query_terms: List[str],
        covered_terms: set,
    ) -> Tuple[float, set]:
        """
        Score based on new query terms covered.
        
        Args:
            passage: Passage text
            query_terms: All query terms
            covered_terms: Already covered terms
            
        Returns:
            Tuple of (coverage_score, newly_covered_terms)
        """
        passage_words = set(passage.lower().split())
        query_set = set(t.lower() for t in query_terms)
        
        # Find new covered terms
        newly_covered = (query_set & passage_words) - covered_terms
        
        if not query_set:
            return 0.0, set()
        
        # Score based on new coverage
        score = len(newly_covered) / len(query_set)
        
        return score, newly_covered


class TokenCounter:
    """Count tokens in text."""
    
    def __init__(
        self,
        method: str = "word",
        chars_per_token: float = 4.0,
    ):
        """
        Initialize token counter.
        
        Args:
            method: Counting method (word, char, tiktoken)
            chars_per_token: Average chars per token for char method
        """
        self.method = method
        self.chars_per_token = chars_per_token
    
    def count(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        if self.method == "word":
            return len(text.split())
        elif self.method == "char":
            return int(len(text) / self.chars_per_token)
        elif self.method == "tiktoken":
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(text))
            except ImportError:
                return len(text.split())
        
        return len(text.split())


class ContextRanker:
    """
    Main context ranking interface.
    
    Example:
        >>> ranker = ContextRanker()
        >>> 
        >>> # Rank passages
        >>> result = ranker.rank(passages, query)
        >>> for p in result.top_k(5):
        ...     print(f"{p.rank}: {p.final_score:.3f}")
        >>> 
        >>> # Select with token budget
        >>> result = ranker.select_context(
        ...     passages, query,
        ...     token_budget=2000,
        ...     strategy="token_budget"
        ... )
    """
    
    def __init__(
        self,
        relevance_weight: float = 0.5,
        quality_weight: float = 0.3,
        position_weight: float = 0.1,
        diversity_weight: float = 0.1,
        use_embeddings: bool = False,
        embedding_func: Optional[Callable] = None,
    ):
        """
        Initialize context ranker.
        
        Args:
            relevance_weight: Weight for relevance score
            quality_weight: Weight for quality score
            position_weight: Weight for position score
            diversity_weight: Weight for diversity score
            use_embeddings: Use embeddings for relevance
            embedding_func: Optional embedding function
        """
        self.relevance_weight = relevance_weight
        self.quality_weight = quality_weight
        self.position_weight = position_weight
        self.diversity_weight = diversity_weight
        
        # Scorers
        self.relevance_scorer = RelevanceScorer(
            use_embedding=use_embeddings,
            embedding_func=embedding_func,
        )
        self.quality_scorer = QualityScorer()
        self.position_scorer = PositionScorer()
        self.diversity_scorer = DiversityScorer()
        self.coverage_scorer = QueryCoverageScorer()
        self.token_counter = TokenCounter()
    
    def rank(
        self,
        passages: List[Union[str, Passage, Dict]],
        query: str,
        max_passages: Optional[int] = None,
    ) -> RankingResult:
        """
        Rank passages by relevance and quality.
        
        Args:
            passages: List of passages
            query: Query text
            max_passages: Maximum passages to rank
            
        Returns:
            RankingResult
        """
        # Convert to Passage objects
        passage_objs = self._to_passages(passages)
        
        if max_passages:
            passage_objs = passage_objs[:max_passages]
        
        total = len(passage_objs)
        
        # Score each passage
        for i, passage in enumerate(passage_objs):
            # Relevance
            passage.relevance_score = self.relevance_scorer.score(
                passage.content, query
            )
            passage.scores['relevance'] = passage.relevance_score
            
            # Quality
            passage.quality_score, quality_components = self.quality_scorer.score(
                passage.content
            )
            passage.scores['quality'] = passage.quality_score
            passage.scores.update({f'quality_{k}': v for k, v in quality_components.items()})
            
            # Position
            passage.position_score = self.position_scorer.score(i, total)
            passage.scores['position'] = passage.position_score
            
            # Calculate final score
            passage.final_score = (
                self.relevance_weight * passage.relevance_score +
                self.quality_weight * passage.quality_score +
                self.position_weight * passage.position_score
            )
        
        # Sort by final score
        passage_objs.sort(key=lambda p: p.final_score, reverse=True)
        
        # Update ranks
        for i, passage in enumerate(passage_objs, 1):
            passage.rank = i
        
        # Calculate statistics
        rel_scores = [p.relevance_score for p in passage_objs]
        qual_scores = [p.quality_score for p in passage_objs]
        
        return RankingResult(
            passages=passage_objs,
            total_passages=total,
            selected_passages=len(passage_objs),
            total_tokens=sum(p.token_count for p in passage_objs),
            relevance_stats={
                'mean': sum(rel_scores) / len(rel_scores) if rel_scores else 0,
                'max': max(rel_scores) if rel_scores else 0,
                'min': min(rel_scores) if rel_scores else 0,
            },
            quality_stats={
                'mean': sum(qual_scores) / len(qual_scores) if qual_scores else 0,
                'max': max(qual_scores) if qual_scores else 0,
                'min': min(qual_scores) if qual_scores else 0,
            },
            strategy=SelectionStrategy.TOP_K,
        )
    
    def select_context(
        self,
        passages: List[Union[str, Passage, Dict]],
        query: str,
        token_budget: Optional[int] = None,
        max_passages: Optional[int] = None,
        strategy: Union[str, SelectionStrategy] = SelectionStrategy.TOKEN_BUDGET,
        diversity_threshold: float = 0.3,
    ) -> RankingResult:
        """
        Select optimal context within constraints.
        
        Args:
            passages: List of passages
            query: Query text
            token_budget: Maximum tokens
            max_passages: Maximum passages
            strategy: Selection strategy
            diversity_threshold: Minimum diversity
            
        Returns:
            RankingResult with selected passages
        """
        if isinstance(strategy, str):
            strategy = SelectionStrategy(strategy)
        
        # First, rank all passages
        ranked = self.rank(passages, query)
        
        if strategy == SelectionStrategy.TOP_K:
            k = max_passages or len(ranked.passages)
            selected = ranked.passages[:k]
        
        elif strategy == SelectionStrategy.TOKEN_BUDGET:
            selected = self._select_by_budget(
                ranked.passages,
                token_budget or 4000,
                max_passages,
            )
        
        elif strategy == SelectionStrategy.DIVERSITY:
            selected = self._select_diverse(
                ranked.passages,
                max_passages or 10,
                diversity_threshold,
            )
        
        elif strategy == SelectionStrategy.COVERAGE:
            selected = self._select_by_coverage(
                ranked.passages,
                query,
                max_passages or 10,
            )
        
        elif strategy == SelectionStrategy.GREEDY:
            selected = self._select_greedy(
                ranked.passages,
                query,
                token_budget or 4000,
                max_passages,
            )
        
        else:
            selected = ranked.passages[:max_passages] if max_passages else ranked.passages
        
        # Update ranks
        for i, passage in enumerate(selected, 1):
            passage.rank = i
        
        return RankingResult(
            passages=selected,
            total_passages=len(ranked.passages),
            selected_passages=len(selected),
            total_tokens=sum(p.token_count for p in selected),
            relevance_stats=ranked.relevance_stats,
            quality_stats=ranked.quality_stats,
            strategy=strategy,
        )
    
    def _to_passages(
        self,
        items: List[Union[str, Passage, Dict]],
    ) -> List[Passage]:
        """Convert items to Passage objects."""
        passages = []
        
        for i, item in enumerate(items):
            if isinstance(item, Passage):
                passages.append(item)
            elif isinstance(item, str):
                passages.append(Passage(
                    id=f"passage_{i}",
                    content=item,
                    position=i,
                ))
            elif isinstance(item, dict):
                passages.append(Passage(
                    id=item.get('id', f"passage_{i}"),
                    content=item.get('content', item.get('text', '')),
                    relevance_score=item.get('score', 0.0),
                    source=item.get('source', ''),
                    position=i,
                    metadata=item.get('metadata', {}),
                ))
        
        return passages
    
    def _select_by_budget(
        self,
        passages: List[Passage],
        budget: int,
        max_passages: Optional[int],
    ) -> List[Passage]:
        """Select passages within token budget."""
        selected = []
        total_tokens = 0
        
        for passage in passages:
            if max_passages and len(selected) >= max_passages:
                break
            
            if total_tokens + passage.token_count <= budget:
                selected.append(passage)
                total_tokens += passage.token_count
        
        return selected
    
    def _select_diverse(
        self,
        passages: List[Passage],
        k: int,
        threshold: float,
    ) -> List[Passage]:
        """Select diverse passages."""
        selected = []
        selected_content = []
        
        for passage in passages:
            if len(selected) >= k:
                break
            
            # Check diversity
            diversity = self.diversity_scorer.score_diversity(
                passage.content,
                selected_content,
            )
            
            if diversity >= threshold or not selected:
                selected.append(passage)
                selected_content.append(passage.content)
        
        return selected
    
    def _select_by_coverage(
        self,
        passages: List[Passage],
        query: str,
        k: int,
    ) -> List[Passage]:
        """Select passages maximizing query coverage."""
        query_terms = query.lower().split()
        covered_terms: set = set()
        selected = []
        
        for passage in passages:
            if len(selected) >= k:
                break
            
            coverage, new_terms = self.coverage_scorer.score_coverage(
                passage.content,
                query_terms,
                covered_terms,
            )
            
            if coverage > 0 or not selected:
                selected.append(passage)
                covered_terms.update(new_terms)
        
        return selected
    
    def _select_greedy(
        self,
        passages: List[Passage],
        query: str,
        budget: int,
        max_passages: Optional[int],
    ) -> List[Passage]:
        """Greedy selection based on marginal utility."""
        selected = []
        selected_content = []
        total_tokens = 0
        
        remaining = list(passages)
        query_terms = set(query.lower().split())
        covered_terms: set = set()
        
        while remaining:
            if max_passages and len(selected) >= max_passages:
                break
            
            best_idx = -1
            best_utility = -1.0
            
            for i, passage in enumerate(remaining):
                if total_tokens + passage.token_count > budget:
                    continue
                
                # Calculate marginal utility
                utility = self._marginal_utility(
                    passage,
                    selected_content,
                    query_terms,
                    covered_terms,
                )
                
                if utility > best_utility:
                    best_utility = utility
                    best_idx = i
            
            if best_idx < 0:
                break
            
            # Add best passage
            best = remaining.pop(best_idx)
            selected.append(best)
            selected_content.append(best.content)
            total_tokens += best.token_count
            
            # Update covered terms
            _, new_terms = self.coverage_scorer.score_coverage(
                best.content,
                list(query_terms),
                covered_terms,
            )
            covered_terms.update(new_terms)
        
        return selected
    
    def _marginal_utility(
        self,
        passage: Passage,
        selected_content: List[str],
        query_terms: set,
        covered_terms: set,
    ) -> float:
        """Calculate marginal utility of adding passage."""
        # Relevance component
        relevance = passage.relevance_score
        
        # Diversity component
        diversity = self.diversity_scorer.score_diversity(
            passage.content,
            selected_content,
        )
        
        # Coverage component
        coverage, _ = self.coverage_scorer.score_coverage(
            passage.content,
            list(query_terms),
            covered_terms,
        )
        
        # Combine
        return (
            0.4 * relevance +
            0.3 * diversity +
            0.3 * coverage
        )


# Convenience functions

def rank_passages(
    passages: List[str],
    query: str,
    top_k: int = 10,
) -> List[Passage]:
    """
    Quick passage ranking.
    
    Args:
        passages: List of passage texts
        query: Query text
        top_k: Number of top passages
        
    Returns:
        Ranked passages
    """
    ranker = ContextRanker()
    result = ranker.rank(passages, query)
    return result.top_k(top_k)


def select_context(
    passages: List[str],
    query: str,
    token_budget: int = 4000,
) -> str:
    """
    Select context within token budget.
    
    Args:
        passages: List of passages
        query: Query text
        token_budget: Token budget
        
    Returns:
        Combined context string
    """
    ranker = ContextRanker()
    result = ranker.select_context(
        passages, query,
        token_budget=token_budget,
        strategy=SelectionStrategy.TOKEN_BUDGET,
    )
    return result.get_content()


def score_passage_relevance(
    passage: str,
    query: str,
) -> float:
    """
    Score passage relevance to query.
    
    Args:
        passage: Passage text
        query: Query text
        
    Returns:
        Relevance score (0-1)
    """
    scorer = RelevanceScorer()
    return scorer.score(passage, query)
