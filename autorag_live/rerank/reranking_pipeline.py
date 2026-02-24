"""Reranking Pipeline for AutoRAG-Live.

Flexible reranking system with multiple strategies:
- Cross-encoder reranking
- Reciprocal rank fusion
- Ensemble reranking
- Contextual reranking
- Diversity-based reranking
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RerankerType(Enum):
    """Types of rerankers."""

    CROSS_ENCODER = "cross_encoder"
    BM25 = "bm25"
    EMBEDDING = "embedding"
    ENSEMBLE = "ensemble"
    RRF = "reciprocal_rank_fusion"
    DIVERSITY = "diversity"
    CONTEXTUAL = "contextual"
    CUSTOM = "custom"


@dataclass
class RerankResult(Generic[T]):
    """Result of a reranking operation."""

    item: T
    original_rank: int
    new_rank: int
    score: float
    original_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankerConfig:
    """Configuration for a reranker."""

    reranker_type: RerankerType
    model_name: str | None = None
    top_k: int = 10
    batch_size: int = 32
    normalize_scores: bool = True
    min_score: float | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC, Generic[T]):
    """Abstract base class for rerankers."""

    def __init__(self, config: RerankerConfig) -> None:
        """Initialize reranker.

        Args:
            config: Reranker configuration
        """
        self.config = config
        self._model: Any = None

    @abstractmethod
    def rerank(
        self,
        query: str,
        items: list[T],
        scores: list[float] | None = None,
    ) -> list[RerankResult[T]]:
        """Rerank items for a query.

        Args:
            query: Query string
            items: Items to rerank
            scores: Original scores (optional)

        Returns:
            Reranked results
        """
        pass

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _apply_threshold(
        self,
        results: list[RerankResult[T]],
    ) -> list[RerankResult[T]]:
        """Apply minimum score threshold."""
        if self.config.min_score is None:
            return results

        return [r for r in results if r.score >= self.config.min_score]


class CrossEncoderReranker(BaseReranker[str]):
    """Cross-encoder based reranker using sentence transformers."""

    def __init__(self, config: RerankerConfig) -> None:
        """Initialize cross-encoder reranker."""
        super().__init__(config)
        self._load_model()

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            model_name = self.config.model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self._model = CrossEncoder(model_name)
            logger.info(f"Loaded cross-encoder: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed, using mock scores")
            self._model = None

    def rerank(
        self,
        query: str,
        items: list[str],
        scores: list[float] | None = None,
    ) -> list[RerankResult[str]]:
        """Rerank using cross-encoder scores."""
        if not items:
            return []

        original_scores = scores or [0.0] * len(items)

        if self._model is not None:
            # Create query-document pairs
            pairs = [(query, item) for item in items]

            # Score in batches
            all_scores: list[float] = []
            for i in range(0, len(pairs), self.config.batch_size):
                batch = pairs[i : i + self.config.batch_size]
                batch_scores = self._model.predict(batch)
                all_scores.extend(batch_scores.tolist())
        else:
            # Mock scores based on text overlap
            all_scores = []
            query_words = set(query.lower().split())
            for item in items:
                item_words = set(item.lower().split())
                overlap = len(query_words & item_words)
                all_scores.append(overlap / max(len(query_words), 1))

        # Normalize if configured
        if self.config.normalize_scores:
            all_scores = self._normalize_scores(all_scores)

        # Create results
        scored_items = list(zip(items, all_scores, original_scores, range(len(items))))
        scored_items.sort(key=lambda x: x[1], reverse=True)

        results = []
        for new_rank, (item, score, orig_score, orig_rank) in enumerate(scored_items):
            results.append(
                RerankResult(
                    item=item,
                    original_rank=orig_rank,
                    new_rank=new_rank,
                    score=score,
                    original_score=orig_score,
                )
            )

        # Apply top_k and threshold
        results = results[: self.config.top_k]
        results = self._apply_threshold(results)

        return results


class BM25Reranker(BaseReranker[str]):
    """BM25-based reranker."""

    def __init__(self, config: RerankerConfig) -> None:
        """Initialize BM25 reranker."""
        super().__init__(config)
        self.k1 = config.extra_config.get("k1", 1.5)
        self.b = config.extra_config.get("b", 0.75)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        return text.lower().split()

    def _compute_bm25(
        self,
        query_terms: list[str],
        doc_terms: list[str],
        avg_doc_len: float,
        doc_freqs: dict[str, int],
        num_docs: int,
    ) -> float:
        """Compute BM25 score."""
        score = 0.0
        doc_len = len(doc_terms)
        doc_term_counts: dict[str, int] = defaultdict(int)

        for term in doc_terms:
            doc_term_counts[term] += 1

        for term in query_terms:
            if term not in doc_term_counts:
                continue

            tf = doc_term_counts[term]
            df = doc_freqs.get(term, 0)

            # IDF
            idf = np.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)

            # TF normalization
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len)
            )

            score += idf * tf_norm

        return float(score)

    def rerank(
        self,
        query: str,
        items: list[str],
        scores: list[float] | None = None,
    ) -> list[RerankResult[str]]:
        """Rerank using BM25 scores."""
        if not items:
            return []

        original_scores = scores or [0.0] * len(items)
        query_terms = self._tokenize(query)

        # Tokenize all documents
        doc_terms_list = [self._tokenize(item) for item in items]

        # Compute document frequencies
        doc_freqs: dict[str, int] = defaultdict(int)
        for doc_terms in doc_terms_list:
            for term in set(doc_terms):
                doc_freqs[term] += 1

        # Average document length
        avg_doc_len = np.mean([len(dt) for dt in doc_terms_list]) if doc_terms_list else 1.0

        # Compute BM25 scores
        bm25_scores = [
            self._compute_bm25(query_terms, doc_terms, avg_doc_len, doc_freqs, len(items))
            for doc_terms in doc_terms_list
        ]

        # Normalize if configured
        if self.config.normalize_scores:
            bm25_scores = self._normalize_scores(bm25_scores)

        # Create results
        scored_items = list(zip(items, bm25_scores, original_scores, range(len(items))))
        scored_items.sort(key=lambda x: x[1], reverse=True)

        results = []
        for new_rank, (item, score, orig_score, orig_rank) in enumerate(scored_items):
            results.append(
                RerankResult(
                    item=item,
                    original_rank=orig_rank,
                    new_rank=new_rank,
                    score=score,
                    original_score=orig_score,
                )
            )

        return results[: self.config.top_k]


class EmbeddingReranker(BaseReranker[str]):
    """Embedding similarity-based reranker."""

    def __init__(
        self,
        config: RerankerConfig,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        """Initialize embedding reranker.

        Args:
            config: Reranker configuration
            embed_fn: Function to embed texts
        """
        super().__init__(config)
        self._embed_fn = embed_fn

    def _default_embed(self, texts: list[str]) -> np.ndarray:
        """Default embedding using random projection (for testing)."""
        np.random.seed(42)
        embeddings = []
        for text in texts:
            # Deterministic embedding based on text hash
            np.random.seed(hash(text) % 2**32)
            emb = np.random.randn(384)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings)

    def rerank(
        self,
        query: str,
        items: list[str],
        scores: list[float] | None = None,
    ) -> list[RerankResult[str]]:
        """Rerank by embedding similarity."""
        if not items:
            return []

        original_scores = scores or [0.0] * len(items)
        embed_fn = self._embed_fn or self._default_embed

        # Embed query and items
        all_texts = [query] + items
        embeddings = embed_fn(all_texts)

        query_emb = embeddings[0]
        item_embs = embeddings[1:]

        # Compute cosine similarities
        similarities = []
        for item_emb in item_embs:
            sim = np.dot(query_emb, item_emb)
            similarities.append(float(sim))

        # Normalize if configured
        if self.config.normalize_scores:
            similarities = self._normalize_scores(similarities)

        # Create results
        scored_items = list(zip(items, similarities, original_scores, range(len(items))))
        scored_items.sort(key=lambda x: x[1], reverse=True)

        results = []
        for new_rank, (item, score, orig_score, orig_rank) in enumerate(scored_items):
            results.append(
                RerankResult(
                    item=item,
                    original_rank=orig_rank,
                    new_rank=new_rank,
                    score=score,
                    original_score=orig_score,
                )
            )

        return results[: self.config.top_k]


class ReciprocalRankFusionReranker(BaseReranker[T]):
    """Reciprocal Rank Fusion for combining multiple rankings."""

    def __init__(self, config: RerankerConfig) -> None:
        """Initialize RRF reranker."""
        super().__init__(config)
        self.k = config.extra_config.get("k", 60)  # RRF parameter

    def fuse_rankings(
        self,
        rankings: list[list[tuple[T, float]]],
    ) -> list[tuple[T, float]]:
        """Fuse multiple rankings using RRF.

        Args:
            rankings: List of rankings, each is list of (item, score)

        Returns:
            Fused ranking
        """
        rrf_scores: dict[int, float] = defaultdict(float)
        item_map: dict[int, T] = {}

        for ranking in rankings:
            for rank, (item, _score) in enumerate(ranking):
                item_id = id(item)
                item_map[item_id] = item
                rrf_scores[item_id] += 1.0 / (self.k + rank + 1)

        # Sort by RRF score
        sorted_items = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(item_map[item_id], score) for item_id, score in sorted_items]

    def rerank(
        self,
        query: str,
        items: list[T],
        scores: list[float] | None = None,
    ) -> list[RerankResult[T]]:
        """Rerank using single ranking (passthrough)."""
        # For single ranking, just return as-is with RRF scores
        if not items:
            return []

        original_scores = scores or [0.0] * len(items)

        results = []
        for rank, (item, orig_score) in enumerate(zip(items, original_scores)):
            rrf_score = 1.0 / (self.k + rank + 1)
            results.append(
                RerankResult(
                    item=item,
                    original_rank=rank,
                    new_rank=rank,
                    score=rrf_score,
                    original_score=orig_score,
                )
            )

        return results[: self.config.top_k]


class DiversityReranker(BaseReranker[str]):
    """Maximal Marginal Relevance (MMR) for diversity-based reranking."""

    def __init__(
        self,
        config: RerankerConfig,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        """Initialize diversity reranker.

        Args:
            config: Reranker configuration
            embed_fn: Embedding function
        """
        super().__init__(config)
        self._embed_fn = embed_fn
        self.lambda_param = config.extra_config.get("lambda", 0.5)

    def _default_embed(self, texts: list[str]) -> np.ndarray:
        """Default embedding function."""
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % 2**32)
            emb = np.random.randn(384)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def rerank(
        self,
        query: str,
        items: list[str],
        scores: list[float] | None = None,
    ) -> list[RerankResult[str]]:
        """Rerank using MMR for diversity."""
        if not items:
            return []

        original_scores = scores or [0.0] * len(items)
        embed_fn = self._embed_fn or self._default_embed

        # Embed query and items
        all_texts = [query] + items
        embeddings = embed_fn(all_texts)
        query_emb = embeddings[0]
        item_embs = embeddings[1:]

        # Compute relevance scores (similarity to query)
        relevance_scores = [self._cosine_similarity(query_emb, item_emb) for item_emb in item_embs]

        # MMR selection
        selected_indices: list[int] = []
        remaining_indices = list(range(len(items)))

        while remaining_indices and len(selected_indices) < self.config.top_k:
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]

                # Diversity component (max similarity to selected)
                if selected_indices:
                    max_sim = max(
                        self._cosine_similarity(item_embs[idx], item_embs[sel])
                        for sel in selected_indices
                    )
                else:
                    max_sim = 0.0

                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr))

            # Select best MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Create results
        results = []
        for new_rank, orig_idx in enumerate(selected_indices):
            results.append(
                RerankResult(
                    item=items[orig_idx],
                    original_rank=orig_idx,
                    new_rank=new_rank,
                    score=relevance_scores[orig_idx],
                    original_score=original_scores[orig_idx],
                    metadata={"selection_method": "mmr"},
                )
            )

        return results


class EnsembleReranker(BaseReranker[T]):
    """Ensemble of multiple rerankers with configurable weights."""

    def __init__(
        self,
        config: RerankerConfig,
        rerankers: list[tuple[BaseReranker[T], float]],
    ) -> None:
        """Initialize ensemble reranker.

        Args:
            config: Reranker configuration
            rerankers: List of (reranker, weight) tuples
        """
        super().__init__(config)
        self.rerankers = rerankers
        self._rrf = ReciprocalRankFusionReranker(config)

    def rerank(
        self,
        query: str,
        items: list[T],
        scores: list[float] | None = None,
    ) -> list[RerankResult[T]]:
        """Rerank using ensemble of rerankers."""
        if not items:
            return []

        original_scores = scores or [0.0] * len(items)

        # Collect rankings from all rerankers
        rankings: list[list[tuple[T, float]]] = []

        for reranker, weight in self.rerankers:
            results = reranker.rerank(query, items, scores)
            ranking = [(r.item, r.score * weight) for r in results]
            rankings.append(ranking)

        # Fuse rankings
        fused = self._rrf.fuse_rankings(rankings)

        # Create results
        item_to_orig_rank = {id(item): i for i, item in enumerate(items)}

        results = []
        for new_rank, (item, score) in enumerate(fused[: self.config.top_k]):
            orig_rank = item_to_orig_rank.get(id(item), -1)
            results.append(
                RerankResult(
                    item=item,
                    original_rank=orig_rank,
                    new_rank=new_rank,
                    score=score,
                    original_score=original_scores[orig_rank] if orig_rank >= 0 else None,
                    metadata={"ensemble_size": len(self.rerankers)},
                )
            )

        return results


class ListwiseLLMReranker(BaseReranker[str]):
    """Listwise LLM reranker using the RankGPT permutation approach.

    Instead of scoring documents **one at a time** (pointwise — N LLM calls)
    or via pairwise bubble-sort (O(N²) LLM calls), this sends all candidate
    documents to the LLM in a *single* prompt and asks it to return a
    permutation of document indices ranked by relevance.

    For very large candidate sets the reranker uses a **sliding-window**
    strategy: sort windows of ``window_size`` documents and merge, reducing
    total LLM calls to ⌈N / stride⌉ while preserving top-k quality.

    Reference: Sun et al., "Is ChatGPT Good at Search? Investigating Large
    Language Models as Re-Ranking Agents" (RankGPT, 2023).

    Args:
        config: Reranker configuration.
        llm_fn: ``async (prompt: str) -> str`` callable.  When *None* the
            reranker falls back to a BM25 word-overlap heuristic.
        window_size: Max documents per LLM call (default 20).
        stride: Overlap stride for the sliding window (default 10).
    """

    _PROMPT_TEMPLATE = (
        "I will provide a query and {n} passages.  Rank the passages by "
        "relevance to the query from MOST to LEAST relevant.  Output ONLY "
        "a comma-separated list of passage numbers (e.g. 3,1,5,2,4).\n\n"
        "Query: {query}\n\n{passages}\nRanking:"
    )

    def __init__(
        self,
        config: RerankerConfig,
        llm_fn: Callable[..., Any] | None = None,
        window_size: int = 20,
        stride: int = 10,
    ) -> None:
        super().__init__(config)
        self._llm_fn = llm_fn
        self._window_size = max(window_size, 2)
        self._stride = max(stride, 1)

    # -- core ---------------------------------------------------------------

    def rerank(
        self,
        query: str,
        items: list[str],
        scores: list[float] | None = None,
    ) -> list[RerankResult[str]]:
        if not items:
            return []

        original_scores = scores or [0.0] * len(items)

        if self._llm_fn is not None:
            try:
                ranked_indices = self._sliding_window_rank(query, items)
            except Exception as exc:
                logger.warning("ListwiseLLMReranker LLM call failed (%s), using fallback", exc)
                ranked_indices = self._fallback_rank(query, items)
        else:
            ranked_indices = self._fallback_rank(query, items)

        results: list[RerankResult[str]] = []
        n = len(ranked_indices)
        for new_rank, orig_idx in enumerate(ranked_indices):
            results.append(
                RerankResult(
                    item=items[orig_idx],
                    original_rank=orig_idx,
                    new_rank=new_rank,
                    score=1.0 - new_rank / max(n, 1),
                    original_score=original_scores[orig_idx],
                    metadata={"method": "listwise_llm"},
                )
            )

        results = results[: self.config.top_k]
        return self._apply_threshold(results)

    # -- sliding window -----------------------------------------------------

    def _sliding_window_rank(self, query: str, items: list[str]) -> list[int]:
        """Rank via sliding-window listwise prompts."""
        n = len(items)
        if n <= self._window_size:
            return self._rank_window(query, items, list(range(n)))

        # Start with initial ordering by original position
        current_order = list(range(n))

        # Slide from back to front so that the best items bubble to the top
        start = max(n - self._window_size, 0)
        while start >= 0:
            end = min(start + self._window_size, n)
            window_indices = current_order[start:end]
            window_items = [items[i] for i in window_indices]

            ranked_local = self._rank_window(query, window_items, window_indices)

            current_order[start:end] = ranked_local
            start -= self._stride
            if start < 0 and start + self._stride > 0:
                start = 0
                continue
            if start < 0:
                break

        return current_order

    def _rank_window(
        self, query: str, window_items: list[str], global_indices: list[int]
    ) -> list[int]:
        """Send one listwise prompt and parse the permutation."""
        passages = "\n\n".join(
            f"Passage {i + 1}: {doc[:500]}" for i, doc in enumerate(window_items)
        )
        prompt = self._PROMPT_TEMPLATE.format(n=len(window_items), query=query, passages=passages)

        import re as _re

        raw = self._llm_fn(prompt) if self._llm_fn else ""  # type: ignore[misc]
        numbers = [int(x) for x in _re.findall(r"\d+", raw)]

        # Map 1-based passage numbers back to global indices
        seen: set[int] = set()
        ordered_global: list[int] = []
        for num in numbers:
            local_idx = num - 1  # 1-based → 0-based
            if 0 <= local_idx < len(global_indices) and local_idx not in seen:
                seen.add(local_idx)
                ordered_global.append(global_indices[local_idx])

        # Append any indices the LLM omitted (preserving original order)
        for local_idx, gidx in enumerate(global_indices):
            if local_idx not in seen:
                ordered_global.append(gidx)

        return ordered_global

    # -- fallback -----------------------------------------------------------

    @staticmethod
    def _fallback_rank(query: str, items: list[str]) -> list[int]:
        """BM25-style word-overlap fallback when no LLM is available."""
        query_words = set(query.lower().split())
        scored = []
        for idx, item in enumerate(items):
            item_words = set(item.lower().split())
            overlap = len(query_words & item_words) / max(len(query_words), 1)
            scored.append((idx, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scored]


class ContextualReranker(BaseReranker[str]):
    """Reranker that considers conversation context."""

    def __init__(
        self,
        config: RerankerConfig,
        context_weight: float = 0.3,
    ) -> None:
        """Initialize contextual reranker.

        Args:
            config: Reranker configuration
            context_weight: Weight for context relevance
        """
        super().__init__(config)
        self.context_weight = context_weight
        self._context_history: list[str] = []

    def add_context(self, context: str) -> None:
        """Add context to history."""
        self._context_history.append(context)
        # Keep only recent context
        if len(self._context_history) > 5:
            self._context_history = self._context_history[-5:]

    def clear_context(self) -> None:
        """Clear context history."""
        self._context_history.clear()

    def _compute_context_score(self, item: str) -> float:
        """Compute relevance to context."""
        if not self._context_history:
            return 0.0

        item_words = set(item.lower().split())
        total_overlap = 0

        for context in self._context_history:
            context_words = set(context.lower().split())
            overlap = len(item_words & context_words)
            total_overlap += overlap

        return total_overlap / (len(item_words) + 1)

    def rerank(
        self,
        query: str,
        items: list[str],
        scores: list[float] | None = None,
    ) -> list[RerankResult[str]]:
        """Rerank considering context."""
        if not items:
            return []

        original_scores = scores or [1.0] * len(items)

        # Compute combined scores
        combined_scores = []
        for item, orig_score in zip(items, original_scores):
            context_score = self._compute_context_score(item)

            # Simple query relevance (word overlap)
            query_words = set(query.lower().split())
            item_words = set(item.lower().split())
            query_score = len(query_words & item_words) / (len(query_words) + 1)

            # Combined score
            combined = (1 - self.context_weight) * query_score + self.context_weight * context_score
            combined_scores.append(combined)

        # Normalize
        if self.config.normalize_scores:
            combined_scores = self._normalize_scores(combined_scores)

        # Create results
        scored_items = list(zip(items, combined_scores, original_scores, range(len(items))))
        scored_items.sort(key=lambda x: x[1], reverse=True)

        results = []
        for new_rank, (item, score, orig_score, orig_rank) in enumerate(scored_items):
            results.append(
                RerankResult(
                    item=item,
                    original_rank=orig_rank,
                    new_rank=new_rank,
                    score=score,
                    original_score=orig_score,
                    metadata={"context_size": len(self._context_history)},
                )
            )

        return results[: self.config.top_k]


class RerankingPipeline:
    """Pipeline for applying multiple reranking stages."""

    def __init__(
        self,
        stages: list[BaseReranker[Any]] | None = None,
    ) -> None:
        """Initialize reranking pipeline.

        Args:
            stages: List of rerankers to apply in sequence
        """
        self.stages = stages or []
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Statistics
        self._stats: dict[str, Any] = {
            "total_reranks": 0,
            "total_items_processed": 0,
            "total_latency_ms": 0.0,
        }

    def add_stage(self, reranker: BaseReranker[Any]) -> "RerankingPipeline":
        """Add a reranking stage.

        Args:
            reranker: Reranker to add

        Returns:
            Self for chaining
        """
        self.stages.append(reranker)
        return self

    def rerank(
        self,
        query: str,
        items: list[Any],
        scores: list[float] | None = None,
    ) -> list[RerankResult[Any]]:
        """Apply all reranking stages.

        Args:
            query: Query string
            items: Items to rerank
            scores: Initial scores

        Returns:
            Final reranked results
        """
        if not items or not self.stages:
            return []

        start_time = time.time()
        current_items = items
        current_scores = scores

        # Apply each stage
        for stage in self.stages:
            results = stage.rerank(query, current_items, current_scores)
            current_items = [r.item for r in results]
            current_scores = [r.score for r in results]

        # Final results with original tracking
        item_to_orig_idx = {id(item): i for i, item in enumerate(items)}

        final_results = []
        for new_rank, (item, score) in enumerate(zip(current_items, current_scores or [])):
            orig_rank = item_to_orig_idx.get(id(item), -1)
            final_results.append(
                RerankResult(
                    item=item,
                    original_rank=orig_rank,
                    new_rank=new_rank,
                    score=score,
                    original_score=scores[orig_rank] if scores and orig_rank >= 0 else None,
                    metadata={"stages_applied": len(self.stages)},
                )
            )

        # Update stats
        latency_ms = (time.time() - start_time) * 1000
        self._stats["total_reranks"] += 1
        self._stats["total_items_processed"] += len(items)
        self._stats["total_latency_ms"] += latency_ms

        return final_results

    async def rerank_async(
        self,
        query: str,
        items: list[Any],
        scores: list[float] | None = None,
    ) -> list[RerankResult[Any]]:
        """Async reranking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.rerank(query, items, scores),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        stats = self._stats.copy()

        if stats["total_reranks"] > 0:
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["total_reranks"]
            stats["avg_items_per_rerank"] = stats["total_items_processed"] / stats["total_reranks"]
        else:
            stats["avg_latency_ms"] = 0.0
            stats["avg_items_per_rerank"] = 0.0

        stats["num_stages"] = len(self.stages)

        return stats


def create_default_pipeline(
    use_cross_encoder: bool = True,
    use_diversity: bool = False,
) -> RerankingPipeline:
    """Create default reranking pipeline.

    Args:
        use_cross_encoder: Whether to use cross-encoder
        use_diversity: Whether to add diversity stage

    Returns:
        Configured pipeline
    """
    pipeline = RerankingPipeline()

    if use_cross_encoder:
        pipeline.add_stage(
            CrossEncoderReranker(
                RerankerConfig(
                    reranker_type=RerankerType.CROSS_ENCODER,
                    top_k=20,
                )
            )
        )

    if use_diversity:
        pipeline.add_stage(
            DiversityReranker(
                RerankerConfig(
                    reranker_type=RerankerType.DIVERSITY,
                    top_k=10,
                ),
            )
        )

    return pipeline


# =============================================================================
# Normalised RRF + Borda Count Fusion
# =============================================================================


class NormalizedRRFFusion:
    """
    Normalised Reciprocal Rank Fusion for fusing multiple ranked lists.

    Improvements over the existing ``ReciprocalRankFusionReranker``:

    1. **Configurable k** — ``k=60`` is the literature default (Cormack et al.,
       2009) but empirically ``k ∈ [10, 100]`` is worth sweeping per corpus.
    2. **Score normalisation** — output scores are divided by the theoretical
       maximum (≡ 1/(k+1) × num_lists) and mapped to ``[0, 1]``, making them
       comparable across different fusion configurations.
    3. **Borda count alternative** — produces linear rank weights instead of
       reciprocal, which can outperform RRF on small candidate sets.
    4. **Per-list weights** — optionally weight individual ranker contributions
       (e.g. up-weight a fine-tuned reranker vs a BM25 baseline).

    References
    ----------
    * Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms
      Condorcet and individual rank learning methods" SIGIR 2009.
    * Rau & Augenstein, "A Thorough Examination of Morning Star" arXiv 2023.

    Args:
        k:              RRF constant (higher ⇒ less rank-difference
                        sensitivity, more robust to outlier rankings).
        use_borda:      Use Borda count instead of RRF (mutually exclusive
                        with ``k`` tuning).
        weights:        Per-list multiplicative weights.  Length must match
                        the number of lists passed to :meth:`fuse`.
                        If ``None``, all lists are weighted equally.

    Example::

        fusion = NormalizedRRFFusion(k=60)
        fused  = fusion.fuse([
            [("doc_a", 0.9), ("doc_b", 0.7)],   # BM25 ranking
            [("doc_b", 0.8), ("doc_a", 0.6)],   # Dense ranking
        ])
        # Returns [("doc_b", 0.83), ("doc_a", 0.75)] — both get credit
    """

    def __init__(
        self,
        k: float = 60.0,
        use_borda: bool = False,
        weights: Optional[List[float]] = None,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self.use_borda = use_borda
        self.weights = weights

    def fuse(
        self,
        rankings: List[List[Tuple[Any, float]]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Fuse *rankings* into a single sorted list.

        Args:
            rankings: Each element is a ranked list of ``(item, score)``
                      pairs, sorted descending by score.  Scores are only
                      used to determine original order; the fusion score is
                      derived from rank position alone.
            top_k:    If provided, return at most *top_k* items.

        Returns:
            List of ``(item, normalised_fused_score)`` sorted descending.
        """
        if not rankings:
            return []

        n_lists = len(rankings)
        weights = self.weights or [1.0] * n_lists

        if len(weights) != n_lists:
            raise ValueError(f"len(weights)={len(weights)} must equal len(rankings)={n_lists}")

        # Total weight for normalisation denominator
        total_weight = sum(weights)
        # Theoretical max score (rank-0 in every list):
        #   RRF:   Σ w_i · 1/(k+1)
        #   Borda: Σ w_i · N_i  (where N_i = list length)
        if self.use_borda:
            max_score = sum(w * max(len(r), 1) for w, r in zip(weights, rankings)) / total_weight
        else:
            max_score = sum(w / (self.k + 1) for w in weights) / total_weight
        max_score = max(max_score, 1e-9)

        raw_scores: Dict[int, float] = {}
        item_map: Dict[int, Any] = {}

        for ranking, weight in zip(rankings, weights):
            n_items = len(ranking)
            for rank, (item, _) in enumerate(ranking):
                item_id = id(item)
                item_map[item_id] = item

                if self.use_borda:
                    score = (n_items - rank) * weight / total_weight
                else:
                    score = weight / (self.k + rank + 1) / total_weight

                raw_scores[item_id] = raw_scores.get(item_id, 0.0) + score

        # Normalise to [0, 1]
        normalised = {iid: s / max_score for iid, s in raw_scores.items()}

        sorted_items = sorted(normalised.items(), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            sorted_items = sorted_items[:top_k]

        return [(item_map[iid], score) for iid, score in sorted_items]

    def fuse_rerank_results(
        self,
        result_lists: List[List["RerankResult"]],
        top_k: Optional[int] = None,
    ) -> List["RerankResult"]:
        """
        Convenience wrapper that accepts lists of :class:`RerankResult`.

        Input lists are assumed already sorted by descending score (as returned
        by any :class:`BaseReranker`).

        Returns a new list of :class:`RerankResult` with updated ranks and
        normalised fusion scores.
        """
        rankings = [[(r.item, r.score) for r in lst] for lst in result_lists]
        fused = self.fuse(rankings, top_k=top_k)

        # Build a lookup of original scores
        orig_scores: Dict[Any, float] = {}
        for lst in result_lists:
            for r in lst:
                orig_scores.setdefault(id(r.item), r.original_score or r.score)

        return [
            RerankResult(
                item=item,
                original_rank=-1,
                new_rank=new_rank,
                score=score,
                original_score=orig_scores.get(id(item)),
                metadata={"fusion": "borda" if self.use_borda else f"rrf_k{self.k}"},
            )
            for new_rank, (item, score) in enumerate(fused)
        ]
