"""
Lost-in-the-Middle Position-Aware Reranker.

Addresses the "Lost in the Middle" problem documented in:
  "Lost in the Middle: How Language Models Use Long Contexts"
  (Liu et al., 2023 - https://arxiv.org/abs/2307.03172)

Key finding: LLMs perform best when relevant context is at the
beginning or end of the context window. Performance degrades for
documents placed in the middle of long contexts.

This reranker reorders retrieved documents to place the highest-scoring
ones at the front and back of the context, pushing lower-ranked docs
to the middle — inverting the typical descending score order.

Strategy (default "sandwich"):
    [rank 1, rank 3, rank 5, ..., rank 6, rank 4, rank 2]
    ↑ most relevant                            most relevant ↑
    (placed at start)                          (placed at end)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ReorderStrategy(str, Enum):
    """Available reordering strategies."""

    SANDWICH = "sandwich"  # Best docs at start + end; rest in middle
    PRIMACY = "primacy"  # Best docs at start only (primacy bias)
    RECENCY = "recency"  # Best docs at end only  (recency bias)
    INTERLEAVE = "interleave"  # Alternate high/low across positions


@dataclass
class ScoredDocument:
    """A document with an associated relevance score."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    original_rank: int = 0


@dataclass
class RerankedResult:
    """Result from the position-aware reranker."""

    documents: List[ScoredDocument]
    strategy: ReorderStrategy
    original_order: List[int]  # Maps new_position → original_rank
    n_docs: int


class LostInMiddleReranker:
    """
    Position-aware reranker that mitigates the Lost-in-the-Middle effect.

    Example::

        reranker = LostInMiddleReranker(strategy=ReorderStrategy.SANDWICH)
        docs = [ScoredDocument(content="...", score=0.9, original_rank=0), ...]
        result = reranker.rerank(docs)
        context = "\\n\\n".join(d.content for d in result.documents)
    """

    def __init__(
        self,
        strategy: ReorderStrategy = ReorderStrategy.SANDWICH,
        top_k: int | None = None,
    ) -> None:
        """
        Initialize the reranker.

        Args:
            strategy: Reordering strategy to apply.
            top_k: If set, only the top_k documents by score are kept before reordering.
        """
        self.strategy = strategy
        self.top_k = top_k

    def rerank(self, documents: Sequence[ScoredDocument]) -> RerankedResult:
        """
        Reorder documents to mitigate position bias in LLM context processing.

        Args:
            documents: Documents to reorder, each with a relevance score.

        Returns:
            RerankedResult with the reordered document list and metadata.
        """
        if not documents:
            return RerankedResult(
                documents=[],
                strategy=self.strategy,
                original_order=[],
                n_docs=0,
            )

        # Sort descending by score, assign original ranks
        sorted_docs = sorted(enumerate(documents), key=lambda x: x[1].score, reverse=True)
        ranked: List[ScoredDocument] = []
        for new_rank, (orig_idx, doc) in enumerate(sorted_docs):
            d = ScoredDocument(
                content=doc.content,
                score=doc.score,
                metadata=doc.metadata,
                original_rank=orig_idx,
            )
            ranked.append(d)

        # Apply top_k filter
        if self.top_k is not None:
            ranked = ranked[: self.top_k]

        reordered = self._apply_strategy(ranked)
        original_order = [d.original_rank for d in reordered]

        logger.debug(
            "LostInMiddleReranker: %s strategy applied to %d docs → order %s",
            self.strategy.value,
            len(ranked),
            original_order,
        )

        return RerankedResult(
            documents=reordered,
            strategy=self.strategy,
            original_order=original_order,
            n_docs=len(reordered),
        )

    def _apply_strategy(self, ranked: List[ScoredDocument]) -> List[ScoredDocument]:
        """Dispatch to the chosen reordering strategy."""
        if self.strategy == ReorderStrategy.SANDWICH:
            return self._sandwich(ranked)
        if self.strategy == ReorderStrategy.PRIMACY:
            return ranked  # Already sorted best-first
        if self.strategy == ReorderStrategy.RECENCY:
            return list(reversed(ranked))
        if self.strategy == ReorderStrategy.INTERLEAVE:
            return self._interleave(ranked)
        return ranked

    @staticmethod
    def _sandwich(ranked: List[ScoredDocument]) -> List[ScoredDocument]:
        """
        Interleave top-ranked docs at start/end, lower-ranked in middle.

        For n docs with ranks [1..n]:
          - Odd-indexed (1, 3, 5, ...) → front, in order
          - Even-indexed (2, 4, 6, ...) → back, in reverse order

        Result: [r1, r3, r5, ..., r6, r4, r2]
        """
        front: List[ScoredDocument] = []
        back: List[ScoredDocument] = []

        for i, doc in enumerate(ranked):
            if i % 2 == 0:
                front.append(doc)
            else:
                back.insert(0, doc)  # Prepend so highest goes to end

        return front + back

    @staticmethod
    def _interleave(ranked: List[ScoredDocument]) -> List[ScoredDocument]:
        """
        Alternate high and low ranked documents.

        Result: [r1, r_n, r2, r_n-1, ...]
        This ensures no two adjacent positions are both low-quality.
        """
        high = ranked[: len(ranked) // 2 + len(ranked) % 2]
        low = list(reversed(ranked[len(ranked) // 2 + len(ranked) % 2 :]))

        result: List[ScoredDocument] = []
        for i, h in enumerate(high):
            result.append(h)
            if i < len(low):
                result.append(low[i])
        return result

    def format_context(self, result: RerankedResult, separator: str = "\n\n---\n\n") -> str:
        """
        Format reranked documents into a single context string.

        Args:
            result: Reranked result from :meth:`rerank`.
            separator: String placed between documents.

        Returns:
            Concatenated context string ready for LLM input.
        """
        parts = []
        for i, doc in enumerate(result.documents):
            parts.append(f"[Document {i + 1}]\n{doc.content}")
        return separator.join(parts)
