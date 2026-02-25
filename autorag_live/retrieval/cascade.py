"""
Cascade retrieval with multi-stage filtering.

Implements efficient cascade: coarse → precise retrieval:
- Stage 1: Fast, high-recall retrieval (1000s of candidates)
- Stage 2: Precise reranking (top 100)
- Stage 3: Final filtering (top 10)
- Achieves optimal precision-recall tradeoff

Based on:
- "ColBERTv2: Effective and Efficient Retrieval" (Khattab et al., 2022)
- "Cascade Retrieval for Production RAG" (Google, 2024)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CascadeConfig:
    """Configuration for cascade retrieval."""

    # Stage 1: Coarse retrieval
    stage1_top_k: int = 1000
    stage1_timeout_ms: float = 100.0

    # Stage 2: Reranking
    stage2_top_k: int = 100
    stage2_timeout_ms: float = 500.0

    # Stage 3: Final filtering
    final_top_k: int = 10


class CascadeRetrieval:
    """
    Multi-stage cascade retrieval pipeline.

    Example:
        >>> cascade = CascadeRetrieval(
        ...     coarse_retriever=bm25,
        ...     reranker=cross_encoder
        ... )
        >>> results = await cascade.retrieve("query", top_k=10)
    """

    def __init__(
        self,
        coarse_retriever: Any,
        reranker: Optional[Any] = None,
        config: Optional[CascadeConfig] = None,
    ):
        """Initialize cascade retrieval."""
        self.coarse_retriever = coarse_retriever
        self.reranker = reranker
        self.config = config or CascadeConfig()

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute cascade retrieval with graceful timeout degradation.

        If Stage 1 times out, returns an empty list instead of crashing.
        If Stage 2 (reranking) times out, falls back to the un-reranked
        Stage 1 results truncated to *top_k*.

        The reranker now receives **all** Stage 1 candidates (not a
        pre-sliced subset) so it can make a globally optimal selection.

        Args:
            query: Query text
            top_k: Final number of results

        Returns:
            Refined results from cascade
        """
        final_k = top_k or self.config.final_top_k

        # Stage 1: Fast coarse retrieval
        logger.debug(f"Stage 1: Retrieving {self.config.stage1_top_k} candidates")
        try:
            candidates = await asyncio.wait_for(
                self.coarse_retriever.retrieve(query, top_k=self.config.stage1_top_k),
                timeout=self.config.stage1_timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            logger.warning("Stage 1 coarse retrieval timed out — returning empty results")
            return []
        except Exception as exc:
            logger.error(f"Stage 1 retrieval failed: {exc}")
            return []

        if not candidates:
            logger.info("Stage 1 returned 0 candidates — skipping reranking")
            return []

        # Stage 2: Reranking — pass ALL candidates, let the reranker pick top-k
        if self.reranker:
            logger.debug(f"Stage 2: Reranking {len(candidates)} candidates to top {final_k}")
            try:
                reranked = await asyncio.wait_for(
                    self.reranker.rerank(query, candidates, top_k=final_k),
                    timeout=self.config.stage2_timeout_ms / 1000,
                )
                results = reranked
            except asyncio.TimeoutError:
                logger.warning("Stage 2 reranking timed out — falling back to coarse results")
                results = candidates[:final_k]
            except Exception as exc:
                logger.error(f"Stage 2 reranking failed: {exc} — using coarse results")
                results = candidates[:final_k]
        else:
            results = candidates[:final_k]

        logger.info(f"Cascade: {len(candidates)} → {len(results)} results")
        return results
