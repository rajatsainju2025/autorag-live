"""
Listwise LLM Reranker (RankGPT).

Implements listwise passage reranking where ALL candidate documents are
scored jointly in a single LLM call, producing a direct ranked permutation.

Unlike pointwise rerankers (one score per doc) or pairwise rerankers
(tournament of comparisons), listwise ranking:
- Observes all documents simultaneously → captures inter-document context
- Produces consistent global ordering in one pass
- Achieves better nDCG@10 on TREC/BEIR benchmarks (Sun et al., 2023)

Sliding Window Strategy
-----------------------
For more than ``window_size`` documents, a sliding window is applied:
documents outside the window are fixed; the model reranks within the
window, then the window slides toward the top. This is equivalent to
bubble-sort with an LLM comparator — O(n/step) passes for a list of n docs.

References
----------
- "Is ChatGPT Good at Search? Investigating Large Language Models as
  Re-Ranking Agents" Sun et al., 2023 (https://arxiv.org/abs/2304.09542)
- "RankVicuna: Zero-Shot Listwise Document Reranking" Pradeep et al., 2023
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

LLMFn = Callable[[str], Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RankedDocument:
    """A document with listwise reranking metadata."""

    doc_id: str
    text: str
    original_rank: int
    final_rank: int = 0
    retrieval_score: float = 0.0
    listwise_rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def promoted(self) -> bool:
        """True if listwise rank is better than original."""
        if self.listwise_rank is None:
            return False
        return self.final_rank < self.original_rank


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are RankGPT, an expert in ranking passages by relevance to a query. "
    "You will be shown a query and a list of passages. "
    "Rank the passages from most to least relevant. "
    "Output ONLY the passage identifiers in ranked order, "
    "separated by ' > '. Do not output anything else."
)

_RANKING_TEMPLATE = (
    "Query: {query}\n\n"
    "Passages:\n{passages}\n\n"
    "Rank the passages above by relevance to the query. "
    "Output the passage identifiers in ranked order (most relevant first), "
    "separated by ' > '. Example format: [2] > [1] > [3]\n\n"
    "Ranking:"
)


def _build_ranking_prompt(query: str, docs: List[RankedDocument]) -> str:
    """Build the listwise ranking prompt for a window of documents."""
    passages = "\n".join(f"[{i + 1}] {doc.text[:400]}" for i, doc in enumerate(docs))
    return _RANKING_TEMPLATE.format(query=query, passages=passages)


def _parse_ranking(raw: str, n_docs: int) -> List[int]:
    """
    Parse LLM output into a 0-indexed permutation list.

    Returns indices into the *docs* list (0-based), most-relevant first.
    Falls back to identity permutation on parse failure.
    """
    # Extract numeric identifiers in order of appearance
    matches = re.findall(r"\[(\d+)\]", raw)
    if not matches:
        # Try plain numbers separated by '>'
        matches = re.findall(r"\b(\d+)\b", raw)

    seen: set[int] = set()
    perm: List[int] = []
    for m in matches:
        idx = int(m) - 1  # 1-based → 0-based
        if 0 <= idx < n_docs and idx not in seen:
            seen.add(idx)
            perm.append(idx)

    # Append any missing indices at the end (keep original relative order)
    for i in range(n_docs):
        if i not in seen:
            perm.append(i)

    return perm


# ---------------------------------------------------------------------------
# Core reranker
# ---------------------------------------------------------------------------


class ListwiseLLMReranker:
    """
    Listwise LLM reranker using a sliding window strategy.

    Args:
        llm_fn: Async ``(prompt: str) → str`` callable.
        window_size: Number of documents visible per LLM call (default 20).
        step_size: Sliding window step (default 10 = 50% overlap).
        top_k: Final number of documents to return.
        concurrency: Maximum parallel LLM calls during windowed passes.

    Example::

        reranker = ListwiseLLMReranker(llm_fn=my_llm)
        results = await reranker.rerank("query", documents, top_k=10)
    """

    def __init__(
        self,
        llm_fn: LLMFn,
        window_size: int = 20,
        step_size: int = 10,
        top_k: int = 10,
        concurrency: int = 4,
    ) -> None:
        self.llm_fn = llm_fn
        self.window_size = window_size
        self.step_size = step_size
        self.top_k = top_k
        self._semaphore = asyncio.Semaphore(concurrency)

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        text_key: str = "text",
        id_key: str = "doc_id",
    ) -> List[RankedDocument]:
        """
        Rerank *documents* for *query* using the listwise LLM strategy.

        Args:
            query: User query.
            documents: Candidate documents (each a dict with at least
                ``text_key`` and optionally ``id_key`` and ``score``).
            top_k: Number of results to return (defaults to ``self.top_k``).
            text_key: Key for document text.
            id_key: Key for document identifier.

        Returns:
            List of :class:`RankedDocument` sorted by final rank (best first).
        """
        k = top_k or self.top_k
        if not documents:
            return []

        # Wrap input dicts into RankedDocument objects
        ranked: List[RankedDocument] = [
            RankedDocument(
                doc_id=str(d.get(id_key, f"doc_{i}")),
                text=str(d.get(text_key, "")),
                original_rank=i,
                retrieval_score=float(d.get("score", 0.0)),
                metadata={kk: vv for kk, vv in d.items() if kk not in (text_key, id_key)},
            )
            for i, d in enumerate(documents)
        ]

        # Single-pass if documents fit in one window
        if len(ranked) <= self.window_size:
            ranked = await self._rank_window(query, ranked)
        else:
            ranked = await self._sliding_window_rank(query, ranked)

        # Assign final ranks
        for i, doc in enumerate(ranked):
            doc.final_rank = i
            doc.listwise_rank = i

        return ranked[:k]

    async def _rank_window(
        self,
        query: str,
        docs: List[RankedDocument],
    ) -> List[RankedDocument]:
        """Rank a single window of documents with one LLM call."""
        prompt = _build_ranking_prompt(query, docs)
        async with self._semaphore:
            try:
                raw = await self.llm_fn(prompt)
            except Exception as exc:
                logger.warning("ListwiseLLMReranker: LLM call failed: %s", exc)
                return docs  # return unchanged on failure

        perm = _parse_ranking(raw, len(docs))
        return [docs[i] for i in perm]

    async def _sliding_window_rank(
        self,
        query: str,
        docs: List[RankedDocument],
    ) -> List[RankedDocument]:
        """
        Apply sliding window reranking (bottom-up bubble approach).

        Starting from the bottom of the list, each window is reranked
        and the window slides toward the top by ``step_size`` positions.
        """
        current = list(docs)
        n = len(current)

        # Slide from bottom to top
        start = max(0, n - self.window_size)
        while start >= 0:
            end = min(start + self.window_size, n)
            window = current[start:end]
            reranked_window = await self._rank_window(query, window)
            current[start:end] = reranked_window

            if start == 0:
                break
            start = max(0, start - self.step_size)

        return current

    async def rerank_with_scores(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        alpha: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Rerank and blend listwise rank with original retrieval score.

        The combined score = alpha × listwise_position_score + (1-alpha) × retrieval_score,
        where listwise_position_score = 1 - rank/n.

        Args:
            query: User query.
            documents: Candidate document dicts.
            top_k: Final number to return.
            alpha: Weight of listwise rank vs retrieval score.

        Returns:
            Original document dicts with ``listwise_rank`` and
            ``combined_score`` fields added, sorted by ``combined_score``.
        """
        reranked = await self.rerank(query, documents, top_k)
        n = len(reranked)

        results: List[Dict[str, Any]] = []
        for doc in reranked:
            listwise_score = 1.0 - (doc.final_rank / max(n, 1))
            ret_score = doc.retrieval_score
            # Normalise retrieval score to [0,1] if it looks like cosine
            ret_score = float(np.clip(ret_score, 0.0, 1.0))
            combined = alpha * listwise_score + (1.0 - alpha) * ret_score
            result = dict(doc.metadata)
            result.update(
                {
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "listwise_rank": doc.final_rank,
                    "retrieval_score": doc.retrieval_score,
                    "combined_score": round(combined, 4),
                }
            )
            results.append(result)

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results
