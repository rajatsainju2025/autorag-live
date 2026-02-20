"""
MMR Diversity Reranker.

Maximal Marginal Relevance (MMR) post-retrieval deduplication and diversity
re-ranking for agentic RAG pipelines.

Why MMR matters
---------------
Standard top-k retrieval returns the k most relevant chunks, but relevance
alone favours *near-duplicate* passages about the same sentence.  The LLM
then sees the same fact repeated multiple times while missing other relevant
perspectives.  MMR trades off relevance vs. redundancy:

    score(d) = λ · sim(d, query) - (1 - λ) · max_{s ∈ S} sim(d, s)

where S is the set of already-selected documents.

References
----------
- Carbonell & Goldstein, 1998. "The use of MMR, diversity-based reranking for
  reordering documents and producing summaries." SIGIR '98.
- Ye et al., 2023. "Complementary Explanations for Effective In-Context
  Learning." ACL 2023 (extends MMR to RAG few-shot).

Features
--------
- Async embedding with configurable concurrency semaphore
- Plug-in ``EmbedFn`` — no hard dependency on any provider
- ``alpha`` (λ) parameter: 1.0 = pure relevance, 0.0 = pure diversity
- ``cross_encoder_scores`` optional dict for hybrid reranking:
  ``score = alpha · mmr + (1 - alpha) · cross_encoder_score``
- Returns same dict format as input with ``mmr_rank`` and ``mmr_score`` fields
- ``MMRRerankerConfig`` dataclass for typed config
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

EmbedFn = Callable[[str], Coroutine[Any, Any, List[float]]]
"""Async function: text → dense embedding vector."""

Document = Dict[str, Any]
"""Retrieved document dict.  Must have a ``"text"`` key (str)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MMRConfig:
    """Typed configuration for :class:`MMRReranker`."""

    alpha: float = 0.7
    """λ in the MMR formula.  Higher → more relevance-focused."""

    top_k: Optional[int] = None
    """Return at most *top_k* documents.  ``None`` returns all ranked docs."""

    max_concurrency: int = 8
    """Maximum concurrent embedding calls."""

    text_field: str = "text"
    """Key in each document dict that holds the passage text."""

    score_field: str = "score"
    """Key in each document dict that holds the original retrieval score."""

    embed_batch_size: int = 32
    """Number of texts per embedding call (single-text embed called in parallel)."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class MMRResult:
    """Single ranked document with MMR metadata."""

    document: Document
    mmr_rank: int
    mmr_score: float
    original_score: float
    marginal_relevance: float
    max_similarity_to_selected: float


# ---------------------------------------------------------------------------
# Core MMR logic
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Numerically stable cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def _cosine_matrix(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """
    Return cosine similarities of *query* (shape D,) against each row of
    *docs* (shape N × D).
    """
    norms = np.linalg.norm(docs, axis=1, keepdims=True).clip(min=1e-10)
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-10:
        return np.zeros(len(docs))
    return (docs / norms) @ (query / query_norm)


def _mmr_select(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    alpha: float,
    k: int,
) -> list[int]:
    """
    Greedy MMR selection returning *k* document indices in ranked order.

    Parameters
    ----------
    query_emb:  shape (D,)
    doc_embs:   shape (N, D) — rows are candidate document embeddings
    alpha:      λ parameter (relevance weight)
    k:          number of items to select

    Returns
    -------
    List of selected indices (0-based into *doc_embs*), length ≤ min(k, N).
    """
    n = len(doc_embs)
    k = min(k, n)
    if k <= 0:
        return []

    relevance = _cosine_matrix(query_emb, doc_embs)  # shape (N,)
    selected: list[int] = []
    remaining = list(range(n))

    for _ in range(k):
        if not remaining:
            break
        if not selected:
            # First pick: pure relevance
            best_idx = max(remaining, key=lambda i: relevance[i])
        else:
            sel_embs = doc_embs[selected]  # shape (|S|, D)
            best_score = -1e18
            best_idx = remaining[0]
            for i in remaining:
                rel_score = alpha * relevance[i]
                sim_to_selected = max(_cosine(doc_embs[i], s_emb) for s_emb in sel_embs)
                div_score = (1.0 - alpha) * sim_to_selected
                mmr_score = rel_score - div_score
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class MMRReranker:
    """
    Post-retrieval MMR diversity reranker.

    Embeds the query and all retrieved documents, then applies greedy MMR
    to return a diversity-aware ranking that reduces near-duplicate exposure.

    Args:
        embed_fn: Async callable returning a dense embedding for a text string.
        config: :class:`MMRConfig` controlling λ, top_k, concurrency, etc.

    Example::

        reranker = MMRReranker(embed_fn=my_embed, config=MMRConfig(alpha=0.7, top_k=5))
        docs = [{"text": "...", "score": 0.9}, ...]
        results = await reranker.rerank("my query", docs)
        for r in results:
            print(r.mmr_rank, r.mmr_score, r.document["text"][:60])
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        config: Optional[MMRConfig] = None,
    ) -> None:
        self._embed = embed_fn
        self.config = config or MMRConfig()
        self._sem = asyncio.Semaphore(self.config.max_concurrency)

    async def _embed_safe(self, text: str) -> np.ndarray:
        async with self._sem:
            vec = await self._embed(text)
        return np.asarray(vec, dtype=np.float32)

    async def _embed_all(self, texts: list[str]) -> np.ndarray:
        """Embed all texts concurrently (semaphore-bounded)."""
        tasks = [asyncio.create_task(self._embed_safe(t)) for t in texts]
        arrays = await asyncio.gather(*tasks)
        # Pad to common dimension if providers return different sizes (rare)
        max_d = max(a.shape[0] for a in arrays)
        padded = np.zeros((len(arrays), max_d), dtype=np.float32)
        for i, a in enumerate(arrays):
            padded[i, : a.shape[0]] = a
        return padded

    async def rerank(
        self,
        query: str,
        documents: List[Document],
        cross_encoder_scores: Optional[Dict[int, float]] = None,
        alpha: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[MMRResult]:
        """
        Rerank *documents* using MMR against *query*.

        Parameters
        ----------
        query:
            The original retrieval query string.
        documents:
            List of document dicts; each must contain a text field
            (``config.text_field``).  Retrieval score is read from
            ``config.score_field`` (defaults to 1.0 if missing).
        cross_encoder_scores:
            Optional mapping ``{doc_index: score}`` from a cross-encoder.
            When provided, ``final_score = alpha·mmr + (1-alpha)·ce``.
        alpha:
            Override ``config.alpha`` for this call.
        top_k:
            Override ``config.top_k`` for this call.

        Returns
        -------
        List of :class:`MMRResult` in MMR-ranked order (rank 1 = best).
        """
        if not documents:
            return []

        lam = alpha if alpha is not None else self.config.alpha
        k = top_k if top_k is not None else (self.config.top_k or len(documents))

        text_field = self.config.text_field
        score_field = self.config.score_field

        # ------------------------------------------------------------------
        # Embed query + all documents concurrently
        # ------------------------------------------------------------------
        texts = [query] + [d.get(text_field, "") for d in documents]
        all_embs = await self._embed_all(texts)
        query_emb = all_embs[0]
        doc_embs = all_embs[1:]

        # ------------------------------------------------------------------
        # Greedy MMR selection
        # ------------------------------------------------------------------
        selected_indices = _mmr_select(query_emb, doc_embs, lam, k)

        # ------------------------------------------------------------------
        # Build result objects with scores
        # ------------------------------------------------------------------
        relevance = _cosine_matrix(query_emb, doc_embs)
        results: list[MMRResult] = []
        already_selected: list[int] = []

        for rank, idx in enumerate(selected_indices, start=1):
            rel = float(relevance[idx])
            if already_selected:
                sel_embs = doc_embs[already_selected]
                max_sim = max(_cosine(doc_embs[idx], s_emb) for s_emb in sel_embs)
            else:
                max_sim = 0.0

            mmr_score_val = lam * rel - (1.0 - lam) * max_sim

            if cross_encoder_scores is not None and idx in cross_encoder_scores:
                ce = cross_encoder_scores[idx]
                final_score = lam * mmr_score_val + (1.0 - lam) * ce
            else:
                final_score = mmr_score_val

            orig = float(documents[idx].get(score_field, 1.0))

            results.append(
                MMRResult(
                    document=documents[idx],
                    mmr_rank=rank,
                    mmr_score=round(final_score, 6),
                    original_score=orig,
                    marginal_relevance=round(rel, 6),
                    max_similarity_to_selected=round(max_sim, 6),
                )
            )
            already_selected.append(idx)

        logger.debug(
            "MMRReranker: %d docs → %d selected (λ=%.2f)",
            len(documents),
            len(results),
            lam,
        )
        return results

    async def rerank_to_dicts(
        self,
        query: str,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[Document]:
        """
        Convenience wrapper: returns plain dicts with ``mmr_rank`` and
        ``mmr_score`` injected.  Useful for dropping in as a pipeline stage.
        """
        results = await self.rerank(query, documents, **kwargs)
        enriched: list[Document] = []
        for r in results:
            doc = dict(r.document)
            doc["mmr_rank"] = r.mmr_rank
            doc["mmr_score"] = r.mmr_score
            enriched.append(doc)
        return enriched


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_mmr_reranker(
    embed_fn: EmbedFn,
    *,
    alpha: float = 0.7,
    top_k: Optional[int] = None,
    max_concurrency: int = 8,
    text_field: str = "text",
    score_field: str = "score",
) -> MMRReranker:
    """
    Factory for :class:`MMRReranker` with keyword-only config.

    Example::

        reranker = create_mmr_reranker(embed_fn, alpha=0.6, top_k=8)
        docs = await reranker.rerank_to_dicts(query, retrieved_docs)
    """
    cfg = MMRConfig(
        alpha=alpha,
        top_k=top_k,
        max_concurrency=max_concurrency,
        text_field=text_field,
        score_field=score_field,
    )
    return MMRReranker(embed_fn, cfg)
