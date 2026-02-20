"""
HyDE: Hypothetical Document Embeddings for zero-shot dense retrieval.

HyDE bridges the vocabulary gap between sparse queries and dense document
embeddings by:
  1. Asking the LLM to generate a *hypothetical* document that would answer
     the query.
  2. Embedding the hypothetical document (not the query) in the same space
     as the corpus.
  3. Using that embedding as the retrieval vector — it is "denser" and more
     similar to actual relevant documents than a raw query embedding.

This implementation extends the original HyDE paper with:
- **Multi-hypothesis ensembling**: generate k hypotheses and average their
  embeddings (reduces variance from single-hypothesis noise).
- **Hybrid HyDE**: linearly interpolates the HyDE embedding with the raw
  query embedding so that exact-match queries are not degraded.
- **Async-native**: all LLM calls and embedding calls are fully concurrent.
- **Fallback**: if the LLM call fails, silently falls back to raw query embedding.

References:
- Precise Zero-Shot Dense Retrieval without Relevance Labels (Gao et al., 2022)
  https://arxiv.org/abs/2212.10496
- HyDE++ (Zheng et al., 2023) — multi-hypothesis ensembling extension.

Example:
    >>> hyde = HyDERetriever(
    ...     llm=my_llm, embedder=my_embedder, base_retriever=my_retriever,
    ...     n_hypotheses=3, alpha=0.6
    ... )
    >>> results = await hyde.retrieve("Why did Rome fall?", top_k=5)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
LLMFn = Callable[[str], Coroutine[Any, Any, str]]
EmbedFn = Callable[[List[str]], Coroutine[Any, Any, np.ndarray]]
RetrieveFn = Callable[[np.ndarray, int], Coroutine[Any, Any, List[Dict[str, Any]]]]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_HYDE_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Write a short, factual document (2-4 sentences) "
    "that directly and completely answers the following question. "
    "Do NOT include the question itself; write as if you are the document being retrieved."
)

_HYDE_PROMPT_TEMPLATE = "{system}\n\nQuestion: {query}\n\nDocument:"


def build_hyde_prompt(query: str, domain_hint: str = "") -> str:
    """
    Build the HyDE generation prompt.

    Args:
        query: The user query.
        domain_hint: Optional domain context (e.g. "medical", "legal") to
                     steer the hypothetical document style.

    Returns:
        Full prompt string for the LLM.
    """
    system = _HYDE_SYSTEM_PROMPT
    if domain_hint:
        system = f"{system} Focus on {domain_hint} knowledge."
    return _HYDE_PROMPT_TEMPLATE.format(system=system, query=query)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HyDEResult:
    """Result from a HyDE retrieval call."""

    query: str
    hypotheses: List[str]
    """The k generated hypothetical documents."""

    hyde_embedding: np.ndarray
    """The averaged (and optionally interpolated) retrieval embedding."""

    documents: List[Dict[str, Any]]
    """Retrieved documents, each with at least a 'text' key."""

    latency_ms: float = 0.0
    n_hypotheses_used: int = 0
    fallback_used: bool = False
    """True if HyDE failed and raw query embedding was used instead."""

    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HyDE Retriever
# ---------------------------------------------------------------------------


class HyDERetriever:
    """
    Two-stage retriever: HyDE embedding → dense retrieval.

    Args:
        llm: Async callable (prompt) → hypothetical document text.
        embedder: Async callable (texts) → (N, D) embedding matrix.
        retrieve_by_vector: Async callable (embedding, top_k) → list of docs.
        n_hypotheses: Number of hypothetical docs to generate (default 3).
        alpha: Interpolation weight for HyDE embedding vs raw query embedding.
               1.0 = pure HyDE, 0.0 = pure query (default 0.7).
        domain_hint: Optional domain label for prompt steering.
        timeout_s: Per-hypothesis LLM timeout (default 20.0).
        max_parallel: Max concurrent LLM generation tasks (default 3).
    """

    def __init__(
        self,
        llm: LLMFn,
        embedder: EmbedFn,
        retrieve_by_vector: RetrieveFn,
        n_hypotheses: int = 3,
        alpha: float = 0.7,
        domain_hint: str = "",
        timeout_s: float = 20.0,
        max_parallel: int = 3,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.retrieve_by_vector = retrieve_by_vector
        self.n_hypotheses = n_hypotheses
        self.alpha = alpha
        self.domain_hint = domain_hint
        self.timeout_s = timeout_s
        self._semaphore = asyncio.Semaphore(max_parallel)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> HyDEResult:
        """
        Retrieve documents using HyDE.

        Args:
            query: The user query string.
            top_k: Number of documents to return.

        Returns:
            HyDEResult with documents and generation metadata.
        """
        start = time.perf_counter()

        # ── Step 1: Generate hypothetical documents in parallel ────────
        hypotheses = await self._generate_hypotheses(query)
        fallback_used = False

        # ── Step 2: Embed hypotheses + query ──────────────────────────
        if hypotheses:
            hyde_embedding = await self._build_hyde_embedding(query, hypotheses)
        else:
            logger.warning("HyDE: no hypotheses generated, falling back to raw query embedding")
            fallback_used = True
            query_emb = await self.embedder([query])
            hyde_embedding = query_emb[0]

        # ── Step 3: Retrieve by embedding ─────────────────────────────
        documents = await self.retrieve_by_vector(hyde_embedding, top_k)

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "HyDE: %d hypotheses → %d docs (alpha=%.2f, fallback=%s, %.0fms)",
            len(hypotheses),
            len(documents),
            self.alpha,
            fallback_used,
            latency_ms,
        )

        return HyDEResult(
            query=query,
            hypotheses=hypotheses,
            hyde_embedding=hyde_embedding,
            documents=documents,
            latency_ms=latency_ms,
            n_hypotheses_used=len(hypotheses),
            fallback_used=fallback_used,
        )

    async def retrieve_hybrid(
        self,
        query: str,
        base_results: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Merge HyDE results with pre-fetched base retrieval results.

        Useful when BM25 / sparse retrieval results are already available:
        deduplicates and re-ranks by combining HyDE and base scores.

        Args:
            query: Query string.
            base_results: Existing retrieval results (each with 'text' key).
            top_k: Final number of results.

        Returns:
            Merged and de-duplicated list of documents.
        """
        hyde_result = await self.retrieve(query, top_k=top_k)
        seen_texts: set = set()
        merged: List[Dict[str, Any]] = []

        for doc in hyde_result.documents:
            t = doc.get("text", "")
            if t not in seen_texts:
                doc["hyde_rank"] = len(merged)
                merged.append(doc)
                seen_texts.add(t)

        for doc in base_results:
            t = doc.get("text", "")
            if t not in seen_texts:
                merged.append(doc)
                seen_texts.add(t)

        return merged[:top_k]

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _generate_hypotheses(self, query: str) -> List[str]:
        """Generate n_hypotheses hypothetical documents concurrently."""
        prompt = build_hyde_prompt(query, self.domain_hint)
        tasks = [
            asyncio.create_task(self._safe_generate(prompt, i)) for i in range(self.n_hypotheses)
        ]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]

    async def _safe_generate(self, prompt: str, idx: int) -> Optional[str]:
        """Single hypothesis generation with semaphore + timeout guard."""
        async with self._semaphore:
            try:
                text = await asyncio.wait_for(self.llm(prompt), timeout=self.timeout_s)
                return text.strip() or None
            except asyncio.TimeoutError:
                logger.warning("HyDE hypothesis %d timed out", idx)
                return None
            except Exception as exc:
                logger.warning("HyDE hypothesis %d failed: %s", idx, exc)
                return None

    async def _build_hyde_embedding(
        self,
        query: str,
        hypotheses: List[str],
    ) -> np.ndarray:
        """
        Embed all hypotheses, average them, and interpolate with query embedding.

        The interpolation formula is:
            hyde_vec = alpha * mean(hypothesis_embeddings) + (1-alpha) * query_vec

        All normalised to unit length after combination.
        """
        # Embed hypotheses and query in one batched call
        all_texts = hypotheses + [query]
        emb_matrix = await self.embedder(all_texts)  # (N+1, D)

        hyp_embs = emb_matrix[: len(hypotheses)]  # (N, D)
        query_emb = emb_matrix[-1]  # (D,)

        # Average hypothesis embeddings
        mean_hyp = hyp_embs.mean(axis=0)  # (D,)

        # Interpolate
        combined = self.alpha * mean_hyp + (1.0 - self.alpha) * query_emb

        # Normalise to unit sphere
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm

        return combined


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_hyde_retriever(
    llm: LLMFn,
    embedder: EmbedFn,
    retrieve_by_vector: RetrieveFn,
    n_hypotheses: int = 3,
    alpha: float = 0.7,
    domain_hint: str = "",
) -> HyDERetriever:
    """
    Create a HyDERetriever with sensible defaults.

    Args:
        llm: Async LLM callable.
        embedder: Async batch embedder.
        retrieve_by_vector: Async vector retrieval function.
        n_hypotheses: Number of hypotheses (default 3).
        alpha: HyDE / query blend weight (default 0.7).
        domain_hint: Optional domain label.

    Returns:
        Configured HyDERetriever.
    """
    return HyDERetriever(
        llm=llm,
        embedder=embedder,
        retrieve_by_vector=retrieve_by_vector,
        n_hypotheses=n_hypotheses,
        alpha=alpha,
        domain_hint=domain_hint,
    )
