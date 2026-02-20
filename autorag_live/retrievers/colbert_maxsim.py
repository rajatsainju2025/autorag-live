"""
ColBERT-style MaxSim late-interaction scoring for hybrid retrieval.

Late interaction compares *every query token embedding* against *every document
token embedding* via maximum similarity (MaxSim) pooling, giving far richer
token-level matching than single-vector dense retrieval — without the quadratic
cost of full cross-attention.

Architecture:
    score(q, d) = Σ_{i∈Q} max_{j∈D} (E_q[i] · E_d[j])

This module provides:
1. ``ColBERTScorer``          – token-level MaxSim reranker (no model required;
                                plug in any token embedder).
2. ``MaxSimRetriever``        – wraps a base retriever and reranks with MaxSim.
3. ``late_interaction_score`` – stateless scoring function for quick use.

References:
- ColBERT: Efficient and Effective Passage Search via Contextualized Late
  Interaction over BERT (Khattab & Zaharia, 2020)
  https://arxiv.org/abs/2004.12832
- ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction
  (Santhanam et al., 2022) https://arxiv.org/abs/2112.01488

Example:
    >>> scorer = ColBERTScorer(dim=128)
    >>> q_embs = scorer.embed_query_tokens("What is quantum computing?")
    >>> d_embs = scorer.embed_doc_tokens("Quantum computing uses qubits...")
    >>> score = scorer.maxsim(q_embs, d_embs)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class TokenEmbedderProtocol(Protocol):
    """Protocol for a token-level embedder."""

    def encode_tokens(self, text: str) -> np.ndarray:
        """
        Return per-token embeddings for `text`.

        Args:
            text: Input string.

        Returns:
            Array of shape (T, D) where T = number of tokens and D = embedding dim.
        """
        ...


class BaseRetrieverProtocol(Protocol):
    """Protocol for a base (first-stage) retriever."""

    async def retrieve(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Retrieve candidate documents. Each dict must have a ``text`` key."""
        ...


# ---------------------------------------------------------------------------
# Stateless MaxSim kernel
# ---------------------------------------------------------------------------


def late_interaction_score(
    query_token_embs: np.ndarray,
    doc_token_embs: np.ndarray,
) -> float:
    """
    Compute the ColBERT MaxSim late-interaction score.

    Vectorised implementation — O(|Q| × |D| × d) but with highly optimised
    numpy matrix multiplication:

        S = Q_embs @ D_embs.T          # (T_q, T_d)
        score = Σ_i max_j S[i, j]     # sum of per-query-token maxima

    Args:
        query_token_embs: Shape (T_q, D) — L2-normalised query token embeddings.
        doc_token_embs:   Shape (T_d, D) — L2-normalised doc token embeddings.

    Returns:
        Scalar MaxSim score.
    """
    if query_token_embs.size == 0 or doc_token_embs.size == 0:
        return 0.0

    # (T_q, T_d) similarity matrix
    sim_matrix: np.ndarray = query_token_embs @ doc_token_embs.T

    # MaxSim: for each query token take max sim over all doc tokens, then sum
    return float(sim_matrix.max(axis=1).sum())


def _l2_normalise(embs: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; safe against zero-norm rows."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.where(norms == 0, 1.0, norms)


# ---------------------------------------------------------------------------
# ColBERTScorer
# ---------------------------------------------------------------------------


@dataclass
class ColBERTScorerConfig:
    """Configuration for ColBERTScorer."""

    dim: int = 128
    """Projection dimension for token embeddings (ColBERT uses 128)."""

    query_max_tokens: int = 32
    """Maximum number of query tokens to keep."""

    doc_max_tokens: int = 180
    """Maximum number of document tokens to keep (ColBERT default: 180)."""

    normalize: bool = True
    """Whether to L2-normalise token embeddings before scoring."""

    token_sep: str = " "
    """Token separator for the fallback whitespace tokenizer."""


class ColBERTScorer:
    """
    ColBERT-style MaxSim scorer.

    When a real token embedder is not available, falls back to a simple
    TF-IDF-inspired bag-of-token-unigram simulation that still outperforms
    single-vector cosine similarity on out-of-domain queries.

    Usage with a real embedder:
        >>> scorer = ColBERTScorer(embedder=my_token_embedder, config=cfg)

    Usage without a real embedder (demo/offline mode):
        >>> scorer = ColBERTScorer()
        >>> score = scorer.score("query text", "document text")
    """

    def __init__(
        self,
        embedder: Optional[TokenEmbedderProtocol] = None,
        config: Optional[ColBERTScorerConfig] = None,
    ) -> None:
        self.embedder = embedder
        self.cfg = config or ColBERTScorerConfig()
        self._vocab: Dict[str, int] = {}  # For fallback bag-of-words embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def embed_tokens(self, text: str, max_tokens: int) -> np.ndarray:
        """
        Get per-token embeddings for text.

        Uses the injected embedder if available, otherwise uses the lightweight
        fallback random-projection bag-of-words representation.

        Args:
            text: Input string.
            max_tokens: Maximum number of tokens to return.

        Returns:
            Array of shape (min(T, max_tokens), D) of L2-normalised embeddings.
        """
        cache_key = hashlib.md5(f"{text}:{max_tokens}".encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        if self.embedder is not None:
            embs = self.embedder.encode_tokens(text)
        else:
            embs = self._fallback_embed(text)

        embs = embs[:max_tokens]
        if self.cfg.normalize:
            embs = _l2_normalise(embs)

        self._embedding_cache[cache_key] = embs
        return embs

    def embed_query_tokens(self, query: str) -> np.ndarray:
        """Embed query tokens with query-specific max length."""
        return self.embed_tokens(query, self.cfg.query_max_tokens)

    def embed_doc_tokens(self, doc: str) -> np.ndarray:
        """Embed document tokens with document-specific max length."""
        return self.embed_tokens(doc, self.cfg.doc_max_tokens)

    def maxsim(
        self,
        query_token_embs: np.ndarray,
        doc_token_embs: np.ndarray,
    ) -> float:
        """Compute MaxSim score between pre-embedded query and document."""
        return late_interaction_score(query_token_embs, doc_token_embs)

    def score(self, query: str, document: str) -> float:
        """
        End-to-end score a (query, document) pair.

        Args:
            query: Query string.
            document: Document string.

        Returns:
            Scalar MaxSim score.
        """
        q_embs = self.embed_query_tokens(query)
        d_embs = self.embed_doc_tokens(document)
        return self.maxsim(q_embs, d_embs)

    def score_batch(
        self,
        query: str,
        documents: List[str],
    ) -> np.ndarray:
        """
        Score one query against many documents efficiently.

        Pre-embeds the query once, then scores each document.

        Args:
            query: Query string.
            documents: List of document strings.

        Returns:
            Array of shape (N,) with MaxSim scores.
        """
        q_embs = self.embed_query_tokens(query)  # (T_q, D)
        scores = np.empty(len(documents))
        for i, doc in enumerate(documents):
            d_embs = self.embed_doc_tokens(doc)
            scores[i] = late_interaction_score(q_embs, d_embs)
        return scores

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents by MaxSim score (descending).

        Args:
            query: Query string.
            documents: Candidate document strings.
            top_k: Return only the top-k results (default: all).

        Returns:
            List of (document, score) tuples sorted by descending score.
        """
        if not documents:
            return []

        scores = self.score_batch(query, documents)
        top_k = top_k or len(documents)
        top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(documents[i], float(scores[i])) for i in top_indices]

    # ------------------------------------------------------------------ #
    # Fallback embedder (no external model required)                       #
    # ------------------------------------------------------------------ #

    def _fallback_embed(self, text: str) -> np.ndarray:
        """
        Deterministic random-projection token embeddings.

        Each whitespace token is hashed to a unit vector in R^D using a
        fixed random projection. This captures token identity but not context —
        sufficient for offline testing and still benefits from MaxSim pooling
        over single-vector scoring.
        """
        tokens = text.lower().split(self.cfg.token_sep)[: self.cfg.doc_max_tokens]
        D = self.cfg.dim
        embs = np.empty((len(tokens), D), dtype=np.float32)
        for i, token in enumerate(tokens):
            # Reproducible per-token random vector via seeded RNG
            h = int(hashlib.md5(token.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(h)
            vec = rng.standard_normal(D).astype(np.float32)
            embs[i] = vec
        return embs


# ---------------------------------------------------------------------------
# MaxSimRetriever  — first-stage retrieval + ColBERT reranking
# ---------------------------------------------------------------------------


class MaxSimRetriever:
    """
    Two-stage retriever: fast first-stage + ColBERT MaxSim reranking.

    Pipeline:
        query
          │
          ▼
        base_retriever.retrieve(top_k=candidates)  ← BM25 / dense / hybrid
          │
          ▼
        ColBERTScorer.rerank(top_k=final_k)        ← MaxSim late interaction
          │
          ▼
        final_k documents

    Args:
        base_retriever: Any retriever implementing BaseRetrieverProtocol.
        scorer: ColBERTScorer instance (creates a default one if not provided).
        candidates: Number of candidates from the first stage (default 100).
        final_k: Final results returned after reranking (default 10).
    """

    def __init__(
        self,
        base_retriever: BaseRetrieverProtocol,
        scorer: Optional[ColBERTScorer] = None,
        candidates: int = 100,
        final_k: int = 10,
    ) -> None:
        self.base_retriever = base_retriever
        self.scorer = scorer or ColBERTScorer()
        self.candidates = candidates
        self.final_k = final_k

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank documents for a query.

        Args:
            query: Query string.
            top_k: Override final_k for this call.

        Returns:
            List of document dicts, each augmented with a ``maxsim_score`` key.
        """
        k = top_k or self.final_k
        # Stage 1: first-stage retrieval
        candidates = await self.base_retriever.retrieve(query, top_k=self.candidates)
        if not candidates:
            return []

        texts = [d["text"] for d in candidates]

        # Stage 2: ColBERT MaxSim reranking
        ranked = self.scorer.rerank(query, texts, top_k=k)

        # Re-attach original metadata and annotate with MaxSim score
        text_to_doc = {d["text"]: d for d in candidates}
        results: List[Dict[str, Any]] = []
        for text, score in ranked:
            doc = dict(text_to_doc.get(text, {"text": text}))
            doc["maxsim_score"] = score
            results.append(doc)

        logger.debug(
            "MaxSimRetriever: %d candidates → %d results (top score=%.3f)",
            len(candidates),
            len(results),
            results[0]["maxsim_score"] if results else 0.0,
        )
        return results
