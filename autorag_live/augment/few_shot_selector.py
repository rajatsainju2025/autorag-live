"""
Dynamic Few-Shot Example Selector (DSPy-style).

Selects the most informative and diverse demonstrations from an example
store for in-context learning.  Unlike static few-shot prompts, this
module retrieves examples that are:

1. **Semantically similar** to the current query (retrieved via cosine
   similarity on cached embeddings).
2. **Diverse** — Maximum Marginal Relevance (MMR) ensures the selected
   set is not redundant, covering different aspects of the query space.
3. **Scored** — examples with better known outcomes (e.g. verified
   correct answers) are preferred via a quality-weighted selection.

This follows the retrieval-augmented prompting paradigm from DSPy
(Khattab et al., 2023) but without requiring the full DSPy framework —
the entire module is pure Python + NumPy.

References
----------
- "DSPy: Compiling Declarative Language Model Calls" Khattab et al., 2023
- "In-Context Learning" (Brown et al., 2020, GPT-3)
- "Diverse Demonstrations Improve In-Context Compositional Generalisation"
  (Su et al., 2023)
- "MMR: Maximal Marginal Relevance" (Carbonell & Goldstein, 1998)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Coroutine[Any, Any, List[float]]]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FewShotExample:
    """
    A single demonstration example.

    Attributes:
        query: The example input query.
        answer: The expected / gold answer.
        context: Optional retrieved context used in this example.
        quality_score: Known quality of this example (0–1; default 0.5).
        metadata: Arbitrary metadata (source, date, domain, etc.).
        embedding: Cached embedding vector (populated lazily).
    """

    query: str
    answer: str
    context: str = ""
    quality_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def example_id(self) -> str:
        """Stable ID derived from query content."""
        return hashlib.md5(self.query.encode(), usedforsecurity=False).hexdigest()[:12]

    def to_prompt_str(self, include_context: bool = True) -> str:
        """Format as a prompt-ready string."""
        parts: List[str] = []
        if include_context and self.context:
            parts.append(f"Context: {self.context[:400]}")
        parts.append(f"Question: {self.query}")
        parts.append(f"Answer: {self.answer}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# MMR selection
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _mmr_select(
    query_emb: np.ndarray,
    candidate_embs: np.ndarray,
    candidate_indices: List[int],
    k: int,
    lambda_mmr: float = 0.7,
) -> List[int]:
    """
    Select *k* indices from candidates using Maximum Marginal Relevance.

    MMR balances relevance to the query (λ) with diversity from already-
    selected examples (1-λ).  Higher λ → more similar to query; lower
    λ → more diverse.

    Args:
        query_emb: Query embedding vector.
        candidate_embs: (n, d) matrix of candidate embeddings.
        candidate_indices: Mapping from row index → original index.
        k: Number of examples to select.
        lambda_mmr: Trade-off parameter ∈ [0, 1].

    Returns:
        List of selected indices (into the original example list).
    """
    if k <= 0 or len(candidate_indices) == 0:
        return []

    # Query similarity scores for all candidates
    q_norm = np.linalg.norm(query_emb)
    if q_norm > 0:
        q_sims = candidate_embs @ (query_emb / q_norm)
        norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        q_sims = q_sims / norms.squeeze()
    else:
        q_sims = np.zeros(len(candidate_indices))

    selected: List[int] = []
    remaining = list(range(len(candidate_indices)))

    for _ in range(min(k, len(remaining))):
        best_idx: int = -1
        best_score = float("-inf")

        for ri in remaining:
            relevance = float(q_sims[ri])
            if selected:
                # Max similarity to already-selected examples
                sel_embs = candidate_embs[selected]
                cand_emb = candidate_embs[ri]
                cn = np.linalg.norm(cand_emb)
                if cn > 0:
                    sn = np.linalg.norm(sel_embs, axis=1, keepdims=True)
                    sn = np.where(sn == 0, 1.0, sn)
                    sim_to_sel = float((sel_embs @ cand_emb / cn / sn.squeeze()).max())
                else:
                    sim_to_sel = 0.0
                score = lambda_mmr * relevance - (1.0 - lambda_mmr) * sim_to_sel
            else:
                score = relevance

            if score > best_score:
                best_score = score
                best_idx = ri

        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidate_indices[i] for i in selected]


# ---------------------------------------------------------------------------
# Example store
# ---------------------------------------------------------------------------


class FewShotExampleStore:
    """
    In-memory store of :class:`FewShotExample` objects.

    Supports async embedding (lazy, cached) and O(1) lookup by ID.

    Args:
        embed_fn: Async ``(text: str) → List[float]`` callable.
            If ``None``, a deterministic hash-based fallback is used
            (not suitable for production — always provide a real embedder).
        embed_query_only: If True, embed the query field; otherwise
            embed ``query + answer`` for richer matching.
    """

    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = None,
        embed_query_only: bool = True,
    ) -> None:
        self._embed_fn = embed_fn
        self.embed_query_only = embed_query_only
        self._examples: List[FewShotExample] = []
        self._id_index: Dict[str, int] = {}

    def add(self, example: FewShotExample) -> None:
        """Add a single example to the store."""
        eid = example.example_id
        if eid in self._id_index:
            return  # dedup by query content
        self._id_index[eid] = len(self._examples)
        self._examples.append(example)

    def add_batch(self, examples: List[FewShotExample]) -> None:
        """Add multiple examples."""
        for ex in examples:
            self.add(ex)

    def __len__(self) -> int:
        return len(self._examples)

    async def _embed(self, text: str) -> np.ndarray:
        """Return embedding as numpy array."""
        if self._embed_fn is not None:
            vec = await self._embed_fn(text)
            return np.asarray(vec, dtype=float)
        # Hash-based fallback (deterministic, low quality)
        digest = hashlib.sha256(text.encode()).digest()
        arr = np.frombuffer(digest, dtype=np.uint8).astype(float)
        arr = arr / (arr.max() + 1e-9)
        return arr

    async def _ensure_embeddings(self) -> np.ndarray:
        """Compute missing embeddings, return (n, d) matrix."""
        import asyncio

        tasks = []
        needs_embed: List[int] = []
        for i, ex in enumerate(self._examples):
            if ex.embedding is None:
                text = ex.query if self.embed_query_only else f"{ex.query} {ex.answer}"
                needs_embed.append(i)
                tasks.append(self._embed(text))

        if tasks:
            results = await asyncio.gather(*tasks)
            for i, emb in zip(needs_embed, results):
                self._examples[i].embedding = emb

        # Stack into matrix — all embeddings must have same shape
        embs = [ex.embedding for ex in self._examples if ex.embedding is not None]
        if not embs:
            return np.zeros((0, 1))

        # Pad to common dimension
        max_dim = max(e.shape[0] for e in embs)
        padded = [np.pad(e, (0, max_dim - e.shape[0])) for e in embs]
        return np.stack(padded)


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


class DynamicFewShotSelector:
    """
    Selects diverse, relevant few-shot examples for a given query.

    Args:
        store: :class:`FewShotExampleStore` instance.
        k: Number of examples to select (default 5).
        lambda_mmr: MMR diversity weight ∈ [0, 1] (default 0.7).
        quality_boost: Factor by which high-quality examples are upweighted
            when scoring (default 0.2).

    Example::

        store = FewShotExampleStore(embed_fn=my_embed)
        store.add_batch(training_examples)
        selector = DynamicFewShotSelector(store, k=4)
        examples = await selector.select("What is quantum entanglement?")
        prompt_prefix = selector.build_prompt(examples)
    """

    def __init__(
        self,
        store: FewShotExampleStore,
        k: int = 5,
        lambda_mmr: float = 0.7,
        quality_boost: float = 0.2,
    ) -> None:
        self.store = store
        self.k = k
        self.lambda_mmr = lambda_mmr
        self.quality_boost = quality_boost

    async def select(
        self,
        query: str,
        k: Optional[int] = None,
        filter_fn: Optional[Callable[[FewShotExample], bool]] = None,
    ) -> List[FewShotExample]:
        """
        Select the best few-shot examples for *query*.

        Args:
            query: Current user query.
            k: Number of examples to return (defaults to ``self.k``).
            filter_fn: Optional predicate to exclude certain examples
                (e.g. filter by domain metadata).

        Returns:
            List of selected :class:`FewShotExample` objects, ordered
            from most to least relevant after MMR diversification.
        """
        n_select = k or self.k
        if not self.store._examples:
            return []

        # Apply optional filter
        candidates = self.store._examples
        if filter_fn is not None:
            candidates = [ex for ex in candidates if filter_fn(ex)]
        if not candidates:
            return []

        # Embed query and ensure candidate embeddings are ready
        query_emb = await self.store._embed(query)
        emb_matrix = await self.store._ensure_embeddings()

        if emb_matrix.shape[0] == 0:
            return candidates[:n_select]

        # Get candidate row indices (may be a subset if filter_fn used)
        all_examples = self.store._examples
        cand_indices = [all_examples.index(ex) for ex in candidates if ex in all_examples]
        if not cand_indices:
            return candidates[:n_select]

        # Sub-matrix for candidates
        cand_embs = emb_matrix[cand_indices]

        # Adjust query embedding dimension to match
        dim = cand_embs.shape[1]
        if query_emb.shape[0] < dim:
            query_emb = np.pad(query_emb, (0, dim - query_emb.shape[0]))
        else:
            query_emb = query_emb[:dim]

        # Apply quality boost to query similarity
        if self.quality_boost > 0:
            quality_weights = np.array(
                [
                    candidates[i].quality_score if i < len(candidates) else 0.5
                    for i in range(len(cand_indices))
                ]
            )
            # Scale candidate embeddings by quality weight (soft reweighting)
            boost = 1.0 + self.quality_boost * (quality_weights - 0.5)
            cand_embs = cand_embs * boost[:, np.newaxis]

        # MMR selection
        local_indices = list(range(len(cand_indices)))
        selected_local = _mmr_select(query_emb, cand_embs, local_indices, n_select, self.lambda_mmr)
        return [candidates[i] for i in selected_local if i < len(candidates)]

    @staticmethod
    def build_prompt(
        examples: List[FewShotExample],
        include_context: bool = True,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Format selected examples into a few-shot prompt prefix.

        Args:
            examples: Selected examples.
            include_context: Whether to include example contexts.
            separator: String separating examples.

        Returns:
            Formatted few-shot prompt string.
        """
        if not examples:
            return ""
        parts = [ex.to_prompt_str(include_context) for ex in examples]
        return separator.join(parts) + separator
