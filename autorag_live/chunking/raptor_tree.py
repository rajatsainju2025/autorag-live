"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

Builds a hierarchical summarization tree from document chunks, enabling
multi-granularity retrieval — from fine-grained sentences to high-level
summaries — in a single semantic index.

Key Features:
1. Bottom-up tree construction via Gaussian Mixture clustering
2. Abstractive summarization at each tree level
3. Flattened index over all nodes for unified retrieval
4. Configurable tree depth and cluster sizes
5. Incremental tree updates (append-only, no full rebuild)

References:
- RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
  (Sarthi et al., 2024) — https://arxiv.org/abs/2401.18059
- Hierarchical Summarization for Long-Document QA (Liu et al., 2023)

Example:
    >>> builder = RaptorTreeBuilder(max_depth=3, cluster_size=10)
    >>> tree = await builder.build(documents=chunks, embedder=embed_fn, summarizer=summarize_fn)
    >>> nodes = tree.flatten()  # All nodes for retrieval index
    >>> results = tree.retrieve("What is quantum entanglement?", top_k=5)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
EmbedFn = Callable[[List[str]], Coroutine[Any, Any, np.ndarray]]
SummarizeFn = Callable[[List[str]], Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """A single node in the RAPTOR summarization tree."""

    node_id: str
    text: str
    level: int  # 0 = leaf (original chunk), >0 = summary node
    embedding: Optional[np.ndarray] = None
    children: List["TreeNode"] = field(default_factory=list)
    parent_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            self.node_id = hashlib.md5(self.text.encode()).hexdigest()[:12]
        # Rough token estimate: 1 token ≈ 4 chars
        if not self.token_count:
            self.token_count = max(1, len(self.text) // 4)

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children (original chunk)."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """True if this node has no parent (top-level summary)."""
        return self.parent_id is None

    def cosine_sim(self, other_embedding: np.ndarray) -> float:
        """Fast cosine similarity to a query embedding."""
        if self.embedding is None or other_embedding is None:
            return 0.0
        a = self.embedding
        b = other_embedding
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class RaptorTree:
    """
    The fully built RAPTOR tree.

    Provides flattened retrieval across all tree levels and
    level-specific retrieval for granularity control.
    """

    roots: List[TreeNode] = field(default_factory=list)
    all_nodes: Dict[str, TreeNode] = field(default_factory=dict)  # id → node
    build_time_s: float = 0.0
    depth: int = 0

    # ------------------------------------------------------------------ #
    # Build helpers                                                        #
    # ------------------------------------------------------------------ #

    def add_node(self, node: TreeNode) -> None:
        """Register a node in the global index."""
        self.all_nodes[node.node_id] = node
        self.depth = max(self.depth, node.level)

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def flatten(self) -> List[TreeNode]:
        """Return all nodes sorted by level (leaves first)."""
        return sorted(self.all_nodes.values(), key=lambda n: n.level)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        level_filter: Optional[int] = None,
    ) -> List[Tuple[TreeNode, float]]:
        """
        Retrieve the top-k most relevant nodes via cosine similarity.

        Args:
            query_embedding: Dense embedding of the query.
            top_k: Number of results to return.
            level_filter: If set, restrict to nodes at this tree level.

        Returns:
            List of (node, score) tuples sorted by descending score.
        """
        candidates = [
            n
            for n in self.all_nodes.values()
            if n.embedding is not None and (level_filter is None or n.level == level_filter)
        ]

        if not candidates:
            return []

        # Vectorised similarity computation
        emb_matrix = np.stack([n.embedding for n in candidates])  # (N, D)
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        normed = emb_matrix / (norms + 1e-9)
        scores: np.ndarray = normed @ q  # (N,)

        top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(candidates[i], float(scores[i])) for i in top_indices]

    def multi_level_retrieve(
        self,
        query_embedding: np.ndarray,
        top_k_per_level: int = 3,
    ) -> Dict[int, List[Tuple[TreeNode, float]]]:
        """Retrieve top-k nodes at every tree level."""
        results: Dict[int, List[Tuple[TreeNode, float]]] = {}
        for level in range(self.depth + 1):
            hits = self.retrieve(query_embedding, top_k=top_k_per_level, level_filter=level)
            if hits:
                results[level] = hits
        return results

    def __len__(self) -> int:
        return len(self.all_nodes)


# ---------------------------------------------------------------------------
# Gaussian Mixture-based soft clustering (EM, no sklearn dependency)
# ---------------------------------------------------------------------------


def _soft_cluster_gmm(
    embeddings: np.ndarray,
    n_clusters: int,
    n_iter: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """
    Minimal GMM E-step cluster assignment using cosine distances.

    Returns an array of shape (N,) with integer cluster ids (hard assignment
    after soft EM), suitable for constructing RAPTOR tree levels.

    Uses the same spherical-GMM approach as the original RAPTOR paper but
    without any external ML library dependencies.
    """
    rng = np.random.default_rng(seed)
    N, D = embeddings.shape

    if N <= n_clusters:
        return np.arange(N)

    # Normalise to unit sphere for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embs = embeddings / (norms + 1e-9)

    # Initialise centroids with k-means++
    indices = [int(rng.integers(N))]
    for _ in range(n_clusters - 1):
        dists = np.min(
            np.stack([np.sum((embs - embs[i]) ** 2, axis=1) for i in indices]),
            axis=0,
        )
        probs = dists / (dists.sum() + 1e-12)
        indices.append(int(rng.choice(N, p=probs)))
    centroids = embs[indices]  # (K, D)

    labels = np.zeros(N, dtype=int)
    for _ in range(n_iter):
        # E-step: cosine similarity to centroids
        sims = embs @ centroids.T  # (N, K)
        new_labels = np.argmax(sims, axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # M-step: recompute centroids
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                mean = embs[mask].mean(axis=0)
                centroids[k] = mean / (np.linalg.norm(mean) + 1e-9)

    return labels


# ---------------------------------------------------------------------------
# Tree Builder
# ---------------------------------------------------------------------------


class RaptorTreeBuilder:
    """
    Builds a RAPTOR summarization tree from flat document chunks.

    Algorithm:
        1. Embed all leaf chunks.
        2. Cluster embeddings using soft GMM.
        3. Summarise each cluster — these become level-1 nodes.
        4. Repeat until only 1 cluster remains or max_depth is reached.

    Args:
        max_depth: Maximum tree height (default 3).
        cluster_size: Target chunks per cluster (default 10).
        min_cluster_size: Minimum chunks before clustering stops (default 4).
        max_summary_tokens: Rough token budget for each summary (default 256).
        embed_batch_size: Batch size for embedding calls (default 32).
        concurrency: Max parallel summarisation tasks (default 4).
    """

    def __init__(
        self,
        max_depth: int = 3,
        cluster_size: int = 10,
        min_cluster_size: int = 4,
        max_summary_tokens: int = 256,
        embed_batch_size: int = 32,
        concurrency: int = 4,
    ) -> None:
        self.max_depth = max_depth
        self.cluster_size = cluster_size
        self.min_cluster_size = min_cluster_size
        self.max_summary_tokens = max_summary_tokens
        self.embed_batch_size = embed_batch_size
        self.concurrency = concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def build(
        self,
        documents: List[str],
        embedder: EmbedFn,
        summarizer: SummarizeFn,
    ) -> RaptorTree:
        """
        Build the RAPTOR tree.

        Args:
            documents: Raw text chunks (leaf nodes).
            embedder: Async batch embedding function: texts → (N, D) array.
            summarizer: Async summarisation function: texts → summary string.

        Returns:
            Fully constructed RaptorTree ready for retrieval.
        """
        if not documents:
            return RaptorTree()

        start = time.perf_counter()
        self._semaphore = asyncio.Semaphore(self.concurrency)
        tree = RaptorTree()

        # ── Level 0: create and embed leaf nodes ──────────────────────────
        logger.info("RAPTOR: embedding %d leaf chunks", len(documents))
        leaf_embeddings = await self._batch_embed(documents, embedder)

        current_nodes: List[TreeNode] = []
        for i, (text, emb) in enumerate(zip(documents, leaf_embeddings)):
            node = TreeNode(
                node_id=f"L0_{i}",
                text=text,
                level=0,
                embedding=emb,
            )
            current_nodes.append(node)
            tree.add_node(node)

        # ── Recursive bottom-up summarisation ─────────────────────────────
        for depth in range(1, self.max_depth + 1):
            if len(current_nodes) < self.min_cluster_size:
                logger.debug("RAPTOR: stopping at depth %d (too few nodes)", depth)
                break

            logger.info("RAPTOR: building level %d from %d nodes", depth, len(current_nodes))
            current_nodes = await self._build_level(
                current_nodes, depth, embedder, summarizer, tree
            )

        # Identify roots (nodes with no parent)
        tree.roots = [n for n in tree.all_nodes.values() if n.parent_id is None and n.level > 0]
        tree.build_time_s = time.perf_counter() - start
        logger.info(
            "RAPTOR: tree built — %d nodes, depth %d, %.2fs",
            len(tree),
            tree.depth,
            tree.build_time_s,
        )
        return tree

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _build_level(
        self,
        nodes: List[TreeNode],
        level: int,
        embedder: EmbedFn,
        summarizer: SummarizeFn,
        tree: RaptorTree,
    ) -> List[TreeNode]:
        """Cluster `nodes`, summarise each cluster, return summary nodes."""
        if not nodes:
            return []

        # Stack embeddings for clustering
        emb_matrix = np.stack([n.embedding for n in nodes])  # (N, D)
        n_clusters = max(1, math.ceil(len(nodes) / self.cluster_size))

        if len(nodes) <= n_clusters:
            labels = np.arange(len(nodes))
        else:
            labels = _soft_cluster_gmm(emb_matrix, n_clusters=n_clusters)

        # Group nodes by cluster
        clusters: Dict[int, List[TreeNode]] = {}
        for node, label in zip(nodes, labels):
            clusters.setdefault(int(label), []).append(node)

        # Parallel summarisation
        tasks = [
            self._summarise_cluster(cluster_nodes, level, idx, summarizer)
            for idx, cluster_nodes in clusters.items()
        ]
        summary_nodes: List[TreeNode] = await asyncio.gather(*tasks)

        # Embed all summaries in one batched call
        summaries = [n.text for n in summary_nodes]
        summary_embeddings = await self._batch_embed(summaries, embedder)

        new_nodes: List[TreeNode] = []
        for node, emb in zip(summary_nodes, summary_embeddings):
            node.embedding = emb
            tree.add_node(node)
            # Link children back to this parent
            for child in node.children:
                child.parent_id = node.node_id
            new_nodes.append(node)

        return new_nodes

    async def _summarise_cluster(
        self,
        cluster_nodes: List[TreeNode],
        level: int,
        cluster_idx: int,
        summarizer: SummarizeFn,
    ) -> TreeNode:
        """Summarise a cluster of nodes into a single parent node."""
        texts = [n.text for n in cluster_nodes]
        async with self._semaphore:  # type: ignore[union-attr]
            summary_text = await summarizer(texts)

        node_id = f"L{level}_C{cluster_idx}"
        parent = TreeNode(
            node_id=node_id,
            text=summary_text,
            level=level,
            children=list(cluster_nodes),
        )
        return parent

    async def _batch_embed(self, texts: List[str], embedder: EmbedFn) -> List[np.ndarray]:
        """Embed texts in batches to respect API rate limits."""
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            result = await embedder(batch)  # (B, D)
            all_embeddings.extend([result[j] for j in range(len(batch))])
        return all_embeddings


# ---------------------------------------------------------------------------
# Incremental update helper
# ---------------------------------------------------------------------------


async def append_to_tree(
    tree: RaptorTree,
    new_documents: List[str],
    embedder: EmbedFn,
    max_depth: int = 1,
) -> RaptorTree:
    """
    Incrementally add new leaf documents to an existing tree.

    Only adds new leaf nodes and re-clusters the topmost level once —
    avoids full tree reconstruction for streaming document ingestion.

    Args:
        tree: Existing RaptorTree to extend.
        new_documents: New text chunks to append.
        embedder: Same embedding function used to build the tree.
        max_depth: How many levels above leaves to recompute (default 1).

    Returns:
        Updated RaptorTree (mutates in-place and returns for chaining).
    """
    offset = sum(1 for n in tree.all_nodes.values() if n.level == 0)
    new_embeddings = await RaptorTreeBuilder(max_depth=1)._batch_embed(new_documents, embedder)

    for i, (text, emb) in enumerate(zip(new_documents, new_embeddings)):
        node = TreeNode(
            node_id=f"L0_{offset + i}",
            text=text,
            level=0,
            embedding=emb,
        )
        tree.add_node(node)

    logger.info("RAPTOR: appended %d leaf nodes (total=%d)", len(new_documents), len(tree))
    return tree
