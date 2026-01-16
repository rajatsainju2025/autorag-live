"""
Late Interaction Reranking Module.

Implements ColBERT-style token-level late interaction for
fine-grained relevance matching between queries and documents.

Key Features:
1. Token-level embedding comparison
2. MaxSim aggregation for relevance scoring
3. Efficient batch processing
4. Multi-vector representations
5. Deferred interaction for efficiency

References:
- ColBERT: Efficient and Effective Passage Search (Khattab & Zaharia, 2020)
- ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction
- PLAID: An Efficient Engine for Late Interaction Retrieval

Example:
    >>> reranker = LateInteractionReranker(embedder)
    >>> ranked = await reranker.rerank(query, documents, top_k=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# OPTIMIZATION 9: PLAID-Style Efficient Late Interaction
# Based on: "PLAID: An Efficient Engine for Late Interaction Retrieval"
# (Santhanam et al., 2022) and ColBERTv2 optimizations
#
# Key optimizations:
# 1. Centroid-based pre-filtering (10-100x candidate reduction)
# 2. Product quantization for memory efficiency
# 3. Batched SIMD-friendly similarity computation
# 4. Score-aware pruning during MaxSim
# =============================================================================


@dataclass
class PLAIDConfig:
    """Configuration for PLAID-style efficient late interaction."""

    # Centroid clustering
    num_centroids: int = 256
    centroid_score_threshold: float = 0.5

    # Quantization
    enable_quantization: bool = True
    quantization_bits: int = 4  # 4-bit quantization
    residual_compression: bool = True

    # Pruning
    enable_score_pruning: bool = True
    max_candidate_tokens: int = 1000
    prune_threshold: float = 0.3

    # Batching
    batch_size: int = 32
    use_simd: bool = True


class TokenCentroidIndex:
    """
    Centroid-based index for efficient token matching.

    Uses k-means clustering to group similar token embeddings,
    enabling fast pre-filtering before exact MaxSim computation.
    """

    def __init__(self, config: PLAIDConfig):
        """Initialize centroid index."""
        self.config = config
        self.centroids: Optional[np.ndarray] = None
        self.token_to_centroid: Dict[int, int] = {}
        self.centroid_to_tokens: Dict[int, List[int]] = {}
        self._is_trained = False

    def train(self, embeddings: np.ndarray) -> None:
        """
        Train centroids using k-means.

        Args:
            embeddings: Token embeddings (num_tokens x dim)
        """
        from sklearn.cluster import MiniBatchKMeans

        n_samples = embeddings.shape[0]
        n_clusters = min(self.config.num_centroids, n_samples)

        if n_clusters < 2:
            # Not enough samples for clustering
            self.centroids = embeddings.mean(axis=0, keepdims=True)
            self.token_to_centroid = {i: 0 for i in range(n_samples)}
            self.centroid_to_tokens = {0: list(range(n_samples))}
            self._is_trained = True
            return

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=min(1000, n_samples),
            max_iter=100,
        )
        labels = kmeans.fit_predict(embeddings)
        self.centroids = kmeans.cluster_centers_

        # Build mappings
        self.token_to_centroid = {i: int(labels[i]) for i in range(n_samples)}
        self.centroid_to_tokens = {}
        for i, label in enumerate(labels):
            if label not in self.centroid_to_tokens:
                self.centroid_to_tokens[label] = []
            self.centroid_to_tokens[label].append(i)

        self._is_trained = True
        logger.debug(f"Trained centroid index with {n_clusters} centroids")

    def get_candidate_tokens(
        self,
        query_embedding: np.ndarray,
        top_centroids: int = 16,
    ) -> List[int]:
        """
        Get candidate token indices using centroid filtering.

        Args:
            query_embedding: Single query token embedding
            top_centroids: Number of top centroids to consider

        Returns:
            List of candidate token indices
        """
        if not self._is_trained or self.centroids is None:
            return list(self.token_to_centroid.keys())

        # Score query against centroids
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        centroid_norms = self.centroids / (
            np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-8
        )
        scores = np.dot(centroid_norms, query_norm)

        # Get top centroids
        top_k = min(top_centroids, len(scores))
        top_indices = np.argsort(scores)[-top_k:]

        # Collect tokens from top centroids
        candidates = []
        for idx in top_indices:
            if scores[idx] >= self.config.centroid_score_threshold:
                candidates.extend(self.centroid_to_tokens.get(int(idx), []))

        return candidates[: self.config.max_candidate_tokens]


class ProductQuantizer:
    """
    Product quantization for compressed token embeddings.

    Enables 4-8x memory reduction with minimal accuracy loss.
    """

    def __init__(self, config: PLAIDConfig):
        """Initialize quantizer."""
        self.config = config
        self.subquantizers: List[np.ndarray] = []
        self.dim_per_subspace: int = 0
        self.num_subspaces: int = 8
        self._is_trained = False

    def train(self, embeddings: np.ndarray) -> None:
        """
        Train product quantizer on embeddings.

        Args:
            embeddings: Training embeddings
        """
        from sklearn.cluster import MiniBatchKMeans

        dim = embeddings.shape[1]
        self.dim_per_subspace = dim // self.num_subspaces
        num_codewords = 2**self.config.quantization_bits

        self.subquantizers = []
        for i in range(self.num_subspaces):
            start = i * self.dim_per_subspace
            end = start + self.dim_per_subspace
            subspace = embeddings[:, start:end]

            kmeans = MiniBatchKMeans(
                n_clusters=min(num_codewords, len(subspace)),
                random_state=42,
            )
            kmeans.fit(subspace)
            self.subquantizers.append(kmeans.cluster_centers_)

        self._is_trained = True

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode embeddings to quantized codes.

        Args:
            embeddings: Original embeddings

        Returns:
            Quantized codes (num_embeddings x num_subspaces)
        """
        if not self._is_trained:
            return np.zeros((len(embeddings), self.num_subspaces), dtype=np.uint8)

        codes = np.zeros((len(embeddings), self.num_subspaces), dtype=np.uint8)
        for i in range(self.num_subspaces):
            start = i * self.dim_per_subspace
            end = start + self.dim_per_subspace
            subspace = embeddings[:, start:end]

            # Find nearest codeword
            dists = np.linalg.norm(subspace[:, np.newaxis] - self.subquantizers[i], axis=2)
            codes[:, i] = np.argmin(dists, axis=1).astype(np.uint8)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode quantized codes to approximate embeddings.

        Args:
            codes: Quantized codes

        Returns:
            Approximate embeddings
        """
        if not self._is_trained:
            return np.zeros((len(codes), self.dim_per_subspace * self.num_subspaces))

        embeddings = np.zeros((len(codes), self.dim_per_subspace * self.num_subspaces))
        for i in range(self.num_subspaces):
            start = i * self.dim_per_subspace
            end = start + self.dim_per_subspace
            embeddings[:, start:end] = self.subquantizers[i][codes[:, i]]

        return embeddings

    def asymmetric_distance(
        self,
        query: np.ndarray,
        codes: np.ndarray,
    ) -> np.ndarray:
        """
        Compute asymmetric distance using lookup tables.

        Much faster than decode + distance for large sets.

        Args:
            query: Query embedding (not quantized)
            codes: Quantized document codes

        Returns:
            Distance scores
        """
        if not self._is_trained:
            return np.zeros(len(codes))

        # Build lookup tables
        tables = []
        for i in range(self.num_subspaces):
            start = i * self.dim_per_subspace
            end = start + self.dim_per_subspace
            query_sub = query[start:end]

            # Distance from query to each codeword
            dists = np.linalg.norm(self.subquantizers[i] - query_sub, axis=1)
            tables.append(dists)

        # Lookup and sum
        distances = np.zeros(len(codes))
        for i in range(self.num_subspaces):
            distances += tables[i][codes[:, i]]

        return distances


class BatchedMaxSimComputer:
    """
    SIMD-friendly batched MaxSim computation.

    Optimized for modern CPU vector instructions.
    """

    def __init__(self, config: PLAIDConfig):
        """Initialize computer."""
        self.config = config

    def compute_batched(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
        candidate_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, List[float], List[int]]:
        """
        Compute MaxSim with batched operations.

        Args:
            query_embeddings: Query token embeddings (Q x D)
            doc_embeddings: Document token embeddings (T x D)
            candidate_mask: Optional mask for candidate pruning

        Returns:
            Tuple of (score, per_token_scores, matched_indices)
        """
        if len(query_embeddings) == 0 or len(doc_embeddings) == 0:
            return 0.0, [], []

        # Normalize (batch operation)
        q_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        d_norm = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)

        # Apply candidate mask if provided
        if candidate_mask is not None:
            d_norm = d_norm[candidate_mask]

        # Batched matrix multiplication (SIMD-friendly)
        # Shape: (Q, T)
        sim_matrix = np.dot(q_norm, d_norm.T)

        # Score-aware pruning: early termination for low scores
        if self.config.enable_score_pruning:
            row_maxes = np.max(sim_matrix, axis=1)
            if np.max(row_maxes) < self.config.prune_threshold:
                return 0.0, row_maxes.tolist(), [0] * len(row_maxes)

        # MaxSim: max over document tokens for each query token
        max_sims = np.max(sim_matrix, axis=1)
        matched_idx = np.argmax(sim_matrix, axis=1)

        # Remap indices if mask was used
        if candidate_mask is not None:
            actual_indices = np.where(candidate_mask)[0]
            matched_idx = actual_indices[matched_idx]

        score = float(np.sum(max_sims))
        return score, max_sims.tolist(), matched_idx.tolist()


class PLAIDLateInteractionReranker:
    """
    PLAID-optimized late interaction reranker.

    Implements state-of-the-art optimizations from PLAID paper:
    - Centroid-based candidate pre-filtering
    - Product quantization for memory efficiency
    - Batched SIMD-friendly computation
    - Score-aware pruning

    Example:
        >>> config = PLAIDConfig(num_centroids=256)
        >>> reranker = PLAIDLateInteractionReranker(embedder, config)
        >>> ranked = await reranker.rerank(query, documents, top_k=10)
    """

    def __init__(
        self,
        embedder: Optional["EmbedderProtocol"] = None,
        config: Optional[PLAIDConfig] = None,
    ):
        """
        Initialize PLAID reranker.

        Args:
            embedder: Embedding model
            config: PLAID configuration
        """
        self.embedder = embedder
        self.config = config or PLAIDConfig()
        self.token_embedder = SimpleTokenEmbedder(embedder)

        # Optimization components
        self.centroid_index = TokenCentroidIndex(self.config)
        self.quantizer = ProductQuantizer(self.config)
        self.batched_computer = BatchedMaxSimComputer(self.config)

        # Cache for indexed documents
        self._doc_embeddings: Dict[str, np.ndarray] = {}
        self._doc_codes: Dict[str, np.ndarray] = {}
        self._indexed = False

    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> None:
        """
        Pre-index documents for efficient reranking.

        Args:
            documents: Documents to index
        """
        all_embeddings = []
        doc_boundaries = [0]

        for doc in documents:
            content = doc.get("content", doc.get("text", str(doc)))
            doc_id = doc.get("id", str(len(self._doc_embeddings)))

            emb = await self.token_embedder.embed_tokens(content)
            self._doc_embeddings[doc_id] = emb.embeddings
            all_embeddings.append(emb.embeddings)
            doc_boundaries.append(doc_boundaries[-1] + len(emb.embeddings))

        if all_embeddings:
            # Train optimization structures
            combined = np.vstack(all_embeddings)
            self.centroid_index.train(combined)

            if self.config.enable_quantization:
                self.quantizer.train(combined)
                for doc_id, embs in self._doc_embeddings.items():
                    self._doc_codes[doc_id] = self.quantizer.encode(embs)

        self._indexed = True
        logger.info(f"Indexed {len(documents)} documents with PLAID optimizations")

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
        use_prefiltering: bool = True,
    ) -> List["RankedDocument"]:
        """
        Rerank documents using PLAID-optimized late interaction.

        Args:
            query: Query text
            documents: Documents to rerank
            top_k: Number of top documents to return
            use_prefiltering: Use centroid pre-filtering

        Returns:
            List of reranked documents
        """
        if not documents:
            return []

        # Get query embeddings
        query_emb = await self.token_embedder.embed_tokens(query)

        # Score documents
        ranked_docs = []
        for i, doc in enumerate(documents):
            content = doc.get("content", doc.get("text", str(doc)))
            doc_id = doc.get("id", str(i))
            original_score = doc.get("score", 0.0)

            # Get document embeddings (cached or compute)
            if doc_id in self._doc_embeddings:
                doc_embs = self._doc_embeddings[doc_id]
            else:
                doc_emb = await self.token_embedder.embed_tokens(content)
                doc_embs = doc_emb.embeddings

            # Apply optimizations
            if use_prefiltering and self._indexed:
                score = await self._score_with_prefiltering(query_emb.embeddings, doc_embs)
            else:
                score, _, _ = self.batched_computer.compute_batched(query_emb.embeddings, doc_embs)

            ranked_doc = RankedDocument(
                content=content,
                doc_id=doc_id,
                original_score=original_score,
                rerank_score=score,
                metadata=doc.get("metadata", {}),
            )
            ranked_docs.append(ranked_doc)

        # Sort by combined score
        ranked_docs.sort(key=lambda d: d.combined_score, reverse=True)

        if top_k:
            ranked_docs = ranked_docs[:top_k]

        return ranked_docs

    async def _score_with_prefiltering(
        self,
        query_embs: np.ndarray,
        doc_embs: np.ndarray,
    ) -> float:
        """Score using centroid pre-filtering."""
        total_score = 0.0

        for q_emb in query_embs:
            # Get candidate tokens using centroid filtering
            candidates = self.centroid_index.get_candidate_tokens(q_emb)

            if not candidates:
                continue

            # Create candidate mask
            candidate_mask = np.zeros(len(doc_embs), dtype=bool)
            valid_candidates = [c for c in candidates if c < len(doc_embs)]
            if valid_candidates:
                candidate_mask[valid_candidates] = True

            # Compute MaxSim only for candidates
            _, token_scores, _ = self.batched_computer.compute_batched(
                q_emb.reshape(1, -1),
                doc_embs,
                candidate_mask if candidate_mask.any() else None,
            )

            if token_scores:
                total_score += token_scores[0]

        return total_score

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "indexed_documents": len(self._doc_embeddings),
            "centroid_trained": self.centroid_index._is_trained,
            "quantizer_trained": self.quantizer._is_trained,
            "config": {
                "num_centroids": self.config.num_centroids,
                "quantization_bits": self.config.quantization_bits,
                "enable_pruning": self.config.enable_score_pruning,
            },
        }


# =============================================================================
# Protocols
# =============================================================================


class TokenEmbedderProtocol(Protocol):
    """Protocol for token-level embedding."""

    async def embed_tokens(self, text: str) -> List[List[float]]:
        """Get per-token embeddings. Returns list of token embeddings."""
        ...


class EmbedderProtocol(Protocol):
    """Protocol for standard embedding."""

    async def embed(self, text: str) -> List[float]:
        """Get single embedding for text."""
        ...


# =============================================================================
# Data Structures
# =============================================================================


class AggregationMethod(str, Enum):
    """Methods for aggregating token scores."""

    MAX_SIM = "max_sim"  # ColBERT-style MaxSim
    AVG_SIM = "avg_sim"  # Average similarity
    SUM_SIM = "sum_sim"  # Sum of max similarities
    WEIGHTED_SUM = "weighted_sum"  # IDF-weighted sum


@dataclass
class TokenEmbedding:
    """
    Token-level embedding representation.

    Attributes:
        text: Original text
        tokens: List of tokens
        embeddings: Per-token embeddings (num_tokens x embedding_dim)
        attention_weights: Optional attention weights per token
    """

    text: str
    tokens: List[str]
    embeddings: np.ndarray  # Shape: (num_tokens, embedding_dim)
    attention_weights: Optional[np.ndarray] = None

    @property
    def num_tokens(self) -> int:
        """Get number of tokens."""
        return len(self.tokens)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else 0


@dataclass
class LateInteractionScore:
    """
    Score from late interaction computation.

    Attributes:
        document_id: Document identifier
        score: Final relevance score
        token_scores: Per-query-token max scores
        matched_tokens: Best matching document tokens
        metadata: Additional scoring details
    """

    document_id: str
    score: float
    token_scores: List[float] = field(default_factory=list)
    matched_tokens: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedDocument:
    """
    Document with late interaction ranking.

    Attributes:
        content: Document text
        doc_id: Document identifier
        original_score: Original retrieval score
        rerank_score: Late interaction score
        token_matches: Token-level match details
    """

    content: str
    doc_id: str = ""
    original_score: float = 0.0
    rerank_score: float = 0.0
    token_matches: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def combined_score(self) -> float:
        """Combine original and rerank scores."""
        return 0.3 * self.original_score + 0.7 * self.rerank_score


# =============================================================================
# Token Embedder
# =============================================================================


class SimpleTokenEmbedder:
    """
    Simple token embedder using word-level embeddings.

    For production, use sentence-transformers with token-level output.
    """

    def __init__(
        self,
        embedder: Optional[EmbedderProtocol] = None,
        embedding_dim: int = 128,
    ):
        """Initialize embedder."""
        self.embedder = embedder
        self.embedding_dim = embedding_dim
        self._cache: Dict[str, np.ndarray] = {}

    async def embed_tokens(self, text: str) -> TokenEmbedding:
        """
        Get token-level embeddings.

        Args:
            text: Input text

        Returns:
            TokenEmbedding with per-token vectors
        """
        # Simple tokenization
        tokens = self._tokenize(text)

        if not tokens:
            return TokenEmbedding(
                text=text,
                tokens=[],
                embeddings=np.zeros((0, self.embedding_dim)),
            )

        # Get embeddings for each token
        embeddings = []
        for token in tokens:
            emb = await self._get_token_embedding(token)
            embeddings.append(emb)

        return TokenEmbedding(
            text=text,
            tokens=tokens,
            embeddings=np.array(embeddings),
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        import re

        # Basic word tokenization
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    async def _get_token_embedding(self, token: str) -> np.ndarray:
        """Get embedding for single token."""
        if token in self._cache:
            return self._cache[token]

        if self.embedder:
            try:
                emb = await self.embedder.embed(token)
                arr = np.array(emb[: self.embedding_dim])
                # Pad or truncate to embedding_dim
                if len(arr) < self.embedding_dim:
                    arr = np.pad(arr, (0, self.embedding_dim - len(arr)))
                self._cache[token] = arr
                return arr
            except Exception:
                pass

        # Fallback: hash-based pseudo-embedding
        return self._hash_embedding(token)

    def _hash_embedding(self, token: str) -> np.ndarray:
        """Generate hash-based embedding for token."""
        import hashlib

        h = hashlib.md5(token.encode()).digest()
        # Convert to float array
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        # Expand to embedding_dim
        arr = np.tile(arr, self.embedding_dim // len(arr) + 1)[: self.embedding_dim]
        # Normalize
        arr = arr / np.linalg.norm(arr)
        return arr


# =============================================================================
# Late Interaction Computation
# =============================================================================


class MaxSimComputer:
    """
    Computes MaxSim (Maximum Similarity) scores.

    ColBERT-style late interaction that computes the maximum
    similarity between each query token and all document tokens.
    """

    def __init__(
        self,
        aggregation: AggregationMethod = AggregationMethod.MAX_SIM,
    ):
        """Initialize computer."""
        self.aggregation = aggregation

    def compute(
        self,
        query_emb: TokenEmbedding,
        doc_emb: TokenEmbedding,
    ) -> LateInteractionScore:
        """
        Compute late interaction score.

        Args:
            query_emb: Query token embeddings
            doc_emb: Document token embeddings

        Returns:
            LateInteractionScore with relevance score
        """
        if query_emb.num_tokens == 0 or doc_emb.num_tokens == 0:
            return LateInteractionScore(
                document_id=doc_emb.text[:50],
                score=0.0,
            )

        # Compute similarity matrix: (query_tokens x doc_tokens)
        sim_matrix = self._compute_similarity_matrix(
            query_emb.embeddings,
            doc_emb.embeddings,
        )

        # Aggregate based on method
        if self.aggregation == AggregationMethod.MAX_SIM:
            score, token_scores, matched_idx = self._max_sim_aggregate(sim_matrix)
        elif self.aggregation == AggregationMethod.AVG_SIM:
            score, token_scores, matched_idx = self._avg_sim_aggregate(sim_matrix)
        elif self.aggregation == AggregationMethod.SUM_SIM:
            score, token_scores, matched_idx = self._sum_sim_aggregate(sim_matrix)
        else:
            score, token_scores, matched_idx = self._max_sim_aggregate(sim_matrix)

        # Get matched token strings
        matched_tokens = [doc_emb.tokens[i] if i < len(doc_emb.tokens) else "" for i in matched_idx]

        return LateInteractionScore(
            document_id=doc_emb.text[:50],
            score=score,
            token_scores=token_scores,
            matched_tokens=matched_tokens,
        )

    def _compute_similarity_matrix(
        self,
        query_emb: np.ndarray,
        doc_emb: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity matrix."""
        # Normalize
        query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
        doc_norm = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity: (Q, D)
        return np.dot(query_norm, doc_norm.T)

    def _max_sim_aggregate(
        self,
        sim_matrix: np.ndarray,
    ) -> Tuple[float, List[float], List[int]]:
        """MaxSim aggregation (ColBERT)."""
        # For each query token, find max similarity across doc tokens
        max_sims = np.max(sim_matrix, axis=1)
        matched_idx = np.argmax(sim_matrix, axis=1).tolist()

        # Sum of max similarities
        score = float(np.sum(max_sims))

        return score, max_sims.tolist(), matched_idx

    def _avg_sim_aggregate(
        self,
        sim_matrix: np.ndarray,
    ) -> Tuple[float, List[float], List[int]]:
        """Average similarity aggregation."""
        max_sims = np.max(sim_matrix, axis=1)
        matched_idx = np.argmax(sim_matrix, axis=1).tolist()

        score = float(np.mean(max_sims))

        return score, max_sims.tolist(), matched_idx

    def _sum_sim_aggregate(
        self,
        sim_matrix: np.ndarray,
    ) -> Tuple[float, List[float], List[int]]:
        """Sum similarity aggregation."""
        max_sims = np.max(sim_matrix, axis=1)
        matched_idx = np.argmax(sim_matrix, axis=1).tolist()

        score = float(np.sum(max_sims))

        return score, max_sims.tolist(), matched_idx


# =============================================================================
# Late Interaction Reranker
# =============================================================================


class LateInteractionReranker:
    """
    ColBERT-style late interaction reranker.

    Reranks documents using token-level similarity matching
    for fine-grained relevance scoring.

    Example:
        >>> reranker = LateInteractionReranker(embedder)
        >>> ranked = await reranker.rerank(
        ...     "What is machine learning?",
        ...     documents,
        ...     top_k=10
        ... )
    """

    def __init__(
        self,
        embedder: Optional[EmbedderProtocol] = None,
        token_embedder: Optional[SimpleTokenEmbedder] = None,
        aggregation: AggregationMethod = AggregationMethod.MAX_SIM,
        max_query_tokens: int = 32,
        max_doc_tokens: int = 512,
    ):
        """
        Initialize reranker.

        Args:
            embedder: Standard embedder (for token embedding fallback)
            token_embedder: Token-level embedder
            aggregation: Score aggregation method
            max_query_tokens: Maximum query tokens
            max_doc_tokens: Maximum document tokens
        """
        self.embedder = embedder
        self.token_embedder = token_embedder or SimpleTokenEmbedder(embedder)
        self.aggregation = aggregation
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens

        self.sim_computer = MaxSimComputer(aggregation=aggregation)

        # Embedding cache
        self._query_cache: Dict[str, TokenEmbedding] = {}
        self._doc_cache: Dict[str, TokenEmbedding] = {}

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
        include_token_matches: bool = False,
    ) -> List[RankedDocument]:
        """
        Rerank documents using late interaction.

        Args:
            query: Query text
            documents: Documents to rerank
            top_k: Number of top documents to return
            include_token_matches: Include token-level match details

        Returns:
            List of reranked documents
        """
        if not documents:
            return []

        # Get query embedding
        query_emb = await self._get_query_embedding(query)

        # Score all documents
        ranked_docs = []
        for i, doc in enumerate(documents):
            content = doc.get("content", doc.get("text", str(doc)))
            doc_id = doc.get("id", str(i))
            original_score = doc.get("score", 0.0)

            # Get document embedding
            doc_emb = await self._get_doc_embedding(content)

            # Compute late interaction score
            li_score = self.sim_computer.compute(query_emb, doc_emb)

            ranked_doc = RankedDocument(
                content=content,
                doc_id=doc_id,
                original_score=original_score,
                rerank_score=li_score.score,
                metadata=doc.get("metadata", {}),
            )

            if include_token_matches:
                ranked_doc.token_matches = [
                    {"query_token": query_emb.tokens[j], "doc_token": t, "score": s}
                    for j, (t, s) in enumerate(zip(li_score.matched_tokens, li_score.token_scores))
                ]

            ranked_docs.append(ranked_doc)

        # Sort by combined score
        ranked_docs.sort(key=lambda d: d.combined_score, reverse=True)

        if top_k:
            ranked_docs = ranked_docs[:top_k]

        return ranked_docs

    async def _get_query_embedding(self, query: str) -> TokenEmbedding:
        """Get or compute query token embedding."""
        cache_key = query[:100]
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]

        emb = await self.token_embedder.embed_tokens(query)

        # Truncate to max tokens
        if emb.num_tokens > self.max_query_tokens:
            emb = TokenEmbedding(
                text=emb.text,
                tokens=emb.tokens[: self.max_query_tokens],
                embeddings=emb.embeddings[: self.max_query_tokens],
            )

        self._query_cache[cache_key] = emb
        return emb

    async def _get_doc_embedding(self, content: str) -> TokenEmbedding:
        """Get or compute document token embedding."""
        cache_key = content[:100]
        if cache_key in self._doc_cache:
            return self._doc_cache[cache_key]

        emb = await self.token_embedder.embed_tokens(content)

        # Truncate to max tokens
        if emb.num_tokens > self.max_doc_tokens:
            emb = TokenEmbedding(
                text=emb.text,
                tokens=emb.tokens[: self.max_doc_tokens],
                embeddings=emb.embeddings[: self.max_doc_tokens],
            )

        # Limit cache size
        if len(self._doc_cache) > 1000:
            # Remove oldest entry
            oldest = next(iter(self._doc_cache))
            del self._doc_cache[oldest]

        self._doc_cache[cache_key] = emb
        return emb

    def clear_cache(self) -> None:
        """Clear embedding caches."""
        self._query_cache.clear()
        self._doc_cache.clear()


# =============================================================================
# Multi-Vector Retriever
# =============================================================================


class MultiVectorRetriever:
    """
    Multi-vector retriever with late interaction.

    Stores and retrieves documents using token-level embeddings
    for fine-grained matching.
    """

    def __init__(
        self,
        embedder: Optional[EmbedderProtocol] = None,
        index_batch_size: int = 100,
    ):
        """
        Initialize retriever.

        Args:
            embedder: Embedding model
            index_batch_size: Batch size for indexing
        """
        self.embedder = embedder
        self.token_embedder = SimpleTokenEmbedder(embedder)
        self.index_batch_size = index_batch_size

        # Document index
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, TokenEmbedding] = {}

        # Reranker for scoring
        self.reranker = LateInteractionReranker(
            embedder=embedder,
            token_embedder=self.token_embedder,
        )

    async def index(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Index documents with token embeddings.

        Args:
            documents: Documents to index

        Returns:
            Number of documents indexed
        """
        count = 0

        for doc in documents:
            doc_id = doc.get("id", str(len(self._documents)))
            content = doc.get("content", doc.get("text", ""))

            if not content:
                continue

            # Store document
            self._documents[doc_id] = doc

            # Compute and store token embedding
            emb = await self.token_embedder.embed_tokens(content)
            self._embeddings[doc_id] = emb

            count += 1

        logger.info(f"Indexed {count} documents with token embeddings")
        return count

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using late interaction.

        Args:
            query: Query text
            top_k: Number of documents to return

        Returns:
            List of relevant documents
        """
        if not self._documents:
            return []

        # Convert stored docs to list format
        doc_list = []
        for doc_id, doc in self._documents.items():
            doc_list.append(
                {
                    "id": doc_id,
                    "content": doc.get("content", doc.get("text", "")),
                    "score": 1.0,  # Default score
                    "metadata": doc.get("metadata", {}),
                }
            )

        # Rerank using late interaction
        ranked = await self.reranker.rerank(query, doc_list, top_k=top_k)

        # Convert back to dict format
        return [
            {
                "id": r.doc_id,
                "content": r.content,
                "score": r.combined_score,
                "rerank_score": r.rerank_score,
                "metadata": r.metadata,
            }
            for r in ranked
        ]

    @property
    def document_count(self) -> int:
        """Get number of indexed documents."""
        return len(self._documents)


# =============================================================================
# Hybrid Late Interaction
# =============================================================================


class HybridLateInteractionReranker:
    """
    Hybrid reranker combining dense and late interaction.

    Uses dense retrieval for initial candidates and late
    interaction for fine-grained reranking.
    """

    def __init__(
        self,
        dense_embedder: Optional[EmbedderProtocol] = None,
        late_interaction_weight: float = 0.6,
    ):
        """
        Initialize hybrid reranker.

        Args:
            dense_embedder: Dense embedding model
            late_interaction_weight: Weight for late interaction score
        """
        self.dense_embedder = dense_embedder
        self.late_interaction_weight = late_interaction_weight
        self.dense_weight = 1.0 - late_interaction_weight

        self.late_reranker = LateInteractionReranker(embedder=dense_embedder)

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[RankedDocument]:
        """
        Hybrid reranking with dense + late interaction.

        Args:
            query: Query text
            documents: Documents with dense scores
            top_k: Number to return

        Returns:
            Hybrid-reranked documents
        """
        # Get late interaction scores
        li_ranked = await self.late_reranker.rerank(
            query,
            documents,
            top_k=None,  # Score all first
        )

        # Combine scores
        for doc in li_ranked:
            doc.rerank_score = (
                self.dense_weight * doc.original_score
                + self.late_interaction_weight * doc.rerank_score
            )

        # Re-sort by combined score
        li_ranked.sort(key=lambda d: d.rerank_score, reverse=True)

        if top_k:
            li_ranked = li_ranked[:top_k]

        return li_ranked


# =============================================================================
# Convenience Functions
# =============================================================================


def create_late_interaction_reranker(
    embedder: Optional[EmbedderProtocol] = None,
    aggregation: AggregationMethod = AggregationMethod.MAX_SIM,
) -> LateInteractionReranker:
    """
    Create a late interaction reranker.

    Args:
        embedder: Embedding model
        aggregation: Score aggregation method

    Returns:
        LateInteractionReranker instance
    """
    return LateInteractionReranker(embedder=embedder, aggregation=aggregation)


async def rerank_with_late_interaction(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Quick late interaction reranking.

    Args:
        query: Query text
        documents: Documents to rerank
        top_k: Number to return

    Returns:
        Reranked documents as dicts
    """
    reranker = LateInteractionReranker()
    ranked = await reranker.rerank(query, documents, top_k=top_k)

    return [
        {
            "content": r.content,
            "id": r.doc_id,
            "score": r.combined_score,
            "rerank_score": r.rerank_score,
        }
        for r in ranked
    ]
