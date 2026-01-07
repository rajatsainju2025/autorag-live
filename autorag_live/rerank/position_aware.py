"""
Position-Aware Reranking Implementation.

Addresses the "Lost in the Middle" problem where LLMs struggle to
use information from middle positions in long contexts.

Reference: Liu et al., 2023 - "Lost in the Middle: How Language Models
Use Long Contexts"

Key strategies:
1. Position-aware scoring
2. Strategic reordering
3. Position-biased fusion
4. Chunked attention reranking
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# ============================================================================
# Protocols
# ============================================================================


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM providers."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    position: int = 0  # Original position in results


# ============================================================================
# Position Strategies
# ============================================================================


class PositionStrategy(Enum):
    """Strategies for position-aware reranking."""

    # Place most relevant at start and end
    PRIMACY_RECENCY = "primacy_recency"
    # Interleave high and low relevance
    INTERLEAVED = "interleaved"
    # Best first (standard)
    BEST_FIRST = "best_first"
    # Reverse (for specific use cases)
    REVERSE = "reverse"
    # Random shuffle with top-k preserved
    SHUFFLE_PRESERVE_TOP = "shuffle_preserve_top"
    # Grouped by topic/relevance tier
    TIERED = "tiered"


class PositionBias(Enum):
    """Position bias profiles."""

    # Strong primacy (first items remembered best)
    PRIMACY = "primacy"
    # Strong recency (last items remembered best)
    RECENCY = "recency"
    # Both primacy and recency (U-shaped)
    U_SHAPED = "u_shaped"
    # Middle focus (inverse U)
    MIDDLE_FOCUS = "middle_focus"
    # Uniform (no bias)
    UNIFORM = "uniform"


# ============================================================================
# Position Bias Models
# ============================================================================


@dataclass
class PositionBiasConfig:
    """Configuration for position bias modeling."""

    bias_type: PositionBias = PositionBias.U_SHAPED
    primacy_strength: float = 0.3  # How much first positions are favored
    recency_strength: float = 0.3  # How much last positions are favored
    decay_rate: float = 0.1  # Rate of attention decay


class PositionBiasModel:
    """Models position-based attention bias."""

    def __init__(self, config: PositionBiasConfig | None = None):
        """Initialize with configuration."""
        self.config = config or PositionBiasConfig()

    def get_position_weight(self, position: int, total: int) -> float:
        """
        Get attention weight for a position.

        Args:
            position: 0-indexed position
            total: Total number of items

        Returns:
            Attention weight (0-1)
        """
        if total <= 1:
            return 1.0

        normalized_pos = position / (total - 1)

        if self.config.bias_type == PositionBias.PRIMACY:
            return self._primacy_weight(normalized_pos)
        elif self.config.bias_type == PositionBias.RECENCY:
            return self._recency_weight(normalized_pos)
        elif self.config.bias_type == PositionBias.U_SHAPED:
            return self._u_shaped_weight(normalized_pos)
        elif self.config.bias_type == PositionBias.MIDDLE_FOCUS:
            return self._middle_focus_weight(normalized_pos)
        else:
            return 1.0  # Uniform

    def _primacy_weight(self, normalized_pos: float) -> float:
        """Weight favoring early positions."""
        import math

        return math.exp(-self.config.decay_rate * normalized_pos * 10)

    def _recency_weight(self, normalized_pos: float) -> float:
        """Weight favoring late positions."""
        import math

        return math.exp(-self.config.decay_rate * (1 - normalized_pos) * 10)

    def _u_shaped_weight(self, normalized_pos: float) -> float:
        """U-shaped weight (primacy + recency)."""
        # Minimum in middle, max at ends
        primacy = self._primacy_weight(normalized_pos)
        recency = self._recency_weight(normalized_pos)
        return max(primacy, recency)

    def _middle_focus_weight(self, normalized_pos: float) -> float:
        """Inverse U (focuses on middle)."""
        import math

        # Maximum in middle
        distance_from_middle = abs(normalized_pos - 0.5) * 2
        return math.exp(-self.config.decay_rate * distance_from_middle * 10)

    def get_optimal_positions(
        self,
        scores: list[float],
        total_positions: int,
    ) -> list[int]:
        """
        Get optimal position ordering for documents.

        Given relevance scores, returns indices in optimal order
        to counteract position bias.

        Args:
            scores: Relevance scores for documents
            total_positions: Total number of positions

        Returns:
            List of indices in optimal order
        """
        # Sort by score
        indexed_scores = [(i, s) for i, s in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if self.config.bias_type == PositionBias.U_SHAPED:
            return self._u_shaped_ordering(indexed_scores, total_positions)
        else:
            # Default: best first
            return [i for i, _ in indexed_scores]

    def _u_shaped_ordering(
        self,
        indexed_scores: list[tuple[int, float]],
        total: int,
    ) -> list[int]:
        """
        Order to counteract U-shaped bias.

        Places most important at start and end, less important in middle.
        """
        result = [0] * len(indexed_scores)

        # Fill from both ends toward middle
        left = 0
        right = len(indexed_scores) - 1

        for i, (idx, _) in enumerate(indexed_scores):
            if i % 2 == 0:
                result[left] = idx
                left += 1
            else:
                result[right] = idx
                right -= 1

        return result


# ============================================================================
# Position-Aware Rerankers
# ============================================================================


class PositionAwareReranker(ABC):
    """Abstract base for position-aware reranking."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Rerank documents considering position effects."""
        ...


class PrimacyRecencyReranker(PositionAwareReranker):
    """
    Reranker that places most relevant at primacy/recency positions.

    Counteracts the "lost in the middle" effect.
    """

    def __init__(
        self,
        bias_config: PositionBiasConfig | None = None,
        preserve_top_n: int = 3,
    ):
        """
        Initialize reranker.

        Args:
            bias_config: Position bias configuration
            preserve_top_n: Always keep top-N at start
        """
        self.bias_model = PositionBiasModel(
            bias_config or PositionBiasConfig(bias_type=PositionBias.U_SHAPED)
        )
        self.preserve_top_n = preserve_top_n

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Rerank documents to counteract position bias.

        Args:
            query: User query
            documents: Documents to rerank

        Returns:
            Reordered documents
        """
        if len(documents) <= 1:
            return documents

        # Record original positions
        for i, doc in enumerate(documents):
            doc.position = i

        # Get optimal ordering
        scores = [doc.score for doc in documents]
        optimal_order = self.bias_model.get_optimal_positions(scores, len(documents))

        # Preserve top-N at start
        if self.preserve_top_n > 0:
            # Sort by score first
            sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
            top_docs = sorted_docs[: self.preserve_top_n]
            remaining_docs = sorted_docs[self.preserve_top_n :]

            # Reorder remaining
            remaining_scores = [d.score for d in remaining_docs]
            if remaining_docs:
                remaining_order = self.bias_model.get_optimal_positions(
                    remaining_scores, len(remaining_docs)
                )
                reordered_remaining = [remaining_docs[i] for i in remaining_order]
            else:
                reordered_remaining = []

            return top_docs + reordered_remaining

        # Apply optimal ordering
        reordered = [documents[i] for i in optimal_order]
        return reordered


class InterleavedReranker(PositionAwareReranker):
    """
    Interleaves high and low relevance documents.

    Helps when middle positions receive less attention.
    """

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Interleave documents by relevance."""
        if len(documents) <= 2:
            return documents

        # Sort by score
        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)

        # Split into high and low relevance
        mid = len(sorted_docs) // 2
        high_relevance = sorted_docs[:mid]
        low_relevance = sorted_docs[mid:]

        # Interleave
        result = []
        for high, low in zip(high_relevance, low_relevance):
            result.append(high)
            result.append(low)

        # Add remaining if odd number
        if len(high_relevance) > len(low_relevance):
            result.append(high_relevance[-1])
        elif len(low_relevance) > len(high_relevance):
            result.append(low_relevance[-1])

        return result


class TieredReranker(PositionAwareReranker):
    """
    Groups documents into relevance tiers.

    Places tier boundaries at attention-favorable positions.
    """

    def __init__(
        self,
        num_tiers: int = 3,
        tier_thresholds: list[float] | None = None,
    ):
        """
        Initialize tiered reranker.

        Args:
            num_tiers: Number of relevance tiers
            tier_thresholds: Custom thresholds (0-1) for tiers
        """
        self.num_tiers = num_tiers
        self.tier_thresholds = tier_thresholds

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Rerank into tiers with strategic placement."""
        if len(documents) <= self.num_tiers:
            return sorted(documents, key=lambda d: d.score, reverse=True)

        # Normalize scores
        max_score = max(d.score for d in documents)
        min_score = min(d.score for d in documents)
        score_range = max_score - min_score or 1.0

        # Assign tiers
        tiers: list[list[RetrievedDocument]] = [[] for _ in range(self.num_tiers)]

        for doc in documents:
            normalized = (doc.score - min_score) / score_range
            tier_idx = min(int(normalized * self.num_tiers), self.num_tiers - 1)
            # Invert so tier 0 is highest
            tier_idx = self.num_tiers - 1 - tier_idx
            tiers[tier_idx].append(doc)

        # Sort within tiers
        for tier in tiers:
            tier.sort(key=lambda d: d.score, reverse=True)

        # Strategic placement: best tier at start, second best at end
        result = []
        if len(tiers) >= 1 and tiers[0]:
            result.extend(tiers[0])  # Best at start
        if len(tiers) >= 3:
            for tier in tiers[2:]:  # Middle tiers
                result.extend(tier)
        if len(tiers) >= 2 and tiers[1]:
            result.extend(tiers[1])  # Second best at end

        return result


# ============================================================================
# LLM-Based Position-Aware Reranker
# ============================================================================


class LLMPositionReranker(PositionAwareReranker):
    """
    Uses LLM to score documents with position debiasing.

    Evaluates each document independently then applies
    position-aware reordering.
    """

    SCORING_PROMPT = """Rate how relevant this passage is to answering the question.

Question: {query}

Passage: {passage}

Rate relevance from 0 to 10 (10 = highly relevant, directly answers question):
Score:"""

    def __init__(
        self,
        llm: LLMProtocol,
        bias_config: PositionBiasConfig | None = None,
    ):
        """Initialize with LLM."""
        self.llm = llm
        self.bias_model = PositionBiasModel(
            bias_config or PositionBiasConfig(bias_type=PositionBias.U_SHAPED)
        )

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Score documents and apply position-aware reordering."""
        # Score each document
        for doc in documents:
            score = await self._score_document(query, doc)
            doc.score = score

        # Get optimal ordering
        scores = [doc.score for doc in documents]
        optimal_order = self.bias_model.get_optimal_positions(scores, len(documents))

        # Reorder
        return [documents[i] for i in optimal_order]

    async def _score_document(
        self,
        query: str,
        document: RetrievedDocument,
    ) -> float:
        """Score a single document."""
        prompt = self.SCORING_PROMPT.format(
            query=query,
            passage=document.content[:500],
        )

        response = await self.llm.generate(prompt, max_tokens=10, temperature=0.0)

        # Parse score
        try:
            score = float("".join(c for c in response if c.isdigit() or c == "."))
            return min(10, max(0, score)) / 10
        except ValueError:
            return 0.5


# ============================================================================
# Chunked Attention Reranker
# ============================================================================


@dataclass
class ChunkConfig:
    """Configuration for chunked attention."""

    chunk_size: int = 5  # Documents per chunk
    overlap: int = 1  # Overlap between chunks
    use_recency_boost: bool = True


class ChunkedAttentionReranker(PositionAwareReranker):
    """
    Processes documents in attention-optimized chunks.

    Breaks long document lists into chunks that fit
    within optimal attention windows.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        config: ChunkConfig | None = None,
    ):
        """Initialize chunked reranker."""
        self.llm = llm
        self.config = config or ChunkConfig()

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Rerank using chunked attention."""
        if len(documents) <= self.config.chunk_size:
            return await self._rank_chunk(query, documents)

        # Create overlapping chunks
        chunks = self._create_chunks(documents)

        # Rank within each chunk
        chunk_results = []
        for chunk in chunks:
            ranked_chunk = await self._rank_chunk(query, chunk)
            chunk_results.append(ranked_chunk)

        # Merge chunk results
        return self._merge_chunks(chunk_results)

    def _create_chunks(
        self,
        documents: list[RetrievedDocument],
    ) -> list[list[RetrievedDocument]]:
        """Create overlapping chunks."""
        chunks = []
        step = self.config.chunk_size - self.config.overlap

        for i in range(0, len(documents), step):
            chunk = documents[i : i + self.config.chunk_size]
            if chunk:
                chunks.append(chunk)

        return chunks

    async def _rank_chunk(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Rank a single chunk using LLM."""
        doc_list = "\n".join(f"[{i+1}] {doc.content[:200]}" for i, doc in enumerate(documents))

        prompt = f"""Rank these passages by relevance to the question.

Question: {query}

Passages:
{doc_list}

Return the passage numbers in order of relevance (most relevant first), comma-separated:"""

        response = await self.llm.generate(prompt, max_tokens=50, temperature=0.0)

        # Parse ranking
        try:
            indices = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]
            indices = [i for i in indices if 0 <= i < len(documents)]

            # Add any missing indices
            seen = set(indices)
            for i in range(len(documents)):
                if i not in seen:
                    indices.append(i)

            return [documents[i] for i in indices]
        except ValueError:
            return documents

    def _merge_chunks(
        self,
        chunk_results: list[list[RetrievedDocument]],
    ) -> list[RetrievedDocument]:
        """Merge ranked chunks."""
        seen_content: set[str] = set()
        merged = []

        # Take top from each chunk in round-robin
        max_len = max(len(chunk) for chunk in chunk_results)

        for pos in range(max_len):
            for chunk in chunk_results:
                if pos < len(chunk):
                    doc = chunk[pos]
                    if doc.content not in seen_content:
                        merged.append(doc)
                        seen_content.add(doc.content)

        return merged


# ============================================================================
# Position-Aware Fusion
# ============================================================================


class PositionAwareFusion:
    """
    Fuses multiple retrieval results with position awareness.

    Applies position weights when combining scores from
    multiple retrieval methods.
    """

    def __init__(
        self,
        bias_config: PositionBiasConfig | None = None,
        rrf_k: int = 60,
    ):
        """
        Initialize fusion.

        Args:
            bias_config: Position bias configuration
            rrf_k: RRF constant
        """
        self.bias_model = PositionBiasModel(
            bias_config or PositionBiasConfig(bias_type=PositionBias.U_SHAPED)
        )
        self.rrf_k = rrf_k

    def fuse(
        self,
        result_lists: list[list[RetrievedDocument]],
        weights: list[float] | None = None,
    ) -> list[RetrievedDocument]:
        """
        Fuse multiple result lists with position awareness.

        Args:
            result_lists: Lists of documents from different retrievers
            weights: Optional weights for each list

        Returns:
            Fused and reordered documents
        """
        if not result_lists:
            return []

        if weights is None:
            weights = [1.0] * len(result_lists)

        # Calculate position-aware RRF scores
        doc_scores: dict[str, tuple[RetrievedDocument, float]] = {}

        for list_idx, doc_list in enumerate(result_lists):
            list_weight = weights[list_idx]
            total_docs = len(doc_list)

            for pos, doc in enumerate(doc_list):
                # Position weight counteracts bias
                pos_weight = self.bias_model.get_position_weight(pos, total_docs)

                # RRF score with position adjustment
                rrf_score = list_weight / (self.rrf_k + pos + 1)
                adjusted_score = rrf_score * pos_weight

                key = doc.content[:100]  # Use content prefix as key
                if key in doc_scores:
                    _, existing_score = doc_scores[key]
                    doc_scores[key] = (doc, existing_score + adjusted_score)
                else:
                    doc_scores[key] = (doc, adjusted_score)

        # Sort by fused score
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)

        # Update scores and apply optimal ordering
        result = []
        for doc, score in sorted_docs:
            doc.score = score
            result.append(doc)

        # Apply position-aware reordering to final result
        scores = [d.score for d in result]
        optimal_order = self.bias_model.get_optimal_positions(scores, len(result))

        return [result[i] for i in optimal_order]


# ============================================================================
# Factory Functions
# ============================================================================


def create_primacy_recency_reranker(
    preserve_top_n: int = 3,
    bias_type: PositionBias = PositionBias.U_SHAPED,
) -> PrimacyRecencyReranker:
    """
    Create primacy-recency reranker.

    Args:
        preserve_top_n: Keep top-N at start
        bias_type: Position bias type

    Returns:
        Configured reranker
    """
    config = PositionBiasConfig(bias_type=bias_type)
    return PrimacyRecencyReranker(config, preserve_top_n)


def create_llm_position_reranker(
    llm: LLMProtocol,
    bias_type: PositionBias = PositionBias.U_SHAPED,
) -> LLMPositionReranker:
    """
    Create LLM-based position-aware reranker.

    Args:
        llm: LLM provider
        bias_type: Position bias type

    Returns:
        Configured reranker
    """
    config = PositionBiasConfig(bias_type=bias_type)
    return LLMPositionReranker(llm, config)


def create_chunked_reranker(
    llm: LLMProtocol,
    chunk_size: int = 5,
    overlap: int = 1,
) -> ChunkedAttentionReranker:
    """
    Create chunked attention reranker.

    Args:
        llm: LLM provider
        chunk_size: Documents per chunk
        overlap: Overlap between chunks

    Returns:
        Configured reranker
    """
    config = ChunkConfig(chunk_size=chunk_size, overlap=overlap)
    return ChunkedAttentionReranker(llm, config)


def create_position_aware_fusion(
    bias_type: PositionBias = PositionBias.U_SHAPED,
    rrf_k: int = 60,
) -> PositionAwareFusion:
    """
    Create position-aware fusion.

    Args:
        bias_type: Position bias type
        rrf_k: RRF constant

    Returns:
        Configured fusion
    """
    config = PositionBiasConfig(bias_type=bias_type)
    return PositionAwareFusion(config, rrf_k)


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example demonstrating position-aware reranking."""
    print("Position-Aware Reranking Implementation Ready")
    print("=" * 50)
    print("\nAddresses: Lost in the Middle (Liu et al., 2023)")
    print("\nFeatures:")
    print("- Position bias modeling (primacy, recency, U-shaped)")
    print("- Primacy-recency reranking")
    print("- Interleaved reranking")
    print("- Tiered grouping")
    print("- LLM-based position-aware scoring")
    print("- Chunked attention processing")
    print("- Position-aware fusion (RRF + position weights)")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
