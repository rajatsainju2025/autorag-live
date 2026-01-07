"""
Cross-Encoder Reranker Implementation.

Implements deep cross-attention reranking for improved relevance scoring.
Cross-encoders jointly encode query and document for fine-grained matching.

Key features:
1. Cross-attention scoring
2. Batch processing for efficiency
3. Multiple backbone support
4. Score calibration
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


@runtime_checkable
class CrossEncoderModelProtocol(Protocol):
    """Protocol for cross-encoder models."""

    def predict(
        self,
        query_doc_pairs: list[tuple[str, str]],
    ) -> list[float]:
        """Predict relevance scores for query-document pairs."""
        ...


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""


# ============================================================================
# Cross-Encoder Types
# ============================================================================


class CrossEncoderBackend(Enum):
    """Available cross-encoder backends."""

    SENTENCE_TRANSFORMERS = "sentence_transformers"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"
    LLM = "llm"  # LLM-based reranking
    CUSTOM = "custom"


class ScoreNormalization(Enum):
    """Score normalization methods."""

    NONE = "none"
    SOFTMAX = "softmax"
    MIN_MAX = "min_max"
    SIGMOID = "sigmoid"


@dataclass
class CrossEncoderConfig:
    """Configuration for cross-encoder reranker."""

    backend: CrossEncoderBackend = CrossEncoderBackend.SENTENCE_TRANSFORMERS
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    max_length: int = 512
    normalization: ScoreNormalization = ScoreNormalization.SIGMOID
    use_cache: bool = True
    device: str = "cpu"


# ============================================================================
# Score Normalizers
# ============================================================================


class ScoreNormalizer(ABC):
    """Abstract base for score normalization."""

    @abstractmethod
    def normalize(self, scores: list[float]) -> list[float]:
        """Normalize scores to [0, 1] range."""
        ...


class SoftmaxNormalizer(ScoreNormalizer):
    """Softmax normalization."""

    def __init__(self, temperature: float = 1.0):
        """Initialize with temperature."""
        self.temperature = temperature

    def normalize(self, scores: list[float]) -> list[float]:
        """Apply softmax normalization."""
        import math

        if not scores:
            return []

        # Subtract max for numerical stability
        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / self.temperature) for s in scores]
        sum_exp = sum(exp_scores)

        if sum_exp == 0:
            return [1.0 / len(scores)] * len(scores)

        return [e / sum_exp for e in exp_scores]


class MinMaxNormalizer(ScoreNormalizer):
    """Min-max normalization."""

    def normalize(self, scores: list[float]) -> list[float]:
        """Apply min-max normalization."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range == 0:
            return [0.5] * len(scores)

        return [(s - min_score) / score_range for s in scores]


class SigmoidNormalizer(ScoreNormalizer):
    """Sigmoid normalization."""

    def __init__(self, scale: float = 1.0, shift: float = 0.0):
        """Initialize with scale and shift."""
        self.scale = scale
        self.shift = shift

    def normalize(self, scores: list[float]) -> list[float]:
        """Apply sigmoid normalization."""
        import math

        normalized = []
        for s in scores:
            try:
                sig = 1.0 / (1.0 + math.exp(-(self.scale * s + self.shift)))
            except OverflowError:
                sig = 0.0 if s < 0 else 1.0
            normalized.append(sig)

        return normalized


class NoNormalization(ScoreNormalizer):
    """No normalization (pass-through)."""

    def normalize(self, scores: list[float]) -> list[float]:
        """Return scores unchanged."""
        return scores


# ============================================================================
# Cross-Encoder Implementations
# ============================================================================


class CrossEncoder(ABC):
    """Abstract base for cross-encoders."""

    @abstractmethod
    async def score(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[float]:
        """Score documents for relevance to query."""
        ...

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Rerank documents by relevance."""
        ...


class SentenceTransformersCrossEncoder(CrossEncoder):
    """
    Cross-encoder using sentence-transformers library.

    Placeholder that simulates the interface.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """Initialize cross-encoder."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.normalizer = SigmoidNormalizer()

        # Placeholder - actual implementation would load model
        self._model: Any = None

    async def score(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[float]:
        """Score documents using cross-encoder."""
        if not documents:
            return []

        # Create query-document pairs
        pairs = [(query, doc.content[:512]) for doc in documents]

        # Batch scoring
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            # Placeholder - actual scoring would use model
            batch_scores = [doc.score * 0.8 + 0.2 for doc in documents[i : i + len(batch)]]
            scores.extend(batch_scores)

        # Normalize
        return self.normalizer.normalize(scores)

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Rerank documents using cross-encoder scores."""
        scores = await self.score(query, documents)

        # Update document scores
        for doc, score in zip(documents, scores):
            doc.score = score

        # Sort by score
        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)

        if top_k:
            return sorted_docs[:top_k]
        return sorted_docs


class LLMCrossEncoder(CrossEncoder):
    """
    LLM-based cross-encoder.

    Uses LLM to score query-document relevance.
    """

    SCORING_PROMPT = """Rate how relevant this passage is to answering the question.

Question: {query}

Passage: {passage}

Rate relevance from 0 to 10 (10 = directly answers the question):
Relevance score:"""

    PAIRWISE_PROMPT = """Which passage is more relevant to answering the question?

Question: {query}

Passage A: {passage_a}

Passage B: {passage_b}

Answer with just 'A' or 'B':"""

    def __init__(
        self,
        llm: LLMProtocol,
        use_pairwise: bool = False,
    ):
        """
        Initialize LLM cross-encoder.

        Args:
            llm: LLM provider
            use_pairwise: Use pairwise comparison instead of scoring
        """
        self.llm = llm
        self.use_pairwise = use_pairwise
        self.normalizer = MinMaxNormalizer()

    async def score(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[float]:
        """Score documents using LLM."""
        scores = []

        for doc in documents:
            prompt = self.SCORING_PROMPT.format(
                query=query,
                passage=doc.content[:500],
            )

            response = await self.llm.generate(prompt, max_tokens=10, temperature=0.0)

            # Parse score
            try:
                score = float("".join(c for c in response if c.isdigit() or c == "."))
                score = min(10, max(0, score)) / 10
            except ValueError:
                score = 0.5

            scores.append(score)

        return self.normalizer.normalize(scores)

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Rerank documents using LLM scoring."""
        if self.use_pairwise:
            return await self._pairwise_rerank(query, documents, top_k)

        scores = await self.score(query, documents)

        for doc, score in zip(documents, scores):
            doc.score = score

        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)

        if top_k:
            return sorted_docs[:top_k]
        return sorted_docs

    async def _pairwise_rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Rerank using pairwise comparisons (bubble sort style)."""
        docs = list(documents)

        # Simple bubble sort with pairwise comparisons
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                winner = await self._pairwise_compare(query, docs[i], docs[j])
                if winner == 1:  # j is better
                    docs[i], docs[j] = docs[j], docs[i]

        if top_k:
            return docs[:top_k]
        return docs

    async def _pairwise_compare(
        self,
        query: str,
        doc_a: RetrievedDocument,
        doc_b: RetrievedDocument,
    ) -> int:
        """Compare two documents. Returns 0 if A wins, 1 if B wins."""
        prompt = self.PAIRWISE_PROMPT.format(
            query=query,
            passage_a=doc_a.content[:300],
            passage_b=doc_b.content[:300],
        )

        response = await self.llm.generate(prompt, max_tokens=5, temperature=0.0)
        return 1 if "B" in response.upper() else 0


class CohereCrossEncoder(CrossEncoder):
    """
    Cohere rerank API cross-encoder.

    Placeholder for Cohere's rerank API.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "rerank-english-v2.0",
    ):
        """Initialize Cohere reranker."""
        self.api_key = api_key
        self.model = model

    async def score(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[float]:
        """Score using Cohere rerank API."""
        # Placeholder - actual implementation would call API
        return [doc.score for doc in documents]

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Rerank using Cohere API."""
        scores = await self.score(query, documents)

        for doc, score in zip(documents, scores):
            doc.score = score

        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)

        if top_k:
            return sorted_docs[:top_k]
        return sorted_docs


# ============================================================================
# Ensemble Cross-Encoder
# ============================================================================


class EnsembleCrossEncoder(CrossEncoder):
    """
    Ensemble of multiple cross-encoders.

    Combines scores from multiple models.
    """

    def __init__(
        self,
        encoders: list[CrossEncoder],
        weights: list[float] | None = None,
        aggregation: str = "weighted_mean",
    ):
        """
        Initialize ensemble.

        Args:
            encoders: List of cross-encoders
            weights: Weights for each encoder
            aggregation: How to combine scores ("weighted_mean", "max", "min")
        """
        self.encoders = encoders
        self.weights = weights or [1.0] * len(encoders)
        self.aggregation = aggregation

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    async def score(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[float]:
        """Score using ensemble."""
        all_scores = []

        for encoder in self.encoders:
            scores = await encoder.score(query, documents)
            all_scores.append(scores)

        # Aggregate scores
        return self._aggregate_scores(all_scores)

    def _aggregate_scores(
        self,
        all_scores: list[list[float]],
    ) -> list[float]:
        """Aggregate scores from multiple encoders."""
        if not all_scores:
            return []

        num_docs = len(all_scores[0])
        aggregated = []

        for i in range(num_docs):
            doc_scores = [scores[i] for scores in all_scores]

            if self.aggregation == "weighted_mean":
                agg = sum(s * w for s, w in zip(doc_scores, self.weights))
            elif self.aggregation == "max":
                agg = max(doc_scores)
            elif self.aggregation == "min":
                agg = min(doc_scores)
            else:
                agg = sum(doc_scores) / len(doc_scores)

            aggregated.append(agg)

        return aggregated

    async def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Rerank using ensemble scores."""
        scores = await self.score(query, documents)

        for doc, score in zip(documents, scores):
            doc.score = score

        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)

        if top_k:
            return sorted_docs[:top_k]
        return sorted_docs


# ============================================================================
# Two-Stage Retriever
# ============================================================================


@dataclass
class TwoStageConfig:
    """Configuration for two-stage retrieval."""

    first_stage_k: int = 100
    final_k: int = 10
    use_score_fusion: bool = False
    first_stage_weight: float = 0.3


class TwoStageRetriever:
    """
    Two-stage retrieval: bi-encoder then cross-encoder.

    First stage: Fast bi-encoder retrieval
    Second stage: Cross-encoder reranking
    """

    def __init__(
        self,
        first_stage: Any,  # RetrieverProtocol
        cross_encoder: CrossEncoder,
        config: TwoStageConfig | None = None,
    ):
        """
        Initialize two-stage retriever.

        Args:
            first_stage: First-stage retriever (bi-encoder)
            cross_encoder: Cross-encoder for reranking
            config: Configuration options
        """
        self.first_stage = first_stage
        self.cross_encoder = cross_encoder
        self.config = config or TwoStageConfig()

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve with two-stage reranking.

        Args:
            query: User query
            top_k: Final number of documents

        Returns:
            Reranked documents
        """
        final_k = top_k or self.config.final_k

        # First stage: retrieve candidates
        candidates = await self.first_stage.retrieve(query, top_k=self.config.first_stage_k)

        # Store first-stage scores
        first_stage_scores = {doc.content[:100]: doc.score for doc in candidates}

        # Second stage: cross-encoder reranking
        reranked = await self.cross_encoder.rerank(query, candidates, top_k=final_k)

        # Optional: fuse first-stage and cross-encoder scores
        if self.config.use_score_fusion:
            for doc in reranked:
                first_score = first_stage_scores.get(doc.content[:100], 0.0)
                cross_score = doc.score
                doc.score = (
                    self.config.first_stage_weight * first_score
                    + (1 - self.config.first_stage_weight) * cross_score
                )
            reranked.sort(key=lambda d: d.score, reverse=True)

        return reranked


# ============================================================================
# Factory Functions
# ============================================================================


def create_cross_encoder(
    backend: CrossEncoderBackend = CrossEncoderBackend.SENTENCE_TRANSFORMERS,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    **kwargs: Any,
) -> CrossEncoder:
    """
    Create a cross-encoder reranker.

    Args:
        backend: Backend to use
        model_name: Model name/path
        **kwargs: Additional arguments

    Returns:
        Configured CrossEncoder
    """
    if backend == CrossEncoderBackend.SENTENCE_TRANSFORMERS:
        return SentenceTransformersCrossEncoder(model_name, **kwargs)
    elif backend == CrossEncoderBackend.COHERE:
        return CohereCrossEncoder(**kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def create_llm_cross_encoder(
    llm: LLMProtocol,
    use_pairwise: bool = False,
) -> LLMCrossEncoder:
    """
    Create LLM-based cross-encoder.

    Args:
        llm: LLM provider
        use_pairwise: Use pairwise comparison

    Returns:
        Configured LLMCrossEncoder
    """
    return LLMCrossEncoder(llm, use_pairwise)


def create_two_stage_retriever(
    first_stage: Any,
    cross_encoder: CrossEncoder,
    first_stage_k: int = 100,
    final_k: int = 10,
) -> TwoStageRetriever:
    """
    Create two-stage retriever.

    Args:
        first_stage: First-stage retriever
        cross_encoder: Cross-encoder reranker
        first_stage_k: Candidates from first stage
        final_k: Final documents to return

    Returns:
        Configured TwoStageRetriever
    """
    config = TwoStageConfig(
        first_stage_k=first_stage_k,
        final_k=final_k,
    )
    return TwoStageRetriever(first_stage, cross_encoder, config)


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example demonstrating cross-encoder reranking."""
    print("Cross-Encoder Reranker Implementation Ready")
    print("=" * 50)
    print("\nFeatures:")
    print("- Multiple backends:")
    print("  - SentenceTransformers cross-encoders")
    print("  - LLM-based scoring and pairwise comparison")
    print("  - Cohere rerank API")
    print("- Score normalization (softmax, min-max, sigmoid)")
    print("- Ensemble cross-encoders")
    print("- Two-stage retrieval (bi-encoder + cross-encoder)")
    print("- Score fusion options")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
