"""
RAG Fusion Implementation.

Implements multi-perspective query generation and retrieval fusion
for improved recall and diversity.

Reference: Raudaschl, 2023 - "Forget RAG, the Future is RAG-Fusion"

Key features:
1. Multi-query generation from different perspectives
2. Parallel retrieval for each query
3. Reciprocal Rank Fusion (RRF) for result merging
4. Query clustering and diversity optimization
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
class RetrieverProtocol(Protocol):
    """Protocol for retrieval backends."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list["RetrievedDocument"]:
        """Retrieve documents for query."""
        ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts."""
        ...


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""
    source_query: str = ""


# ============================================================================
# RAG Fusion Types
# ============================================================================


class QueryGenerationStrategy(Enum):
    """Strategies for generating alternative queries."""

    PERSPECTIVE = "perspective"  # Different perspectives
    PARAPHRASE = "paraphrase"  # Paraphrases
    DECOMPOSE = "decompose"  # Break into sub-questions
    EXPAND = "expand"  # Add context
    ABSTRACT = "abstract"  # More abstract version
    SPECIFIC = "specific"  # More specific version


class FusionMethod(Enum):
    """Methods for fusing retrieval results."""

    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED_RRF = "weighted_rrf"  # Weighted RRF
    COMBSUM = "combsum"  # Sum of scores
    COMBMNZ = "combmnz"  # Sum * count
    MAX = "max"  # Maximum score
    BORDA = "borda"  # Borda count


@dataclass
class RAGFusionConfig:
    """Configuration for RAG Fusion."""

    num_queries: int = 4
    queries_per_strategy: int = 2
    strategies: list[QueryGenerationStrategy] = field(default_factory=list)
    fusion_method: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60  # RRF constant
    top_k_per_query: int = 5
    final_top_k: int = 10
    deduplicate: bool = True
    min_query_diversity: float = 0.3

    def __post_init__(self):
        if not self.strategies:
            self.strategies = [
                QueryGenerationStrategy.PERSPECTIVE,
                QueryGenerationStrategy.PARAPHRASE,
            ]


@dataclass
class FusionResult:
    """Result from RAG Fusion."""

    original_query: str
    generated_queries: list[str]
    per_query_results: dict[str, list[RetrievedDocument]]
    fused_results: list[RetrievedDocument]
    fusion_method: FusionMethod
    query_diversity_score: float


# ============================================================================
# Query Generators
# ============================================================================


class QueryGenerator(ABC):
    """Abstract base for query generation."""

    @abstractmethod
    async def generate(
        self,
        query: str,
        num_queries: int = 4,
    ) -> list[str]:
        """Generate alternative queries."""
        ...


class PerspectiveQueryGenerator(QueryGenerator):
    """Generate queries from different perspectives."""

    PROMPT = """Generate {num} different search queries that would help answer this question from different perspectives.
Each query should approach the question from a different angle.

Original question: {query}

Different perspective queries (one per line, no numbering):"""

    def __init__(self, llm: LLMProtocol):
        """Initialize with LLM."""
        self.llm = llm

    async def generate(
        self,
        query: str,
        num_queries: int = 4,
    ) -> list[str]:
        """Generate perspective queries."""
        prompt = self.PROMPT.format(query=query, num=num_queries)

        response = await self.llm.generate(prompt, max_tokens=300, temperature=0.7)

        return self._parse_queries(response, num_queries)

    def _parse_queries(self, response: str, max_queries: int) -> list[str]:
        """Parse generated queries."""
        queries = []
        for line in response.split("\n"):
            line = line.strip()
            if line and not line.startswith(("-", "*", "•")):
                # Remove numbering
                if line[0].isdigit():
                    line = line.split(".", 1)[-1].strip()
                    line = line.split(")", 1)[-1].strip()
                if line:
                    queries.append(line)

        return queries[:max_queries]


class ParaphraseQueryGenerator(QueryGenerator):
    """Generate paraphrased queries."""

    PROMPT = """Rewrite this search query {num} different ways while keeping the same meaning.
Use different words and sentence structures.

Original query: {query}

Paraphrased queries (one per line):"""

    def __init__(self, llm: LLMProtocol):
        """Initialize with LLM."""
        self.llm = llm

    async def generate(
        self,
        query: str,
        num_queries: int = 4,
    ) -> list[str]:
        """Generate paraphrased queries."""
        prompt = self.PROMPT.format(query=query, num=num_queries)

        response = await self.llm.generate(prompt, max_tokens=300, temperature=0.8)

        return self._parse_queries(response, num_queries)

    def _parse_queries(self, response: str, max_queries: int) -> list[str]:
        """Parse generated queries."""
        queries = []
        for line in response.split("\n"):
            line = line.strip()
            if line:
                if line[0].isdigit():
                    line = line.split(".", 1)[-1].strip()
                    line = line.split(")", 1)[-1].strip()
                if line:
                    queries.append(line)

        return queries[:max_queries]


class DecomposeQueryGenerator(QueryGenerator):
    """Decompose complex queries into sub-questions."""

    PROMPT = """Break down this complex question into {num} simpler sub-questions that together help answer the original.

Original question: {query}

Sub-questions (one per line):"""

    def __init__(self, llm: LLMProtocol):
        """Initialize with LLM."""
        self.llm = llm

    async def generate(
        self,
        query: str,
        num_queries: int = 4,
    ) -> list[str]:
        """Generate decomposed sub-questions."""
        prompt = self.PROMPT.format(query=query, num=num_queries)

        response = await self.llm.generate(prompt, max_tokens=300, temperature=0.5)

        queries = []
        for line in response.split("\n"):
            line = line.strip()
            if line:
                if line[0].isdigit():
                    line = line.split(".", 1)[-1].strip()
                if line:
                    queries.append(line)

        return queries[:num_queries]


class MultiStrategyQueryGenerator(QueryGenerator):
    """Generate queries using multiple strategies."""

    def __init__(
        self,
        llm: LLMProtocol,
        strategies: list[QueryGenerationStrategy] | None = None,
    ):
        """
        Initialize multi-strategy generator.

        Args:
            llm: LLM provider
            strategies: List of strategies to use
        """
        self.llm = llm
        self.strategies = strategies or [
            QueryGenerationStrategy.PERSPECTIVE,
            QueryGenerationStrategy.PARAPHRASE,
        ]

        # Initialize strategy generators
        self.generators: dict[QueryGenerationStrategy, QueryGenerator] = {
            QueryGenerationStrategy.PERSPECTIVE: PerspectiveQueryGenerator(llm),
            QueryGenerationStrategy.PARAPHRASE: ParaphraseQueryGenerator(llm),
            QueryGenerationStrategy.DECOMPOSE: DecomposeQueryGenerator(llm),
        }

    async def generate(
        self,
        query: str,
        num_queries: int = 4,
    ) -> list[str]:
        """Generate queries using multiple strategies."""
        all_queries = []
        queries_per_strategy = max(1, num_queries // len(self.strategies))

        for strategy in self.strategies:
            generator = self.generators.get(strategy)
            if generator:
                strategy_queries = await generator.generate(query, queries_per_strategy)
                all_queries.extend(strategy_queries)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_queries = []
        for q in all_queries:
            if q.lower() not in seen:
                unique_queries.append(q)
                seen.add(q.lower())

        return unique_queries[:num_queries]


# ============================================================================
# Fusion Methods
# ============================================================================


class ResultFuser(ABC):
    """Abstract base for result fusion."""

    @abstractmethod
    def fuse(
        self,
        query_results: dict[str, list[RetrievedDocument]],
    ) -> list[RetrievedDocument]:
        """Fuse results from multiple queries."""
        ...


class RRFFuser(ResultFuser):
    """Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        k: int = 60,
        query_weights: dict[str, float] | None = None,
    ):
        """
        Initialize RRF fuser.

        Args:
            k: RRF constant (higher = less aggressive rank weighting)
            query_weights: Optional weights for each query
        """
        self.k = k
        self.query_weights = query_weights or {}

    def fuse(
        self,
        query_results: dict[str, list[RetrievedDocument]],
    ) -> list[RetrievedDocument]:
        """Fuse using RRF scoring."""
        doc_scores: dict[str, tuple[RetrievedDocument, float]] = {}

        for query, docs in query_results.items():
            weight = self.query_weights.get(query, 1.0)

            for rank, doc in enumerate(docs, 1):
                key = doc.content[:100]  # Use content prefix as key
                rrf_score = weight / (self.k + rank)

                if key in doc_scores:
                    existing_doc, existing_score = doc_scores[key]
                    doc_scores[key] = (existing_doc, existing_score + rrf_score)
                else:
                    doc.source_query = query
                    doc_scores[key] = (doc, rrf_score)

        # Sort by fused score
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)

        # Update scores
        result = []
        for doc, score in sorted_docs:
            doc.score = score
            result.append(doc)

        return result


class CombSumFuser(ResultFuser):
    """CombSUM fusion (sum of normalized scores)."""

    def fuse(
        self,
        query_results: dict[str, list[RetrievedDocument]],
    ) -> list[RetrievedDocument]:
        """Fuse using score summation."""
        doc_scores: dict[str, tuple[RetrievedDocument, float, int]] = {}

        for query, docs in query_results.items():
            # Normalize scores within each result list
            if docs:
                max_score = max(d.score for d in docs)
                min_score = min(d.score for d in docs)
                score_range = max_score - min_score or 1.0

            for doc in docs:
                normalized_score = (doc.score - min_score) / score_range
                key = doc.content[:100]

                if key in doc_scores:
                    existing_doc, existing_score, count = doc_scores[key]
                    doc_scores[key] = (
                        existing_doc,
                        existing_score + normalized_score,
                        count + 1,
                    )
                else:
                    doc.source_query = query
                    doc_scores[key] = (doc, normalized_score, 1)

        # Sort by fused score
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)

        result = []
        for doc, score, _ in sorted_docs:
            doc.score = score
            result.append(doc)

        return result


class CombMNZFuser(ResultFuser):
    """CombMNZ fusion (sum * count)."""

    def fuse(
        self,
        query_results: dict[str, list[RetrievedDocument]],
    ) -> list[RetrievedDocument]:
        """Fuse using CombMNZ (sum * count)."""
        doc_scores: dict[str, tuple[RetrievedDocument, float, int]] = {}

        for query, docs in query_results.items():
            for doc in docs:
                key = doc.content[:100]

                if key in doc_scores:
                    existing_doc, existing_score, count = doc_scores[key]
                    doc_scores[key] = (
                        existing_doc,
                        existing_score + doc.score,
                        count + 1,
                    )
                else:
                    doc.source_query = query
                    doc_scores[key] = (doc, doc.score, 1)

        # Apply MNZ: score * count
        sorted_docs = sorted(
            [(doc, score * count, count) for doc, score, count in doc_scores.values()],
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for doc, score, _ in sorted_docs:
            doc.score = score
            result.append(doc)

        return result


class BordaFuser(ResultFuser):
    """Borda count fusion."""

    def fuse(
        self,
        query_results: dict[str, list[RetrievedDocument]],
    ) -> list[RetrievedDocument]:
        """Fuse using Borda count."""
        doc_scores: dict[str, tuple[RetrievedDocument, float]] = {}

        for query, docs in query_results.items():
            num_docs = len(docs)

            for rank, doc in enumerate(docs):
                # Borda score: n - rank (higher rank = higher score)
                borda_score = num_docs - rank
                key = doc.content[:100]

                if key in doc_scores:
                    existing_doc, existing_score = doc_scores[key]
                    doc_scores[key] = (existing_doc, existing_score + borda_score)
                else:
                    doc.source_query = query
                    doc_scores[key] = (doc, borda_score)

        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)

        result = []
        for doc, score in sorted_docs:
            doc.score = score
            result.append(doc)

        return result


# ============================================================================
# Query Diversity
# ============================================================================


class QueryDiversityOptimizer:
    """Optimize query diversity to maximize coverage."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        min_diversity: float = 0.3,
    ):
        """
        Initialize diversity optimizer.

        Args:
            embedder: Embedding provider
            min_diversity: Minimum pairwise diversity threshold
        """
        self.embedder = embedder
        self.min_diversity = min_diversity

    async def calculate_diversity(
        self,
        queries: list[str],
    ) -> float:
        """Calculate average pairwise diversity."""
        if len(queries) < 2:
            return 1.0

        embeddings = await self.embedder.embed(queries)

        total_diversity = 0.0
        num_pairs = 0

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                diversity = 1.0 - similarity
                total_diversity += diversity
                num_pairs += 1

        return total_diversity / max(num_pairs, 1)

    async def select_diverse(
        self,
        queries: list[str],
        num_select: int,
    ) -> list[str]:
        """Select most diverse subset of queries."""
        if len(queries) <= num_select:
            return queries

        embeddings = await self.embedder.embed(queries)

        # Greedy selection for diversity
        selected_indices = [0]  # Start with first query

        while len(selected_indices) < num_select:
            best_idx = -1
            best_min_diversity = -1

            for i in range(len(queries)):
                if i in selected_indices:
                    continue

                # Calculate minimum diversity to selected queries
                min_div = float("inf")
                for j in selected_indices:
                    sim = self._cosine_similarity(embeddings[i], embeddings[j])
                    div = 1.0 - sim
                    min_div = min(min_div, div)

                if min_div > best_min_diversity:
                    best_min_diversity = min_div
                    best_idx = i

            if best_idx >= 0:
                selected_indices.append(best_idx)

        return [queries[i] for i in selected_indices]

    def _cosine_similarity(self, emb1: list[float], emb2: list[float]) -> float:
        """Calculate cosine similarity."""
        import math

        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)


# ============================================================================
# RAG Fusion Pipeline
# ============================================================================


class RAGFusion:
    """
    RAG Fusion pipeline.

    Generates multiple queries, retrieves for each, and fuses results.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol,
        config: RAGFusionConfig | None = None,
        embedder: EmbedderProtocol | None = None,
    ):
        """
        Initialize RAG Fusion.

        Args:
            llm: LLM for query generation
            retriever: Retrieval backend
            config: Configuration options
            embedder: Optional embedder for diversity optimization
        """
        self.llm = llm
        self.retriever = retriever
        self.config = config or RAGFusionConfig()
        self.embedder = embedder

        # Initialize query generator
        self.query_generator = MultiStrategyQueryGenerator(llm, self.config.strategies)

        # Initialize fuser
        self.fuser = self._get_fuser()

        # Optional diversity optimizer
        self.diversity_optimizer = (
            QueryDiversityOptimizer(embedder, self.config.min_query_diversity) if embedder else None
        )

    def _get_fuser(self) -> ResultFuser:
        """Get appropriate fuser for config."""
        fusers = {
            FusionMethod.RRF: RRFFuser(self.config.rrf_k),
            FusionMethod.COMBSUM: CombSumFuser(),
            FusionMethod.COMBMNZ: CombMNZFuser(),
            FusionMethod.BORDA: BordaFuser(),
        }
        return fusers.get(self.config.fusion_method, RRFFuser(self.config.rrf_k))

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> FusionResult:
        """
        Retrieve with RAG Fusion.

        Args:
            query: Original user query
            top_k: Final number of documents

        Returns:
            FusionResult with fused documents
        """
        final_k = top_k or self.config.final_top_k

        # Generate alternative queries
        generated_queries = await self.query_generator.generate(query, self.config.num_queries)

        # Always include original query
        all_queries = [query] + generated_queries

        # Optimize for diversity if embedder available
        diversity_score = 1.0
        if self.diversity_optimizer:
            diversity_score = await self.diversity_optimizer.calculate_diversity(all_queries)

            if len(all_queries) > self.config.num_queries + 1:
                all_queries = await self.diversity_optimizer.select_diverse(
                    all_queries, self.config.num_queries + 1
                )

        # Retrieve for each query
        per_query_results: dict[str, list[RetrievedDocument]] = {}
        for q in all_queries:
            results = await self.retriever.retrieve(q, top_k=self.config.top_k_per_query)
            per_query_results[q] = results

        # Fuse results
        fused = self.fuser.fuse(per_query_results)

        # Deduplicate if configured
        if self.config.deduplicate:
            fused = self._deduplicate(fused)

        # Truncate to final_k
        fused = fused[:final_k]

        return FusionResult(
            original_query=query,
            generated_queries=generated_queries,
            per_query_results=per_query_results,
            fused_results=fused,
            fusion_method=self.config.fusion_method,
            query_diversity_score=diversity_score,
        )

    def _deduplicate(
        self,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Remove duplicate documents."""
        seen: set[str] = set()
        unique = []

        for doc in documents:
            key = doc.content[:200]
            if key not in seen:
                unique.append(doc)
                seen.add(key)

        return unique


# ============================================================================
# Factory Functions
# ============================================================================


def create_rag_fusion(
    llm: LLMProtocol,
    retriever: RetrieverProtocol,
    num_queries: int = 4,
    fusion_method: FusionMethod = FusionMethod.RRF,
) -> RAGFusion:
    """
    Create RAG Fusion pipeline.

    Args:
        llm: LLM for query generation
        retriever: Retrieval backend
        num_queries: Number of alternative queries
        fusion_method: Result fusion method

    Returns:
        Configured RAGFusion
    """
    config = RAGFusionConfig(
        num_queries=num_queries,
        fusion_method=fusion_method,
    )
    return RAGFusion(llm, retriever, config)


def create_diverse_rag_fusion(
    llm: LLMProtocol,
    retriever: RetrieverProtocol,
    embedder: EmbedderProtocol,
    num_queries: int = 4,
    min_diversity: float = 0.3,
) -> RAGFusion:
    """
    Create RAG Fusion with diversity optimization.

    Args:
        llm: LLM for query generation
        retriever: Retrieval backend
        embedder: Embedder for diversity calculation
        num_queries: Number of alternative queries
        min_diversity: Minimum query diversity

    Returns:
        Configured RAGFusion with diversity
    """
    config = RAGFusionConfig(
        num_queries=num_queries,
        min_query_diversity=min_diversity,
    )
    return RAGFusion(llm, retriever, config, embedder)


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example demonstrating RAG Fusion."""
    print("RAG Fusion Implementation Ready")
    print("=" * 50)
    print("\nReference: Raudaschl, 2023")
    print("\nFeatures:")
    print("- Multi-strategy query generation:")
    print("  - Perspective-based")
    print("  - Paraphrase")
    print("  - Decomposition")
    print("- Fusion methods:")
    print("  - RRF (Reciprocal Rank Fusion)")
    print("  - CombSUM (normalized score sum)")
    print("  - CombMNZ (sum * count)")
    print("  - Borda count")
    print("- Query diversity optimization")
    print("- Result deduplication")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
