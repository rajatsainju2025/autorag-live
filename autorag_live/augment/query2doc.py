"""
Query2Doc Implementation.

Implements query expansion via pseudo-document generation.
Based on: "Query2Doc: Query Expansion with Large Language Models"
(Wang et al., 2023)

Key idea: Generate a pseudo-document from the query using an LLM,
then use the expanded query (original + pseudo-doc) for retrieval.
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


# ============================================================================
# Query2Doc Configuration
# ============================================================================


class ExpansionStrategy(Enum):
    """Strategy for expanding queries."""

    PREPEND = "prepend"  # Pseudo-doc prepended to query
    APPEND = "append"  # Pseudo-doc appended to query
    WEIGHTED_CONCAT = "weighted_concat"  # Weighted combination
    SEPARATE = "separate"  # Separate retrievals merged


class PseudoDocStyle(Enum):
    """Style of pseudo-document generation."""

    PASSAGE = "passage"  # Wikipedia-style passage
    QA = "qa"  # Question-answer format
    DEFINITION = "definition"  # Definition style
    NARRATIVE = "narrative"  # Narrative explanation
    TECHNICAL = "technical"  # Technical documentation


@dataclass
class Query2DocConfig:
    """Configuration for Query2Doc."""

    expansion_strategy: ExpansionStrategy = ExpansionStrategy.PREPEND
    pseudo_doc_style: PseudoDocStyle = PseudoDocStyle.PASSAGE
    pseudo_doc_length: int = 200  # Target length in tokens
    num_pseudo_docs: int = 1  # Number of pseudo-docs to generate
    temperature: float = 0.7
    query_weight: float = 0.5  # For weighted concat
    top_k: int = 10


# ============================================================================
# Pseudo-Document Generators
# ============================================================================


class PseudoDocGenerator(ABC):
    """Abstract base for pseudo-document generation."""

    @abstractmethod
    async def generate(self, query: str) -> str:
        """Generate pseudo-document for query."""
        ...


class LLMPseudoDocGenerator(PseudoDocGenerator):
    """LLM-based pseudo-document generator."""

    STYLE_PROMPTS = {
        PseudoDocStyle.PASSAGE: """Write a detailed Wikipedia-style passage that would contain the answer to this question:

Question: {query}

Passage:""",
        PseudoDocStyle.QA: """Generate a question-answer pair where the question is similar to the input and the answer is comprehensive.

Input: {query}

Q: {query}
A:""",
        PseudoDocStyle.DEFINITION: """Write a clear, encyclopedic definition that would help answer this question:

Question: {query}

Definition:""",
        PseudoDocStyle.NARRATIVE: """Write a brief explanatory narrative that addresses this question:

Question: {query}

Explanation:""",
        PseudoDocStyle.TECHNICAL: """Write technical documentation that would contain information relevant to:

Question: {query}

Documentation:""",
    }

    def __init__(
        self,
        llm: LLMProtocol,
        style: PseudoDocStyle = PseudoDocStyle.PASSAGE,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ):
        """
        Initialize generator.

        Args:
            llm: LLM provider
            style: Style of pseudo-document
            max_tokens: Maximum tokens in pseudo-doc
            temperature: Generation temperature
        """
        self.llm = llm
        self.style = style
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def generate(self, query: str) -> str:
        """Generate pseudo-document for query."""
        prompt = self.STYLE_PROMPTS[self.style].format(query=query)

        pseudo_doc = await self.llm.generate(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return pseudo_doc.strip()


class MultiStylePseudoDocGenerator(PseudoDocGenerator):
    """Generate pseudo-docs in multiple styles."""

    def __init__(
        self,
        llm: LLMProtocol,
        styles: list[PseudoDocStyle] | None = None,
        max_tokens: int = 150,
    ):
        """Initialize with multiple styles."""
        self.llm = llm
        self.styles = styles or [
            PseudoDocStyle.PASSAGE,
            PseudoDocStyle.DEFINITION,
        ]
        self.max_tokens = max_tokens

    async def generate(self, query: str) -> str:
        """Generate pseudo-docs in multiple styles and combine."""
        generators = [
            LLMPseudoDocGenerator(self.llm, style, self.max_tokens) for style in self.styles
        ]

        pseudo_docs = []
        for gen in generators:
            doc = await gen.generate(query)
            pseudo_docs.append(doc)

        return "\n\n".join(pseudo_docs)


# ============================================================================
# Query Expanders
# ============================================================================


class QueryExpander(ABC):
    """Abstract base for query expansion."""

    @abstractmethod
    async def expand(self, query: str) -> str:
        """Expand query with pseudo-document."""
        ...


class Query2DocExpander(QueryExpander):
    """Main Query2Doc expander."""

    def __init__(
        self,
        generator: PseudoDocGenerator,
        strategy: ExpansionStrategy = ExpansionStrategy.PREPEND,
    ):
        """
        Initialize expander.

        Args:
            generator: Pseudo-document generator
            strategy: Expansion strategy
        """
        self.generator = generator
        self.strategy = strategy

    async def expand(self, query: str) -> str:
        """Expand query with pseudo-document."""
        pseudo_doc = await self.generator.generate(query)

        if self.strategy == ExpansionStrategy.PREPEND:
            return f"{pseudo_doc}\n\n{query}"
        elif self.strategy == ExpansionStrategy.APPEND:
            return f"{query}\n\n{pseudo_doc}"
        elif self.strategy == ExpansionStrategy.WEIGHTED_CONCAT:
            # For embedding-based, return combined
            return f"{query} {pseudo_doc}"
        else:
            # Separate - return just pseudo_doc
            return pseudo_doc


class MultiPseudoDocExpander(QueryExpander):
    """Generate multiple pseudo-docs and combine."""

    def __init__(
        self,
        generator: PseudoDocGenerator,
        num_docs: int = 3,
        combine_strategy: str = "concat",
    ):
        """
        Initialize with multiple pseudo-docs.

        Args:
            generator: Pseudo-document generator
            num_docs: Number of pseudo-docs to generate
            combine_strategy: How to combine ("concat" or "separate")
        """
        self.generator = generator
        self.num_docs = num_docs
        self.combine_strategy = combine_strategy

    async def expand(self, query: str) -> str:
        """Expand with multiple pseudo-documents."""
        pseudo_docs = []
        for _ in range(self.num_docs):
            doc = await self.generator.generate(query)
            pseudo_docs.append(doc)

        combined_docs = "\n\n".join(pseudo_docs)
        return f"{query}\n\n{combined_docs}"


# ============================================================================
# Weighted Query Embedding
# ============================================================================


@dataclass
class WeightedQueryEmbedding:
    """Result of weighted query embedding."""

    combined_embedding: list[float]
    query_embedding: list[float]
    pseudo_doc_embedding: list[float]
    query_weight: float


class WeightedEmbedder:
    """Embed query with weighted pseudo-doc contribution."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        query_weight: float = 0.5,
    ):
        """
        Initialize weighted embedder.

        Args:
            embedder: Base embedding provider
            query_weight: Weight for original query (0-1)
        """
        self.embedder = embedder
        self.query_weight = query_weight

    async def embed_expanded(
        self,
        query: str,
        pseudo_doc: str,
    ) -> WeightedQueryEmbedding:
        """
        Create weighted embedding of query + pseudo-doc.

        Args:
            query: Original query
            pseudo_doc: Generated pseudo-document

        Returns:
            WeightedQueryEmbedding with combined embedding
        """
        embeddings = await self.embedder.embed([query, pseudo_doc])
        query_emb = embeddings[0]
        doc_emb = embeddings[1]

        # Weighted combination
        doc_weight = 1 - self.query_weight
        combined = [self.query_weight * q + doc_weight * d for q, d in zip(query_emb, doc_emb)]

        return WeightedQueryEmbedding(
            combined_embedding=combined,
            query_embedding=query_emb,
            pseudo_doc_embedding=doc_emb,
            query_weight=self.query_weight,
        )


# ============================================================================
# Query2Doc Retriever
# ============================================================================


@dataclass
class Query2DocResult:
    """Result from Query2Doc retrieval."""

    query: str
    expanded_query: str
    pseudo_doc: str
    documents: list[RetrievedDocument]
    expansion_strategy: ExpansionStrategy


class Query2DocRetriever:
    """
    Query2Doc enhanced retriever.

    Generates pseudo-documents to expand queries before retrieval.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol,
        config: Query2DocConfig | None = None,
    ):
        """
        Initialize Query2Doc retriever.

        Args:
            llm: LLM for pseudo-doc generation
            retriever: Base retriever
            config: Configuration options
        """
        self.llm = llm
        self.retriever = retriever
        self.config = config or Query2DocConfig()

        # Initialize generator
        self.generator = LLMPseudoDocGenerator(
            llm,
            style=self.config.pseudo_doc_style,
            max_tokens=self.config.pseudo_doc_length,
            temperature=self.config.temperature,
        )

        # Initialize expander
        self.expander = Query2DocExpander(
            self.generator,
            strategy=self.config.expansion_strategy,
        )

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> Query2DocResult:
        """
        Retrieve documents using Query2Doc expansion.

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Query2DocResult with expanded query and documents
        """
        k = top_k or self.config.top_k

        # Generate pseudo-document
        pseudo_doc = await self.generator.generate(query)

        # Expand query
        expanded_query = await self.expander.expand(query)

        # Retrieve with expanded query
        if self.config.expansion_strategy == ExpansionStrategy.SEPARATE:
            # Separate retrieval for query and pseudo-doc
            docs = await self._separate_retrieval(query, pseudo_doc, k)
        else:
            docs = await self.retriever.retrieve(expanded_query, top_k=k)

        return Query2DocResult(
            query=query,
            expanded_query=expanded_query,
            pseudo_doc=pseudo_doc,
            documents=docs,
            expansion_strategy=self.config.expansion_strategy,
        )

    async def _separate_retrieval(
        self,
        query: str,
        pseudo_doc: str,
        top_k: int,
    ) -> list[RetrievedDocument]:
        """Retrieve separately and merge results."""
        # Retrieve for original query
        query_docs = await self.retriever.retrieve(query, top_k=top_k)

        # Retrieve for pseudo-doc
        doc_docs = await self.retriever.retrieve(pseudo_doc, top_k=top_k)

        # Merge with deduplication (favor query results)
        seen_content: set[str] = set()
        merged = []

        for doc in query_docs:
            if doc.content not in seen_content:
                merged.append(doc)
                seen_content.add(doc.content)

        for doc in doc_docs:
            if doc.content not in seen_content:
                # Apply discount to pseudo-doc retrievals
                doc.score *= 0.8
                merged.append(doc)
                seen_content.add(doc.content)

        # Sort by score and truncate
        merged.sort(key=lambda d: d.score, reverse=True)
        return merged[:top_k]


# ============================================================================
# HyDE-Style Query2Doc (using embeddings)
# ============================================================================


class EmbeddingQuery2Doc:
    """
    Query2Doc with embedding-based retrieval.

    Similar to HyDE but uses Query2Doc style expansion.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        embedder: EmbedderProtocol,
        retriever: RetrieverProtocol,
        query_weight: float = 0.5,
        style: PseudoDocStyle = PseudoDocStyle.PASSAGE,
    ):
        """
        Initialize embedding-based Query2Doc.

        Args:
            llm: LLM for pseudo-doc generation
            embedder: Embedding provider
            retriever: Base retriever (for vector search)
            query_weight: Weight for original query in combined embedding
            style: Style of pseudo-document
        """
        self.llm = llm
        self.embedder = embedder
        self.retriever = retriever
        self.query_weight = query_weight

        self.generator = LLMPseudoDocGenerator(llm, style)
        self.weighted_embedder = WeightedEmbedder(embedder, query_weight)

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> Query2DocResult:
        """
        Retrieve using weighted embedding combination.

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Query2DocResult
        """
        # Generate pseudo-doc
        pseudo_doc = await self.generator.generate(query)

        # Create weighted embedding (stored for potential vector retrieval)
        _ = await self.weighted_embedder.embed_expanded(query, pseudo_doc)

        # Use combined embedding for retrieval
        # Note: This requires retriever to support embedding-based search
        # For now, we fall back to text-based search with expanded query
        expanded = f"{query} {pseudo_doc}"
        docs = await self.retriever.retrieve(expanded, top_k=top_k)

        return Query2DocResult(
            query=query,
            expanded_query=expanded,
            pseudo_doc=pseudo_doc,
            documents=docs,
            expansion_strategy=ExpansionStrategy.WEIGHTED_CONCAT,
        )


# ============================================================================
# Query2Doc with Re-ranking
# ============================================================================


@dataclass
class RerankConfig:
    """Configuration for re-ranking."""

    initial_top_k: int = 50
    final_top_k: int = 10
    use_original_query: bool = True


class Query2DocReranker:
    """
    Query2Doc with two-stage retrieval and re-ranking.

    First retrieves with expanded query, then re-ranks with original.
    """

    RERANK_PROMPT = """Rate the relevance of this document to the query.

Query: {query}

Document: {document}

Rate from 1-10 (10 being most relevant):
Relevance score:"""

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol,
        config: RerankConfig | None = None,
    ):
        """Initialize with re-ranking."""
        self.llm = llm
        self.retriever = retriever
        self.config = config or RerankConfig()

        self.query2doc = Query2DocRetriever(llm, retriever)

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> Query2DocResult:
        """
        Retrieve with expansion then re-rank.

        Args:
            query: User query
            top_k: Final number of documents

        Returns:
            Query2DocResult with re-ranked documents
        """
        final_k = top_k or self.config.final_top_k

        # First stage: retrieve with Query2Doc
        result = await self.query2doc.retrieve(query, top_k=self.config.initial_top_k)

        # Second stage: re-rank with original query
        if self.config.use_original_query:
            reranked_docs = await self._rerank(query, result.documents)
        else:
            reranked_docs = result.documents

        result.documents = reranked_docs[:final_k]
        return result

    async def _rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """Re-rank documents by relevance to original query."""
        scored_docs = []

        for doc in documents:
            prompt = self.RERANK_PROMPT.format(
                query=query,
                document=doc.content[:500],
            )

            response = await self.llm.generate(prompt, max_tokens=10, temperature=0.0)

            # Parse score
            try:
                score = float("".join(c for c in response if c.isdigit() or c == "."))
                score = min(10, max(1, score)) / 10  # Normalize to 0-1
            except ValueError:
                score = 0.5

            doc.score = score
            scored_docs.append(doc)

        scored_docs.sort(key=lambda d: d.score, reverse=True)
        return scored_docs


# ============================================================================
# Iterative Query2Doc
# ============================================================================


class IterativeQuery2Doc:
    """
    Iterative Query2Doc with refinement.

    Generates pseudo-docs, retrieves, then refines based on results.
    """

    REFINEMENT_PROMPT = """Based on these retrieved documents, refine this pseudo-document to be more accurate and relevant.

Original Query: {query}
Original Pseudo-Document: {pseudo_doc}

Retrieved Documents:
{retrieved_docs}

Refined Pseudo-Document:"""

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol,
        iterations: int = 2,
    ):
        """Initialize iterative Query2Doc."""
        self.llm = llm
        self.retriever = retriever
        self.iterations = iterations
        self.generator = LLMPseudoDocGenerator(llm)

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> Query2DocResult:
        """
        Iteratively refine pseudo-doc and retrieve.

        Args:
            query: User query
            top_k: Documents to retrieve

        Returns:
            Query2DocResult after iteration
        """
        # Initial pseudo-doc
        pseudo_doc = await self.generator.generate(query)

        for _ in range(self.iterations):
            # Retrieve with current pseudo-doc
            expanded = f"{query}\n\n{pseudo_doc}"
            docs = await self.retriever.retrieve(expanded, top_k=top_k)

            # Refine pseudo-doc
            pseudo_doc = await self._refine(query, pseudo_doc, docs)

        # Final retrieval
        final_expanded = f"{query}\n\n{pseudo_doc}"
        final_docs = await self.retriever.retrieve(final_expanded, top_k=top_k)

        return Query2DocResult(
            query=query,
            expanded_query=final_expanded,
            pseudo_doc=pseudo_doc,
            documents=final_docs,
            expansion_strategy=ExpansionStrategy.PREPEND,
        )

    async def _refine(
        self,
        query: str,
        pseudo_doc: str,
        docs: list[RetrievedDocument],
    ) -> str:
        """Refine pseudo-doc based on retrieved documents."""
        retrieved_text = "\n\n".join(
            f"[{i}] {doc.content[:200]}" for i, doc in enumerate(docs[:5], 1)
        )

        prompt = self.REFINEMENT_PROMPT.format(
            query=query,
            pseudo_doc=pseudo_doc,
            retrieved_docs=retrieved_text,
        )

        refined = await self.llm.generate(prompt, max_tokens=200, temperature=0.5)
        return refined.strip()


# ============================================================================
# Factory Functions
# ============================================================================


def create_query2doc_retriever(
    llm: LLMProtocol,
    retriever: RetrieverProtocol,
    style: PseudoDocStyle = PseudoDocStyle.PASSAGE,
    strategy: ExpansionStrategy = ExpansionStrategy.PREPEND,
) -> Query2DocRetriever:
    """
    Create Query2Doc retriever.

    Args:
        llm: LLM provider
        retriever: Base retriever
        style: Pseudo-document style
        strategy: Query expansion strategy

    Returns:
        Configured Query2DocRetriever
    """
    config = Query2DocConfig(
        pseudo_doc_style=style,
        expansion_strategy=strategy,
    )
    return Query2DocRetriever(llm, retriever, config)


def create_iterative_query2doc(
    llm: LLMProtocol,
    retriever: RetrieverProtocol,
    iterations: int = 2,
) -> IterativeQuery2Doc:
    """
    Create iterative Query2Doc retriever.

    Args:
        llm: LLM provider
        retriever: Base retriever
        iterations: Number of refinement iterations

    Returns:
        Configured IterativeQuery2Doc
    """
    return IterativeQuery2Doc(llm, retriever, iterations)


def create_query2doc_reranker(
    llm: LLMProtocol,
    retriever: RetrieverProtocol,
    initial_k: int = 50,
    final_k: int = 10,
) -> Query2DocReranker:
    """
    Create Query2Doc with re-ranking.

    Args:
        llm: LLM provider
        retriever: Base retriever
        initial_k: Initial retrieval count
        final_k: Final count after re-ranking

    Returns:
        Configured Query2DocReranker
    """
    config = RerankConfig(initial_top_k=initial_k, final_top_k=final_k)
    return Query2DocReranker(llm, retriever, config)


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example demonstrating Query2Doc."""
    print("Query2Doc Implementation Ready")
    print("=" * 50)
    print("\nFeatures:")
    print("- Multiple pseudo-document styles (passage, QA, definition, etc.)")
    print("- Multiple expansion strategies (prepend, append, weighted)")
    print("- Multi-style pseudo-doc generation")
    print("- Weighted embedding combination")
    print("- Two-stage retrieval with re-ranking")
    print("- Iterative refinement")
    print("\nReference: Wang et al., 2023 - Query2Doc")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
