"""
Contextual Retrieval Implementation.

Based on: Anthropic's "Contextual Retrieval" (2024)

Contextual Retrieval improves chunk retrieval by:
1. Prepending document-level context to each chunk before embedding
2. Using LLM to generate "situating context" explaining chunk's role
3. Achieving up to 49% improvement in retrieval accuracy

This addresses the problem of chunks losing context when split from documents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# ============================================================================
# Protocols and Types
# ============================================================================


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM providers."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Embed text into vector."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        ...


@dataclass
class Document:
    """A source document."""

    content: str
    doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    title: str = ""


@dataclass
class Chunk:
    """A chunk of a document."""

    content: str
    chunk_id: str
    doc_id: str
    start_idx: int
    end_idx: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextualChunk:
    """A chunk with added contextual information."""

    chunk: Chunk
    context: str
    contextualized_content: str
    embedding: list[float] | None = None


# ============================================================================
# Context Generation Strategies
# ============================================================================


class ContextStrategy(Enum):
    """Strategies for generating chunk context."""

    DOCUMENT_SUMMARY = "document_summary"  # Use document summary
    SITUATING = "situating"  # LLM-generated situating context
    HIERARCHICAL = "hierarchical"  # Section/subsection context
    NEIGHBOR = "neighbor"  # Include neighboring chunks
    HYBRID = "hybrid"  # Combine multiple strategies


@dataclass
class ContextConfig:
    """Configuration for context generation."""

    strategy: ContextStrategy = ContextStrategy.SITUATING
    max_context_length: int = 200
    include_title: bool = True
    include_section: bool = True
    neighbor_chunks: int = 0  # Number of neighbor chunks to include
    cache_contexts: bool = True


# ============================================================================
# Context Generators
# ============================================================================


class ContextGenerator(ABC):
    """Abstract base for context generation."""

    @abstractmethod
    async def generate_context(
        self,
        chunk: Chunk,
        document: Document,
    ) -> str:
        """Generate context for a chunk."""
        ...


class SituatingContextGenerator(ContextGenerator):
    """
    LLM-based situating context generator.

    Generates context that explains what the chunk is about
    within the larger document.
    """

    PROMPT_TEMPLATE = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

    def __init__(
        self,
        llm: LLMProtocol,
        max_doc_length: int = 8000,
    ):
        """Initialize with LLM provider."""
        self.llm = llm
        self.max_doc_length = max_doc_length
        self._cache: dict[str, str] = {}

    async def generate_context(
        self,
        chunk: Chunk,
        document: Document,
    ) -> str:
        """Generate situating context using LLM."""
        cache_key = f"{document.doc_id}:{chunk.chunk_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Truncate document if needed
        doc_content = document.content
        if len(doc_content) > self.max_doc_length:
            # Keep beginning and end, include chunk location
            start = doc_content[: self.max_doc_length // 3]
            end = doc_content[-self.max_doc_length // 3 :]

            # Include context around the chunk
            chunk_start = max(0, chunk.start_idx - 500)
            chunk_end = min(len(doc_content), chunk.end_idx + 500)
            middle = doc_content[chunk_start:chunk_end]

            doc_content = f"{start}\n...\n{middle}\n...\n{end}"

        prompt = self.PROMPT_TEMPLATE.format(
            document=doc_content,
            chunk=chunk.content,
        )

        context = await self.llm.generate(
            prompt,
            max_tokens=150,
            temperature=0.0,
        )

        self._cache[cache_key] = context.strip()
        return self._cache[cache_key]


class DocumentSummaryContextGenerator(ContextGenerator):
    """Generate context using document summary."""

    SUMMARY_PROMPT = """Summarize the following document in 2-3 sentences,
focusing on the main topics and key information.

Document:
{document}

Summary:"""

    def __init__(
        self,
        llm: LLMProtocol,
    ):
        """Initialize with LLM provider."""
        self.llm = llm
        self._summary_cache: dict[str, str] = {}

    async def generate_context(
        self,
        chunk: Chunk,
        document: Document,
    ) -> str:
        """Generate context from document summary."""
        if document.doc_id not in self._summary_cache:
            summary = await self._generate_summary(document)
            self._summary_cache[document.doc_id] = summary

        summary = self._summary_cache[document.doc_id]

        # Build context
        context_parts = []
        if document.title:
            context_parts.append(f"From: {document.title}")
        context_parts.append(f"Document summary: {summary}")

        return " ".join(context_parts)

    async def _generate_summary(self, document: Document) -> str:
        """Generate document summary."""
        prompt = self.SUMMARY_PROMPT.format(
            document=document.content[:4000],
        )

        summary = await self.llm.generate(
            prompt,
            max_tokens=100,
            temperature=0.0,
        )
        return summary.strip()


class HierarchicalContextGenerator(ContextGenerator):
    """Generate context using document hierarchy (sections, subsections)."""

    def __init__(self):
        """Initialize hierarchical generator."""
        self._section_cache: dict[str, dict[int, str]] = {}

    async def generate_context(
        self,
        chunk: Chunk,
        document: Document,
    ) -> str:
        """Generate hierarchical context."""
        # Extract section information
        sections = self._extract_sections(document)

        # Find section containing this chunk
        chunk_section = self._find_section(chunk, sections)

        context_parts = []
        if document.title:
            context_parts.append(f"Document: {document.title}")
        if chunk_section:
            context_parts.append(f"Section: {chunk_section}")

        return " | ".join(context_parts) if context_parts else ""

    def _extract_sections(
        self,
        document: Document,
    ) -> list[tuple[int, str]]:
        """Extract section headers and their positions."""
        import re

        sections = []
        # Match markdown headers
        pattern = r"^(#{1,6})\s+(.+)$"

        for i, line in enumerate(document.content.split("\n")):
            match = re.match(pattern, line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                # Approximate character position
                pos = sum(len(line) + 1 for line in document.content.split("\n")[:i])
                sections.append((pos, f"{'  ' * (level - 1)}{title}"))

        return sections

    def _find_section(
        self,
        chunk: Chunk,
        sections: list[tuple[int, str]],
    ) -> str:
        """Find the section containing a chunk."""
        current_section = ""
        for pos, title in sections:
            if pos <= chunk.start_idx:
                current_section = title
            else:
                break
        return current_section


class NeighborContextGenerator(ContextGenerator):
    """Generate context using neighboring chunks."""

    def __init__(self, num_neighbors: int = 1):
        """Initialize with number of neighbors to include."""
        self.num_neighbors = num_neighbors

    async def generate_context(
        self,
        chunk: Chunk,
        document: Document,
        all_chunks: list[Chunk] | None = None,
    ) -> str:
        """Generate context from neighboring chunks."""
        if all_chunks is None:
            return ""

        # Find chunk index
        chunk_idx = None
        for i, c in enumerate(all_chunks):
            if c.chunk_id == chunk.chunk_id:
                chunk_idx = i
                break

        if chunk_idx is None:
            return ""

        context_parts = []

        # Add previous chunks
        for i in range(max(0, chunk_idx - self.num_neighbors), chunk_idx):
            prev_chunk = all_chunks[i]
            context_parts.append(f"[Previous: {prev_chunk.content[:100]}...]")

        # Add next chunks
        for i in range(chunk_idx + 1, min(len(all_chunks), chunk_idx + 1 + self.num_neighbors)):
            next_chunk = all_chunks[i]
            context_parts.append(f"[Next: {next_chunk.content[:100]}...]")

        return " ".join(context_parts)


class HybridContextGenerator(ContextGenerator):
    """Combine multiple context generation strategies."""

    def __init__(
        self,
        llm: LLMProtocol,
        strategies: list[ContextStrategy] | None = None,
    ):
        """Initialize with multiple strategies."""
        self.strategies = strategies or [
            ContextStrategy.SITUATING,
            ContextStrategy.HIERARCHICAL,
        ]

        self.generators: dict[ContextStrategy, ContextGenerator] = {
            ContextStrategy.SITUATING: SituatingContextGenerator(llm),
            ContextStrategy.DOCUMENT_SUMMARY: DocumentSummaryContextGenerator(llm),
            ContextStrategy.HIERARCHICAL: HierarchicalContextGenerator(),
            ContextStrategy.NEIGHBOR: NeighborContextGenerator(),
        }

    async def generate_context(
        self,
        chunk: Chunk,
        document: Document,
    ) -> str:
        """Generate context using multiple strategies."""
        context_parts = []

        for strategy in self.strategies:
            if strategy in self.generators:
                generator = self.generators[strategy]
                context = await generator.generate_context(chunk, document)
                if context:
                    context_parts.append(context)

        return " | ".join(context_parts)


# ============================================================================
# Contextual Chunker
# ============================================================================


class ContextualChunker:
    """
    Creates contextualized chunks from documents.

    Adds situating context to each chunk for improved retrieval.
    """

    def __init__(
        self,
        context_generator: ContextGenerator,
        config: ContextConfig | None = None,
    ):
        """
        Initialize contextual chunker.

        Args:
            context_generator: Strategy for generating context
            config: Chunking configuration
        """
        self.context_generator = context_generator
        self.config = config or ContextConfig()

    async def create_contextual_chunks(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> list[ContextualChunk]:
        """
        Create contextualized versions of chunks.

        Args:
            document: Source document
            chunks: Pre-split chunks

        Returns:
            List of contextual chunks
        """
        contextual_chunks = []

        for chunk in chunks:
            context = await self.context_generator.generate_context(chunk, document)

            # Build contextualized content
            contextualized = self._build_contextualized_content(chunk, document, context)

            contextual_chunks.append(
                ContextualChunk(
                    chunk=chunk,
                    context=context,
                    contextualized_content=contextualized,
                )
            )

        return contextual_chunks

    def _build_contextualized_content(
        self,
        chunk: Chunk,
        document: Document,
        context: str,
    ) -> str:
        """Build the final contextualized content."""
        parts = []

        if self.config.include_title and document.title:
            parts.append(f"[Document: {document.title}]")

        if context:
            parts.append(f"[Context: {context}]")

        parts.append(chunk.content)

        return "\n".join(parts)


# ============================================================================
# Contextual Embedder
# ============================================================================


class ContextualEmbedder:
    """
    Embeds chunks with their contextual information.

    Can embed both contextualized and original versions for hybrid search.
    """

    def __init__(
        self,
        embedder: EmbeddingProtocol,
        embed_original: bool = False,
    ):
        """
        Initialize contextual embedder.

        Args:
            embedder: Base embedding provider
            embed_original: Also embed original content
        """
        self.embedder = embedder
        self.embed_original = embed_original

    async def embed_contextual_chunks(
        self,
        chunks: list[ContextualChunk],
    ) -> list[ContextualChunk]:
        """
        Embed contextualized chunks.

        Args:
            chunks: Contextual chunks to embed

        Returns:
            Chunks with embeddings
        """
        # Extract texts to embed
        texts = [chunk.contextualized_content for chunk in chunks]

        # Batch embed
        embeddings = await self.embedder.embed_batch(texts)

        # Assign embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        return chunks

    async def embed_hybrid(
        self,
        chunks: list[ContextualChunk],
    ) -> list[tuple[ContextualChunk, list[float]]]:
        """
        Embed both contextualized and original for hybrid search.

        Returns:
            List of (chunk, original_embedding) tuples
        """
        # Embed contextualized
        chunks = await self.embed_contextual_chunks(chunks)

        # Embed original if requested
        if self.embed_original:
            original_texts = [chunk.chunk.content for chunk in chunks]
            original_embeddings = await self.embedder.embed_batch(original_texts)
            return list(zip(chunks, original_embeddings))

        return [(chunk, []) for chunk in chunks]


# ============================================================================
# Contextual Retrieval Pipeline
# ============================================================================


@dataclass
class ContextualRetrievalResult:
    """Result from contextual retrieval."""

    chunks: list[ContextualChunk]
    scores: list[float]
    query: str
    method: str = "contextual"


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector stores."""

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search by vector, return (id, score) pairs."""
        ...

    async def add(
        self,
        id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Add vector to store."""
        ...


class ContextualRetriever:
    """
    Contextual Retrieval system.

    Full pipeline from document processing to retrieval.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        embedder: EmbeddingProtocol,
        vector_store: VectorStoreProtocol | None = None,
        config: ContextConfig | None = None,
    ):
        """
        Initialize contextual retriever.

        Args:
            llm: LLM for context generation
            embedder: Embedding provider
            vector_store: Vector store for retrieval
            config: Configuration
        """
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config or ContextConfig()

        # Initialize components based on strategy
        self.context_generator = self._create_generator()
        self.chunker = ContextualChunker(self.context_generator, self.config)
        self.contextual_embedder = ContextualEmbedder(embedder)

        # Storage for chunks
        self._chunk_store: dict[str, ContextualChunk] = {}

    def _create_generator(self) -> ContextGenerator:
        """Create context generator based on strategy."""
        if self.config.strategy == ContextStrategy.SITUATING:
            return SituatingContextGenerator(self.llm)
        elif self.config.strategy == ContextStrategy.DOCUMENT_SUMMARY:
            return DocumentSummaryContextGenerator(self.llm)
        elif self.config.strategy == ContextStrategy.HIERARCHICAL:
            return HierarchicalContextGenerator()
        elif self.config.strategy == ContextStrategy.NEIGHBOR:
            return NeighborContextGenerator(self.config.neighbor_chunks)
        elif self.config.strategy == ContextStrategy.HYBRID:
            return HybridContextGenerator(self.llm)
        else:
            return SituatingContextGenerator(self.llm)

    async def index_document(
        self,
        document: Document,
        chunks: list[Chunk],
    ) -> list[ContextualChunk]:
        """
        Index a document with contextual chunks.

        Args:
            document: Source document
            chunks: Pre-split chunks

        Returns:
            Indexed contextual chunks
        """
        # Create contextual chunks
        contextual_chunks = await self.chunker.create_contextual_chunks(document, chunks)

        # Embed
        contextual_chunks = await self.contextual_embedder.embed_contextual_chunks(
            contextual_chunks
        )

        # Store in vector store
        for chunk in contextual_chunks:
            if self.vector_store and chunk.embedding:
                await self.vector_store.add(
                    id=chunk.chunk.chunk_id,
                    vector=chunk.embedding,
                    metadata={
                        "doc_id": document.doc_id,
                        "content": chunk.chunk.content,
                        "context": chunk.context,
                        "contextualized": chunk.contextualized_content,
                    },
                )
            self._chunk_store[chunk.chunk.chunk_id] = chunk

        return contextual_chunks

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> ContextualRetrievalResult:
        """
        Retrieve relevant contextual chunks.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            ContextualRetrievalResult with relevant chunks
        """
        # Embed query
        query_embedding = await self.embedder.embed(query)

        if self.vector_store:
            # Search vector store
            results = await self.vector_store.search(query_embedding, top_k)

            chunks = []
            scores = []
            for chunk_id, score in results:
                if chunk_id in self._chunk_store:
                    chunks.append(self._chunk_store[chunk_id])
                    scores.append(score)

            return ContextualRetrievalResult(
                chunks=chunks,
                scores=scores,
                query=query,
            )

        # Fallback: in-memory search
        return await self._in_memory_search(query_embedding, top_k, query)

    async def _in_memory_search(
        self,
        query_embedding: list[float],
        top_k: int,
        query: str,
    ) -> ContextualRetrievalResult:
        """In-memory cosine similarity search."""
        import math

        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        scored_chunks = []
        for chunk in self._chunk_store.values():
            if chunk.embedding:
                score = cosine_similarity(query_embedding, chunk.embedding)
                scored_chunks.append((chunk, score))

        # Sort by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_chunks[:top_k]

        return ContextualRetrievalResult(
            chunks=[c for c, _ in top_results],
            scores=[s for _, s in top_results],
            query=query,
        )


# ============================================================================
# Contextual BM25 Hybrid
# ============================================================================


class ContextualBM25Hybrid:
    """
    Hybrid retrieval combining contextual embeddings with BM25.

    As recommended by Anthropic for best results.
    """

    def __init__(
        self,
        contextual_retriever: ContextualRetriever,
        bm25_weight: float = 0.3,
        contextual_weight: float = 0.7,
    ):
        """
        Initialize hybrid retriever.

        Args:
            contextual_retriever: Contextual retrieval system
            bm25_weight: Weight for BM25 scores
            contextual_weight: Weight for contextual scores
        """
        self.contextual_retriever = contextual_retriever
        self.bm25_weight = bm25_weight
        self.contextual_weight = contextual_weight

        # BM25 index (simple implementation)
        self._bm25_index: dict[str, dict[str, float]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._avg_doc_length: float = 0.0

    def index_for_bm25(self, chunk_id: str, content: str):
        """Index content for BM25."""
        tokens = self._tokenize(content)
        self._doc_lengths[chunk_id] = len(tokens)

        # Update term frequencies
        tf: dict[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        self._bm25_index[chunk_id] = tf

        # Update average document length
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths.values()) / len(self._doc_lengths)

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> ContextualRetrievalResult:
        """Retrieve using hybrid scoring."""
        # Get contextual results
        contextual_result = await self.contextual_retriever.retrieve(query, top_k * 2)

        # Get BM25 scores
        bm25_scores = self._bm25_score(query)

        # Combine scores
        combined_scores: dict[str, float] = {}

        for chunk, score in zip(contextual_result.chunks, contextual_result.scores):
            chunk_id = chunk.chunk.chunk_id
            contextual_score = score * self.contextual_weight
            bm25_score = bm25_scores.get(chunk_id, 0) * self.bm25_weight
            combined_scores[chunk_id] = contextual_score + bm25_score

        # Sort by combined score
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True,
        )[:top_k]

        # Build result
        chunks = []
        scores = []
        for chunk_id in sorted_ids:
            for c in contextual_result.chunks:
                if c.chunk.chunk_id == chunk_id:
                    chunks.append(c)
                    scores.append(combined_scores[chunk_id])
                    break

        return ContextualRetrievalResult(
            chunks=chunks,
            scores=scores,
            query=query,
            method="contextual_bm25_hybrid",
        )

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        import re

        return re.findall(r"\w+", text.lower())

    def _bm25_score(
        self,
        query: str,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> dict[str, float]:
        """Calculate BM25 scores for query."""
        query_tokens = self._tokenize(query)
        scores: dict[str, float] = {}

        n_docs = len(self._bm25_index)
        if n_docs == 0:
            return scores

        for chunk_id, tf in self._bm25_index.items():
            score = 0.0
            doc_len = self._doc_lengths.get(chunk_id, 0)

            for token in query_tokens:
                if token not in tf:
                    continue

                # Term frequency in document
                term_freq = tf[token]

                # Document frequency (number of docs containing term)
                df = sum(1 for tfs in self._bm25_index.values() if token in tfs)

                # IDF
                idf = max(0, (n_docs - df + 0.5) / (df + 0.5))
                import math

                idf = math.log(1 + idf)

                # TF normalization
                tf_norm = term_freq / (
                    term_freq + k1 * (1 - b + b * doc_len / max(1, self._avg_doc_length))
                )

                score += idf * tf_norm

            scores[chunk_id] = score

        return scores


# ============================================================================
# Factory Functions
# ============================================================================


def create_contextual_retriever(
    llm: LLMProtocol,
    embedder: EmbeddingProtocol,
    strategy: ContextStrategy = ContextStrategy.SITUATING,
    vector_store: VectorStoreProtocol | None = None,
) -> ContextualRetriever:
    """
    Create a contextual retriever.

    Args:
        llm: LLM for context generation
        embedder: Embedding provider
        strategy: Context generation strategy
        vector_store: Optional vector store

    Returns:
        Configured ContextualRetriever
    """
    config = ContextConfig(strategy=strategy)
    return ContextualRetriever(llm, embedder, vector_store, config)


def create_hybrid_contextual_retriever(
    llm: LLMProtocol,
    embedder: EmbeddingProtocol,
    bm25_weight: float = 0.3,
) -> ContextualBM25Hybrid:
    """
    Create a hybrid contextual + BM25 retriever.

    Args:
        llm: LLM for context generation
        embedder: Embedding provider
        bm25_weight: Weight for BM25 (contextual gets 1 - bm25_weight)

    Returns:
        Configured ContextualBM25Hybrid
    """
    contextual = create_contextual_retriever(llm, embedder)
    return ContextualBM25Hybrid(
        contextual_retriever=contextual,
        bm25_weight=bm25_weight,
        contextual_weight=1 - bm25_weight,
    )


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example demonstrating contextual retrieval."""
    # This would use actual implementations
    # llm = OpenAILLM(api_key="...")
    # embedder = OpenAIEmbeddings(api_key="...")

    # Create contextual retriever
    # retriever = create_contextual_retriever(llm, embedder)

    # Index a document
    # doc = Document(content="...", doc_id="doc1", title="My Document")
    # chunks = [Chunk(content="...", chunk_id="c1", ...)]
    # await retriever.index_document(doc, chunks)

    # Retrieve
    # result = await retriever.retrieve("What is...?")

    print("Contextual Retrieval Implementation Ready")
    print("=" * 50)
    print("Features:")
    print("- Situating context generation (LLM-based)")
    print("- Document summary context")
    print("- Hierarchical section context")
    print("- Neighbor chunk context")
    print("- Hybrid multi-strategy context")
    print("- Contextual + BM25 hybrid retrieval")
    print("- Up to 49% retrieval improvement (per Anthropic)")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
