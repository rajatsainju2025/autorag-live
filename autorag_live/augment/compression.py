"""
Context Compression Module.

Implements LLM-based context compression and distillation to optimize
retrieved context before generation.

Key Features:
1. Extractive compression (select important passages)
2. Abstractive compression (summarize context)
3. Query-focused compression
4. Hierarchical summarization
5. Information density scoring

Example:
    >>> compressor = ContextCompressor(llm)
    >>> compressed = await compressor.compress(docs, query)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from autorag_live.core.protocols import BaseLLM, Document, Message
from autorag_live.utils.tokenizer import TokenCounter

logger = logging.getLogger(__name__)

# Module-level shared token counter (uses tiktoken when available)
_token_counter = TokenCounter()


# =============================================================================
# Compression Strategies
# =============================================================================


class CompressionStrategy(str, Enum):
    """Available compression strategies."""

    EXTRACTIVE = "extractive"  # Select important sentences
    ABSTRACTIVE = "abstractive"  # Summarize content
    QUERY_FOCUSED = "query_focused"  # Focus on query-relevant content
    HIERARCHICAL = "hierarchical"  # Multi-level summarization
    HYBRID = "hybrid"  # Combine extractive + abstractive


@dataclass
class CompressionConfig:
    """
    Configuration for context compression.

    Attributes:
        strategy: Compression strategy to use
        max_tokens: Maximum output tokens
        compression_ratio: Target compression ratio (0-1)
        preserve_structure: Maintain document structure
        query_focus_weight: Weight for query relevance (0-1)
        min_sentence_score: Minimum score to include sentence
        chunk_overlap: Overlap between chunks for hierarchical
    """

    strategy: CompressionStrategy = CompressionStrategy.QUERY_FOCUSED
    max_tokens: int = 2000
    compression_ratio: float = 0.5
    preserve_structure: bool = True
    query_focus_weight: float = 0.7
    min_sentence_score: float = 0.3
    chunk_overlap: int = 50


@dataclass
class SentenceInfo:
    """
    Information about a sentence.

    Attributes:
        text: Sentence text
        position: Position in original document
        doc_id: Source document ID
        score: Importance score
        query_relevance: Query relevance score
        information_density: Information density score
    """

    text: str
    position: int
    doc_id: str
    score: float = 0.5
    query_relevance: float = 0.5
    information_density: float = 0.5

    @property
    def combined_score(self) -> float:
        """Compute combined importance score."""
        return self.score * 0.4 + self.query_relevance * 0.4 + self.information_density * 0.2


@dataclass
class CompressionResult:
    """
    Result of context compression.

    Attributes:
        compressed_text: Compressed context
        compressed_documents: Compressed document objects
        original_tokens: Original token count estimate
        compressed_tokens: Compressed token count estimate
        compression_ratio: Actual compression ratio achieved
        preserved_info: Information preserved (0-1 estimate)
        strategy_used: Strategy that was applied
        metadata: Additional metadata
    """

    compressed_text: str
    compressed_documents: List[Document] = field(default_factory=list)
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.5
    preserved_info: float = 0.8
    strategy_used: CompressionStrategy = CompressionStrategy.EXTRACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Extractive Compressor
# =============================================================================


class ExtractiveCompressor:
    """
    Extracts important sentences from documents.

    Uses statistical and semantic signals to select key sentences.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        config: Optional[CompressionConfig] = None,
    ):
        """Initialize compressor."""
        self.llm = llm
        self.config = config or CompressionConfig()

    def extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def estimate_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (exact) or calibrated heuristic."""
        return _token_counter.count_tokens(text)

    def score_sentence_position(
        self,
        position: int,
        total: int,
    ) -> float:
        """Score based on position (first/last sentences often important)."""
        if total <= 1:
            return 1.0

        # First and last sentences get higher scores
        relative_pos = position / (total - 1) if total > 1 else 0
        if relative_pos < 0.2 or relative_pos > 0.8:
            return 0.8
        return 0.5

    def score_sentence_length(self, sentence: str) -> float:
        """Score based on length (very short/long less useful)."""
        length = len(sentence.split())
        if length < 5:
            return 0.3  # Too short
        elif length > 50:
            return 0.5  # Too long
        elif 10 <= length <= 25:
            return 0.9  # Ideal length
        return 0.7

    def score_information_density(self, sentence: str) -> float:
        """
        Estimate information density.

        Higher score for sentences with:
        - Named entities (capitalized words)
        - Numbers and dates
        - Technical terms
        """
        score = 0.5

        # Check for named entities (simple heuristic)
        caps = re.findall(r"\b[A-Z][a-z]+\b", sentence)
        if caps:
            score += min(len(caps) * 0.1, 0.2)

        # Check for numbers
        numbers = re.findall(r"\b\d+[\d,.]*\b", sentence)
        if numbers:
            score += min(len(numbers) * 0.05, 0.15)

        # Check for technical indicators
        if re.search(r"\b(e\.g\.|i\.e\.|etc\.|vs\.|namely|specifically)\b", sentence.lower()):
            score += 0.1

        return min(score, 1.0)

    def score_query_relevance(
        self,
        sentence: str,
        query: str,
    ) -> float:
        """Score based on query term overlap."""
        query_terms = set(query.lower().split())
        sentence_terms = set(sentence.lower().split())

        if not query_terms:
            return 0.5

        overlap = query_terms & sentence_terms
        return min(len(overlap) / len(query_terms), 1.0)

    def analyze_sentences(
        self,
        documents: List[Document],
        query: str,
    ) -> List[SentenceInfo]:
        """
        Analyze all sentences from documents.

        Args:
            documents: Source documents
            query: User query

        Returns:
            List of scored sentences
        """
        all_sentences = []

        for doc in documents:
            sentences = self.extract_sentences(doc.content)
            total = len(sentences)

            for i, sent in enumerate(sentences):
                # Compute scores
                pos_score = self.score_sentence_position(i, total)
                len_score = self.score_sentence_length(sent)
                info_score = self.score_information_density(sent)
                query_score = self.score_query_relevance(sent, query)

                # Combined base score
                base_score = (pos_score + len_score + info_score) / 3

                sent_info = SentenceInfo(
                    text=sent,
                    position=i,
                    doc_id=doc.id,
                    score=base_score,
                    query_relevance=query_score,
                    information_density=info_score,
                )
                all_sentences.append(sent_info)

        return all_sentences

    def select_sentences(
        self,
        sentences: List[SentenceInfo],
        max_tokens: int,
    ) -> List[SentenceInfo]:
        """
        Select top sentences within token budget.

        Args:
            sentences: Scored sentences
            max_tokens: Token budget

        Returns:
            Selected sentences
        """
        # Sort by combined score
        sorted_sents = sorted(sentences, key=lambda s: s.combined_score, reverse=True)

        selected = []
        token_count = 0

        for sent in sorted_sents:
            sent_tokens = self.estimate_tokens(sent.text)
            if token_count + sent_tokens <= max_tokens:
                if sent.combined_score >= self.config.min_sentence_score:
                    selected.append(sent)
                    token_count += sent_tokens

        # Re-order by original position for coherence
        selected.sort(key=lambda s: (s.doc_id, s.position))

        return selected

    def compress(
        self,
        documents: List[Document],
        query: str,
    ) -> CompressionResult:
        """
        Perform extractive compression.

        Args:
            documents: Source documents
            query: User query

        Returns:
            CompressionResult
        """
        # Analyze sentences
        sentences = self.analyze_sentences(documents, query)

        # Compute original tokens
        original_text = " ".join(doc.content for doc in documents)
        original_tokens = self.estimate_tokens(original_text)

        # Compute target tokens
        target_tokens = min(
            self.config.max_tokens,
            int(original_tokens * self.config.compression_ratio),
        )

        # Select sentences
        selected = self.select_sentences(sentences, target_tokens)

        # Build compressed text
        compressed_parts = []
        current_doc = None

        for sent in selected:
            if self.config.preserve_structure and sent.doc_id != current_doc:
                if current_doc is not None:
                    compressed_parts.append("\n---\n")
                current_doc = sent.doc_id
            compressed_parts.append(sent.text)

        compressed_text = " ".join(compressed_parts)
        compressed_tokens = self.estimate_tokens(compressed_text)

        # Build compressed documents
        compressed_docs = self._build_compressed_docs(selected, documents)

        return CompressionResult(
            compressed_text=compressed_text,
            compressed_documents=compressed_docs,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            preserved_info=len(selected) / len(sentences) if sentences else 1.0,
            strategy_used=CompressionStrategy.EXTRACTIVE,
            metadata={
                "sentences_selected": len(selected),
                "sentences_total": len(sentences),
            },
        )

    def _build_compressed_docs(
        self,
        sentences: List[SentenceInfo],
        original_docs: List[Document],
    ) -> List[Document]:
        """Build compressed document objects."""
        # Group sentences by document
        doc_sentences: Dict[str, List[SentenceInfo]] = {}
        for sent in sentences:
            if sent.doc_id not in doc_sentences:
                doc_sentences[sent.doc_id] = []
            doc_sentences[sent.doc_id].append(sent)

        # Build compressed docs
        compressed_docs = []
        doc_map = {d.id: d for d in original_docs}

        for doc_id, sents in doc_sentences.items():
            orig_doc = doc_map.get(doc_id)
            if orig_doc:
                compressed_content = " ".join(s.text for s in sents)
                compressed_docs.append(
                    Document(
                        id=f"{doc_id}_compressed",
                        content=compressed_content,
                        metadata={
                            **orig_doc.metadata,
                            "compression": "extractive",
                            "original_id": doc_id,
                        },
                        score=orig_doc.score,
                    )
                )

        return compressed_docs


# =============================================================================
# Abstractive Compressor
# =============================================================================


class AbstractiveCompressor:
    """
    Compresses context using LLM-based summarization.

    Generates concise summaries that preserve key information.
    """

    SUMMARY_PROMPT = """Compress the following context into a concise summary that preserves the key information relevant to answering questions.

Context:
{context}

Instructions:
- Focus on facts, definitions, and relationships
- Remove redundant information
- Maintain accuracy - don't add information not in the context
- Target length: {target_length} words

Summary:"""

    QUERY_FOCUSED_PROMPT = """Compress the following context, focusing on information relevant to this query:

Query: {query}

Context:
{context}

Instructions:
- Extract information directly relevant to the query
- Remove tangential or unrelated content
- Preserve specific facts, numbers, and definitions
- Target length: {target_length} words

Compressed context:"""

    def __init__(
        self,
        llm: BaseLLM,
        config: Optional[CompressionConfig] = None,
    ):
        """Initialize compressor."""
        self.llm = llm
        self.config = config or CompressionConfig()

    def estimate_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (exact) or calibrated heuristic."""
        return _token_counter.count_tokens(text)

    async def compress(
        self,
        documents: List[Document],
        query: str,
    ) -> CompressionResult:
        """
        Perform abstractive compression.

        Args:
            documents: Source documents
            query: User query

        Returns:
            CompressionResult
        """
        # Combine context
        context = "\n\n".join(f"[{i+1}] {doc.content}" for i, doc in enumerate(documents))
        original_tokens = self.estimate_tokens(context)

        # Compute target length
        target_words = int(original_tokens * self.config.compression_ratio * 0.75)
        target_words = max(100, min(target_words, 500))

        # Choose prompt based on query
        if query:
            prompt = self.QUERY_FOCUSED_PROMPT.format(
                query=query,
                context=context,
                target_length=target_words,
            )
        else:
            prompt = self.SUMMARY_PROMPT.format(
                context=context,
                target_length=target_words,
            )

        # Generate summary
        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.3,
            max_tokens=self.config.max_tokens,
        )

        compressed_text = result.content.strip()
        compressed_tokens = self.estimate_tokens(compressed_text)

        # Create compressed document
        compressed_doc = Document(
            id="compressed_summary",
            content=compressed_text,
            metadata={
                "compression": "abstractive",
                "source_count": len(documents),
            },
            score=1.0,
        )

        return CompressionResult(
            compressed_text=compressed_text,
            compressed_documents=[compressed_doc],
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            preserved_info=0.85,  # Estimate
            strategy_used=CompressionStrategy.ABSTRACTIVE,
            metadata={
                "model_used": True,
                "target_words": target_words,
            },
        )


# =============================================================================
# Hierarchical Compressor
# =============================================================================


class HierarchicalCompressor:
    """
    Compresses context using hierarchical summarization.

    Creates multi-level summaries for very large contexts.
    """

    CHUNK_SUMMARY_PROMPT = """Summarize this text chunk concisely:

{chunk}

Summary (max {max_words} words):"""

    COMBINE_PROMPT = """Combine these summaries into a coherent, comprehensive summary:

{summaries}

Combined summary (max {max_words} words):"""

    def __init__(
        self,
        llm: BaseLLM,
        config: Optional[CompressionConfig] = None,
        chunk_size: int = 500,
    ):
        """Initialize compressor."""
        self.llm = llm
        self.config = config or CompressionConfig()
        self.chunk_size = chunk_size

    def estimate_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (exact) or calibrated heuristic."""
        return _token_counter.count_tokens(text)

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.config.chunk_overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    async def summarize_chunk(self, chunk: str, max_words: int) -> str:
        """Summarize a single chunk."""
        prompt = self.CHUNK_SUMMARY_PROMPT.format(
            chunk=chunk,
            max_words=max_words,
        )

        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.3,
            max_tokens=max_words * 2,
        )

        return result.content.strip()

    async def combine_summaries(self, summaries: List[str], max_words: int) -> str:
        """Combine multiple summaries."""
        numbered = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(summaries))

        prompt = self.COMBINE_PROMPT.format(
            summaries=numbered,
            max_words=max_words,
        )

        result = await self.llm.generate(
            [Message.user(prompt)],
            temperature=0.3,
            max_tokens=max_words * 2,
        )

        return result.content.strip()

    async def compress(
        self,
        documents: List[Document],
        query: str,
    ) -> CompressionResult:
        """
        Perform hierarchical compression.

        Args:
            documents: Source documents
            query: User query

        Returns:
            CompressionResult
        """
        # Combine all content
        full_text = "\n\n".join(doc.content for doc in documents)
        original_tokens = self.estimate_tokens(full_text)

        # Chunk the text
        chunks = self.chunk_text(full_text)

        if len(chunks) <= 1:
            # Small enough for single summary
            abstractive = AbstractiveCompressor(self.llm, self.config)
            return await abstractive.compress(documents, query)

        # First level: summarize each chunk
        target_chunk_words = int(self.chunk_size * self.config.compression_ratio)
        chunk_summaries = await asyncio.gather(
            *[self.summarize_chunk(chunk, target_chunk_words) for chunk in chunks]
        )

        # Combine summaries
        combined = "\n\n".join(chunk_summaries)

        # If still too large, recurse
        if self.estimate_tokens(combined) > self.config.max_tokens:
            final_words = int(self.config.max_tokens * 0.75)
            compressed_text = await self.combine_summaries(chunk_summaries, final_words)
        else:
            compressed_text = combined

        compressed_tokens = self.estimate_tokens(compressed_text)

        compressed_doc = Document(
            id="hierarchical_summary",
            content=compressed_text,
            metadata={
                "compression": "hierarchical",
                "chunks_processed": len(chunks),
                "levels": 2 if len(chunks) > 1 else 1,
            },
            score=1.0,
        )

        return CompressionResult(
            compressed_text=compressed_text,
            compressed_documents=[compressed_doc],
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            preserved_info=0.75,  # Multi-level loses some info
            strategy_used=CompressionStrategy.HIERARCHICAL,
            metadata={
                "chunks_processed": len(chunks),
            },
        )


# =============================================================================
# Unified Context Compressor
# =============================================================================


class ContextCompressor:
    """
    Unified context compressor supporting multiple strategies.

    Features:
    - Strategy selection based on context size
    - Query-focused compression
    - Hybrid extractive + abstractive

    Example:
        >>> compressor = ContextCompressor(llm)
        >>> result = await compressor.compress(documents, query)
        >>> print(f"Compressed {result.compression_ratio:.0%}")
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        config: Optional[CompressionConfig] = None,
    ):
        """
        Initialize compressor.

        Args:
            llm: LLM for abstractive/query-focused compression
            config: Compression configuration
        """
        self.llm = llm
        self.config = config or CompressionConfig()

        # Initialize sub-compressors
        self.extractive = ExtractiveCompressor(llm, self.config)
        if llm:
            self.abstractive = AbstractiveCompressor(llm, self.config)
            self.hierarchical = HierarchicalCompressor(llm, self.config)
        else:
            self.abstractive = None
            self.hierarchical = None

    def select_strategy(
        self,
        documents: List[Document],
    ) -> CompressionStrategy:
        """
        Select compression strategy based on context.

        Args:
            documents: Source documents

        Returns:
            Recommended strategy
        """
        # Estimate total tokens
        total_tokens = sum(self.extractive.estimate_tokens(doc.content) for doc in documents)

        # Use configured strategy if LLM available
        if self.llm:
            if total_tokens > 8000:
                return CompressionStrategy.HIERARCHICAL
            elif self.config.strategy != CompressionStrategy.EXTRACTIVE:
                return self.config.strategy
            return CompressionStrategy.QUERY_FOCUSED

        # Fall back to extractive without LLM
        return CompressionStrategy.EXTRACTIVE

    async def compress(
        self,
        documents: List[Document],
        query: str,
        *,
        strategy: Optional[CompressionStrategy] = None,
    ) -> CompressionResult:
        """
        Compress documents using appropriate strategy.

        Args:
            documents: Source documents
            query: User query for focus
            strategy: Override strategy selection

        Returns:
            CompressionResult
        """
        if not documents:
            return CompressionResult(
                compressed_text="",
                compressed_documents=[],
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
            )

        # Select strategy
        selected = strategy or self.select_strategy(documents)

        # Execute compression
        if selected == CompressionStrategy.EXTRACTIVE:
            return self.extractive.compress(documents, query)

        elif selected == CompressionStrategy.ABSTRACTIVE:
            if not self.abstractive:
                return self.extractive.compress(documents, query)
            return await self.abstractive.compress(documents, query)

        elif selected == CompressionStrategy.HIERARCHICAL:
            if not self.hierarchical:
                return self.extractive.compress(documents, query)
            return await self.hierarchical.compress(documents, query)

        elif selected == CompressionStrategy.QUERY_FOCUSED:
            return await self._query_focused_compress(documents, query)

        elif selected == CompressionStrategy.HYBRID:
            return await self._hybrid_compress(documents, query)

        # Default to extractive
        return self.extractive.compress(documents, query)

    async def _query_focused_compress(
        self,
        documents: List[Document],
        query: str,
    ) -> CompressionResult:
        """Query-focused compression."""
        if not self.llm:
            return self.extractive.compress(documents, query)

        # Use abstractive with query focus
        return await self.abstractive.compress(documents, query)

    async def _hybrid_compress(
        self,
        documents: List[Document],
        query: str,
    ) -> CompressionResult:
        """Hybrid extractive + abstractive compression."""
        # First pass: extractive selection
        extractive_result = self.extractive.compress(documents, query)

        if not self.llm:
            return extractive_result

        # Second pass: abstractive polish
        if extractive_result.compressed_documents:
            abstractive_result = await self.abstractive.compress(
                extractive_result.compressed_documents,
                query,
            )

            # Combine metadata
            abstractive_result.strategy_used = CompressionStrategy.HYBRID
            abstractive_result.original_tokens = extractive_result.original_tokens
            abstractive_result.metadata["extractive_sentences"] = extractive_result.metadata.get(
                "sentences_selected", 0
            )

            return abstractive_result

        return extractive_result

    def compress_sync(
        self,
        documents: List[Document],
        query: str,
    ) -> CompressionResult:
        """
        Synchronous compression (extractive only).

        Args:
            documents: Source documents
            query: User query

        Returns:
            CompressionResult
        """
        return self.extractive.compress(documents, query)


# =============================================================================
# Convenience Functions
# =============================================================================


def compress_context(
    documents: List[Document],
    query: str,
    max_tokens: int = 2000,
) -> CompressionResult:
    """
    Quick compression using extractive strategy.

    Args:
        documents: Source documents
        query: User query
        max_tokens: Token budget

    Returns:
        CompressionResult
    """
    config = CompressionConfig(
        strategy=CompressionStrategy.EXTRACTIVE,
        max_tokens=max_tokens,
    )
    compressor = ExtractiveCompressor(config=config)
    return compressor.compress(documents, query)


async def compress_with_llm(
    documents: List[Document],
    query: str,
    llm: BaseLLM,
    max_tokens: int = 2000,
) -> CompressionResult:
    """
    Compress using LLM-based strategy.

    Args:
        documents: Source documents
        query: User query
        llm: Language model
        max_tokens: Token budget

    Returns:
        CompressionResult
    """
    config = CompressionConfig(
        strategy=CompressionStrategy.QUERY_FOCUSED,
        max_tokens=max_tokens,
    )
    compressor = ContextCompressor(llm, config)
    return await compressor.compress(documents, query)
