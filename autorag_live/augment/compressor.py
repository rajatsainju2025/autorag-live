"""
Context compression for AutoRAG-Live.

Provides utilities to compress and filter retrieved context
to fit within token limits while preserving relevance.

Features:
- Token-aware compression
- Relevance-based filtering
- Extractive compression
- Sentence selection
- Context reordering

Example usage:
    >>> compressor = ContextCompressor(max_tokens=2000)
    >>> compressed = compressor.compress(contexts, query)
    >>> print(f"Reduced from {original_tokens} to {compressed_tokens}")
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CompressionStrategy(str, Enum):
    """Context compression strategies."""
    
    TRUNCATE = "truncate"
    EXTRACTIVE = "extractive"
    RELEVANCE = "relevance"
    SENTENCE = "sentence"
    HYBRID = "hybrid"


@dataclass
class ContextChunk:
    """Represents a chunk of context."""
    
    id: str
    content: str
    
    # Relevance to query
    relevance_score: float = 0.0
    
    # Token count
    token_count: int = 0
    
    # Source info
    source: Optional[str] = None
    position: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressedContext:
    """Result of context compression."""
    
    chunks: List[ContextChunk]
    
    # Statistics
    original_count: int = 0
    compressed_count: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0
    
    # Strategy used
    strategy: CompressionStrategy = CompressionStrategy.TRUNCATE
    
    @property
    def compression_ratio(self) -> float:
        """Get compression ratio."""
        if self.original_tokens == 0:
            return 0.0
        return 1 - (self.compressed_tokens / self.original_tokens)
    
    @property
    def text(self) -> str:
        """Get combined text from chunks."""
        return "\n\n".join(chunk.content for chunk in self.chunks)
    
    def to_list(self) -> List[str]:
        """Get list of chunk texts."""
        return [chunk.content for chunk in self.chunks]


class Tokenizer:
    """Simple tokenizer for token counting."""
    
    def __init__(
        self,
        chars_per_token: float = 4.0,
        tokenizer_func: Optional[Callable[[str], List[str]]] = None,
    ):
        """
        Initialize tokenizer.
        
        Args:
            chars_per_token: Average characters per token
            tokenizer_func: Optional custom tokenizer
        """
        self.chars_per_token = chars_per_token
        self.tokenizer_func = tokenizer_func
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer_func:
            return len(self.tokenizer_func(text))
        
        # Estimate based on character count
        return int(len(text) / self.chars_per_token)
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if self.tokenizer_func:
            tokens = self.tokenizer_func(text)
            if len(tokens) <= max_tokens:
                return text
            # Approximate truncation
            ratio = max_tokens / len(tokens)
            target_chars = int(len(text) * ratio)
            return text[:target_chars]
        
        # Character-based truncation
        max_chars = int(max_tokens * self.chars_per_token)
        if len(text) <= max_chars:
            return text
        
        return text[:max_chars]


class BaseCompressor(ABC):
    """Base class for context compressors."""
    
    @abstractmethod
    def compress(
        self,
        chunks: List[ContextChunk],
        query: str,
        max_tokens: int,
    ) -> List[ContextChunk]:
        """Compress context chunks."""
        pass
    
    @property
    @abstractmethod
    def strategy(self) -> CompressionStrategy:
        """Get compression strategy."""
        pass


class TruncateCompressor(BaseCompressor):
    """Compress by truncating content."""
    
    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        """Initialize truncate compressor."""
        self.tokenizer = tokenizer or Tokenizer()
    
    @property
    def strategy(self) -> CompressionStrategy:
        return CompressionStrategy.TRUNCATE
    
    def compress(
        self,
        chunks: List[ContextChunk],
        query: str,
        max_tokens: int,
    ) -> List[ContextChunk]:
        """Compress by truncating."""
        result = []
        remaining_tokens = max_tokens
        
        for chunk in chunks:
            if remaining_tokens <= 0:
                break
            
            tokens = self.tokenizer.count_tokens(chunk.content)
            
            if tokens <= remaining_tokens:
                result.append(chunk)
                remaining_tokens -= tokens
            else:
                # Truncate this chunk
                truncated = self.tokenizer.truncate_to_tokens(
                    chunk.content, remaining_tokens
                )
                truncated_chunk = ContextChunk(
                    id=chunk.id,
                    content=truncated,
                    relevance_score=chunk.relevance_score,
                    token_count=remaining_tokens,
                    source=chunk.source,
                    position=chunk.position,
                    metadata=chunk.metadata,
                )
                result.append(truncated_chunk)
                break
        
        return result


class RelevanceCompressor(BaseCompressor):
    """Compress by selecting most relevant chunks."""
    
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        min_score: float = 0.0,
    ):
        """
        Initialize relevance compressor.
        
        Args:
            tokenizer: Token counter
            min_score: Minimum relevance score to include
        """
        self.tokenizer = tokenizer or Tokenizer()
        self.min_score = min_score
    
    @property
    def strategy(self) -> CompressionStrategy:
        return CompressionStrategy.RELEVANCE
    
    def compress(
        self,
        chunks: List[ContextChunk],
        query: str,
        max_tokens: int,
    ) -> List[ContextChunk]:
        """Compress by selecting most relevant chunks."""
        # Filter by minimum score
        filtered = [c for c in chunks if c.relevance_score >= self.min_score]
        
        # Sort by relevance
        sorted_chunks = sorted(
            filtered,
            key=lambda c: c.relevance_score,
            reverse=True,
        )
        
        # Select chunks within token budget
        result = []
        total_tokens = 0
        
        for chunk in sorted_chunks:
            tokens = chunk.token_count or self.tokenizer.count_tokens(chunk.content)
            
            if total_tokens + tokens <= max_tokens:
                chunk.token_count = tokens
                result.append(chunk)
                total_tokens += tokens
        
        return result


class SentenceCompressor(BaseCompressor):
    """Compress by selecting most relevant sentences."""
    
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        similarity_func: Optional[Callable[[str, str], float]] = None,
    ):
        """
        Initialize sentence compressor.
        
        Args:
            tokenizer: Token counter
            similarity_func: Function to compute sentence-query similarity
        """
        self.tokenizer = tokenizer or Tokenizer()
        self.similarity_func = similarity_func or self._default_similarity
    
    @property
    def strategy(self) -> CompressionStrategy:
        return CompressionStrategy.SENTENCE
    
    def compress(
        self,
        chunks: List[ContextChunk],
        query: str,
        max_tokens: int,
    ) -> List[ContextChunk]:
        """Compress by selecting relevant sentences."""
        # Extract and score sentences from all chunks
        scored_sentences: List[Tuple[str, float, str]] = []
        
        for chunk in chunks:
            sentences = self._split_sentences(chunk.content)
            for sentence in sentences:
                score = self.similarity_func(query, sentence)
                scored_sentences.append((sentence, score, chunk.id))
        
        # Sort by score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Select sentences within token budget
        selected_sentences: Dict[str, List[str]] = {}
        total_tokens = 0
        
        for sentence, score, chunk_id in scored_sentences:
            tokens = self.tokenizer.count_tokens(sentence)
            
            if total_tokens + tokens <= max_tokens:
                if chunk_id not in selected_sentences:
                    selected_sentences[chunk_id] = []
                selected_sentences[chunk_id].append(sentence)
                total_tokens += tokens
        
        # Rebuild chunks with selected sentences
        result = []
        for chunk in chunks:
            if chunk.id in selected_sentences:
                content = " ".join(selected_sentences[chunk.id])
                compressed_chunk = ContextChunk(
                    id=chunk.id,
                    content=content,
                    relevance_score=chunk.relevance_score,
                    token_count=self.tokenizer.count_tokens(content),
                    source=chunk.source,
                    position=chunk.position,
                    metadata=chunk.metadata,
                )
                result.append(compressed_chunk)
        
        return result
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _default_similarity(self, query: str, sentence: str) -> float:
        """Default similarity using word overlap."""
        query_words = set(query.lower().split())
        sentence_words = set(sentence.lower().split())
        
        intersection = len(query_words & sentence_words)
        union = len(query_words | sentence_words)
        
        return intersection / union if union > 0 else 0.0


class ExtractiveCompressor(BaseCompressor):
    """Compress by extracting key content."""
    
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        keywords_weight: float = 0.3,
        position_weight: float = 0.2,
        length_weight: float = 0.1,
    ):
        """
        Initialize extractive compressor.
        
        Args:
            tokenizer: Token counter
            keywords_weight: Weight for keyword matching
            position_weight: Weight for position (earlier = better)
            length_weight: Weight for length (moderate = better)
        """
        self.tokenizer = tokenizer or Tokenizer()
        self.keywords_weight = keywords_weight
        self.position_weight = position_weight
        self.length_weight = length_weight
    
    @property
    def strategy(self) -> CompressionStrategy:
        return CompressionStrategy.EXTRACTIVE
    
    def compress(
        self,
        chunks: List[ContextChunk],
        query: str,
        max_tokens: int,
    ) -> List[ContextChunk]:
        """Compress by extracting key content."""
        # Extract keywords from query
        query_keywords = self._extract_keywords(query)
        
        # Score sentences from all chunks
        all_sentences: List[Tuple[str, float, int, str]] = []
        
        for chunk_idx, chunk in enumerate(chunks):
            sentences = self._split_sentences(chunk.content)
            
            for sent_idx, sentence in enumerate(sentences):
                score = self._score_sentence(
                    sentence,
                    query_keywords,
                    sent_idx,
                    len(sentences),
                )
                all_sentences.append((sentence, score, chunk_idx, chunk.id))
        
        # Sort by score
        all_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Select sentences
        selected: Dict[str, List[Tuple[str, int]]] = {}
        total_tokens = 0
        
        for sentence, score, chunk_idx, chunk_id in all_sentences:
            tokens = self.tokenizer.count_tokens(sentence)
            
            if total_tokens + tokens <= max_tokens:
                if chunk_id not in selected:
                    selected[chunk_id] = []
                selected[chunk_id].append((sentence, chunk_idx))
                total_tokens += tokens
        
        # Rebuild chunks preserving order
        result = []
        for chunk in chunks:
            if chunk.id in selected:
                # Sort sentences by original position
                chunk_sentences = sorted(selected[chunk.id], key=lambda x: x[1])
                content = " ".join(s for s, _ in chunk_sentences)
                
                compressed_chunk = ContextChunk(
                    id=chunk.id,
                    content=content,
                    relevance_score=chunk.relevance_score,
                    token_count=self.tokenizer.count_tokens(content),
                    source=chunk.source,
                    position=chunk.position,
                    metadata=chunk.metadata,
                )
                result.append(compressed_chunk)
        
        return result
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter stopwords
        stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does",
            "this", "that", "what", "how", "why", "when", "where", "who",
        }
        
        return {w for w in words if w not in stopwords and len(w) > 2}
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _score_sentence(
        self,
        sentence: str,
        query_keywords: Set[str],
        position: int,
        total_sentences: int,
    ) -> float:
        """Score a sentence."""
        # Keyword score
        sentence_words = set(sentence.lower().split())
        keyword_overlap = len(query_keywords & sentence_words)
        keyword_score = keyword_overlap / len(query_keywords) if query_keywords else 0
        
        # Position score (earlier sentences often more important)
        position_score = 1 - (position / total_sentences) if total_sentences > 0 else 0
        
        # Length score (prefer moderate length sentences)
        word_count = len(sentence.split())
        if word_count < 5:
            length_score = 0.3
        elif word_count > 50:
            length_score = 0.5
        else:
            length_score = 1.0
        
        return (
            self.keywords_weight * keyword_score +
            self.position_weight * position_score +
            self.length_weight * length_score
        )


class HybridCompressor(BaseCompressor):
    """Combine multiple compression strategies."""
    
    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        relevance_ratio: float = 0.7,
    ):
        """
        Initialize hybrid compressor.
        
        Args:
            tokenizer: Token counter
            relevance_ratio: Ratio of tokens for relevance-based selection
        """
        self.tokenizer = tokenizer or Tokenizer()
        self.relevance_ratio = relevance_ratio
        
        self.relevance_compressor = RelevanceCompressor(tokenizer)
        self.sentence_compressor = SentenceCompressor(tokenizer)
    
    @property
    def strategy(self) -> CompressionStrategy:
        return CompressionStrategy.HYBRID
    
    def compress(
        self,
        chunks: List[ContextChunk],
        query: str,
        max_tokens: int,
    ) -> List[ContextChunk]:
        """Compress using hybrid approach."""
        # First pass: relevance-based selection
        relevance_tokens = int(max_tokens * self.relevance_ratio)
        relevant_chunks = self.relevance_compressor.compress(
            chunks, query, relevance_tokens
        )
        
        # Calculate remaining tokens
        used_tokens = sum(
            c.token_count or self.tokenizer.count_tokens(c.content)
            for c in relevant_chunks
        )
        remaining_tokens = max_tokens - used_tokens
        
        if remaining_tokens <= 0:
            return relevant_chunks
        
        # Second pass: sentence-level compression on remaining chunks
        remaining_chunks = [c for c in chunks if c not in relevant_chunks]
        
        if remaining_chunks:
            sentence_selected = self.sentence_compressor.compress(
                remaining_chunks, query, remaining_tokens
            )
            return relevant_chunks + sentence_selected
        
        return relevant_chunks


class ContextCompressor:
    """
    Main context compression interface.
    
    Example:
        >>> compressor = ContextCompressor(max_tokens=2000)
        >>> 
        >>> # Compress context
        >>> contexts = [
        ...     {"text": "Long context...", "score": 0.9},
        ...     {"text": "Another context...", "score": 0.7},
        ... ]
        >>> compressed = compressor.compress(contexts, query="What is ML?")
        >>> 
        >>> print(f"Compression ratio: {compressed.compression_ratio:.1%}")
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        strategy: CompressionStrategy = CompressionStrategy.HYBRID,
        tokenizer: Optional[Tokenizer] = None,
        preserve_order: bool = True,
    ):
        """
        Initialize context compressor.
        
        Args:
            max_tokens: Maximum tokens in output
            strategy: Compression strategy
            tokenizer: Token counter
            preserve_order: Preserve original chunk order
        """
        self.max_tokens = max_tokens
        self.preserve_order = preserve_order
        self.tokenizer = tokenizer or Tokenizer()
        
        # Initialize compressor based on strategy
        if strategy == CompressionStrategy.TRUNCATE:
            self._compressor = TruncateCompressor(self.tokenizer)
        elif strategy == CompressionStrategy.RELEVANCE:
            self._compressor = RelevanceCompressor(self.tokenizer)
        elif strategy == CompressionStrategy.SENTENCE:
            self._compressor = SentenceCompressor(self.tokenizer)
        elif strategy == CompressionStrategy.EXTRACTIVE:
            self._compressor = ExtractiveCompressor(self.tokenizer)
        else:
            self._compressor = HybridCompressor(self.tokenizer)
    
    def compress(
        self,
        contexts: List[Dict[str, Any]],
        query: str,
        max_tokens: Optional[int] = None,
    ) -> CompressedContext:
        """
        Compress context to fit token limit.
        
        Args:
            contexts: List of context dicts with 'text' and optional 'score'
            query: Query for relevance scoring
            max_tokens: Override max tokens
            
        Returns:
            CompressedContext
        """
        max_tokens = max_tokens or self.max_tokens
        
        # Convert to ContextChunks
        chunks = []
        total_original_tokens = 0
        
        for i, ctx in enumerate(contexts):
            text = ctx.get("text", ctx.get("content", ""))
            tokens = self.tokenizer.count_tokens(text)
            total_original_tokens += tokens
            
            chunk = ContextChunk(
                id=str(i),
                content=text,
                relevance_score=ctx.get("score", ctx.get("relevance", 0.0)),
                token_count=tokens,
                source=ctx.get("source"),
                position=i,
                metadata=ctx.get("metadata", {}),
            )
            chunks.append(chunk)
        
        # Check if compression needed
        if total_original_tokens <= max_tokens:
            return CompressedContext(
                chunks=chunks,
                original_count=len(chunks),
                compressed_count=len(chunks),
                original_tokens=total_original_tokens,
                compressed_tokens=total_original_tokens,
                strategy=self._compressor.strategy,
            )
        
        # Compress
        compressed_chunks = self._compressor.compress(chunks, query, max_tokens)
        
        # Reorder if needed
        if self.preserve_order:
            compressed_chunks.sort(key=lambda c: c.position)
        
        # Calculate compressed token count
        compressed_tokens = sum(
            c.token_count or self.tokenizer.count_tokens(c.content)
            for c in compressed_chunks
        )
        
        return CompressedContext(
            chunks=compressed_chunks,
            original_count=len(chunks),
            compressed_count=len(compressed_chunks),
            original_tokens=total_original_tokens,
            compressed_tokens=compressed_tokens,
            strategy=self._compressor.strategy,
        )
    
    def fits_in_limit(
        self,
        contexts: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
    ) -> bool:
        """Check if contexts fit within token limit."""
        max_tokens = max_tokens or self.max_tokens
        
        total_tokens = sum(
            self.tokenizer.count_tokens(ctx.get("text", ctx.get("content", "")))
            for ctx in contexts
        )
        
        return total_tokens <= max_tokens
    
    def estimate_tokens(
        self,
        contexts: List[Dict[str, Any]],
    ) -> int:
        """Estimate total tokens in contexts."""
        return sum(
            self.tokenizer.count_tokens(ctx.get("text", ctx.get("content", "")))
            for ctx in contexts
        )


class ContextReorderer:
    """
    Reorder context chunks for optimal presentation.
    
    Example:
        >>> reorderer = ContextReorderer()
        >>> reordered = reorderer.reorder(chunks, query, strategy="relevance_first")
    """
    
    def __init__(self):
        """Initialize reorderer."""
        pass
    
    def reorder(
        self,
        chunks: List[ContextChunk],
        query: str,
        strategy: str = "relevance_first",
    ) -> List[ContextChunk]:
        """
        Reorder chunks.
        
        Args:
            chunks: Chunks to reorder
            query: Query for relevance
            strategy: Reordering strategy
            
        Returns:
            Reordered chunks
        """
        if strategy == "relevance_first":
            return sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
        
        elif strategy == "relevance_last":
            # Most relevant at end (for recency bias in attention)
            return sorted(chunks, key=lambda c: c.relevance_score)
        
        elif strategy == "diversity":
            return self._diversity_reorder(chunks)
        
        elif strategy == "original":
            return sorted(chunks, key=lambda c: c.position)
        
        return chunks
    
    def _diversity_reorder(
        self,
        chunks: List[ContextChunk],
    ) -> List[ContextChunk]:
        """Reorder for diversity (alternate high/low relevance)."""
        sorted_by_relevance = sorted(
            chunks,
            key=lambda c: c.relevance_score,
            reverse=True,
        )
        
        result = []
        high_idx = 0
        low_idx = len(sorted_by_relevance) - 1
        
        while high_idx <= low_idx:
            result.append(sorted_by_relevance[high_idx])
            high_idx += 1
            
            if high_idx <= low_idx:
                result.append(sorted_by_relevance[low_idx])
                low_idx -= 1
        
        return result


class ContextFilter:
    """
    Filter context chunks based on criteria.
    
    Example:
        >>> filter = ContextFilter(min_score=0.5, min_length=50)
        >>> filtered = filter.filter(chunks)
    """
    
    def __init__(
        self,
        min_score: float = 0.0,
        max_score: float = 1.0,
        min_length: int = 0,
        max_length: int = float('inf'),
        exclude_patterns: Optional[List[str]] = None,
        require_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize filter.
        
        Args:
            min_score: Minimum relevance score
            max_score: Maximum relevance score
            min_length: Minimum character length
            max_length: Maximum character length
            exclude_patterns: Regex patterns to exclude
            require_patterns: Regex patterns to require
        """
        self.min_score = min_score
        self.max_score = max_score
        self.min_length = min_length
        self.max_length = max_length
        
        self.exclude_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (exclude_patterns or [])
        ]
        self.require_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (require_patterns or [])
        ]
    
    def filter(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Filter chunks based on criteria."""
        result = []
        
        for chunk in chunks:
            if not self._passes_criteria(chunk):
                continue
            result.append(chunk)
        
        return result
    
    def _passes_criteria(self, chunk: ContextChunk) -> bool:
        """Check if chunk passes all criteria."""
        # Score filter
        if not (self.min_score <= chunk.relevance_score <= self.max_score):
            return False
        
        # Length filter
        length = len(chunk.content)
        if not (self.min_length <= length <= self.max_length):
            return False
        
        # Exclude patterns
        for pattern in self.exclude_patterns:
            if pattern.search(chunk.content):
                return False
        
        # Require patterns
        for pattern in self.require_patterns:
            if not pattern.search(chunk.content):
                return False
        
        return True


# Convenience functions

def compress_context(
    contexts: List[Dict[str, Any]],
    query: str,
    max_tokens: int = 4000,
    strategy: CompressionStrategy = CompressionStrategy.HYBRID,
) -> CompressedContext:
    """
    Compress context to fit token limit.
    
    Args:
        contexts: List of context dicts
        query: Query for relevance
        max_tokens: Maximum tokens
        strategy: Compression strategy
        
    Returns:
        CompressedContext
    """
    compressor = ContextCompressor(
        max_tokens=max_tokens,
        strategy=strategy,
    )
    return compressor.compress(contexts, query)


def filter_context(
    contexts: List[Dict[str, Any]],
    min_score: float = 0.3,
    min_length: int = 50,
) -> List[Dict[str, Any]]:
    """
    Filter contexts by score and length.
    
    Args:
        contexts: Contexts to filter
        min_score: Minimum relevance score
        min_length: Minimum character length
        
    Returns:
        Filtered contexts
    """
    filtered = []
    
    for ctx in contexts:
        score = ctx.get("score", ctx.get("relevance", 0.0))
        text = ctx.get("text", ctx.get("content", ""))
        
        if score >= min_score and len(text) >= min_length:
            filtered.append(ctx)
    
    return filtered


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate token count for text.
    
    Args:
        text: Input text
        chars_per_token: Average characters per token
        
    Returns:
        Estimated token count
    """
    return int(len(text) / chars_per_token)
