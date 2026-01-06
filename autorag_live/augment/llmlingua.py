"""
LLMLingua-style Context Compression Module.

Implements selective token pruning and context compression
for efficient LLM inference with long contexts.

Key Features:
1. Perplexity-based token importance scoring
2. Selective token/sentence pruning
3. Budget-aware compression
4. Coarse-to-fine compression pipeline
5. Query-aware compression

References:
- LLMLingua: Compressing Prompts for Accelerated Inference (Jiang et al., 2023)
- LongLLMLingua: Accelerating Long-Context LLM Inference
- Selective Context: Efficient Inference for Long Documents

Example:
    >>> compressor = LLMLinguaCompressor(llm)
    >>> compressed = await compressor.compress(context, target_ratio=0.5)
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class LLMProtocol(Protocol):
    """Protocol for LLM interface."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response from prompt."""
        ...

    async def get_logprobs(self, prompt: str, **kwargs: Any) -> List[Tuple[str, float]]:
        """Get log probabilities for tokens."""
        ...


# =============================================================================
# Data Structures
# =============================================================================


class CompressionStrategy(str, Enum):
    """Strategy for context compression."""

    TOKEN_LEVEL = "token_level"  # Individual token pruning
    SENTENCE_LEVEL = "sentence_level"  # Sentence-level pruning
    PARAGRAPH_LEVEL = "paragraph_level"  # Paragraph-level pruning
    COARSE_TO_FINE = "coarse_to_fine"  # Multi-level compression
    QUERY_AWARE = "query_aware"  # Query-focused compression


@dataclass
class TokenScore:
    """
    Score for a token in the context.

    Attributes:
        token: The token text
        position: Position in context
        perplexity: Perplexity score (lower = more predictable)
        importance: Importance score (higher = keep)
        keep: Whether to keep this token
    """

    token: str
    position: int
    perplexity: float = 0.0
    importance: float = 0.5
    keep: bool = True


@dataclass
class SentenceScore:
    """
    Score for a sentence in the context.

    Attributes:
        text: Sentence text
        index: Sentence index
        tokens: Token scores in sentence
        avg_perplexity: Average token perplexity
        importance: Overall importance score
        query_relevance: Relevance to query (if query-aware)
    """

    text: str
    index: int
    tokens: List[TokenScore] = field(default_factory=list)
    avg_perplexity: float = 0.0
    importance: float = 0.5
    query_relevance: float = 0.5


@dataclass
class CompressionResult:
    """
    Result of context compression.

    Attributes:
        original: Original context
        compressed: Compressed context
        compression_ratio: Ratio of compressed to original length
        tokens_removed: Number of tokens removed
        strategy: Strategy used
        metadata: Additional details
    """

    original: str
    compressed: str
    compression_ratio: float = 1.0
    tokens_removed: int = 0
    strategy: CompressionStrategy = CompressionStrategy.SENTENCE_LEVEL
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def original_length(self) -> int:
        """Get original character length."""
        return len(self.original)

    @property
    def compressed_length(self) -> int:
        """Get compressed character length."""
        return len(self.compressed)

    @property
    def savings_percent(self) -> float:
        """Get compression savings as percentage."""
        if self.original_length == 0:
            return 0.0
        return (1 - self.compression_ratio) * 100


# =============================================================================
# Perplexity Calculator
# =============================================================================


class PerplexityCalculator:
    """
    Calculates perplexity scores for tokens.

    Uses LLM logprobs or heuristics to estimate token predictability.
    """

    def __init__(self, llm: Optional[LLMProtocol] = None):
        """Initialize calculator."""
        self.llm = llm
        self._stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set:
        """Load common stopwords."""
        return {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "although",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "them",
            "their",
        }

    async def calculate_token_scores(
        self,
        text: str,
    ) -> List[TokenScore]:
        """
        Calculate perplexity scores for tokens.

        Args:
            text: Input text

        Returns:
            List of TokenScore for each token
        """
        tokens = self._tokenize(text)

        if self.llm and hasattr(self.llm, "get_logprobs"):
            return await self._calculate_with_llm(tokens, text)
        else:
            return self._calculate_heuristic(tokens)

    async def _calculate_with_llm(
        self,
        tokens: List[str],
        text: str,
    ) -> List[TokenScore]:
        """Calculate using LLM logprobs."""
        try:
            logprobs = await self.llm.get_logprobs(text)

            scores = []
            for i, (token, logprob) in enumerate(logprobs):
                perplexity = math.exp(-logprob) if logprob else 1.0
                importance = 1.0 / (1.0 + perplexity)  # Lower perplexity = higher importance

                scores.append(
                    TokenScore(
                        token=token,
                        position=i,
                        perplexity=perplexity,
                        importance=importance,
                    )
                )

            return scores
        except Exception as e:
            logger.warning(f"LLM logprobs failed: {e}, using heuristics")
            return self._calculate_heuristic(tokens)

    def _calculate_heuristic(self, tokens: List[str]) -> List[TokenScore]:
        """Calculate using heuristics."""
        scores = []

        for i, token in enumerate(tokens):
            token_lower = token.lower()

            # Heuristic importance scoring
            importance = 0.5

            # Stopwords are less important
            if token_lower in self._stopwords:
                importance = 0.2

            # Punctuation is less important
            if token in ".,;:!?\"'()[]{}":
                importance = 0.1

            # Capitalized words (names, acronyms) are more important
            if token[0].isupper() and len(token) > 1:
                importance = 0.8

            # Numbers may be important
            if token.isdigit():
                importance = 0.7

            # Longer words tend to be more meaningful
            if len(token) > 8:
                importance = min(importance + 0.2, 1.0)

            # Estimate perplexity (inverse of importance for heuristic)
            perplexity = 1.0 / max(importance, 0.1)

            scores.append(
                TokenScore(
                    token=token,
                    position=i,
                    perplexity=perplexity,
                    importance=importance,
                )
            )

        return scores

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        return re.findall(r"\S+", text)


# =============================================================================
# Sentence Scorer
# =============================================================================


class SentenceScorer:
    """Scores sentences for compression."""

    def __init__(
        self,
        perplexity_calc: PerplexityCalculator,
    ):
        """Initialize scorer."""
        self.perplexity_calc = perplexity_calc

    async def score_sentences(
        self,
        text: str,
        query: Optional[str] = None,
    ) -> List[SentenceScore]:
        """
        Score sentences in text.

        Args:
            text: Input text
            query: Optional query for relevance scoring

        Returns:
            List of SentenceScore
        """
        sentences = self._split_sentences(text)
        scores = []

        for i, sentence in enumerate(sentences):
            token_scores = await self.perplexity_calc.calculate_token_scores(sentence)

            # Calculate average perplexity
            if token_scores:
                avg_ppl = sum(t.perplexity for t in token_scores) / len(token_scores)
                avg_imp = sum(t.importance for t in token_scores) / len(token_scores)
            else:
                avg_ppl = 1.0
                avg_imp = 0.5

            # Calculate query relevance if query provided
            query_rel = 0.5
            if query:
                query_rel = self._calculate_query_relevance(sentence, query)

            scores.append(
                SentenceScore(
                    text=sentence,
                    index=i,
                    tokens=token_scores,
                    avg_perplexity=avg_ppl,
                    importance=avg_imp,
                    query_relevance=query_rel,
                )
            )

        return scores

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_query_relevance(
        self,
        sentence: str,
        query: str,
    ) -> float:
        """Calculate relevance of sentence to query."""
        sentence_words = set(sentence.lower().split())
        query_words = set(query.lower().split())

        if not query_words:
            return 0.5

        # Word overlap
        overlap = len(sentence_words & query_words)
        relevance = overlap / len(query_words)

        return min(relevance, 1.0)


# =============================================================================
# Compression Strategies
# =============================================================================


class CompressionStrategyBase(ABC):
    """Base class for compression strategies."""

    @abstractmethod
    async def compress(
        self,
        text: str,
        target_ratio: float,
        **kwargs: Any,
    ) -> str:
        """Compress text to target ratio."""
        pass


class TokenLevelCompression(CompressionStrategyBase):
    """Token-level compression via pruning."""

    def __init__(self, perplexity_calc: PerplexityCalculator):
        """Initialize strategy."""
        self.perplexity_calc = perplexity_calc

    async def compress(
        self,
        text: str,
        target_ratio: float,
        **kwargs: Any,
    ) -> str:
        """Compress by removing low-importance tokens."""
        token_scores = await self.perplexity_calc.calculate_token_scores(text)

        if not token_scores:
            return text

        # Calculate number of tokens to keep
        num_keep = max(1, int(len(token_scores) * target_ratio))

        # Sort by importance and keep top tokens
        sorted_scores = sorted(token_scores, key=lambda t: t.importance, reverse=True)

        keep_positions = set(t.position for t in sorted_scores[:num_keep])

        # Reconstruct text keeping original order
        compressed_tokens = [t.token for t in token_scores if t.position in keep_positions]

        return " ".join(compressed_tokens)


class SentenceLevelCompression(CompressionStrategyBase):
    """Sentence-level compression via pruning."""

    def __init__(self, sentence_scorer: SentenceScorer):
        """Initialize strategy."""
        self.sentence_scorer = sentence_scorer

    async def compress(
        self,
        text: str,
        target_ratio: float,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Compress by removing low-importance sentences."""
        sentence_scores = await self.sentence_scorer.score_sentences(text, query)

        if not sentence_scores:
            return text

        # Calculate target length
        original_length = sum(len(s.text) for s in sentence_scores)
        target_length = int(original_length * target_ratio)

        # Score sentences (combine importance and query relevance)
        for s in sentence_scores:
            s.importance = 0.5 * s.importance + 0.5 * s.query_relevance

        # Sort by importance
        sorted_sentences = sorted(sentence_scores, key=lambda s: s.importance, reverse=True)

        # Greedily select sentences up to target length
        selected = []
        current_length = 0

        for sentence in sorted_sentences:
            if current_length + len(sentence.text) <= target_length:
                selected.append(sentence)
                current_length += len(sentence.text)

        # Restore original order
        selected.sort(key=lambda s: s.index)

        return " ".join(s.text for s in selected)


class CoarseToFineCompression(CompressionStrategyBase):
    """
    Coarse-to-fine compression (LLMLingua style).

    First removes whole paragraphs/sentences, then prunes tokens.
    """

    def __init__(
        self,
        sentence_scorer: SentenceScorer,
        perplexity_calc: PerplexityCalculator,
    ):
        """Initialize strategy."""
        self.sentence_scorer = sentence_scorer
        self.perplexity_calc = perplexity_calc
        self.sentence_compressor = SentenceLevelCompression(sentence_scorer)
        self.token_compressor = TokenLevelCompression(perplexity_calc)

    async def compress(
        self,
        text: str,
        target_ratio: float,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Two-stage coarse-to-fine compression."""
        # Stage 1: Coarse compression (sentence level)
        # Remove ~30% at sentence level
        coarse_ratio = min(target_ratio * 1.5, 0.9)
        coarse_text = await self.sentence_compressor.compress(text, coarse_ratio, query=query)

        # Stage 2: Fine compression (token level)
        # Remaining compression at token level
        fine_ratio = target_ratio / coarse_ratio if coarse_ratio > 0 else target_ratio
        fine_text = await self.token_compressor.compress(coarse_text, fine_ratio)

        return fine_text


# =============================================================================
# Main LLMLingua Compressor
# =============================================================================


class LLMLinguaCompressor:
    """
    LLMLingua-style context compressor.

    Compresses prompts/contexts for efficient LLM inference
    while preserving essential information.

    Example:
        >>> compressor = LLMLinguaCompressor(llm)
        >>> result = await compressor.compress(
        ...     context="Long document text...",
        ...     target_ratio=0.5,
        ...     query="What is the main topic?"
        ... )
        >>> print(f"Compressed to {result.compression_ratio:.1%}")
    """

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        strategy: CompressionStrategy = CompressionStrategy.COARSE_TO_FINE,
    ):
        """
        Initialize compressor.

        Args:
            llm: Language model for perplexity calculation
            strategy: Default compression strategy
        """
        self.llm = llm
        self.default_strategy = strategy

        # Initialize components
        self.perplexity_calc = PerplexityCalculator(llm)
        self.sentence_scorer = SentenceScorer(self.perplexity_calc)

        # Initialize strategies
        self._strategies: Dict[CompressionStrategy, CompressionStrategyBase] = {
            CompressionStrategy.TOKEN_LEVEL: TokenLevelCompression(self.perplexity_calc),
            CompressionStrategy.SENTENCE_LEVEL: SentenceLevelCompression(self.sentence_scorer),
            CompressionStrategy.COARSE_TO_FINE: CoarseToFineCompression(
                self.sentence_scorer, self.perplexity_calc
            ),
        }

    async def compress(
        self,
        context: str,
        *,
        target_ratio: float = 0.5,
        query: Optional[str] = None,
        strategy: Optional[CompressionStrategy] = None,
        min_length: int = 100,
    ) -> CompressionResult:
        """
        Compress context to target ratio.

        Args:
            context: Text to compress
            target_ratio: Target compression ratio (0.5 = 50% of original)
            query: Optional query for query-aware compression
            strategy: Compression strategy
            min_length: Minimum output length

        Returns:
            CompressionResult with compressed text
        """
        if not context or len(context) < min_length:
            return CompressionResult(
                original=context,
                compressed=context,
                compression_ratio=1.0,
                strategy=strategy or self.default_strategy,
            )

        strategy = strategy or self.default_strategy

        # Use query-aware if query provided
        if query and strategy != CompressionStrategy.QUERY_AWARE:
            strategy = CompressionStrategy.SENTENCE_LEVEL

        # Get strategy implementation
        strategy_impl = self._strategies.get(
            strategy,
            self._strategies[CompressionStrategy.SENTENCE_LEVEL],
        )

        # Compress
        compressed = await strategy_impl.compress(
            context,
            target_ratio,
            query=query,
        )

        # Ensure minimum length
        if len(compressed) < min_length and len(context) >= min_length:
            # Re-compress with higher ratio
            new_ratio = min_length / len(context)
            compressed = await strategy_impl.compress(context, new_ratio, query=query)

        # Calculate stats
        original_tokens = len(context.split())
        compressed_tokens = len(compressed.split())
        tokens_removed = original_tokens - compressed_tokens

        actual_ratio = len(compressed) / len(context) if len(context) > 0 else 1.0

        return CompressionResult(
            original=context,
            compressed=compressed,
            compression_ratio=actual_ratio,
            tokens_removed=tokens_removed,
            strategy=strategy,
            metadata={
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "query_used": query is not None,
            },
        )

    async def compress_documents(
        self,
        documents: List[Dict[str, Any]],
        *,
        target_ratio: float = 0.5,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compress multiple documents.

        Args:
            documents: Documents with 'content' key
            target_ratio: Target compression ratio
            query: Optional query for relevance

        Returns:
            Documents with compressed content
        """
        results = []

        for doc in documents:
            content = doc.get("content", doc.get("text", ""))

            result = await self.compress(
                content,
                target_ratio=target_ratio,
                query=query,
            )

            compressed_doc = dict(doc)
            compressed_doc["content"] = result.compressed
            compressed_doc["original_content"] = content
            compressed_doc["compression_ratio"] = result.compression_ratio

            results.append(compressed_doc)

        return results


# =============================================================================
# Budget-Aware Compressor
# =============================================================================


class BudgetAwareCompressor:
    """
    Compressor that respects token budgets.

    Distributes compression across multiple documents to fit
    within a total token budget.
    """

    def __init__(
        self,
        compressor: LLMLinguaCompressor,
        max_tokens: int = 4096,
    ):
        """
        Initialize budget-aware compressor.

        Args:
            compressor: Base compressor
            max_tokens: Maximum token budget
        """
        self.compressor = compressor
        self.max_tokens = max_tokens

    async def compress_to_budget(
        self,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compress documents to fit within budget.

        Args:
            documents: Documents to compress
            query: Optional query

        Returns:
            Compressed documents fitting budget
        """
        if not documents:
            return []

        # Calculate current total tokens
        total_tokens = sum(len(doc.get("content", "").split()) for doc in documents)

        if total_tokens <= self.max_tokens:
            # Already within budget
            return documents

        # Calculate required compression ratio
        target_ratio = self.max_tokens / total_tokens

        # Compress all documents
        return await self.compressor.compress_documents(
            documents,
            target_ratio=target_ratio,
            query=query,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple word-based estimation
        # More accurate would use actual tokenizer
        return len(text.split())


# =============================================================================
# Convenience Functions
# =============================================================================


def create_compressor(
    llm: Optional[LLMProtocol] = None,
    strategy: CompressionStrategy = CompressionStrategy.COARSE_TO_FINE,
) -> LLMLinguaCompressor:
    """
    Create an LLMLingua-style compressor.

    Args:
        llm: Language model
        strategy: Compression strategy

    Returns:
        LLMLinguaCompressor instance
    """
    return LLMLinguaCompressor(llm=llm, strategy=strategy)


async def compress_context(
    context: str,
    target_ratio: float = 0.5,
    query: Optional[str] = None,
) -> str:
    """
    Quick context compression.

    Args:
        context: Text to compress
        target_ratio: Target ratio
        query: Optional query

    Returns:
        Compressed text
    """
    compressor = LLMLinguaCompressor()
    result = await compressor.compress(context, target_ratio=target_ratio, query=query)
    return result.compressed
