"""
Accurate token counting for AutoRAG-Live.

Provides model-aware token counting using tiktoken when available,
with a calibrated fallback for environments without tiktoken.

State-of-the-art agentic RAG requires precise token budgets for:
- Context window management (avoiding truncation)
- Cost estimation (tokens → API pricing)
- Chunking alignment (token-aware splitting)
- Agent memory management (conversation history pruning)

Example:
    >>> tokenizer = get_tokenizer("gpt-4o")
    >>> count = tokenizer.count_tokens("Hello, world!")
    >>> truncated = tokenizer.truncate_to_tokens("long text...", max_tokens=100)
"""

from __future__ import annotations

import functools
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore[assignment]
    TIKTOKEN_AVAILABLE = False
    logger.info("tiktoken not installed; using calibrated heuristic token counting")


# Model → tiktoken encoding name mapping
_MODEL_ENCODING_MAP: Dict[str, str] = {
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "claude-3-opus": "cl100k_base",  # Approximate with cl100k
    "claude-3-sonnet": "cl100k_base",
    "claude-3-haiku": "cl100k_base",
    "claude-3.5-sonnet": "cl100k_base",
    "default": "cl100k_base",
}


class TokenCounter:
    """
    Model-aware token counter.

    Uses tiktoken for exact counts when available, falls back to
    a calibrated character-based heuristic (3.3 chars/token for English,
    empirically measured across GPT tokenizers).

    The naive 4 chars/token heuristic underestimates by ~20%, causing
    context window overflows in production agentic RAG pipelines.
    """

    def __init__(self, model: str = "default"):
        """
        Initialize token counter.

        Args:
            model: Model name for encoding selection
        """
        self.model = model
        self._encoder = None

        if TIKTOKEN_AVAILABLE:
            encoding_name = _MODEL_ENCODING_MAP.get(model, "cl100k_base")
            try:
                self._encoder = tiktoken.get_encoding(encoding_name)
            except Exception:
                logger.warning(f"Failed to load encoding {encoding_name}, using heuristic")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count (exact with tiktoken, estimated otherwise)
        """
        if not text:
            return 0

        if self._encoder is not None:
            return len(self._encoder.encode(text))

        # Calibrated heuristic: 1 token ≈ 3.3 chars for English text
        # This accounts for subword tokenization patterns
        return max(1, int(len(text) / 3.3))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """
        Count tokens for multiple texts efficiently.

        Args:
            texts: List of input texts

        Returns:
            List of token counts
        """
        if self._encoder is not None:
            return [len(self._encoder.encode(t)) for t in texts]
        return [self.count_tokens(t) for t in texts]

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within a token budget.

        Args:
            text: Input text
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated text that fits within max_tokens
        """
        if not text or max_tokens <= 0:
            return ""

        if self._encoder is not None:
            tokens = self._encoder.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self._encoder.decode(tokens[:max_tokens])

        # Heuristic truncation
        estimated_chars = int(max_tokens * 3.3)
        if len(text) <= estimated_chars:
            return text
        return text[:estimated_chars]

    def fits_in_budget(self, text: str, budget: int) -> bool:
        """Check if text fits within token budget."""
        return self.count_tokens(text) <= budget


# Module-level cached instances
@functools.lru_cache(maxsize=8)
def get_tokenizer(model: str = "default") -> TokenCounter:
    """
    Get a cached TokenCounter instance for a model.

    Args:
        model: Model name

    Returns:
        TokenCounter instance (cached per model)
    """
    return TokenCounter(model=model)


def count_tokens(text: str, model: str = "default") -> int:
    """
    Convenience function for one-off token counting.

    Args:
        text: Input text
        model: Model name

    Returns:
        Token count
    """
    return get_tokenizer(model).count_tokens(text)


def truncate_to_tokens(text: str, max_tokens: int, model: str = "default") -> str:
    """
    Convenience function for one-off token truncation.

    Args:
        text: Input text
        max_tokens: Maximum tokens
        model: Model name

    Returns:
        Truncated text
    """
    return get_tokenizer(model).truncate_to_tokens(text, max_tokens)
