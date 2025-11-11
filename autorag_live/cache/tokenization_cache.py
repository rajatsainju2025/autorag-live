"""Efficient tokenization caching for BM25 and TF-IDF operations.

This module provides thread-safe tokenization caching with:
- Fast regex-based tokenization
- LRU cache with size limits
- Memory-efficient token storage
- Batch tokenization support

Example:
    >>> tokenizer = TokenizationCache()
    >>> tokens = tokenizer.tokenize("query text")
    >>> batch_tokens = tokenizer.tokenize_batch(["text1", "text2"])
"""

import re
import threading
from collections import Counter, OrderedDict
from typing import Dict, List

# Pre-compiled token pattern for fast tokenization
_TOKEN_PATTERN = re.compile(r"\w+")


class TokenizationCache:
    """Thread-safe tokenization cache with LRU eviction."""

    def __init__(
        self,
        max_size: int = 512,
        lowercase: bool = True,
        remove_empty: bool = True,
    ):
        """Initialize tokenization cache.

        Args:
            max_size: Maximum number of cached tokenizations
            lowercase: Whether to lowercase tokens
            remove_empty: Whether to remove empty token lists
        """
        self.max_size = max_size
        self.lowercase = lowercase
        self.remove_empty = remove_empty
        self.cache: OrderedDict[str, List[str]] = OrderedDict()
        self.lock = threading.RLock()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text or retrieve from cache.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        with self.lock:
            if text in self.cache:
                # Move to end (LRU)
                self.cache.move_to_end(text)
                return self.cache[text]

            tokens = self._tokenize_impl(text)

            # Cache result
            self.cache[text] = tokens

            # Evict oldest if over capacity
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

            return tokens

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Tokenize batch of texts.

        Args:
            texts: Texts to tokenize

        Returns:
            List of token lists
        """
        return [self.tokenize(text) for text in texts]

    def get_term_frequencies(self, text: str) -> Dict[str, int]:
        """Get term frequency counter for text.

        Args:
            text: Text to analyze

        Returns:
            Counter with token frequencies
        """
        tokens = self.tokenize(text)
        return dict(Counter(tokens)) if tokens else {}

    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()

    def _tokenize_impl(self, text: str) -> List[str]:
        """Implement tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Fast regex tokenization
        tokens = _TOKEN_PATTERN.findall(text)

        # Optional lowercasing
        if self.lowercase:
            tokens = [t.lower() for t in tokens]

        # Optional empty removal
        if self.remove_empty:
            tokens = [t for t in tokens if t]

        return tokens

    def __len__(self) -> int:
        """Return number of cached entries."""
        with self.lock:
            return len(self.cache)

    def __contains__(self, text: str) -> bool:
        """Check if text is cached."""
        with self.lock:
            return text in self.cache


class TFIDFVectorizer:
    """Efficient TF-IDF vectorization with caching.

    Computes term frequencies with memoized tokenization for performance.
    """

    def __init__(self, max_cache_size: int = 512):
        """Initialize TF-IDF vectorizer.

        Args:
            max_cache_size: Max tokenization cache size
        """
        self.tokenizer = TokenizationCache(max_size=max_cache_size)
        self.idf_cache: Dict[str, float] = {}

    def compute_tf(self, text: str) -> Dict[str, float]:
        """Compute term frequencies for text.

        Args:
            text: Text to compute TF for

        Returns:
            Dictionary of term -> frequency
        """
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return {}

        n_tokens = len(tokens)
        tf = Counter(tokens)

        # Normalize by document length
        return {term: count / n_tokens for term, count in tf.items()}

    def compute_tf_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Compute TF for batch of texts.

        Args:
            texts: Texts to process

        Returns:
            List of TF dictionaries
        """
        return [self.compute_tf(text) for text in texts]

    def fit_idf(self, corpus: List[str]) -> None:
        """Fit IDF from corpus.

        Args:
            corpus: Documents to compute IDF from
        """
        import math

        n_docs = len(corpus)
        doc_freq: Dict[str, int] = {}

        for text in corpus:
            tokens_set = set(self.tokenizer.tokenize(text))
            for token in tokens_set:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # IDF = log(N / df)
        self.idf_cache = {term: math.log(n_docs / freq) for term, freq in doc_freq.items()}

    def compute_tfidf(self, text: str) -> Dict[str, float]:
        """Compute TF-IDF for text (requires fit_idf first).

        Args:
            text: Text to compute TF-IDF for

        Returns:
            Dictionary of term -> TF-IDF score
        """
        tf = self.compute_tf(text)
        return {term: tf_val * self.idf_cache.get(term, 0.0) for term, tf_val in tf.items()}

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.tokenizer.clear()
        self.idf_cache.clear()
