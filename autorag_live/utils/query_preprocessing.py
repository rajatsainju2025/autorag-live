"""
Query preprocessing utilities for AutoRAG-Live.

Utilities for normalizing, cleaning, and preprocessing queries before
retrieval to improve search quality and consistency.

Example:
    >>> from autorag_live.utils.query_preprocessing import preprocess_query
    >>>
    >>> query = "  What is AI?!?  "
    >>> clean = preprocess_query(query)
    >>> print(clean)  # "what is ai"
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set


def preprocess_query(
    query: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_extra_whitespace: bool = True,
    remove_stopwords: bool = False,
    stopwords: Optional[Set[str]] = None,
) -> str:
    """
    Preprocess query text for retrieval.

    Args:
        query: Query string to preprocess
        lowercase: Convert to lowercase
        remove_punctuation: Remove punctuation marks
        remove_extra_whitespace: Normalize whitespace
        remove_stopwords: Remove common stopwords
        stopwords: Custom stopword set (uses default if None)

    Returns:
        Preprocessed query string

    Example:
        >>> query = "What's the BEST way???"
        >>> clean = preprocess_query(query)
        >>> print(clean)  # "whats the best way"
    """
    if not query:
        return ""

    # Normalize unicode
    query = unicodedata.normalize("NFKD", query)

    # Lowercase
    if lowercase:
        query = query.lower()

    # Remove punctuation
    if remove_punctuation:
        query = re.sub(r"[^\w\s]", " ", query)

    # Remove extra whitespace
    if remove_extra_whitespace:
        query = " ".join(query.split())

    # Remove stopwords
    if remove_stopwords:
        if stopwords is None:
            stopwords = get_default_stopwords()
        words = query.split()
        words = [w for w in words if w not in stopwords]
        query = " ".join(words)

    return query.strip()


def get_default_stopwords() -> Set[str]:
    """
    Get default English stopwords.

    Returns:
        Set of common stopwords

    Example:
        >>> stopwords = get_default_stopwords()
        >>> assert "the" in stopwords
    """
    return {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
    }


def expand_query(query: str, synonyms: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """
    Expand query with synonyms for broader matching.

    Args:
        query: Original query
        synonyms: Dictionary mapping words to synonyms

    Returns:
        List of expanded queries

    Example:
        >>> query = "fast car"
        >>> synonyms = {"fast": ["quick", "rapid"], "car": ["vehicle", "auto"]}
        >>> expanded = expand_query(query, synonyms)
        >>> assert len(expanded) > 1
    """
    if not synonyms:
        return [query]

    queries = [query]
    words = query.split()

    # Replace each word with synonyms
    for i, word in enumerate(words):
        if word.lower() in synonyms:
            for synonym in synonyms[word.lower()]:
                new_words = words.copy()
                new_words[i] = synonym
                queries.append(" ".join(new_words))

    return queries


def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace to single spaces.

    Args:
        text: Text to normalize

    Returns:
        Normalized text

    Example:
        >>> text = "hello\\t\\nworld   "
        >>> normalized = normalize_whitespace(text)
        >>> assert normalized == "hello world"
    """
    return " ".join(text.split())


def remove_urls(text: str) -> str:
    """
    Remove URLs from text.

    Args:
        text: Text containing URLs

    Returns:
        Text with URLs removed

    Example:
        >>> text = "Visit https://example.com for more"
        >>> clean = remove_urls(text)
        >>> assert "https" not in clean
    """
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "", text)


def remove_emails(text: str) -> str:
    """
    Remove email addresses from text.

    Args:
        text: Text containing emails

    Returns:
        Text with emails removed

    Example:
        >>> text = "Contact user@example.com"
        >>> clean = remove_emails(text)
        >>> assert "@" not in clean
    """
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return re.sub(email_pattern, "", text)


def extract_keywords(query: str, min_length: int = 3, max_keywords: int = 5) -> List[str]:
    """
    Extract important keywords from query.

    Args:
        query: Query string
        min_length: Minimum keyword length
        max_keywords: Maximum number of keywords

    Returns:
        List of keywords

    Example:
        >>> query = "machine learning algorithms for data science"
        >>> keywords = extract_keywords(query, max_keywords=3)
        >>> assert len(keywords) <= 3
    """
    # Simple keyword extraction (frequency-based)
    words = preprocess_query(query, remove_stopwords=True).split()
    words = [w for w in words if len(w) >= min_length]

    # Count frequencies
    freq: Dict[str, int] = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    # Sort by frequency and return top keywords
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]
