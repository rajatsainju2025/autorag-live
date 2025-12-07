"""
Centralized input validation utilities for AutoRAG-Live.

This module provides consistent validation for common input patterns across
the codebase, ensuring robust handling of edge cases like None values,
empty strings, and invalid data types.

Functions:
    - validate_query: Validate query strings
    - validate_corpus: Validate document corpus lists
    - validate_k_parameter: Validate k (top-k) parameters
    - validate_score_range: Validate score values
    - sanitize_text: Clean and normalize text inputs

Example:
    >>> from autorag_live.utils.input_validation import validate_query, validate_corpus
    >>>
    >>> query = validate_query("  example query  ")  # Strips whitespace
    >>> corpus = validate_corpus(["doc1", "doc2", None])  # Filters None
    >>>
    >>> # Validation with custom error messages
    >>> k = validate_k_parameter(5, max_value=100)
"""

from typing import Any, List, Optional, Union

from autorag_live.types import DocumentText, QueryText, ValidationError


def validate_query(
    query: Optional[str],
    allow_empty: bool = False,
    strip_whitespace: bool = True,
    max_length: Optional[int] = None,
) -> QueryText:
    """
    Validate and normalize query strings.

    Args:
        query: Query string to validate
        allow_empty: Whether to allow empty queries after stripping
        strip_whitespace: Whether to strip leading/trailing whitespace
        max_length: Maximum allowed query length (None for unlimited)

    Returns:
        Validated and normalized query string

    Raises:
        ValidationError: If query is invalid

    Example:
        >>> query = validate_query("  test  ")
        >>> assert query == "test"
    """
    if query is None:
        raise ValidationError(
            message="Query cannot be None",
            context={"query": query},
        )

    if not isinstance(query, str):
        raise ValidationError(
            message=f"Query must be a string, got {type(query).__name__}",
            context={"query": query, "type": type(query).__name__},
        )

    if strip_whitespace:
        query = query.strip()

    if not query and not allow_empty:
        raise ValidationError(
            message="Query cannot be empty",
            context={"query": query},
        )

    if max_length is not None and len(query) > max_length:
        raise ValidationError(
            message=f"Query exceeds maximum length of {max_length}",
            context={"query_length": len(query), "max_length": max_length},
        )

    return query


def validate_corpus(
    corpus: Optional[List[Any]],
    allow_empty: bool = False,
    filter_none: bool = True,
    filter_empty: bool = True,
    min_size: Optional[int] = None,
) -> List[DocumentText]:
    """
    Validate and clean document corpus.

    Args:
        corpus: List of documents to validate
        allow_empty: Whether to allow empty corpus
        filter_none: Whether to filter out None values
        filter_empty: Whether to filter out empty strings
        min_size: Minimum required corpus size after filtering

    Returns:
        Validated and cleaned corpus

    Raises:
        ValidationError: If corpus is invalid

    Example:
        >>> corpus = validate_corpus(["doc1", None, "", "doc2"])
        >>> assert corpus == ["doc1", "doc2"]
    """
    if corpus is None:
        raise ValidationError(
            message="Corpus cannot be None",
            context={"corpus": corpus},
        )

    if not isinstance(corpus, (list, tuple)):
        raise ValidationError(
            message=f"Corpus must be a list or tuple, got {type(corpus).__name__}",
            context={"type": type(corpus).__name__},
        )

    # Convert to list and filter
    cleaned_corpus: List[str] = []
    for idx, doc in enumerate(corpus):
        # Filter None values
        if doc is None:
            if not filter_none:
                raise ValidationError(
                    message=f"Document at index {idx} is None",
                    context={"index": idx, "document": doc},
                )
            continue

        # Convert to string
        if not isinstance(doc, str):
            doc = str(doc)

        # Filter empty strings
        if not doc.strip():
            if not filter_empty:
                raise ValidationError(
                    message=f"Document at index {idx} is empty",
                    context={"index": idx, "document": doc},
                )
            continue

        cleaned_corpus.append(doc)

    # Check minimum size
    if not cleaned_corpus and not allow_empty:
        raise ValidationError(
            message="Corpus is empty after filtering",
            context={"original_size": len(corpus), "filtered_size": 0},
        )

    if min_size is not None and len(cleaned_corpus) < min_size:
        raise ValidationError(
            message=f"Corpus size {len(cleaned_corpus)} is less than minimum {min_size}",
            context={
                "corpus_size": len(cleaned_corpus),
                "min_size": min_size,
            },
        )

    return cleaned_corpus


def validate_k_parameter(
    k: Union[int, float],
    min_value: int = 1,
    max_value: Optional[int] = None,
    corpus_size: Optional[int] = None,
) -> int:
    """
    Validate k (top-k) parameter.

    Args:
        k: Number of results to retrieve
        min_value: Minimum allowed value
        max_value: Maximum allowed value (None for unlimited)
        corpus_size: Size of corpus to validate against

    Returns:
        Validated k value as integer

    Raises:
        ValidationError: If k is invalid

    Example:
        >>> k = validate_k_parameter(5, corpus_size=10)
        >>> assert k == 5
    """
    if not isinstance(k, (int, float)):
        raise ValidationError(
            message=f"k must be numeric, got {type(k).__name__}",
            context={"k": k, "type": type(k).__name__},
        )

    k = int(k)

    if k < min_value:
        raise ValidationError(
            message=f"k must be at least {min_value}, got {k}",
            context={"k": k, "min_value": min_value},
        )

    if max_value is not None and k > max_value:
        raise ValidationError(
            message=f"k cannot exceed {max_value}, got {k}",
            context={"k": k, "max_value": max_value},
        )

    if corpus_size is not None and k > corpus_size:
        # This is a warning, not an error - just clamp to corpus size
        k = corpus_size

    return k


def validate_score_range(
    score: float,
    min_score: float = 0.0,
    max_score: float = 1.0,
    allow_negative: bool = False,
) -> float:
    """
    Validate score values.

    Args:
        score: Score value to validate
        min_score: Minimum allowed score
        max_score: Maximum allowed score
        allow_negative: Whether to allow negative scores

    Returns:
        Validated score

    Raises:
        ValidationError: If score is invalid

    Example:
        >>> score = validate_score_range(0.85)
        >>> assert 0.0 <= score <= 1.0
    """
    if not isinstance(score, (int, float)):
        raise ValidationError(
            message=f"Score must be numeric, got {type(score).__name__}",
            context={"score": score, "type": type(score).__name__},
        )

    if not allow_negative and score < 0:
        raise ValidationError(
            message=f"Score cannot be negative, got {score}",
            context={"score": score},
        )

    if score < min_score:
        raise ValidationError(
            message=f"Score {score} is below minimum {min_score}",
            context={"score": score, "min_score": min_score},
        )

    if score > max_score:
        raise ValidationError(
            message=f"Score {score} exceeds maximum {max_score}",
            context={"score": score, "max_score": max_score},
        )

    return float(score)


def sanitize_text(
    text: str,
    remove_special_chars: bool = False,
    normalize_whitespace: bool = True,
    lowercase: bool = False,
    max_length: Optional[int] = None,
) -> str:
    """
    Sanitize and normalize text inputs.

    Args:
        text: Text to sanitize
        remove_special_chars: Whether to remove special characters
        normalize_whitespace: Whether to normalize whitespace
        lowercase: Whether to convert to lowercase
        max_length: Maximum text length (truncates if exceeded)

    Returns:
        Sanitized text

    Example:
        >>> text = sanitize_text("  Hello   World!  ", normalize_whitespace=True)
        >>> assert text == "Hello World!"
    """
    if not isinstance(text, str):
        text = str(text)

    # Normalize whitespace
    if normalize_whitespace:
        text = " ".join(text.split())

    # Remove special characters (keep alphanumeric and basic punctuation)
    if remove_special_chars:
        import re

        text = re.sub(r"[^a-zA-Z0-9\s.,!?-]", "", text)

    # Convert to lowercase
    if lowercase:
        text = text.lower()

    # Truncate if needed
    if max_length is not None and len(text) > max_length:
        text = text[:max_length]

    return text


# Convenience validators for common patterns
def validate_retrieval_inputs(
    query: str,
    corpus: List[Any],
    k: int,
) -> tuple[QueryText, List[DocumentText], int]:
    """
    Validate all inputs for a retrieval operation.

    Args:
        query: Query string
        corpus: Document corpus
        k: Number of results

    Returns:
        Tuple of (validated_query, validated_corpus, validated_k)

    Raises:
        ValidationError: If any input is invalid

    Example:
        >>> query, corpus, k = validate_retrieval_inputs("test", ["doc1"], 5)
        >>> assert k == 1  # Clamped to corpus size
    """
    validated_query = validate_query(query)
    validated_corpus = validate_corpus(corpus)
    validated_k = validate_k_parameter(k, corpus_size=len(validated_corpus))

    return validated_query, validated_corpus, validated_k
