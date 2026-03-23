"""Vectorized document filtering for efficient corpus preprocessing."""

import re
from typing import Callable, List


def filter_documents_by_length(
    documents: List[str],
    min_length: int = 10,
    max_length: int = 10000,
) -> tuple[List[str], List[int]]:
    """
    Filter documents by character length.

    Args:
        documents: List of documents
        min_length: Minimum document length
        max_length: Maximum document length

    Returns:
        Tuple of (filtered_documents, kept_indices)
    """
    filtered = []
    indices = []
    for i, doc in enumerate(documents):
        if min_length <= len(doc) <= max_length:
            filtered.append(doc)
            indices.append(i)
    return filtered, indices


def filter_documents_by_pattern(
    documents: List[str],
    pattern: str,
    keep_matching: bool = True,
) -> tuple[List[str], List[int]]:
    """
    Filter documents by regex pattern.

    Args:
        documents: List of documents
        pattern: Regex pattern to match
        keep_matching: If True, keep docs matching pattern; else discard

    Returns:
        Tuple of (filtered_documents, kept_indices)
    """
    regex = re.compile(pattern)
    filtered = []
    indices = []
    for i, doc in enumerate(documents):
        match = bool(regex.search(doc))
        if match == keep_matching:
            filtered.append(doc)
            indices.append(i)
    return filtered, indices


def filter_documents_by_predicate(
    documents: List[str],
    predicate: Callable[[str], bool],
) -> tuple[List[str], List[int]]:
    """
    Filter documents using a custom predicate function.

    Args:
        documents: List of documents
        predicate: Function that returns True for docs to keep

    Returns:
        Tuple of (filtered_documents, kept_indices)
    """
    filtered = []
    indices = []
    for i, doc in enumerate(documents):
        if predicate(doc):
            filtered.append(doc)
            indices.append(i)
    return filtered, indices


def remove_empty_documents(documents: List[str]) -> tuple[List[str], List[int]]:
    """
    Remove empty or whitespace-only documents.

    Args:
        documents: List of documents

    Returns:
        Tuple of (non_empty_documents, kept_indices)
    """
    filtered = []
    indices = []
    for i, doc in enumerate(documents):
        if doc.strip():
            filtered.append(doc)
            indices.append(i)
    return filtered, indices


def remove_duplicate_documents(documents: List[str]) -> tuple[List[str], List[int]]:
    """
    Remove exact duplicate documents.

    Args:
        documents: List of documents

    Returns:
        Tuple of (unique_documents, kept_indices)
    """
    seen = set()
    unique_docs = []
    unique_indices = []

    for idx, doc in enumerate(documents):
        if doc not in seen:
            seen.add(doc)
            unique_docs.append(doc)
            unique_indices.append(idx)

    return unique_docs, unique_indices
