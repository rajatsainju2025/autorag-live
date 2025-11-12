"""Vectorized document filtering for efficient corpus preprocessing."""

import re
from typing import Callable, List

import numpy as np


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
    lengths = np.array([len(doc) for doc in documents])
    mask = (lengths >= min_length) & (lengths <= max_length)
    indices = np.where(mask)[0]

    filtered = [documents[i] for i in indices]
    return filtered, indices.tolist()


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
    mask = np.array([bool(regex.search(doc)) for doc in documents])

    if not keep_matching:
        mask = ~mask

    indices = np.where(mask)[0]
    filtered = [documents[i] for i in indices]
    return filtered, indices.tolist()


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
    mask = np.array([predicate(doc) for doc in documents])
    indices = np.where(mask)[0]
    filtered = [documents[i] for i in indices]
    return filtered, indices.tolist()


def remove_empty_documents(documents: List[str]) -> tuple[List[str], List[int]]:
    """
    Remove empty or whitespace-only documents.

    Args:
        documents: List of documents

    Returns:
        Tuple of (non_empty_documents, kept_indices)
    """
    mask = np.array([bool(doc.strip()) for doc in documents])
    indices = np.where(mask)[0]
    filtered = [documents[i] for i in indices]
    return filtered, indices.tolist()


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
