"""Fast-path evaluation metrics for simple cases with early exit."""

from typing import List


def hit_rate_fast(
    retrieved_docs: List[str],
    relevant_docs: List[str],
    k: int = 10,
) -> float:
    """
    Fast hit rate computation with early exit for all hits found.

    Args:
        retrieved_docs: Retrieved documents
        relevant_docs: Relevant documents
        k: Consider top-k results

    Returns:
        Hit rate (1.0 if any relevant found, 0.0 otherwise)
    """
    # Early exit: if no relevant docs, return 0
    if not relevant_docs:
        return 0.0

    # Fast path: convert to set for O(1) lookup
    relevant_set = set(relevant_docs)
    top_k_docs = retrieved_docs[:k]

    # Check if any retrieved doc is relevant
    for doc in top_k_docs:
        if doc in relevant_set:
            return 1.0

    return 0.0


def precision_at_k_fast(
    retrieved_docs: List[str],
    relevant_docs: List[str],
    k: int = 10,
) -> float:
    """
    Fast precision@k computation optimized for common cases.

    Args:
        retrieved_docs: Retrieved documents
        relevant_docs: Relevant documents
        k: Compute precision at k

    Returns:
        Precision@k score
    """
    # Early exit for edge cases
    if not retrieved_docs or not relevant_docs:
        return 0.0

    k = min(k, len(retrieved_docs))

    # Convert to set for fast lookup
    relevant_set = set(relevant_docs)

    # Count relevant in top-k
    relevant_count = sum(1 for doc in retrieved_docs[:k] if doc in relevant_set)

    return relevant_count / k if k > 0 else 0.0


def recall_at_k_fast(
    retrieved_docs: List[str],
    relevant_docs: List[str],
    k: int = 10,
) -> float:
    """
    Fast recall@k computation with early exit if not enough results.

    Args:
        retrieved_docs: Retrieved documents
        relevant_docs: Relevant documents
        k: Compute recall at k

    Returns:
        Recall@k score
    """
    # Early exit for edge cases
    if not relevant_docs:
        return 0.0

    if not retrieved_docs:
        return 0.0

    # Convert to set for fast lookup
    relevant_set = set(relevant_docs)

    # Check only top-k
    top_k_docs = retrieved_docs[:k]

    # Count how many relevant docs are in top-k
    relevant_count = sum(1 for doc in top_k_docs if doc in relevant_set)

    return relevant_count / len(relevant_docs)


def f1_score_fast(
    retrieved_docs: List[str],
    relevant_docs: List[str],
    k: int = 10,
) -> float:
    """
    Fast F1 score computation combining precision and recall.
    Optimized to compute both metrics in a single pass.

    Args:
        retrieved_docs: Retrieved documents
        relevant_docs: Relevant documents
        k: Compute F1@k

    Returns:
        F1 score
    """
    # Early exit for edge cases
    if not retrieved_docs or not relevant_docs:
        return 0.0

    k = min(k, len(retrieved_docs))
    if k == 0:
        return 0.0

    # Single set conversion and pass
    relevant_set = set(relevant_docs)
    top_k_docs = retrieved_docs[:k]

    # Count relevant in single pass
    relevant_count = sum(1 for doc in top_k_docs if doc in relevant_set)

    if relevant_count == 0:
        return 0.0

    # Compute precision and recall
    precision = relevant_count / k
    recall = relevant_count / len(relevant_docs)

    return 2 * (precision * recall) / (precision + recall)
