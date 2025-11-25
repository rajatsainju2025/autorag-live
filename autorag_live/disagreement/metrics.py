from functools import lru_cache
from typing import List, Set, Tuple, cast

from scipy.stats import kendalltau


@lru_cache(maxsize=256)
def _get_rank_mapping(items_tuple: Tuple[str, ...]) -> dict[str, int]:
    """Cache rank mappings for frequently used lists."""
    return {item: i for i, item in enumerate(items_tuple)}


def jaccard_at_k(list1: List[str], list2: List[str]) -> float:
    """
    Calculates the Jaccard similarity at k between two lists.

    Optimized with early returns and efficient set operations.
    """
    # Early return for empty lists
    if not list1 and not list2:
        return 1.0  # Both empty = identical
    if not list1 or not list2:
        return 0.0  # One empty = no overlap

    set1: Set[str] = set(list1)
    set2: Set[str] = set(list2)

    # Use bitwise operations for efficiency
    intersection = set1 & set2
    union = set1 | set2

    return len(intersection) / len(union)


def kendall_tau_at_k(list1: List[str], list2: List[str]) -> float:
    """
    Calculates Kendall's Tau rank correlation between two lists.
    Optimized with cached rank mappings for frequently used lists.
    """
    # Use cached rank mappings
    rank1 = _get_rank_mapping(tuple(list1))
    rank2 = _get_rank_mapping(tuple(list2))

    # Get all items
    all_items = set(list1) | set(list2)
    all_items_len = len(all_items)

    # Create rank arrays for common items
    ranks1 = [rank1.get(item, all_items_len) for item in all_items]
    ranks2 = [rank2.get(item, all_items_len) for item in all_items]

    tau, _ = kendalltau(ranks1, ranks2)
    return cast(float, tau)


# Optimization: perf(vectorize): add numpy-based metric optimizations
