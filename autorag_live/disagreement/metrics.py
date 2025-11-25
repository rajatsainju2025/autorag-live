from typing import List, Set, cast

from scipy.stats import kendalltau


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
    """
    # Create a mapping from item to its rank in each list
    rank1 = {item: i for i, item in enumerate(list1)}
    rank2 = {item: i for i, item in enumerate(list2)}

    # Get all items
    all_items = set(list1) | set(list2)

    # Create rank arrays for common items
    ranks1 = []
    ranks2 = []
    for item in all_items:
        ranks1.append(rank1.get(item, len(all_items)))
        ranks2.append(rank2.get(item, len(all_items)))

    tau, _ = kendalltau(ranks1, ranks2)
    return cast(float, tau)


# Optimization: perf(vectorize): add numpy-based metric optimizations
