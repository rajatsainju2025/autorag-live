from typing import List, Set, cast
from scipy.stats import kendalltau

def jaccard_at_k(list1: List[str], list2: List[str]) -> float:
    """
    Calculates the Jaccard similarity at k between two lists.
    """
    set1: Set[str] = set(list1)
    set2: Set[str] = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

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

