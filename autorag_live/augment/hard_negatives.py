from typing import List, Set


def sample_hard_negatives(
    positive_results: List[str], negative_pool: List[List[str]], num_negatives: int = 5
) -> List[str]:
    """
    Samples hard negatives from a pool of negative results.
    Hard negatives are defined as documents that appear in the negative pool
    but not in the positive results.
    """
    positive_set: Set[str] = set(positive_results)

    hard_negatives: Set[str] = set()
    for result_list in negative_pool:
        for doc in result_list:
            if doc not in positive_set:
                hard_negatives.add(doc)

    return list(hard_negatives)[:num_negatives]
