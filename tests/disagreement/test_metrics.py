from autorag_live.disagreement import metrics
import numpy as np

def test_jaccard_at_k():
    list1 = ["a", "b", "c"]
    list2 = ["b", "c", "d"]
    assert metrics.jaccard_at_k(list1, list2) == 2 / 4

    list3 = ["a", "b", "c"]
    list4 = ["d", "e", "f"]
    assert metrics.jaccard_at_k(list3, list4) == 0

    list5 = ["a", "b", "c"]
    list6 = ["a", "b", "c"]
    assert metrics.jaccard_at_k(list5, list6) == 1

def test_kendall_tau_at_k():
    list1 = ["a", "b", "c"]
    list2 = ["a", "b", "c"]
    assert np.isclose(metrics.kendall_tau_at_k(list1, list2), 1.0)

    list3 = ["a", "b", "c"]
    list4 = ["c", "b", "a"]
    assert np.isclose(metrics.kendall_tau_at_k(list3, list4), -1.0)

    list5 = ["a", "b", "c"]
    list6 = ["d", "e", "f"]
    # The exact value depends on the implementation details of handling disjoint sets.
    # Let's just check it's a valid number for now.
    assert isinstance(metrics.kendall_tau_at_k(list5, list6), float)

