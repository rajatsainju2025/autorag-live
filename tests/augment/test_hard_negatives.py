from autorag_live.augment import hard_negatives

def test_sample_hard_negatives():
    positive_results = ["a", "b", "c"]
    negative_pool = [
        ["b", "c", "d"],
        ["c", "e", "f"],
    ]
    
    negatives = hard_negatives.sample_hard_negatives(positive_results, negative_pool)
    
    assert set(negatives) == {"d", "e", "f"}
    
    negatives_with_limit = hard_negatives.sample_hard_negatives(positive_results, negative_pool, num_negatives=2)
    
    assert len(negatives_with_limit) == 2
    assert set(negatives_with_limit).issubset({"d", "e", "f"})
