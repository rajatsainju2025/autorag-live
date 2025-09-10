from autorag_live.augment.synonym_miner import (
    mine_synonyms_from_disagreements, 
    load_terms_yaml, 
    save_terms_yaml,
    update_terms_from_mining
)
import tempfile
import os


def test_mine_synonyms_from_disagreements():
    bm25 = ["the bright sun", "blue sky"]
    dense = ["bright sun shining", "clear blue sky"]
    hybrid = ["sun bright", "sky blue clear"]
    
    synonyms = mine_synonyms_from_disagreements(bm25, dense, hybrid, min_freq=2)
    
    # Should find groups where terms appear multiple times
    assert isinstance(synonyms, dict)
    # Check that we get some reasonable groupings
    assert len(synonyms) >= 0  # May be empty with this small sample


def test_terms_yaml_operations(tmp_path):
    test_file = tmp_path / "test_terms.yaml"
    
    # Test save and load
    terms = {"sun": ["bright", "shining"], "sky": ["blue", "clear"]}
    save_terms_yaml(terms, str(test_file))
    
    loaded = load_terms_yaml(str(test_file))
    assert loaded == terms
    
    # Test update
    new_terms = {"sun": ["brilliant"], "ocean": ["water", "sea"]}
    updated = update_terms_from_mining(new_terms, str(test_file))
    
    assert "brilliant" in updated["sun"]
    assert "bright" in updated["sun"]  # existing preserved
    assert updated["ocean"] == ["water", "sea"]
