from autorag_live.augment import query_rewrites

def test_rewrite_query():
    query = "test query"
    rewrites = query_rewrites.rewrite_query(query, num_rewrites=3)
    
    assert len(rewrites) == 3
    assert "test query (rewrite 1)" in rewrites
    assert "test query (rewrite 2)" in rewrites
    assert "test query (rewrite 3)" in rewrites
