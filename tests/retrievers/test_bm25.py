from autorag_live.retrievers import bm25

def test_bm25_retrieve():
    corpus = ["hello world", "this is a test", "another document"]
    query = "hello"
    results = bm25.bm25_retrieve(query, corpus, k=2)
    assert len(results) == 2
    assert "hello world" in results
