from autorag_live.retrievers import hybrid

def test_hybrid_retrieve():
    corpus = ["hello world", "this is a test", "another document", "fourth doc"]
    query = "hello"
    results = hybrid.hybrid_retrieve(query, corpus, k=3)
    assert len(results) == 3
