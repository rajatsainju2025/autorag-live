from autorag_live.retrievers import dense


def test_dense_retrieve():
    corpus = ["hello world", "this is a test", "another document"]
    query = "hello"
    results = dense.dense_retrieve(query, corpus, k=2)
    assert len(results) == 2
