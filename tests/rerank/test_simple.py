from autorag_live.rerank.simple import SimpleReranker


def test_simple_reranker_scores_and_reranks():
    rr = SimpleReranker(seed=123, lr=0.1, epochs=3)
    q = "bright sun in the sky"
    pos = "the sun in the sky is bright"
    negs = ["the quick brown fox jumps", "lazy dog sleeps"]

    # train on a single pair deterministically
    rr.fit([{"query": q, "positive": pos, "negatives": negs}])

    docs = [pos] + negs
    reranked = rr.rerank(q, docs)

    # positive should appear first after training
    assert reranked[0] == pos

    # score should increase for pos vs negs
    sp = rr.score(q, pos)
    sn = rr.score(q, negs[0])
    assert sp > sn
