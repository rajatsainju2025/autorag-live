
from autorag_live.pipeline.hybrid_optimizer import (
    HybridWeights,
    grid_search_hybrid_weights,
    hybrid_retrieve_weighted,
    load_hybrid_config,
    save_hybrid_config,
)


def test_hybrid_weights():
    w = HybridWeights(0.7, 0.3)
    normalized = w.normalize()
    assert abs(normalized.bm25_weight + normalized.dense_weight - 1.0) < 1e-6


def test_hybrid_retrieve_weighted():
    corpus = ["the bright sun", "blue sky", "clear weather", "sunny day"]
    query = "bright sun"
    weights = HybridWeights(0.8, 0.2)

    results = hybrid_retrieve_weighted(query, corpus, k=2, weights=weights)
    assert len(results) <= 2
    assert all(doc in corpus for doc in results)


def test_grid_search_hybrid_weights():
    corpus = ["the bright sun", "blue sky", "clear weather", "sunny day"]
    queries = ["bright sun", "blue sky"]

    best_weights, score = grid_search_hybrid_weights(queries, corpus, k=2, grid_size=3)

    assert isinstance(best_weights, HybridWeights)
    assert 0.0 <= best_weights.bm25_weight <= 1.0
    assert 0.0 <= best_weights.dense_weight <= 1.0
    assert score >= 0.0


def test_config_save_load(tmp_path):
    weights = HybridWeights(0.6, 0.4)
    config_path = tmp_path / "test_config.json"

    save_hybrid_config(weights, str(config_path))
    loaded = load_hybrid_config(str(config_path))

    assert abs(loaded.bm25_weight - 0.6) < 1e-6
    assert abs(loaded.dense_weight - 0.4) < 1e-6
