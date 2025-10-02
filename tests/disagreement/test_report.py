import os

from autorag_live.disagreement import report


def test_generate_disagreement_report():
    query = "test query"
    results = {
        "bm25": ["doc a", "doc b"],
        "dense": ["doc b", "doc c"],
    }
    metrics = {
        "jaccard_bm25_vs_dense": 0.3333,
        "kendall_tau_bm25_vs_dense": -1.0,
    }
    output_path = "test_report.html"

    report.generate_disagreement_report(query, results, metrics, output_path)

    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        content = f.read()
        assert "Disagreement Report" in content
        assert "test query" in content
        assert "jaccard_bm25_vs_dense" in content
        assert "0.3333" in content
        assert "bm25" in content
        assert "doc a" in content

    os.remove(output_path)
