from autorag_live.evals.small import exact_match, token_f1, run_small_suite


def test_exact_match_and_f1():
    assert exact_match("Blue", "blue") == 1.0
    assert exact_match("red", "blue") == 0.0

    assert token_f1("blue", "blue") == 1.0
    assert token_f1("blue", "red") == 0.0
    assert 0.0 <= token_f1("blue sky", "blue") <= 1.0


def test_run_small_suite(tmp_path):
    # override runs dir
    summary = run_small_suite(runs_dir=str(tmp_path))
    assert "metrics" in summary
    assert 0.0 <= summary["metrics"]["em"] <= 1.0
    assert 0.0 <= summary["metrics"]["f1"] <= 1.0
