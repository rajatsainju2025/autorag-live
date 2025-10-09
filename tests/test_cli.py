import os
import subprocess
import sys


def test_cli_module_imports(tmp_path):
    """Test that CLI module can be imported and basic functions work."""
    from autorag_live.evals.small import run_small_suite

    # Test evaluation function directly
    result = run_small_suite(str(tmp_path / "test_runs"))
    assert "metrics" in result
    assert "f1" in result["metrics"]


def test_cli_eval_subprocess(tmp_path):
    """Test CLI eval command via subprocess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "/Users/rsainju/Research/1_Research_Personal/autorag-live"

    result = subprocess.run(
        [sys.executable, "-m", "autorag_live.cli", "eval", "--suite", "small"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )

    # Should complete without error
    assert result.returncode == 0
    assert "EM=" in result.stdout or "F1=" in result.stdout
