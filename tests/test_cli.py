import os
import subprocess
import sys


def test_cli_module_imports():
    """Test that CLI module can be imported and basic functions work."""
    from autorag_live.cli import app
    from autorag_live.evals.small import run_small_suite
    
    # Test evaluation function directly
    result = run_small_suite("test_runs")
    assert "metrics" in result
    assert "f1" in result["metrics"]
    
    # Clean up
    if os.path.exists("test_runs"):
        import shutil
        shutil.rmtree("test_runs")


def test_cli_eval_subprocess():
    """Test CLI eval command via subprocess."""
    result = subprocess.run([
        sys.executable, "-m", "autorag_live.cli", "eval", "--suite", "small"
    ], capture_output=True, text=True, cwd=os.getcwd())
    
    # Should complete without error
    assert result.returncode == 0
    assert "EM=" in result.stdout or "F1=" in result.stdout
