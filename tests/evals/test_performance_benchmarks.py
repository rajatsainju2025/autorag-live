"""Tests for performance benchmarks."""

import json
import os
import tempfile
from pathlib import Path

from autorag_live.evals.performance_benchmarks import (
    BenchmarkResult,
    PerformanceBenchmark,
    compare_benchmark_runs,
    run_full_benchmark_suite,
)


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult."""
        result = BenchmarkResult(
            operation="test_op",
            iterations=10,
            total_time=1.0,
            avg_time=0.1,
            min_time=0.08,
            max_time=0.12,
            std_time=0.02,
            memory_usage_mb=5.5,
            throughput=10.0,
            metadata={"test": "data"},
        )

        assert result.operation == "test_op"
        assert result.iterations == 10
        assert result.total_time == 1.0
        assert result.avg_time == 0.1
        assert result.min_time == 0.08
        assert result.max_time == 0.12
        assert result.std_time == 0.02
        assert result.memory_usage_mb == 5.5
        assert result.throughput == 10.0
        assert result.metadata == {"test": "data"}


class TestPerformanceBenchmark:
    """Test PerformanceBenchmark class."""

    def test_initialization(self):
        """Test benchmark initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = PerformanceBenchmark(tmpdir)
            assert benchmark.output_dir == Path(tmpdir)
            assert benchmark.results == []

    def test_benchmark_function(self):
        """Test benchmarking a simple function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = PerformanceBenchmark(tmpdir)

            def simple_func(x, y=10):
                return x + y

            result = benchmark.benchmark_function(
                simple_func, 5, y=15, iterations=5, operation_name="test_add"
            )

            assert result.operation == "test_add"
            assert result.iterations == 5
            assert result.total_time > 0
            assert result.avg_time > 0
            assert result.throughput > 0
            assert result.metadata["args_count"] == 1
            assert result.metadata["kwargs_count"] == 1

    def test_save_results(self):
        """Test saving benchmark results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = PerformanceBenchmark(tmpdir)

            def dummy_func():
                return 42

            benchmark.benchmark_function(dummy_func, iterations=3)

            filepath = benchmark.save_results("test_results.json")
            assert filepath.exists()

            # Verify JSON structure
            with open(filepath, "r") as f:
                data = json.load(f)

            assert "timestamp" in data
            assert "results" in data
            assert len(data["results"]) == 1
            assert data["results"][0]["operation"] == "dummy_func"

    def test_print_summary(self, capsys):
        """Test printing benchmark summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = PerformanceBenchmark(tmpdir)

            def dummy_func():
                return 42

            benchmark.benchmark_function(dummy_func, iterations=3)

            benchmark.print_summary()
            captured = capsys.readouterr()

            assert "PERFORMANCE BENCHMARK SUMMARY" in captured.out
            assert "dummy_func" in captured.out


class TestBenchmarkSuite:
    """Test the full benchmark suite."""

    def test_run_full_benchmark_suite(self):
        """Test running the full benchmark suite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)  # Change to temp dir to avoid cluttering main dir

            results = run_full_benchmark_suite("test_full_suite.json")

            assert len(results) > 0
            assert all(isinstance(r, BenchmarkResult) for r in results)

            # Check that results file was created
            results_file = Path("benchmarks/test_full_suite.json")
            assert results_file.exists()

    def test_compare_benchmark_runs(self):
        """Test comparing benchmark runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            # Create two mock benchmark files
            run1_data = {
                "timestamp": 1000000,
                "results": [
                    {
                        "operation": "test_op",
                        "iterations": 10,
                        "total_time": 1.0,
                        "avg_time": 0.1,
                        "min_time": 0.08,
                        "max_time": 0.12,
                        "std_time": 0.02,
                        "memory_usage_mb": 5.0,
                        "throughput": 10.0,
                        "metadata": {},
                    }
                ],
            }

            run2_data = {
                "timestamp": 1000001,
                "results": [
                    {
                        "operation": "test_op",
                        "iterations": 10,
                        "total_time": 0.8,
                        "avg_time": 0.08,
                        "min_time": 0.07,
                        "max_time": 0.09,
                        "std_time": 0.01,
                        "memory_usage_mb": 4.5,
                        "throughput": 12.5,
                        "metadata": {},
                    }
                ],
            }

            run1_file = Path("run1.json")
            run2_file = Path("run2.json")

            with open(run1_file, "w") as f:
                json.dump(run1_data, f)

            with open(run2_file, "w") as f:
                json.dump(run2_data, f)

            # This should not raise an exception
            compare_benchmark_runs(str(run1_file), str(run2_file))


class TestBenchmarkOptimizations:
    """Test performance optimizations."""

    def test_memory_monitor_context_manager(self):
        """Test the memory monitor context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = PerformanceBenchmark(tmpdir)

            with benchmark.memory_monitor():
                # Simulate some memory usage
                test_list = [i for i in range(1000)]
                _ = test_list * 10

            # Memory usage should be tracked
            assert hasattr(benchmark, "current_memory_usage")
            assert isinstance(benchmark.current_memory_usage, (int, float))

    def test_warmup_iterations(self):
        """Test that warmup iterations are properly handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark = PerformanceBenchmark(tmpdir)

            call_count = 0

            def counting_func():
                nonlocal call_count
                call_count += 1
                return call_count

            result = benchmark.benchmark_function(counting_func, iterations=5, warmup_iterations=3)

            # Should be called iterations + warmup times
            assert call_count == 8  # 5 + 3
            assert result.iterations == 5
