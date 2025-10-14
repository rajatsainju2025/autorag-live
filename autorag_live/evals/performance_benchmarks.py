"""Performance benchmarks for autorag-live components."""

import asyncio
import cProfile
import json
import os
import pstats
import statistics
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import psutil

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    operation: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_time: float
    memory_usage_mb: float
    throughput: float  # operations per second
    metadata: Dict[str, Any]
    timestamp: Optional[datetime] = None
    cpu_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ProfilingResult:
    """Result of profiling a function."""

    operation: str
    profile_stats: pstats.Stats
    total_calls: int
    primitive_calls: int
    total_time: float
    metadata: Dict[str, Any]


class PerformanceBenchmark:
    """Performance benchmarking suite for autorag-live components."""

    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.profiler = cProfile.Profile()
        self.tracemalloc_enabled = False

    @contextmanager
    def memory_monitor(self):
        """Context manager to monitor memory usage."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        max_memory = initial_memory
        peak_memory = initial_memory

        # Track CPU usage
        initial_cpu = psutil.cpu_percent(interval=None)

        # GPU memory tracking
        initial_gpu_memory = 0.0
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB

        yield

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        max_memory = max(max_memory, final_memory)
        peak_memory = max(peak_memory, final_memory)

        final_cpu = psutil.cpu_percent(interval=None)

        final_gpu_memory = 0.0
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB

        self.current_memory_usage = max_memory - initial_memory
        self.peak_memory_usage = peak_memory
        self.cpu_usage = final_cpu - initial_cpu
        self.gpu_memory_usage = final_gpu_memory - initial_gpu_memory

    @contextmanager
    def profiling(self):
        """Context manager for profiling."""
        self.profiler.enable()
        try:
            yield
        finally:
            self.profiler.disable()

    def get_profile_stats(self, operation: str) -> ProfilingResult:
        """Get profiling statistics."""
        stats = pstats.Stats(self.profiler)
        stats.sort_stats("cumulative")

        # Extract summary statistics
        total_calls = 0
        primitive_calls = 0
        total_time = 0.0
        try:
            # Try to access stats if available
            stats_dict = getattr(stats, "stats", {})
            if stats_dict:
                total_calls = sum(stat[0] for stat in stats_dict.values())
                primitive_calls = sum(stat[1] for stat in stats_dict.values())
                total_time = sum(stat[2] for stat in stats_dict.values())
        except Exception:
            pass  # Use defaults if stats access fails

        return ProfilingResult(
            operation=operation,
            profile_stats=stats,
            total_calls=total_calls,
            primitive_calls=primitive_calls,
            total_time=total_time,
            metadata={"profiler": "cProfile"},
        )

    def benchmark_function(
        self,
        func: Callable,
        *args,
        iterations: int = 10,
        warmup_iterations: int = 2,
        operation_name: Optional[str] = None,
        enable_profiling: bool = False,
        enable_tracemalloc: bool = False,
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark a function's performance."""

        if operation_name is None:
            operation_name = func.__name__

        # Setup profiling and memory tracking
        if enable_tracemalloc:
            tracemalloc.start()
            self.tracemalloc_enabled = True

        # Warmup
        for _ in range(warmup_iterations):
            func(*args, **kwargs)

        # Clear caches between warmup and benchmark
        if hasattr(func, "__wrapped__"):  # Check if it's a cached function
            # This is a simple heuristic - in practice you'd need more sophisticated cache clearing
            pass

        # Benchmark
        times = []
        self.current_memory_usage = 0
        self.peak_memory_usage = 0
        self.cpu_usage = 0
        self.gpu_memory_usage = 0

        if enable_profiling:
            self.profiler.clear()

        with self.memory_monitor():
            if enable_profiling:
                with self.profiling():
                    for _ in range(iterations):
                        start_time = time.perf_counter()
                        func(*args, **kwargs)
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
            else:
                for _ in range(iterations):
                    start_time = time.perf_counter()
                    func(*args, **kwargs)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)

        if enable_profiling:
            self.get_profile_stats(operation_name)

        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        throughput = iterations / total_time if total_time > 0 else 0

        # Memory statistics from tracemalloc if enabled
        tracemalloc_stats = None
        if enable_tracemalloc:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_stats = {"current_mb": current / 1024 / 1024, "peak_mb": peak / 1024 / 1024}
            tracemalloc.stop()
            self.tracemalloc_enabled = False

        benchmark_result = BenchmarkResult(
            operation=operation_name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_time=std_time,
            memory_usage_mb=self.current_memory_usage,
            throughput=throughput,
            cpu_percent=self.cpu_usage,
            gpu_memory_mb=self.gpu_memory_usage,
            peak_memory_mb=self.peak_memory_usage,
            metadata={
                "warmup_iterations": warmup_iterations,
                "function_args": len(args),
                "function_kwargs": list(kwargs.keys()),
                "args_count": len(args),  # Backward compatibility
                "kwargs_count": len(kwargs),  # Backward compatibility
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "platform": sys.platform,
                "cpu_count": os.cpu_count(),
                "tracemalloc_stats": tracemalloc_stats,
                "profiling_enabled": enable_profiling,
                "torch_available": TORCH_AVAILABLE,
                "cuda_available": TORCH_AVAILABLE
                and torch is not None
                and torch.cuda.is_available()
                if TORCH_AVAILABLE
                else False,
            },
        )

        self.results.append(benchmark_result)
        return benchmark_result

    async def benchmark_async_function(
        self,
        func: Callable,
        *args,
        iterations: int = 10,
        warmup_iterations: int = 2,
        operation_name: Optional[str] = None,
        concurrent: bool = False,
        concurrency: int = 4,
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark an async function's performance."""

        if operation_name is None:
            operation_name = func.__name__

        # Warmup
        for _ in range(warmup_iterations):
            await func(*args, **kwargs)

        # Benchmark
        times = []
        self.current_memory_usage = 0

        with self.memory_monitor():
            if concurrent:
                # Run multiple concurrent executions
                for _ in range(iterations // concurrency):
                    start_time = time.perf_counter()
                    tasks = [func(*args, **kwargs) for _ in range(concurrency)]
                    await asyncio.gather(*tasks)
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) / concurrency)
            else:
                # Sequential execution
                for _ in range(iterations):
                    start_time = time.perf_counter()
                    await func(*args, **kwargs)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)

        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        throughput = iterations / total_time if total_time > 0 else 0

        benchmark_result = BenchmarkResult(
            operation=operation_name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_time=std_time,
            memory_usage_mb=self.current_memory_usage,
            throughput=throughput,
            metadata={
                "async": True,
                "concurrent": concurrent,
                "concurrency": concurrency if concurrent else 1,
                "warmup_iterations": warmup_iterations,
            },
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def benchmark_gpu_function(
        self,
        func: Callable,
        *args,
        iterations: int = 10,
        warmup_iterations: int = 2,
        operation_name: Optional[str] = None,
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark a GPU function's performance."""

        if not (TORCH_AVAILABLE and torch is not None and torch.cuda.is_available()):
            raise RuntimeError("CUDA not available for GPU benchmarking")

        if operation_name is None:
            operation_name = func.__name__

        # Warmup on GPU
        for _ in range(warmup_iterations):
            func(*args, **kwargs)
            if TORCH_AVAILABLE and torch is not None:
                torch.cuda.synchronize()  # Wait for GPU operations to complete

        # Benchmark
        times = []
        self.current_memory_usage = 0
        self.gpu_memory_usage = 0

        with self.memory_monitor():
            for _ in range(iterations):
                start_time = time.perf_counter()
                func(*args, **kwargs)
                if TORCH_AVAILABLE and torch is not None:
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append(end_time - start_time)

        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        throughput = iterations / total_time if total_time > 0 else 0

        benchmark_result = BenchmarkResult(
            operation=operation_name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_time=std_time,
            memory_usage_mb=self.current_memory_usage,
            throughput=throughput,
            gpu_memory_mb=self.gpu_memory_usage,
            metadata={
                "gpu_enabled": True,
                "cuda_version": getattr(torch, "version", {}).get("cuda", None)
                if TORCH_AVAILABLE and torch is not None
                else None,
                "gpu_name": torch.cuda.get_device_name()
                if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available()
                else None,
                "warmup_iterations": warmup_iterations,
            },
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def save_results(self, filename: Optional[str] = None) -> str:
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        filepath = self.output_dir / filename

        # Convert results to dictionaries
        results_dict = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
                "platform": sys.platform,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
            "results": [asdict(result) for result in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

        return str(filepath)

    def save_profile_stats(self, operation: str, filename: Optional[str] = None) -> str:
        """Save profiling statistics to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"profile_{operation}_{timestamp}.txt"

        filepath = self.output_dir / filename

        profile_result = self.get_profile_stats(operation)
        with open(filepath, "w") as f:
            f.write(f"Profiling results for {operation}\n")
            f.write(f"Total calls: {profile_result.total_calls}\n")
            f.write(f"Primitive calls: {profile_result.primitive_calls}\n")
            f.write(f"Total time: {profile_result.total_time:.4f}s\n\n")

            f.write("Profile statistics summary:\n")
            f.write("(Full stats require manual inspection)\n")

        return str(filepath)

    def print_summary(self, detailed: bool = False) -> None:
        """Print benchmark summary."""
        if not self.results:
            print("No benchmark results to display.")
            return

        print(f"\n{'='*60}")
        print(f"Performance Benchmark Summary ({len(self.results)} operations)")
        print(f"{'='*60}")

        for result in self.results:
            print(f"\nOperation: {result.operation}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Avg Time: {result.avg_time:.4f}s")
            print(f"  Min/Max Time: {result.min_time:.4f}s / {result.max_time:.4f}s")
            print(f"  Std Dev: {result.std_time:.4f}s")
            print(f"  Throughput: {result.throughput:.2f} ops/sec")
            print(f"  Memory Usage: {result.memory_usage_mb:.2f} MB")
            print(f"  Peak Memory: {result.peak_memory_mb:.2f} MB")

            if result.cpu_percent > 0:
                print(f"  CPU Usage: {result.cpu_percent:.1f}%")
            if result.gpu_memory_mb > 0:
                print(f"  GPU Memory: {result.gpu_memory_mb:.2f} MB")

            if detailed and result.metadata:
                print(f"  Metadata: {json.dumps(result.metadata, indent=2, default=str)}")

        print(f"\n{'='*60}")


class ComparativeBenchmark:
    """Compare performance across different configurations."""

    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_results: Dict[str, BenchmarkResult] = {}
        self.comparison_results: Dict[str, BenchmarkResult] = {}

    def set_baseline(self, results: Dict[str, BenchmarkResult]) -> None:
        """Set baseline results for comparison."""
        self.baseline_results = results

    def compare_with_baseline(
        self, results: Dict[str, BenchmarkResult]
    ) -> Dict[str, Dict[str, float]]:
        """Compare results with baseline."""
        self.comparison_results = results
        comparisons = {}

        for operation in set(self.baseline_results.keys()) | set(results.keys()):
            if operation in self.baseline_results and operation in results:
                baseline = self.baseline_results[operation]
                current = results[operation]

                time_diff = current.avg_time - baseline.avg_time
                time_percent = (time_diff / baseline.avg_time) * 100 if baseline.avg_time > 0 else 0

                throughput_diff = current.throughput - baseline.throughput
                throughput_percent = (
                    (throughput_diff / baseline.throughput) * 100 if baseline.throughput > 0 else 0
                )

                memory_diff = current.memory_usage_mb - baseline.memory_usage_mb
                memory_percent = (
                    (memory_diff / baseline.memory_usage_mb) * 100
                    if baseline.memory_usage_mb > 0
                    else 0
                )

                comparisons[operation] = {
                    "time_diff": time_diff,
                    "time_percent": time_percent,
                    "throughput_diff": throughput_diff,
                    "throughput_percent": throughput_percent,
                    "memory_diff": memory_diff,
                    "memory_percent": memory_percent,
                }

        return comparisons

    def print_comparison_report(self) -> None:
        """Print detailed comparison report."""
        if not self.baseline_results or not self.comparison_results:
            print("Need both baseline and comparison results to generate report.")
            return

        comparisons = self.compare_with_baseline(self.comparison_results)

        print(f"\n{'='*80}")
        print("Performance Comparison Report")
        print(f"{'='*80}")

        for operation, metrics in comparisons.items():
            print(f"\nOperation: {operation}")
            print(f"  Time Change: {metrics['time_diff']:+.4f}s ({metrics['time_percent']:+.1f}%)")
            print(
                f"  Throughput Change: {metrics['throughput_diff']:+.2f} ops/sec ({metrics['throughput_percent']:+.1f}%)"
            )
            print(
                f"  Memory Change: {metrics['memory_diff']:+.2f} MB ({metrics['memory_percent']:+.1f}%)"
            )

        print(f"\n{'='*80}")


def run_full_benchmark_suite(
    filename: str = "full_benchmark_suite.json", output_dir: str = "benchmarks"
) -> Dict[str, BenchmarkResult]:
    """Run comprehensive benchmark suite."""
    from autorag_live.evals.small import run_small_suite
    from autorag_live.retrievers import bm25, dense

    benchmark = PerformanceBenchmark(output_dir)

    # Sample data for benchmarking
    corpus = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing handles human language",
        "Computer vision enables machines to interpret visual information",
        "Reinforcement learning learns through trial and error",
    ] * 20  # Make it larger for meaningful benchmarks

    query = "artificial intelligence and machine learning"

    # Benchmark retrievers
    print("Benchmarking BM25 retriever...")
    benchmark.benchmark_function(
        bm25.bm25_retrieve, query, corpus, k=5, operation_name="bm25_retrieve"
    )

    print("Benchmarking dense retriever...")
    try:
        benchmark.benchmark_function(
            dense.dense_retrieve, query, corpus, k=5, operation_name="dense_retrieve"
        )
    except Exception as e:
        print(f"Dense retriever benchmark failed: {e}")

    # Benchmark evaluation
    print("Benchmarking evaluation suite...")
    try:
        benchmark.benchmark_function(run_small_suite, operation_name="small_evaluation_suite")
    except Exception as e:
        print(f"Evaluation benchmark failed: {e}")

    # Save results
    results_file = benchmark.save_results(filename)
    print(f"Benchmark results saved to: {results_file}")

    # Convert to dict for return
    results_dict = {result.operation: result for result in benchmark.results}

    return results_dict


def compare_benchmark_runs(run1_file: str, run2_file: str) -> None:
    """Compare two benchmark runs."""
    # Load results
    with open(run1_file, "r") as f:
        run1_data = json.load(f)
    with open(run2_file, "r") as f:
        run2_data = json.load(f)

    results1 = {r["operation"]: BenchmarkResult(**r) for r in run1_data["results"]}
    results2 = {r["operation"]: BenchmarkResult(**r) for r in run2_data["results"]}

    comparator = ComparativeBenchmark()
    comparator.set_baseline(results1)
    comparator.compare_with_baseline(results2)

    print(f"\nComparing {run1_file} (baseline) vs {run2_file}")
    print(
        f"Run 1 timestamp: {run1_data.get('metadata', {}).get('timestamp', run1_data.get('timestamp', 'unknown'))}"
    )
    print(
        f"Run 2 timestamp: {run2_data.get('metadata', {}).get('timestamp', run2_data.get('timestamp', 'unknown'))}"
    )

    for operation in set(results1.keys()) | set(results2.keys()):
        print(f"\nOperation: {operation}")

        if operation in results1 and operation in results2:
            r1 = results1[operation]
            r2 = results2[operation]

            time_diff = r2.avg_time - r1.avg_time
            time_percent = (time_diff / r1.avg_time) * 100 if r1.avg_time > 0 else 0

            throughput_diff = r2.throughput - r1.throughput
            throughput_percent = (throughput_diff / r1.throughput) * 100 if r1.throughput > 0 else 0

            print(f"  Run 1: {r1.avg_time:.4f}s ({r1.throughput:.2f} ops/sec)")
            print(f"  Run 2: {r2.avg_time:.4f}s ({r2.throughput:.2f} ops/sec)")
            print(f"  Time change: {time_diff:+.4f}s ({time_percent:+.1f}%)")
            print(
                f"  Throughput change: {throughput_diff:+.2f} ops/sec ({throughput_percent:+.1f}%)"
            )

        elif operation in results1:
            r1 = results1[operation]
            print(f"  Only in Run 1: {r1.avg_time:.4f}s ({r1.throughput:.2f} ops/sec)")

        else:
            r2 = results2[operation]
            print(f"  Only in Run 2: {r2.avg_time:.4f}s ({r2.throughput:.2f} ops/sec)")

        print()
