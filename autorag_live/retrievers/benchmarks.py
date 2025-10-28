"""Performance benchmarking and profiling tools for retrievers."""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    median_time: float
    p95_time: float
    p99_time: float
    total_time: float
    iterations: int
    throughput: float  # items per second
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        """Format benchmark results as string."""
        return (
            f"{self.name}:\n"
            f"  Mean: {self.mean_time*1000:.2f}ms ± {self.std_time*1000:.2f}ms\n"
            f"  Median: {self.median_time*1000:.2f}ms\n"
            f"  Min/Max: {self.min_time*1000:.2f}ms / {self.max_time*1000:.2f}ms\n"
            f"  P95/P99: {self.p95_time*1000:.2f}ms / {self.p99_time*1000:.2f}ms\n"
            f"  Throughput: {self.throughput:.2f} items/sec\n"
            f"  Iterations: {self.iterations}"
        )


class RetrieverBenchmark:
    """Benchmark suite for retriever performance."""

    def __init__(self, warmup_iterations: int = 3, min_iterations: int = 10):
        """Initialize benchmark suite.

        Args:
            warmup_iterations: Number of warmup iterations before timing
            min_iterations: Minimum number of timed iterations
        """
        self.warmup_iterations = warmup_iterations
        self.min_iterations = min_iterations
        self.results: List[BenchmarkResult] = []

    def benchmark(
        self,
        name: str,
        func: Callable,
        *args,
        iterations: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark a function.

        Args:
            name: Name of the benchmark
            func: Function to benchmark
            *args: Positional arguments to pass to function
            iterations: Number of iterations (default: min_iterations)
            metadata: Additional metadata to store with results
            **kwargs: Keyword arguments to pass to function

        Returns:
            BenchmarkResult with timing statistics
        """
        if iterations is None:
            iterations = self.min_iterations

        if metadata is None:
            metadata = {}

        # Warmup
        logger.debug(f"Running {self.warmup_iterations} warmup iterations for {name}")
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)

        # Timed runs
        logger.info(f"Benchmarking {name} with {iterations} iterations")
        times = []
        for i in range(iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            # Store result size if it's a list
            if i == 0 and isinstance(result, list):
                metadata["result_size"] = len(result)

        times_array = np.array(times)
        total_time = times_array.sum()

        # Calculate statistics
        result = BenchmarkResult(
            name=name,
            mean_time=float(np.mean(times_array)),
            std_time=float(np.std(times_array)),
            min_time=float(np.min(times_array)),
            max_time=float(np.max(times_array)),
            median_time=float(np.median(times_array)),
            p95_time=float(np.percentile(times_array, 95)),
            p99_time=float(np.percentile(times_array, 99)),
            total_time=total_time,
            iterations=iterations,
            throughput=iterations / total_time if total_time > 0 else 0,
            metadata=metadata,
        )

        self.results.append(result)
        logger.info(f"\n{result}")
        return result

    def compare(self, baseline_name: str, comparison_name: str) -> Dict[str, float]:
        """Compare two benchmark results.

        Args:
            baseline_name: Name of baseline benchmark
            comparison_name: Name of comparison benchmark

        Returns:
            Dictionary with comparison metrics (speedup, etc.)
        """
        baseline = next((r for r in self.results if r.name == baseline_name), None)
        comparison = next((r for r in self.results if r.name == comparison_name), None)

        if baseline is None or comparison is None:
            raise ValueError("Benchmark results not found")

        speedup = baseline.mean_time / comparison.mean_time
        throughput_improvement = (
            (comparison.throughput - baseline.throughput) / baseline.throughput * 100
        )

        comparison_dict = {
            "speedup": speedup,
            "throughput_improvement_pct": throughput_improvement,
            "baseline_mean_ms": baseline.mean_time * 1000,
            "comparison_mean_ms": comparison.mean_time * 1000,
            "time_saved_ms": (baseline.mean_time - comparison.mean_time) * 1000,
        }

        logger.info(
            f"\nComparison ({baseline_name} vs {comparison_name}):\n"
            f"  Speedup: {speedup:.2f}x\n"
            f"  Throughput improvement: {throughput_improvement:.1f}%\n"
            f"  Time saved: {comparison_dict['time_saved_ms']:.2f}ms per operation"
        )

        return comparison_dict

    def get_summary(self) -> str:
        """Get summary of all benchmark results.

        Returns:
            Formatted string with all results
        """
        if not self.results:
            return "No benchmark results available"

        summary = "Benchmark Summary\n" + "=" * 80 + "\n\n"
        for result in self.results:
            summary += str(result) + "\n\n"

        return summary


def profile_retriever_operations(retriever, corpus: List[str], queries: List[str]):
    """Profile common retriever operations.

    Args:
        retriever: Retriever instance to profile
        corpus: List of documents
        queries: List of queries to test

    Returns:
        BenchmarkResult list with profiling data
    """
    benchmark = RetrieverBenchmark(warmup_iterations=2, min_iterations=5)

    # Add documents
    logger.info("Profiling add_documents()")
    benchmark.benchmark(
        "add_documents",
        retriever.add_documents,
        corpus,
        metadata={"num_docs": len(corpus)},
    )

    # Single query retrieval
    logger.info("Profiling single query retrieve()")
    benchmark.benchmark(
        "retrieve_single",
        retriever.retrieve,
        queries[0],
        5,
        metadata={"query": queries[0][:50]},
    )

    # Batch retrieval
    logger.info("Profiling batch retrieve()")
    benchmark.benchmark(
        "retrieve_batch",
        retriever.retrieve_batch,
        queries,
        5,
        metadata={"num_queries": len(queries)},
    )

    # Compare single vs batch
    if len(benchmark.results) >= 2:
        benchmark.compare("retrieve_single", "retrieve_batch")

    print("\n" + benchmark.get_summary())
    return benchmark.results
