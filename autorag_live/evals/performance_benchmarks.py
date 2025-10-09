"""Performance benchmarks for autorag-live components."""

import json
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil


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


class PerformanceBenchmark:
    """Performance benchmarking suite for autorag-live components."""

    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    @contextmanager
    def memory_monitor(self):
        """Context manager to monitor memory usage."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        max_memory = initial_memory

        yield

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        max_memory = max(max_memory, final_memory)

        self.current_memory_usage = max_memory - initial_memory

    def benchmark_function(
        self,
        func: Callable,
        *args,
        iterations: int = 10,
        warmup_iterations: int = 2,
        operation_name: Optional[str] = None,
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark a function's performance."""

        if operation_name is None:
            operation_name = func.__name__

        # Warmup
        for _ in range(warmup_iterations):
            func(*args, **kwargs)

        # Benchmark
        times = []
        self.current_memory_usage = 0
        result = None

        with self.memory_monitor():
            for _ in range(iterations):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
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
                "args_count": len(args),
                "kwargs_count": len(kwargs),
                "result_type": type(result).__name__ if result is not None else "None",
            },
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        filepath = self.output_dir / filename

        results_dict = {
            "timestamp": time.time(),
            "results": [
                {
                    "operation": r.operation,
                    "iterations": r.iterations,
                    "total_time": r.total_time,
                    "avg_time": r.avg_time,
                    "min_time": r.min_time,
                    "max_time": r.max_time,
                    "std_time": r.std_time,
                    "memory_usage_mb": r.memory_usage_mb,
                    "throughput": r.throughput,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
        }

        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"Benchmark results saved to {filepath}")
        return filepath

    def print_summary(self):
        """Print a summary of all benchmark results."""
        if not self.results:
            print("No benchmark results to display.")
            return

        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)

        for result in self.results:
            print(f"\nOperation: {result.operation}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Total Time: {result.total_time:.4f}s")
            print(f"  Avg Time: {result.avg_time:.4f}s ± {result.std_time:.4f}s")
            print(f"  Min/Max Time: {result.min_time:.4f}s / {result.max_time:.4f}s")
            print(f"  Throughput: {result.throughput:.2f} ops/sec")
            print(f"  Memory Usage: {result.memory_usage_mb:.2f} MB")

        print("\n" + "=" * 80)


def benchmark_retrievers(corpus: List[str], queries: List[str], benchmark: PerformanceBenchmark):
    """Benchmark different retriever implementations."""

    print("Benchmarking retrievers...")

    # BM25 retriever (function-based)
    def bm25_retrieve():
        from autorag_live.retrievers import bm25

        return bm25.bm25_retrieve(queries[0], corpus, 10)

    benchmark.benchmark_function(
        bm25_retrieve, iterations=20, operation_name="BM25 Retrieval (function)"
    )

    # BM25 retriever (class-based)
    bm25_retriever = None

    def bm25_class_retrieve():
        nonlocal bm25_retriever
        if bm25_retriever is None:
            from autorag_live.retrievers.bm25 import BM25Retriever

            bm25_retriever = BM25Retriever()
            bm25_retriever.add_documents(corpus)
        return bm25_retriever.retrieve(queries[0], 10)

    benchmark.benchmark_function(
        bm25_class_retrieve, iterations=20, operation_name="BM25 Retrieval (class)"
    )

    # Dense retriever (function-based)
    def dense_retrieve():
        from autorag_live.retrievers import dense

        return dense.dense_retrieve(queries[0], corpus, 10)

    benchmark.benchmark_function(
        dense_retrieve, iterations=5, operation_name="Dense Retrieval (function)"
    )

    # Dense retriever (class-based with caching)
    dense_retriever = None

    def dense_class_retrieve():
        nonlocal dense_retriever
        if dense_retriever is None:
            from autorag_live.retrievers.dense import DenseRetriever

            dense_retriever = DenseRetriever(cache_embeddings=True)
            dense_retriever.add_documents(corpus)
        return dense_retriever.retrieve(queries[0], 10)

    benchmark.benchmark_function(
        dense_class_retrieve, iterations=20, operation_name="Dense Retrieval (class+cached)"
    )

    # Hybrid retriever (function-based)
    def hybrid_retrieve():
        from autorag_live.retrievers import hybrid

        return hybrid.hybrid_retrieve(queries[0], corpus, 10)

    benchmark.benchmark_function(
        hybrid_retrieve, iterations=5, operation_name="Hybrid Retrieval (function)"
    )

    # Hybrid retriever (class-based)
    hybrid_retriever = None

    def hybrid_class_retrieve():
        nonlocal hybrid_retriever
        if hybrid_retriever is None:
            from autorag_live.retrievers.hybrid import HybridRetriever

            hybrid_retriever = HybridRetriever(bm25_weight=0.5)
            hybrid_retriever.add_documents(corpus)
        return hybrid_retriever.retrieve(queries[0], 10)

    benchmark.benchmark_function(
        hybrid_class_retrieve, iterations=20, operation_name="Hybrid Retrieval (class)"
    )


def benchmark_evaluation(corpus: List[str], queries: List[str], benchmark: PerformanceBenchmark):
    """Benchmark evaluation components."""

    print("Benchmarking evaluation components...")

    # Small evaluation suite
    def run_eval_suite():
        from autorag_live.evals.small import run_small_suite

        return run_small_suite(judge_type="deterministic")

    benchmark.benchmark_function(
        run_eval_suite, iterations=5, operation_name="Small Evaluation Suite"
    )

    # Advanced metrics with optimized retriever
    hybrid_retriever = None

    def comprehensive_eval():
        nonlocal hybrid_retriever
        from autorag_live.evals.advanced_metrics import comprehensive_evaluation

        if hybrid_retriever is None:
            from autorag_live.retrievers.hybrid import HybridRetriever

            hybrid_retriever = HybridRetriever(bm25_weight=0.5)
            hybrid_retriever.add_documents(corpus)

        retrieved = [doc for doc, score in hybrid_retriever.retrieve(queries[0], 5)]
        relevant = corpus[:3]  # Assume first 3 are relevant
        return comprehensive_evaluation(retrieved, relevant, queries[0])

    benchmark.benchmark_function(
        comprehensive_eval, iterations=10, operation_name="Comprehensive Evaluation (optimized)"
    )


def benchmark_optimization(corpus: List[str], queries: List[str], benchmark: PerformanceBenchmark):
    """Benchmark optimization components."""

    print("Benchmarking optimization components...")

    # Grid search optimization
    def grid_search_optimize():
        from autorag_live.pipeline.hybrid_optimizer import grid_search_hybrid_weights

        return grid_search_hybrid_weights(queries[:2], corpus, k=5, grid_size=4)

    benchmark.benchmark_function(
        grid_search_optimize, iterations=3, operation_name="Grid Search Optimization"
    )


def benchmark_augmentation(corpus: List[str], queries: List[str], benchmark: PerformanceBenchmark):
    """Benchmark data augmentation components."""

    print("Benchmarking augmentation components...")

    # Get retriever results for synonym mining
    from autorag_live.retrievers import bm25, dense, hybrid

    bm25_results = bm25.bm25_retrieve(queries[0], corpus, 5)
    dense_results = dense.dense_retrieve(queries[0], corpus, 5)
    hybrid_results = hybrid.hybrid_retrieve(queries[0], corpus, 5)

    def synonym_mining():
        from autorag_live.augment.synonym_miner import mine_synonyms_from_disagreements

        return mine_synonyms_from_disagreements(bm25_results, dense_results, hybrid_results)

    benchmark.benchmark_function(synonym_mining, iterations=10, operation_name="Synonym Mining")


def benchmark_reranking(corpus: List[str], queries: List[str], benchmark: PerformanceBenchmark):
    """Benchmark reranking components."""

    print("Benchmarking reranking components...")

    # Get initial retrieval results
    from autorag_live.retrievers import hybrid

    retrieved_docs = hybrid.hybrid_retrieve(queries[0], corpus, 15)

    def simple_rerank():
        from autorag_live.rerank.simple import SimpleReranker

        reranker = SimpleReranker()
        return reranker.rerank(queries[0], retrieved_docs, k=10)

    benchmark.benchmark_function(simple_rerank, iterations=15, operation_name="Simple Reranking")


def benchmark_time_series(corpus: List[str], benchmark: PerformanceBenchmark):
    """Benchmark time-series components."""

    print("Benchmarking time-series components...")

    # Create time-series notes
    from datetime import datetime, timedelta

    from autorag_live.data.time_series import FFTEmbedder, TimeSeriesNote, TimeSeriesRetriever

    notes = []
    base_time = datetime.now()

    for i, doc in enumerate(corpus[:10]):  # Use subset for performance
        timestamp = base_time - timedelta(days=i)
        note = TimeSeriesNote(content=doc, timestamp=timestamp, metadata={"id": f"note_{i}"})
        notes.append(note)

    embedder = FFTEmbedder()
    retriever = TimeSeriesRetriever(embedder=embedder)
    retriever.add_notes(notes)

    def time_series_search():
        return retriever.search(
            query="test query", query_time=base_time, top_k=5, time_window_days=7
        )

    benchmark.benchmark_function(
        time_series_search, iterations=10, operation_name="Time-Series Search"
    )


def benchmark_disagreement_analysis(
    corpus: List[str], queries: List[str], benchmark: PerformanceBenchmark
):
    """Benchmark disagreement analysis components."""

    print("Benchmarking disagreement analysis...")

    # Get retriever results
    from autorag_live.retrievers import bm25, dense, hybrid

    bm25_results = bm25.bm25_retrieve(queries[0], corpus, 10)
    dense_results = dense.dense_retrieve(queries[0], corpus, 10)
    hybrid_results = hybrid.hybrid_retrieve(queries[0], corpus, 10)

    def disagreement_metrics():
        from autorag_live.disagreement import metrics

        return {
            "jaccard_bd": metrics.jaccard_at_k(bm25_results, dense_results),
            "jaccard_bh": metrics.jaccard_at_k(bm25_results, hybrid_results),
            "jaccard_dh": metrics.jaccard_at_k(dense_results, hybrid_results),
            "kendall_bd": metrics.kendall_tau_at_k(bm25_results, dense_results),
            "kendall_bh": metrics.kendall_tau_at_k(bm25_results, hybrid_results),
            "kendall_dh": metrics.kendall_tau_at_k(dense_results, hybrid_results),
        }

    benchmark.benchmark_function(
        disagreement_metrics, iterations=20, operation_name="Disagreement Metrics"
    )


def run_full_benchmark_suite(output_file: Optional[str] = None):
    """Run the complete benchmark suite."""

    print("Starting comprehensive performance benchmark suite...")
    print("=" * 60)

    # Sample data
    corpus = [
        "The sky is blue and beautiful during the day.",
        "The sun rises in the east and sets in the west.",
        "The sun is bright and provides light to Earth.",
        "The sun in the sky is very bright during daytime.",
        "We can see the shining sun, the bright sun in the sky.",
        "The quick brown fox jumps over the lazy dog.",
        "A lazy fox is usually sleeping in its den.",
        "The fox is a mammal that belongs to the canine family.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision enables machines to interpret visual information.",
        "Data science combines statistics, programming, and domain expertise.",
        "Python is a popular programming language for data science.",
        "Jupyter notebooks provide an interactive environment for coding.",
        "Neural networks can learn complex patterns from data.",
        "Supervised learning requires labeled training data.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Reinforcement learning learns through trial and error.",
        "Transfer learning applies knowledge from one task to another.",
    ]

    queries = [
        "bright sun in the sky",
        "fox jumping over dog",
        "machine learning and AI",
        "programming with Python",
        "data science techniques",
        "neural network learning",
    ]

    # Initialize benchmark
    benchmark = PerformanceBenchmark()

    # Run all benchmarks
    benchmark_retrievers(corpus, queries, benchmark)
    benchmark_evaluation(corpus, queries, benchmark)
    benchmark_optimization(corpus, queries, benchmark)
    benchmark_augmentation(corpus, queries, benchmark)
    benchmark_reranking(corpus, queries, benchmark)
    benchmark_time_series(corpus, benchmark)
    benchmark_disagreement_analysis(corpus, queries, benchmark)

    # Print summary
    benchmark.print_summary()

    # Save results
    if output_file:
        filepath = benchmark.save_results(output_file)
    else:
        filepath = benchmark.save_results()

    print(f"\nBenchmark complete! Results saved to {filepath}")

    return benchmark.results


def compare_benchmark_runs(run1_file: str, run2_file: str):
    """Compare two benchmark runs."""

    def load_benchmark_results(filepath: str) -> Dict[str, BenchmarkResult]:
        with open(filepath, "r") as f:
            data = json.load(f)

        results = {}
        for result_data in data["results"]:
            result = BenchmarkResult(**result_data)
            results[result.operation] = result

        return results

    results1 = load_benchmark_results(run1_file)
    results2 = load_benchmark_results(run2_file)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"Comparing: {run1_file} vs {run2_file}")
    print()

    all_operations = set(results1.keys()) | set(results2.keys())

    for operation in sorted(all_operations):
        print(f"Operation: {operation}")

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


if __name__ == "__main__":
    # Run the full benchmark suite
    results = run_full_benchmark_suite()

    print(f"\nBenchmark completed with {len(results)} operations tested.")
