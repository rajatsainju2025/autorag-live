import typer
from typing_extensions import Annotated
from typing import List, Optional
import os
import json
from pathlib import Path
from datetime import datetime
import time

from autorag_live.retrievers import bm25, dense, hybrid
from autorag_live.disagreement import metrics, report
from autorag_live.evals.small import run_small_suite
from autorag_live.evals.advanced_metrics import comprehensive_evaluation, aggregate_metrics
from autorag_live.augment.synonym_miner import (
    mine_synonyms_from_disagreements,
    update_terms_from_mining
)
from autorag_live.pipeline.hybrid_optimizer import (
    grid_search_hybrid_weights,
    save_hybrid_config,
    load_hybrid_config
)
from autorag_live.pipeline.acceptance_policy import AcceptancePolicy, safe_config_update
from autorag_live.pipeline.bandit_optimizer import BanditHybridOptimizer
from autorag_live.rerank.simple import SimpleReranker
from autorag_live.data.time_series import TimeSeriesRetriever, TimeSeriesNote, FFTEmbedder

app = typer.Typer()

# Dummy corpus for demonstration
CORPUS = [
    "The sky is blue.",
    "The sun is bright.",
    "The sun in the sky is bright.",
    "We can see the shining sun, the bright sun.",
    "The quick brown fox jumps over the lazy dog.",
    "A lazy fox is a sleeping fox.",
    "The fox is a mammal.",
]

@app.command()
def disagree(
    query: str,
    k: Annotated[int, typer.Option(help="Number of documents to retrieve.")] = 10,
    report_path: Annotated[str, typer.Option(help="Path to save the HTML report.")] = "reports/disagreement_report.html",
):
    """
    Run disagreement analysis between different retrievers for a given query.
    """
    print(f"Running disagreement analysis for query: '{query}'")

    # Ensure the reports directory exists
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # 1. Run all retrievers
    bm25_results = bm25.bm25_retrieve(query, CORPUS, k)
    dense_results = dense.dense_retrieve(query, CORPUS, k)
    hybrid_results = hybrid.hybrid_retrieve(query, CORPUS, k)

    results = {
        "BM25": bm25_results,
        "Dense": dense_results,
        "Hybrid": hybrid_results,
    }

    # 2. Compute disagreement metrics
    disagreement_metrics = {
        "jaccard_bm25_vs_dense": metrics.jaccard_at_k(bm25_results, dense_results),
        "kendall_tau_bm25_vs_dense": metrics.kendall_tau_at_k(bm25_results, dense_results),
        "jaccard_bm25_vs_hybrid": metrics.jaccard_at_k(bm25_results, hybrid_results),
        "kendall_tau_bm25_vs_hybrid": metrics.kendall_tau_at_k(bm25_results, hybrid_results),
        "jaccard_dense_vs_hybrid": metrics.jaccard_at_k(dense_results, hybrid_results),
        "kendall_tau_dense_vs_hybrid": metrics.kendall_tau_at_k(dense_results, hybrid_results),
    }

    # 3. Generate report
    report.generate_disagreement_report(query, results, disagreement_metrics, report_path)

    # 4. Mine synonyms and update terms.yaml
    mined = mine_synonyms_from_disagreements(bm25_results, dense_results, hybrid_results)
    if mined:
        update_terms_from_mining(mined)
        print(f"Updated terms.yaml with {len(mined)} synonym groups")

    print(f"Disagreement report saved to {report_path}")


@app.command()
def eval(
    suite: Annotated[str, typer.Option(help="Evaluation suite to run")] = "small",
    judge: Annotated[str, typer.Option(help="Judge type: deterministic or openai")] = "deterministic",
):
    """Run evaluation suites (e.g., small)."""
    if suite == "small":
        summary = run_small_suite(judge_type=judge)
        print(f"EM={summary['metrics']['em']:.3f} F1={summary['metrics']['f1']:.3f}")
        print(f"Relevance={summary['metrics']['relevance']:.3f} Faithfulness={summary['metrics']['faithfulness']:.3f}")
        print(f"Wrote runs/{summary['run_id']}.json")
    else:
        raise typer.BadParameter(f"Unknown suite: {suite}")


@app.command() 
def optimize(
    queries: Annotated[List[str], typer.Option(help="Queries for optimization")] = ["bright sun", "blue sky", "fox mammal"],
):
    """Optimize hybrid retriever weights with acceptance policy."""
    
    def update_func():
        weights, score = grid_search_hybrid_weights(queries, CORPUS, k=5, grid_size=4)
        save_hybrid_config(weights)
        print(f"New weights: BM25={weights.bm25_weight:.3f}, Dense={weights.dense_weight:.3f}")
        print(f"Diversity score: {score:.3f}")
    
    policy = AcceptancePolicy(threshold=0.01)
    accepted = safe_config_update(update_func, ["hybrid_config.json"], policy)
    
    if accepted:
        print("✅ Optimization accepted!")
    else:
        print("❌ Optimization reverted.")


@app.command()
def advanced_eval(
    query: Annotated[str, typer.Option(help="Query to evaluate")] = "bright sun in the sky",
    relevant_docs: Annotated[List[str], typer.Option(help="Relevant documents")] = ["The sun is bright.", "The sun in the sky is bright."],
    output_file: Annotated[Optional[str], typer.Option(help="Output file for metrics")] = None,
):
    """Run advanced evaluation metrics for a query."""
    print(f"Running advanced evaluation for query: '{query}'")

    # Get retrieved documents from different retrievers
    bm25_results = bm25.bm25_retrieve(query, CORPUS, 5)
    dense_results = dense.dense_retrieve(query, CORPUS, 5)
    hybrid_results = hybrid.hybrid_retrieve(query, CORPUS, 5)

    # Run comprehensive evaluation for each retriever
    retrievers = {
        "BM25": bm25_results,
        "Dense": dense_results,
        "Hybrid": hybrid_results
    }

    all_metrics = {}
    for name, results in retrievers.items():
        metrics = comprehensive_evaluation(results, relevant_docs, query)
        all_metrics[name] = metrics
        print(f"\n{name} Metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.3f}")

    # Aggregate metrics
    aggregated = aggregate_metrics(list(all_metrics.values()))
    print("\nAggregated Metrics:")
    for metric_name, stats in aggregated.items():
        print(f"  {metric_name}: {stats['mean']:.3f} ± {stats['std']:.3f}")

    # Save to file if requested
    if output_file:
        result = {
            "query": query,
            "relevant_docs": relevant_docs,
            "retriever_metrics": all_metrics,
            "aggregated_metrics": aggregated
        }
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output_file}")


@app.command()
def rerank(
    query: Annotated[str, typer.Option(help="Query for reranking")] = "bright sun",
    docs: Annotated[Optional[List[str]], typer.Option(help="Documents to rerank")] = None,
    k: Annotated[int, typer.Option(help="Number of documents to retrieve initially")] = 10,
):
    """Rerank documents using the simple reranker."""
    if docs is None:
        docs = CORPUS

    print(f"Reranking {len(docs)} documents for query: '{query}'")

    # First retrieve documents
    retrieved_docs = hybrid.hybrid_retrieve(query, docs, k)
    print(f"Initially retrieved {len(retrieved_docs)} documents")

    # Rerank using simple reranker
    reranker = SimpleReranker()
    reranked_docs = reranker.rerank(query, retrieved_docs)

    print("\nReranked Results:")
    for i, doc in enumerate(reranked_docs[:5]):  # Show top 5
        print(f"{i+1}. {doc}")


@app.command()
def bandit_optimize(
    queries: Annotated[List[str], typer.Option(help="Queries for bandit optimization")] = ["bright sun", "blue sky", "fox mammal"],
    algorithm: Annotated[str, typer.Option(help="Bandit algorithm: ucb1 or thompson")] = "ucb1",
    iterations: Annotated[int, typer.Option(help="Number of iterations")] = 10,
):
    """Optimize hybrid weights using bandit algorithms."""
    print(f"Running {algorithm} bandit optimization with {iterations} iterations")

    optimizer = BanditHybridOptimizer(
        retriever_names=["bm25", "dense"],
        bandit_type=algorithm
    )

    # Simple optimization loop
    for i in range(iterations):
        # Suggest weights
        weights = optimizer.suggest_weights()

        # Evaluate weights (simplified)
        bm25_w = weights.get("bm25", 0.5)
        dense_w = weights.get("dense", 0.5)

        # Simple evaluation based on diversity
        test_query = queries[i % len(queries)]
        bm25_results = bm25.bm25_retrieve(test_query, CORPUS, 5)
        dense_results = dense.dense_retrieve(test_query, CORPUS, 5)

        # Calculate diversity as reward
        diversity = metrics.jaccard_at_k(bm25_results, dense_results)
        reward = diversity  # Higher diversity = higher reward

        # Update with reward
        optimizer.update_weights(weights, reward)
        print(f"  Iteration {i+1}: weights=({bm25_w:.3f}, {dense_w:.3f}), reward={reward:.3f}")

    # Get best configuration
    best_weights = optimizer.get_best_weights()
    stats = optimizer.get_statistics()

    print("\nBest Configuration:")
    print(f"  Weights: {best_weights}")
    print(f"  Best Reward: {stats.get('best_reward', 0):.3f}")
    print(f"  Total Pulls: {stats.get('total_pulls', 0)}")


@app.command()
def time_series_retrieve(
    query: Annotated[str, typer.Option(help="Query for time-series retrieval")] = "recent developments",
    time_window: Annotated[str, typer.Option(help="Time window (e.g., 7d, 30d)")] = "7d",
):
    """Retrieve documents using time-series aware retrieval."""
    print(f"Running time-series retrieval for query: '{query}' with window: {time_window}")

    # Create sample time-series notes
    from datetime import datetime, timedelta
    import random

    notes = []
    base_time = datetime.now()

    for i, doc in enumerate(CORPUS):
        # Random timestamp within the last 30 days
        days_ago = random.randint(0, 30)
        timestamp = base_time - timedelta(days=days_ago)
        note = TimeSeriesNote(content=doc, timestamp=timestamp, metadata={"id": f"note_{i}"})
        notes.append(note)

    # Initialize retriever with embedder
    embedder = FFTEmbedder()
    retriever = TimeSeriesRetriever(embedder=embedder)
    retriever.add_notes(notes)

    # Parse time window
    if time_window.endswith('d'):
        window_days = int(time_window[:-1])
    else:
        window_days = 7  # default

    # Retrieve with time window
    results = retriever.search(
        query=query,
        query_time=base_time,
        top_k=5,
        time_window_days=window_days
    )

    print(f"\nRetrieved {len(results)} documents within {time_window}:")
    for i, result in enumerate(results):
        time_diff = result['time_diff_days']
        print(f"{i+1}. ({time_diff:.1f} days ago) {result['content']}")
        print(f"   Combined score: {result['combined_score']:.3f} "
              f"(temporal: {result['temporal_score']:.3f}, content: {result['content_score']:.3f})")


@app.command()
def compare_retrievers(
    query: Annotated[str, typer.Option(help="Query to compare")] = "bright sun",
    k: Annotated[int, typer.Option(help="Number of documents to retrieve")] = 5,
    output_file: Annotated[Optional[str], typer.Option(help="Output file for comparison")] = None,
):
    """Compare all available retrievers on a query."""
    print(f"Comparing retrievers for query: '{query}'")

    retrievers = {
        "BM25": lambda q, c, k: bm25.bm25_retrieve(q, c, k),
        "Dense": lambda q, c, k: dense.dense_retrieve(q, c, k),
        "Hybrid": lambda q, c, k: hybrid.hybrid_retrieve(q, c, k),
    }

    results = {}
    for name, retrieve_func in retrievers.items():
        docs = retrieve_func(query, CORPUS, k)
        results[name] = docs
        print(f"\n{name} Results:")
        for i, doc in enumerate(docs):
            print(f"  {i+1}. {doc}")

    # Save comparison if requested
    if output_file:
        comparison = {
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {output_file}")


@app.command()
def config(
    action: Annotated[str, typer.Option(help="Action: show, reset, or update")] = "show",
    bm25_weight: Annotated[Optional[float], typer.Option(help="BM25 weight for update")] = None,
    dense_weight: Annotated[Optional[float], typer.Option(help="Dense weight for update")] = None,
):
    """Manage hybrid retriever configuration."""
    if action == "show":
        try:
            config = load_hybrid_config()
            print("Current Configuration:")
            print(f"  BM25 Weight: {config.bm25_weight:.3f}")
            print(f"  Dense Weight: {config.dense_weight:.3f}")
        except:
            print("No configuration found. Using defaults:")
            print("  BM25 Weight: 0.500")
            print("  Dense Weight: 0.500")

    elif action == "reset":
        # Reset to default
        from autorag_live.pipeline.hybrid_optimizer import HybridWeights
        default_config = HybridWeights(bm25_weight=0.5, dense_weight=0.5)
        save_hybrid_config(default_config)
        print("Configuration reset to defaults.")

    elif action == "update":
        if bm25_weight is None or dense_weight is None:
            print("Error: Both bm25_weight and dense_weight must be provided for update.")
            return

        from autorag_live.pipeline.hybrid_optimizer import HybridWeights
        new_config = HybridWeights(bm25_weight=bm25_weight, dense_weight=dense_weight)
        save_hybrid_config(new_config)
        print(f"Configuration updated: BM25={bm25_weight:.3f}, Dense={dense_weight:.3f}")

    else:
        print(f"Unknown action: {action}. Use 'show', 'reset', or 'update'.")


@app.command()
def self_improve(
    iterations: Annotated[int, typer.Option(help="Number of improvement iterations")] = 3,
    output_dir: Annotated[str, typer.Option(help="Output directory for results")] = "runs",
):
    """Run self-improvement loop (simplified version)."""
    print(f"Running self-improvement loop for {iterations} iterations")

    # Simple self-improvement simulation
    train_queries = ["bright sun", "blue sky", "fox mammal", "machine learning"]
    eval_queries = ["sun bright sky", "fox jumping", "AI algorithms"]

    history = []
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")

        # Evaluate current performance
        bm25_results = bm25.bm25_retrieve(eval_queries[0], CORPUS, 5)
        dense_results = dense.dense_retrieve(eval_queries[0], CORPUS, 5)
        hybrid_results = hybrid.hybrid_retrieve(eval_queries[0], CORPUS, 5)

        diversity = metrics.jaccard_at_k(bm25_results, dense_results)
        print(f"  Diversity: {diversity:.3f}")

        # Optimize weights
        weights, score = grid_search_hybrid_weights(train_queries, CORPUS, k=5, grid_size=3)
        print(f"  New weights: BM25={weights.bm25_weight:.3f}, Dense={weights.dense_weight:.3f}")

        # Mine synonyms
        synonyms = mine_synonyms_from_disagreements(bm25_results, dense_results, hybrid_results)
        if synonyms:
            update_terms_from_mining(synonyms)
            print(f"  Added {len(synonyms)} synonym groups")

        history.append({
            "iteration": i+1,
            "diversity": diversity,
            "weights": {"bm25": weights.bm25_weight, "dense": weights.dense_weight},
            "synonyms_added": len(synonyms) if synonyms else 0
        })

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/self_improvement_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nSelf-improvement complete! Results saved to {output_file}")


if __name__ == "__main__":
    app()
