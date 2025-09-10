import typer
from typing_extensions import Annotated
from typing import List
import os

from autorag_live.retrievers import bm25, dense, hybrid
from autorag_live.disagreement import metrics, report
from autorag_live.evals.small import run_small_suite

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

    print(f"Disagreement report saved to {report_path}")


@app.command()
def eval(
    suite: Annotated[str, typer.Option(help="Evaluation suite to run")] = "small",
):
    """Run evaluation suites (e.g., small)."""
    if suite == "small":
        summary = run_small_suite()
        print(f"EM={summary['metrics']['em']:.3f} F1={summary['metrics']['f1']:.3f}")
        print(f"Wrote runs/{summary['run_id']}.json")
    else:
        raise typer.BadParameter(f"Unknown suite: {suite}")


if __name__ == "__main__":
    app()
