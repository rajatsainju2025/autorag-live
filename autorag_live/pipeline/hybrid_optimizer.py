import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from autorag_live.disagreement.metrics import jaccard_at_k
from autorag_live.retrievers import bm25, dense
from autorag_live.utils import get_logger

logger = get_logger(__name__)


@dataclass
class HybridWeights:
    bm25_weight: float = 0.5
    dense_weight: float = 0.5

    def normalize(self) -> "HybridWeights":
        total = self.bm25_weight + self.dense_weight
        if total == 0:
            return HybridWeights(0.5, 0.5)
        return HybridWeights(self.bm25_weight / total, self.dense_weight / total)


def hybrid_retrieve_weighted(
    query: str, corpus: List[str], k: int, weights: HybridWeights
) -> List[str]:
    """Enhanced hybrid retrieval with custom weights."""
    # Get results from both retrievers
    bm25_results = bm25.bm25_retrieve(query, corpus, min(k * 2, len(corpus)))
    dense_results = dense.dense_retrieve(query, corpus, min(k * 2, len(corpus)))

    # Score-based combination (simplified)
    doc_scores: Dict[str, float] = {}

    # BM25 scores (inverse rank)
    for i, doc in enumerate(bm25_results):
        doc_scores[doc] = doc_scores.get(doc, 0) + weights.bm25_weight * (len(bm25_results) - i)

    # Dense scores (inverse rank)
    for i, doc in enumerate(dense_results):
        doc_scores[doc] = doc_scores.get(doc, 0) + weights.dense_weight * (len(dense_results) - i)

    # Sort by combined score and return top k
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:k]]


def grid_search_hybrid_weights(
    queries: List[str], corpus: List[str], k: int = 5, grid_size: int = 5
) -> Tuple[HybridWeights, float]:
    """
    Simple grid search for optimal hybrid weights.
    Evaluates based on diversity (disagreement) between methods.
    """
    logger.info(
        f"Starting grid search for hybrid weights with {len(queries)} queries, grid_size={grid_size}"
    )

    best_weights = HybridWeights()
    best_score = -1.0

    for i in range(grid_size + 1):
        bm25_w = i / grid_size
        dense_w = 1.0 - bm25_w
        weights = HybridWeights(bm25_w, dense_w)
        logger.debug(f"Evaluating weights: BM25={bm25_w:.2f}, Dense={dense_w:.2f}")

        total_diversity = 0.0
        for query in queries:
            bm25_res = bm25.bm25_retrieve(query, corpus, k)
            dense_res = dense.dense_retrieve(query, corpus, k)
            hybrid_res = hybrid_retrieve_weighted(query, corpus, k, weights)

            # Measure diversity as average disagreement
            j1 = jaccard_at_k(bm25_res, hybrid_res)
            j2 = jaccard_at_k(dense_res, hybrid_res)
            diversity = 1.0 - (j1 + j2) / 2  # Higher diversity = lower jaccard
            total_diversity += diversity

        avg_diversity = total_diversity / len(queries)
        logger.debug(
            f"Weights BM25={bm25_w:.2f}, Dense={dense_w:.2f}: avg_diversity={avg_diversity:.4f}"
        )

        if avg_diversity > best_score:
            best_score = avg_diversity
            best_weights = weights
            logger.debug(f"New best weights found: diversity={best_score:.4f}")

    logger.info(
        f"Grid search completed. Best weights: BM25={best_weights.bm25_weight:.2f}, Dense={best_weights.dense_weight:.2f}, diversity={best_score:.4f}"
    )
    return best_weights, best_score


def save_hybrid_config(weights: HybridWeights, path: str = "hybrid_config.json") -> None:
    """Save hybrid weights to config file."""
    config = {
        "bm25_weight": weights.bm25_weight,
        "dense_weight": weights.dense_weight,
        "version": 1,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(
        f"Saved hybrid config to {path}: BM25={weights.bm25_weight:.2f}, Dense={weights.dense_weight:.2f}"
    )


def load_hybrid_config(path: str = "hybrid_config.json") -> HybridWeights:
    """Load hybrid weights from config file."""
    if not os.path.exists(path):
        logger.warning(f"Hybrid config file not found: {path}, using default weights")
        return HybridWeights()  # Default weights

    with open(path, "r") as f:
        config = json.load(f)

    weights = HybridWeights(
        bm25_weight=config.get("bm25_weight", 0.5), dense_weight=config.get("dense_weight", 0.5)
    )
    logger.info(
        f"Loaded hybrid config from {path}: BM25={weights.bm25_weight:.2f}, Dense={weights.dense_weight:.2f}"
    )
    return weights


# Optimization: perf(batch-tune): implement adaptive batch sizing
