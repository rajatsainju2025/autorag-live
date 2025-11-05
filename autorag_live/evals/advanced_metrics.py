"""Advanced evaluation metrics for RAG systems."""

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@lru_cache(maxsize=128)
def _get_discount_factors(length: int) -> np.ndarray:
    """Cache discount factors for efficiency."""
    positions = np.arange(1, length + 1, dtype=float)
    return np.log2(positions + 1.0)


def ndcg_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k.

    Optimized version with cached discount factors.

    Args:
        retrieved_docs: List of retrieved document IDs/content
        relevant_docs: List of relevant document IDs/content
        k: Number of top results to consider

    Returns:
        NDCG@k score
    """
    if not isinstance(retrieved_docs, list) or not isinstance(relevant_docs, list):
        raise TypeError("retrieved_docs and relevant_docs must be lists")
    if k <= 0:
        raise ValueError("k must be positive")

    if not retrieved_docs or not relevant_docs:
        return 0.0

    # Relevance vector: 1 if relevant, else 0 (up to k)
    rel_set = set(relevant_docs)
    top_k = retrieved_docs[:k]
    if not top_k:
        return 0.0

    relevance = np.fromiter((1.0 if d in rel_set else 0.0 for d in top_k), dtype=float)

    # Discount factors: log2(positions+1) - cached
    discounts = _get_discount_factors(len(relevance))

    dcg = float(np.sum(relevance / discounts))

    # Ideal DCG with all ones up to min(len(relevant), k)
    ideal_len = min(len(rel_set), len(top_k))
    if ideal_len == 0:
        return 0.0
    ideal_relevance = np.ones(ideal_len, dtype=float)
    ideal_discounts = _get_discount_factors(ideal_len)
    idcg = float(np.sum(ideal_relevance / ideal_discounts))

    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(retrieved_docs: List[List[str]], relevant_docs: List[List[str]]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    Args:
        retrieved_docs: List of retrieved document lists for each query
        relevant_docs: List of relevant document lists for each query

    Returns:
        Mean Reciprocal Rank score
    """
    if not isinstance(retrieved_docs, list) or not isinstance(relevant_docs, list):
        raise TypeError("retrieved_docs and relevant_docs must be lists of lists")
    if len(retrieved_docs) != len(relevant_docs):
        raise ValueError("retrieved_docs and relevant_docs must have the same length")

    if not retrieved_docs:
        return 0.0

    reciprocal_ranks = []

    for ret_docs, rel_docs in zip(retrieved_docs, relevant_docs):
        if not isinstance(ret_docs, list) or not isinstance(rel_docs, list):
            raise TypeError("Each element must be a list")

        # Use a set for O(1) membership checks
        rel_set = set(rel_docs)

        # Find first relevant document
        rr = 0.0
        for i, doc in enumerate(ret_docs):
            if doc in rel_set:
                rr = 1.0 / (i + 1)
                break
        reciprocal_ranks.append(rr)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def mean_average_precision(
    retrieved_docs: List[List[str]], relevant_docs: List[List[str]], k: int = 10
) -> float:
    """Calculate Mean Average Precision (MAP@k).

    Args:
        retrieved_docs: List of retrieved document lists for each query
        relevant_docs: List of relevant document lists for each query
        k: Number of top results to consider

    Returns:
        Mean Average Precision score
    """
    if not retrieved_docs or not relevant_docs:
        return 0.0

    if len(retrieved_docs) != len(relevant_docs):
        raise ValueError("retrieved_docs and relevant_docs must have the same length")

    average_precisions = []

    for ret_docs, rel_docs in zip(retrieved_docs, relevant_docs):
        if not rel_docs:
            continue

        rel_set = set(rel_docs)
        top_k_docs = ret_docs[:k]
        if not top_k_docs:
            average_precisions.append(0.0)
            continue

        # Vectorized precision calculation
        relevant_mask = np.array([doc in rel_set for doc in top_k_docs], dtype=bool)
        if not np.any(relevant_mask):
            average_precisions.append(0.0)
            continue

        relevant_cumsum = np.cumsum(relevant_mask)
        positions = np.arange(1, len(top_k_docs) + 1)
        precisions = relevant_cumsum[relevant_mask] / positions[relevant_mask]

        # Average precision for this query
        ap = np.mean(precisions)
        average_precisions.append(ap)

    return float(np.mean(average_precisions)) if average_precisions else 0.0


def hit_rate_at_k(
    retrieved_docs: List[List[str]], relevant_docs: List[List[str]], k: int = 10
) -> float:
    """Calculate Hit Rate@k (Success Rate).

    Measures the proportion of queries with at least one relevant document in top-k.

    Optimized using pre-converted sets and numpy operations.

    Args:
        retrieved_docs: List of retrieved document lists for each query
        relevant_docs: List of relevant document lists for each query
        k: Number of top results to consider

    Returns:
        Hit rate score (0.0 to 1.0)
    """
    if not retrieved_docs or not relevant_docs:
        return 0.0

    if len(retrieved_docs) != len(relevant_docs):
        raise ValueError("retrieved_docs and relevant_docs must have the same length")

    # Vectorized hit calculation with pre-converted sets
    hit_list = []
    for ret_docs, rel_docs in zip(retrieved_docs, relevant_docs):
        rel_set = set(rel_docs)
        top_k_docs = set(ret_docs[:k])
        hit_list.append(bool(top_k_docs & rel_set))

    return np.sum(hit_list) / len(retrieved_docs)


def coverage_score(retrieved_docs: List[str], corpus: List[str]) -> float:
    """Calculate corpus coverage (fraction of unique corpus docs retrieved).

    Args:
        retrieved_docs: Retrieved documents
        corpus: Full corpus of available documents

    Returns:
        Coverage score (0.0 to 1.0)
    """
    if not corpus:
        return 0.0

    corpus_set = set(corpus)
    retrieved_set = set(retrieved_docs)

    return len(retrieved_set & corpus_set) / len(corpus_set)


def diversity_score(retrieved_docs: List[str], embeddings: Optional[np.ndarray] = None) -> float:
    """Calculate diversity score based on semantic similarity.

    Args:
        retrieved_docs: List of retrieved documents
        embeddings: Pre-computed embeddings for the documents

    Returns:
        Diversity score (1 - average pairwise similarity)
    """
    if len(retrieved_docs) < 2 or embeddings is None:
        return 1.0  # Maximum diversity if insufficient data

    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(embeddings)

    # Get upper triangle (excluding diagonal)
    n = len(similarities)
    upper_triangle = similarities[np.triu_indices(n, k=1)]

    # Diversity = 1 - average similarity
    avg_similarity = np.mean(upper_triangle)
    return float(1.0 - avg_similarity)


def novelty_score(retrieved_docs: List[str], query_history: List[str]) -> float:
    """Calculate novelty score based on query history.

    Args:
        retrieved_docs: Current retrieved documents
        query_history: Previously retrieved documents

    Returns:
        Novelty score (fraction of new documents)
    """
    if not query_history:
        return 1.0  # All documents are novel

    history_set = set(query_history)
    current_set = set(retrieved_docs)

    # Count novel documents
    novel_docs = current_set - history_set
    return len(novel_docs) / len(current_set) if current_set else 0.0


def semantic_coverage(
    retrieved_docs: List[str],
    all_relevant_docs: List[str],
    embeddings: Optional[np.ndarray] = None,
) -> float:
    """Calculate semantic coverage of relevant documents.

    Optimized version with efficient indexing and vectorized operations.

    Args:
        retrieved_docs: Retrieved documents
        all_relevant_docs: All relevant documents
        embeddings: Pre-computed embeddings for all docs

    Returns:
        Semantic coverage score
    """
    if not all_relevant_docs or embeddings is None:
        return 0.0

    num_retrieved = len(retrieved_docs)
    num_relevant = len(all_relevant_docs)

    # Bounds checking to avoid invalid slicing
    if num_retrieved == 0 or num_relevant == 0:
        return 0.0

    if num_retrieved + num_relevant > len(embeddings):
        return 0.0

    # Vectorized similarity computation
    retrieved_embeddings = embeddings[:num_retrieved]
    relevant_embeddings = embeddings[num_retrieved : num_retrieved + num_relevant]

    # Calculate maximum similarity for each relevant doc to any retrieved doc
    max_similarities = []
    for rel_emb in relevant_embeddings:
        similarities = cosine_similarity(np.array([rel_emb]), retrieved_embeddings)[0]
        max_similarities.append(np.max(similarities))

    return float(np.mean(max_similarities))


def robustness_score(retrieved_docs_list: List[List[str]], relevant_docs: List[str]) -> float:
    """Calculate robustness across multiple retrieval runs.

    Args:
        retrieved_docs_list: List of retrieved document lists from different runs
        relevant_docs: Ground truth relevant documents

    Returns:
        Robustness score (consistency across runs)
    """
    if not retrieved_docs_list:
        return 0.0

    # For single run, robustness is perfect (no variation)
    if len(retrieved_docs_list) == 1:
        return 1.0

    # Calculate precision for each run
    precisions = []
    for ret_docs in retrieved_docs_list:
        if ret_docs:
            precision = len(set(ret_docs) & set(relevant_docs)) / len(ret_docs)
            precisions.append(precision)

    if not precisions:
        return 0.0

    # Robustness = 1 - coefficient of variation
    mean_precision = np.mean(precisions)
    std_precision = np.std(precisions)

    cv = std_precision / mean_precision if mean_precision > 0 else 0.0
    return float(max(0.0, 1.0 - cv))


def contextual_relevance(retrieved_docs: List[str], query: str, context_window: int = 3) -> float:
    """Calculate contextual relevance based on term proximity.

    Optimized with vectorized set operations and early termination.

    Args:
        retrieved_docs: Retrieved documents
        query: Original query
        context_window: Size of context window for term proximity

    Returns:
        Contextual relevance score
    """
    if not retrieved_docs or not query:
        return 0.0

    query_terms = set(query.lower().split())
    if not query_terms:
        return 0.0

    total_score = 0.0
    for doc in retrieved_docs:
        doc_terms = doc.lower().split()

        # Early exit for empty docs
        if not doc_terms:
            continue

        # Find positions of query terms efficiently using dict
        term_positions = {term: [] for term in query_terms}
        for i, term in enumerate(doc_terms):
            if term in term_positions:
                term_positions[term].append(i)

        # Vectorized proximity computation using numpy arrays
        relevant_terms = [t for t, pos_list in term_positions.items() if pos_list]
        if not relevant_terms:
            continue

        if len(relevant_terms) > 1:
            # Calculate proximity scores using vectorized approach
            proximity_scores = []
            for i in range(len(relevant_terms)):
                for j in range(i + 1, len(relevant_terms)):
                    pos1_list = np.array(term_positions[relevant_terms[i]])
                    pos2_list = np.array(term_positions[relevant_terms[j]])

                    # Vectorized distance computation
                    distances = np.abs(pos1_list[:, np.newaxis] - pos2_list[np.newaxis, :])
                    min_dist = np.min(distances)

                    if min_dist <= context_window:
                        proximity_scores.append(1.0 / (min_dist + 1))

            proximity_score = float(np.mean(proximity_scores)) if proximity_scores else 0.0
        else:
            proximity_score = 0.0

        # Combine term coverage and proximity
        coverage = len(relevant_terms) / len(query_terms)
        total_score += (coverage + proximity_score) / 2

    return float(total_score / len(retrieved_docs)) if retrieved_docs else 0.0


def fairness_score(retrieved_docs: List[str], document_groups: Dict[str, List[str]]) -> float:
    """Calculate fairness score across different document groups.

    Args:
        retrieved_docs: Retrieved documents
        document_groups: Dictionary mapping group names to document lists

    Returns:
        Fairness score (1 - variance in group representation)
    """
    if not document_groups or not retrieved_docs:
        return 1.0

    retrieved_set = set(retrieved_docs)
    group_representations = []

    for group_name, group_docs in document_groups.items():
        group_set = set(group_docs)
        intersection = retrieved_set & group_set

        if group_set:  # Avoid division by zero
            representation = len(intersection) / len(group_set)
            group_representations.append(representation)

    if not group_representations:
        return 1.0

    # Fairness = 1 - coefficient of variation
    mean_repr = np.mean(group_representations)
    std_repr = np.std(group_representations)

    cv = std_repr / mean_repr if mean_repr > 0 else 0.0
    return float(max(0.0, 1.0 - cv))


def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 10) -> float:
    """Calculate Precision@k.

    Args:
        retrieved_docs: List of retrieved document IDs/content
        relevant_docs: List of relevant document IDs/content
        k: Number of top results to consider

    Returns:
        Precision@k score
    """
    if not retrieved_docs or not relevant_docs or k <= 0:
        return 0.0

    top_k = retrieved_docs[:k]
    relevant_set = set(relevant_docs)

    # Vectorized calculation using NumPy
    relevant_mask = np.array([doc in relevant_set for doc in top_k], dtype=bool)
    return float(np.sum(relevant_mask) / len(top_k))


def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 10) -> float:
    """Calculate Recall@k.

    Args:
        retrieved_docs: List of retrieved document IDs/content
        relevant_docs: List of relevant document IDs/content
        k: Number of top results to consider

    Returns:
        Recall@k score
    """
    if not retrieved_docs or not relevant_docs or k <= 0:
        return 0.0

    top_k = retrieved_docs[:k]
    relevant_set = set(relevant_docs)

    # Vectorized calculation using NumPy
    relevant_mask = np.array([doc in relevant_set for doc in top_k], dtype=bool)
    num_relevant_retrieved = np.sum(relevant_mask)
    total_relevant = len(relevant_set)

    return float(num_relevant_retrieved / total_relevant) if total_relevant > 0 else 0.0


def efficiency_score(retrieval_time: float, num_docs: int, baseline_time: float = 1.0) -> float:
    """Calculate efficiency score based on retrieval time.

    Args:
        retrieval_time: Time taken for retrieval
        num_docs: Number of documents retrieved
        baseline_time: Baseline time for comparison

    Returns:
        Efficiency score (higher is better)
    """
    if retrieval_time <= 0 or num_docs <= 0:
        return 0.0

    # Normalize by baseline and adjust for number of documents
    normalized_time = retrieval_time / baseline_time
    throughput = num_docs / retrieval_time

    # Combine normalized time and throughput
    time_score = 1.0 / (1.0 + normalized_time)
    throughput_score = min(1.0, throughput / 100.0)  # Cap at reasonable throughput

    return (time_score + throughput_score) / 2


def comprehensive_evaluation(
    retrieved_docs: List[str],
    relevant_docs: List[str],
    query: str = "",
    embeddings: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, float]:
    """Run comprehensive evaluation with multiple metrics.

    Args:
        retrieved_docs: Retrieved documents
        relevant_docs: Relevant documents
        query: Original query
        embeddings: Document embeddings
        **kwargs: Additional parameters for specific metrics
            - retrieved_docs_list: For MAP, MRR, Hit Rate (list of lists)
            - relevant_docs_list: For MAP, MRR, Hit Rate (list of lists)
            - corpus: For coverage score
            - query_history: For novelty score
            - all_relevant_docs: For semantic coverage
            - document_groups: For fairness score
            - retrieval_time: For efficiency score
            - num_docs: For efficiency score

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}

    # Basic metrics
    if retrieved_docs and relevant_docs:
        # NDCG at different k values
        metrics["ndcg@1"] = ndcg_at_k(retrieved_docs, relevant_docs, k=1)
        metrics["ndcg@3"] = ndcg_at_k(retrieved_docs, relevant_docs, k=3)
        metrics["ndcg@5"] = ndcg_at_k(retrieved_docs, relevant_docs, k=5)
        metrics["ndcg@10"] = ndcg_at_k(retrieved_docs, relevant_docs, k=10)

        # Precision and Recall at different k values
        relevant_set = set(relevant_docs)

        for k in [1, 3, 5, 10]:
            top_k = retrieved_docs[:k]
            if top_k:
                k_intersection = set(top_k) & relevant_set
                metrics[f"precision@{k}"] = len(k_intersection) / len(top_k)
                metrics[f"recall@{k}"] = (
                    len(k_intersection) / len(relevant_set) if relevant_set else 0.0
                )

    # Multi-query metrics (require lists of lists)
    if "retrieved_docs_list" in kwargs and "relevant_docs_list" in kwargs:
        ret_list = kwargs["retrieved_docs_list"]
        rel_list = kwargs["relevant_docs_list"]

        metrics["map@5"] = mean_average_precision(ret_list, rel_list, k=5)
        metrics["map@10"] = mean_average_precision(ret_list, rel_list, k=10)
        metrics["mrr"] = mean_reciprocal_rank(ret_list, rel_list)
        metrics["hit_rate@1"] = hit_rate_at_k(ret_list, rel_list, k=1)
        metrics["hit_rate@5"] = hit_rate_at_k(ret_list, rel_list, k=5)
        metrics["hit_rate@10"] = hit_rate_at_k(ret_list, rel_list, k=10)

    # Advanced metrics
    if embeddings is not None and len(retrieved_docs) > 1:
        metrics["diversity"] = diversity_score(retrieved_docs, embeddings)

    if "corpus" in kwargs:
        metrics["coverage"] = coverage_score(retrieved_docs, kwargs["corpus"])

    if query:
        metrics["contextual_relevance"] = contextual_relevance(retrieved_docs, query)

    # Additional metrics from kwargs
    if "query_history" in kwargs:
        metrics["novelty"] = novelty_score(retrieved_docs, kwargs["query_history"])

    if "all_relevant_docs" in kwargs and embeddings is not None:
        metrics["semantic_coverage"] = semantic_coverage(
            retrieved_docs, kwargs["all_relevant_docs"], embeddings
        )

    if "retrieved_docs_list" in kwargs and relevant_docs:
        metrics["robustness"] = robustness_score(kwargs["retrieved_docs_list"], relevant_docs)

    if "document_groups" in kwargs:
        metrics["fairness"] = fairness_score(retrieved_docs, kwargs["document_groups"])

    if "retrieval_time" in kwargs and "num_docs" in kwargs:
        metrics["efficiency"] = efficiency_score(kwargs["retrieval_time"], kwargs["num_docs"])

    return metrics


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple evaluation runs.

    Optimized with vectorized NumPy operations.

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not metrics_list:
        return {}

    # Collect all metric names (vectorized approach)
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())

    aggregated = {}
    for metric_name in all_metrics:
        # Vectorized collection of values into numpy array
        values = np.array([m.get(metric_name, 0.0) for m in metrics_list if metric_name in m])

        if len(values) > 0:
            aggregated[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

    return aggregated
