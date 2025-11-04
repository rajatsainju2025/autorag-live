"""Advanced evaluation metrics for RAG systems."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def ndcg_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k.

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

    # Discount factors: log2(positions+1)
    positions = np.arange(1, len(relevance) + 1, dtype=float)
    discounts = np.log2(positions + 1.0)

    dcg = float(np.sum(relevance / discounts))

    # Ideal DCG with all ones up to min(len(relevant), k)
    ideal_len = min(len(rel_set), len(top_k))
    if ideal_len == 0:
        return 0.0
    ideal_relevance = np.ones(ideal_len, dtype=float)
    ideal_discounts = np.log2(np.arange(1, ideal_len + 1, dtype=float) + 1.0)
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

    # Vectorized hit calculation using NumPy
    hit_list = [
        any(doc in set(rel_docs) for doc in ret_docs[:k])
        for ret_docs, rel_docs in zip(retrieved_docs, relevant_docs)
    ]
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
    retrieved_docs: List[str], all_relevant_docs: List[str], embeddings: Optional[np.ndarray] = None
) -> float:
    """Calculate semantic coverage of relevant documents.

    Args:
        retrieved_docs: Retrieved documents
        all_relevant_docs: All relevant documents
        embeddings: Pre-computed embeddings

    Returns:
        Semantic coverage score
    """
    if not all_relevant_docs or embeddings is None:
        return 0.0

    retrieved_embeddings = embeddings[: len(retrieved_docs)]
    relevant_embeddings = embeddings[
        len(retrieved_docs) : len(retrieved_docs) + len(all_relevant_docs)
    ]

    if len(retrieved_embeddings) == 0 or len(relevant_embeddings) == 0:
        return 0.0

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
        term_positions = {}

        # Find positions of query terms
        for i, term in enumerate(doc_terms):
            if term in query_terms:
                if term not in term_positions:
                    term_positions[term] = []
                term_positions[term].append(i)

        # Calculate proximity score
        if len(term_positions) > 1:
            min_distances = []
            terms_list = list(term_positions.keys())

            for i in range(len(terms_list)):
                for j in range(i + 1, len(terms_list)):
                    term1, term2 = terms_list[i], terms_list[j]
                    pos1_list = term_positions[term1]
                    pos2_list = term_positions[term2]

                    # Find minimum distance between any positions
                    min_dist = float("inf")
                    for pos1 in pos1_list:
                        for pos2 in pos2_list:
                            min_dist = min(min_dist, abs(pos1 - pos2))

                    if min_dist <= context_window:
                        min_distances.append(1.0 / (min_dist + 1))

            proximity_score = np.mean(min_distances) if min_distances else 0.0
        else:
            proximity_score = 0.0

        # Combine term coverage and proximity
        coverage = len(term_positions) / len(query_terms)
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

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not metrics_list:
        return {}

    # Collect all metric names
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())

    aggregated = {}
    for metric_name in all_metrics:
        values = [m.get(metric_name, 0.0) for m in metrics_list if metric_name in m]

        if values:
            aggregated[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

    return aggregated
