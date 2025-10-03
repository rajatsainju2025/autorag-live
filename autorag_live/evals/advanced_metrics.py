"""Advanced evaluation metrics for RAG systems."""

import math
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
    if not retrieved_docs or not relevant_docs:
        return 0.0

    # Create relevance scores (1 if relevant, 0 otherwise)
    relevance_scores = []
    for i, doc in enumerate(retrieved_docs[:k]):
        relevance = 1.0 if doc in relevant_docs else 0.0
        relevance_scores.append(relevance)

    # Calculate DCG
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        dcg += rel / math.log2(i + 2)  # i+2 because positions start from 1

    # Calculate IDCG (ideal DCG)
    ideal_relevance = [1.0] * min(len(relevant_docs), k)
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        idcg += rel / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(retrieved_docs: List[List[str]], relevant_docs: List[List[str]]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    Args:
        retrieved_docs: List of retrieved document lists for each query
        relevant_docs: List of relevant document lists for each query

    Returns:
        MRR score
    """
    reciprocal_ranks = []

    for ret_docs, rel_docs in zip(retrieved_docs, relevant_docs):
        rr = 0.0
        for i, doc in enumerate(ret_docs):
            if doc in rel_docs:
                rr = 1.0 / (i + 1)
                break
        reciprocal_ranks.append(rr)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


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
    **kwargs
) -> Dict[str, float]:
    """Run comprehensive evaluation with multiple metrics.

    Args:
        retrieved_docs: Retrieved documents
        relevant_docs: Relevant documents
        query: Original query
        embeddings: Document embeddings
        **kwargs: Additional parameters for specific metrics

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}

    # Basic metrics
    if retrieved_docs and relevant_docs:
        # NDCG@5 and NDCG@10
        metrics["ndcg@5"] = ndcg_at_k(retrieved_docs, relevant_docs, k=5)
        metrics["ndcg@10"] = ndcg_at_k(retrieved_docs, relevant_docs, k=10)

        # Precision and Recall
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        intersection = retrieved_set & relevant_set

        metrics["precision@5"] = (
            len(intersection) / len(retrieved_docs[:5]) if retrieved_docs[:5] else 0.0
        )
        metrics["precision@10"] = (
            len(intersection) / len(retrieved_docs[:10]) if retrieved_docs[:10] else 0.0
        )
        metrics["recall@5"] = len(intersection) / len(relevant_set) if relevant_set else 0.0
        metrics["recall@10"] = len(intersection) / len(relevant_set) if relevant_set else 0.0

    # Advanced metrics
    if embeddings is not None and len(retrieved_docs) > 1:
        metrics["diversity"] = diversity_score(retrieved_docs, embeddings)

    if query:
        metrics["contextual_relevance"] = contextual_relevance(retrieved_docs, query)

    # Additional metrics from kwargs
    if "query_history" in kwargs:
        metrics["novelty"] = novelty_score(retrieved_docs, kwargs["query_history"])

    if "all_relevant_docs" in kwargs and embeddings is not None:
        metrics["semantic_coverage"] = semantic_coverage(
            retrieved_docs, kwargs["all_relevant_docs"], embeddings
        )

    if "retrieved_docs_list" in kwargs:
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
