"""Async retrieval functions for concurrent operations.

This module provides async/await versions of retrieval functions to enable
concurrent query processing and better scalability for high-throughput applications.
"""

import asyncio
from typing import List

from ..types.types import DocumentText, QueryText, RetrieverError
from ..utils import get_logger
from . import bm25, dense, hybrid

logger = get_logger(__name__)


async def bm25_retrieve_async(
    query: QueryText,
    corpus: List[DocumentText],
    k: int = 5,
) -> List[str]:
    """Async BM25 retrieval for concurrent processing.

    Args:
        query: Query text
        corpus: List of documents to search
        k: Number of top documents to return

    Returns:
        List of top-k retrieved documents

    Raises:
        RetrieverError: If retrieval fails

    Example:
        >>> corpus = ["document 1", "document 2"]
        >>> results = await bm25_retrieve_async("query", corpus, k=5)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: bm25.bm25_retrieve(query, corpus, k),
    )


async def dense_retrieve_async(
    query: QueryText,
    corpus: List[DocumentText],
    k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[str]:
    """Async dense retrieval for concurrent processing.

    Args:
        query: Query text
        corpus: List of documents to search
        k: Number of top documents to return
        model_name: Sentence transformer model name

    Returns:
        List of top-k retrieved documents

    Raises:
        RetrieverError: If retrieval fails

    Example:
        >>> corpus = ["document 1", "document 2"]
        >>> results = await dense_retrieve_async("query", corpus, k=5)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: dense.dense_retrieve(query, corpus, k, model_name),
    )


async def hybrid_retrieve_async(
    query: QueryText,
    corpus: List[DocumentText],
    k: int = 5,
    bm25_weight: float = 0.5,
) -> List[str]:
    """Async hybrid retrieval for concurrent processing.

    Args:
        query: Query text
        corpus: List of documents to search
        k: Number of top documents to return
        bm25_weight: Weight for BM25 scores (dense weight = 1 - bm25_weight)

    Returns:
        List of top-k retrieved documents

    Raises:
        RetrieverError: If retrieval fails

    Example:
        >>> corpus = ["document 1", "document 2"]
        >>> results = await hybrid_retrieve_async("query", corpus, k=5)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: hybrid.hybrid_retrieve(query, corpus, k, bm25_weight),
    )


async def batch_retrieve_async(
    queries: List[QueryText],
    corpus: List[DocumentText],
    k: int = 5,
    method: str = "bm25",
    max_concurrency: int = 10,
) -> List[List[str]]:
    """Retrieve results for multiple queries concurrently.

    Args:
        queries: List of query texts
        corpus: List of documents to search
        k: Number of top documents per query
        method: Retrieval method ("bm25", "dense", or "hybrid")
        max_concurrency: Maximum concurrent retrieval operations

    Returns:
        List of retrieval results for each query

    Raises:
        RetrieverError: If retrieval fails
        ValueError: If method is invalid

    Example:
        >>> queries = ["query 1", "query 2", "query 3"]
        >>> corpus = ["doc 1", "doc 2"]
        >>> results = await batch_retrieve_async(queries, corpus, k=5)
    """
    if not queries:
        return []

    # Select retrieval method
    if method == "bm25":
        retrieve_fn = bm25_retrieve_async
    elif method == "dense":
        retrieve_fn = dense_retrieve_async
    elif method == "hybrid":
        retrieve_fn = hybrid_retrieve_async
    else:
        raise ValueError(f"Invalid method: {method}. Must be 'bm25', 'dense', or 'hybrid'")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded_retrieve(query: QueryText) -> List[str]:
        """Retrieve with concurrency limit."""
        async with semaphore:
            try:
                result = await retrieve_fn(query, corpus, k)  # type: ignore
                return result  # type: ignore
            except Exception as e:
                logger.error(f"Retrieval failed for query '{query[:50]}...': {e}")
                raise RetrieverError(f"Batch retrieval failed: {e}")

    # Execute all retrievals concurrently
    tasks = [bounded_retrieve(query) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    logger.info(f"Completed batch retrieval for {len(queries)} queries")
    return results  # type: ignore


async def parallel_retrieval_comparison(
    query: QueryText,
    corpus: List[DocumentText],
    k: int = 5,
) -> dict:
    """Compare retrieval methods concurrently.

    Runs BM25, dense, and hybrid retrieval in parallel for comparison.

    Args:
        query: Query text
        corpus: List of documents to search
        k: Number of top documents to return

    Returns:
        Dictionary with results from each method

    Example:
        >>> corpus = ["document 1", "document 2"]
        >>> comparison = await parallel_retrieval_comparison("query", corpus)
        >>> print(comparison["bm25"][:3])  # Top 3 BM25 results
    """
    # Execute all retrieval methods concurrently
    bm25_task = bm25_retrieve_async(query, corpus, k)
    dense_task = dense_retrieve_async(query, corpus, k)
    hybrid_task = hybrid_retrieve_async(query, corpus, k)

    bm25_results, dense_results, hybrid_results = await asyncio.gather(
        bm25_task,
        dense_task,
        hybrid_task,
        return_exceptions=False,
    )

    return {
        "bm25": bm25_results,
        "dense": dense_results,
        "hybrid": hybrid_results,
    }


__all__ = [
    "bm25_retrieve_async",
    "dense_retrieve_async",
    "hybrid_retrieve_async",
    "batch_retrieve_async",
    "parallel_retrieval_comparison",
]
