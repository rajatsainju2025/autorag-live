"""Memory profiling utilities for retrievers."""

import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryProfile:
    """Memory profile results."""

    name: str
    size_bytes: int
    size_mb: float
    size_gb: float
    item_count: Optional[int] = None
    bytes_per_item: Optional[float] = None

    def __str__(self) -> str:
        """Format memory profile as string."""
        result = f"{self.name}: {self.size_mb:.2f} MB"
        if self.item_count is not None:
            result += f" ({self.item_count} items, {self.bytes_per_item:.2f} bytes/item)"
        return result


def get_object_size(obj: Any) -> int:
    """Get size of Python object in bytes.

    Args:
        obj: Object to measure

    Returns:
        Size in bytes
    """
    size = sys.getsizeof(obj)

    # Handle containers recursively
    if isinstance(obj, dict):
        size += sum(get_object_size(k) + get_object_size(v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(get_object_size(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        size = obj.nbytes

    return size


def profile_retriever_memory(retriever) -> dict:
    """Profile memory usage of a retriever.

    Args:
        retriever: Retriever instance to profile

    Returns:
        Dictionary with memory profiles
    """
    profiles = {}

    # Corpus
    if hasattr(retriever, "corpus") and retriever.corpus:
        corpus_size = get_object_size(retriever.corpus)
        profiles["corpus"] = MemoryProfile(
            name="Corpus",
            size_bytes=corpus_size,
            size_mb=corpus_size / (1024 * 1024),
            size_gb=corpus_size / (1024 * 1024 * 1024),
            item_count=len(retriever.corpus),
            bytes_per_item=corpus_size / len(retriever.corpus),
        )

    # Corpus embeddings
    if hasattr(retriever, "corpus_embeddings") and retriever.corpus_embeddings is not None:
        emb_size = retriever.corpus_embeddings.nbytes
        profiles["corpus_embeddings"] = MemoryProfile(
            name="Corpus Embeddings",
            size_bytes=emb_size,
            size_mb=emb_size / (1024 * 1024),
            size_gb=emb_size / (1024 * 1024 * 1024),
            item_count=len(retriever.corpus_embeddings),
            bytes_per_item=emb_size / len(retriever.corpus_embeddings),
        )

    # Normalized embeddings cache
    if (
        hasattr(retriever, "_corpus_embeddings_normalized")
        and retriever._corpus_embeddings_normalized is not None
    ):
        norm_size = retriever._corpus_embeddings_normalized.nbytes
        profiles["normalized_embeddings"] = MemoryProfile(
            name="Normalized Embeddings Cache",
            size_bytes=norm_size,
            size_mb=norm_size / (1024 * 1024),
            size_gb=norm_size / (1024 * 1024 * 1024),
            item_count=len(retriever._corpus_embeddings_normalized),
            bytes_per_item=norm_size / len(retriever._corpus_embeddings_normalized),
        )

    # Pre-fetch pool
    if hasattr(retriever, "_prefetch_pool"):
        pool_size = (
            sum(emb.nbytes for emb in retriever._prefetch_pool.values())
            if retriever._prefetch_pool
            else 0
        )
        if pool_size > 0:
            profiles["prefetch_pool"] = MemoryProfile(
                name="Pre-fetch Pool",
                size_bytes=pool_size,
                size_mb=pool_size / (1024 * 1024),
                size_gb=pool_size / (1024 * 1024 * 1024),
                item_count=len(retriever._prefetch_pool),
                bytes_per_item=pool_size / len(retriever._prefetch_pool),
            )

    # Total
    total_size = sum(p.size_bytes for p in profiles.values())
    profiles["total"] = MemoryProfile(
        name="Total Memory",
        size_bytes=total_size,
        size_mb=total_size / (1024 * 1024),
        size_gb=total_size / (1024 * 1024 * 1024),
    )

    # Log summary
    logger.info("\nMemory Profile:")
    for profile in profiles.values():
        logger.info(f"  {profile}")

    return profiles


def estimate_memory_requirements(
    num_docs: int, avg_doc_length: int, embedding_dim: int = 384
) -> dict:
    """Estimate memory requirements for a corpus.

    Args:
        num_docs: Number of documents
        avg_doc_length: Average document length in characters
        embedding_dim: Embedding dimension

    Returns:
        Dictionary with memory estimates
    """
    # Estimate text storage (rough approximation)
    text_bytes = num_docs * avg_doc_length * 2  # UTF-8 can use up to 2 bytes per char

    # Embeddings storage (float32)
    embeddings_bytes = num_docs * embedding_dim * 4  # 4 bytes per float32

    # Normalized embeddings (if cached)
    normalized_bytes = embeddings_bytes

    # Total
    total_bytes = text_bytes + embeddings_bytes + normalized_bytes

    estimates = {
        "num_docs": num_docs,
        "text_mb": text_bytes / (1024 * 1024),
        "embeddings_mb": embeddings_bytes / (1024 * 1024),
        "normalized_mb": normalized_bytes / (1024 * 1024),
        "total_mb": total_bytes / (1024 * 1024),
        "total_gb": total_bytes / (1024 * 1024 * 1024),
    }

    logger.info(
        f"\nMemory Requirements Estimate (n={num_docs:,}):\n"
        f"  Text: {estimates['text_mb']:.2f} MB\n"
        f"  Embeddings: {estimates['embeddings_mb']:.2f} MB\n"
        f"  Normalized Cache: {estimates['normalized_mb']:.2f} MB\n"
        f"  Total: {estimates['total_mb']:.2f} MB ({estimates['total_gb']:.2f} GB)"
    )

    return estimates
