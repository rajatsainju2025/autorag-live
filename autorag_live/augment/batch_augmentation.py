"""Batch augmentation pipeline for efficient processing."""

from concurrent.futures import ThreadPoolExecutor
from typing import List


class BatchAugmentationPipeline:
    """Processes augmentations in batches for efficiency."""

    def __init__(self, batch_size: int = 32):
        """Initialize batch augmentation pipeline."""
        self.batch_size = batch_size

    def augment_batch(
        self,
        texts: List[str],
        augmentation_fn,
        max_workers: int = 4,
    ) -> List[str]:
        """
        Apply augmentation to a batch of texts.

        Args:
            texts: List of text strings
            augmentation_fn: Function to apply to each text
            max_workers: Number of parallel workers

        Returns:
            Augmented texts
        """
        if not texts:
            return []

        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

        def process_batch(batch):
            return [augmentation_fn(t) for t in batch]

        augmented = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_batch, batches)
            for res in results:
                augmented.extend(res)

        return augmented

    def augment_with_rewrites(
        self,
        queries: List[str],
        rewrite_fn,
        num_rewrites: int = 3,
        max_workers: int = 4,
    ) -> List[str]:
        """
        Generate query rewrites in batches.

        Args:
            queries: List of query strings
            rewrite_fn: Function to generate rewrites
            num_rewrites: Number of rewrites per query
            max_workers: Number of parallel workers

        Returns:
            Original queries + rewrites
        """
        results = list(queries)  # Include originals

        def process_query(query):
            return rewrite_fn(query, num_rewrites)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process all queries in parallel
            rewrites_list = executor.map(process_query, queries)
            for rewrites in rewrites_list:
                results.extend(rewrites)

        return results
