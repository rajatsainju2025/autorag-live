"""Efficient content-based deduplication using hashing."""

import hashlib
from typing import List, Set, Tuple


class ContentDeduplicator:
    """Deduplicates content efficiently using content hashes."""

    def __init__(self, hash_algorithm: str = "md5"):
        """Initialize content deduplicator.

        Args:
            hash_algorithm: Hash algorithm to use (md5, sha256, etc.)
        """
        self._hash_algorithm = hash_algorithm
        self._hash_cache = {}

    def _compute_hash(self, content: str) -> str:
        """Compute hash of content."""
        if content in self._hash_cache:
            return self._hash_cache[content]

        hash_obj = hashlib.new(self._hash_algorithm)
        hash_obj.update(content.encode("utf-8"))
        hash_value = hash_obj.hexdigest()

        self._hash_cache[content] = hash_value
        return hash_value

    def deduplicate(self, items: List[str]) -> Tuple[List[str], List[int]]:
        """
        Deduplicate items, returning unique items and original indices.

        Args:
            items: List of content strings

        Returns:
            Tuple of (unique_items, unique_indices)
        """
        seen_hashes: Set[str] = set()
        unique_items = []
        unique_indices = []

        for idx, item in enumerate(items):
            item_hash = self._compute_hash(item)
            if item_hash not in seen_hashes:
                seen_hashes.add(item_hash)
                unique_items.append(item)
                unique_indices.append(idx)

        return unique_items, unique_indices

    def is_duplicate(self, content1: str, content2: str) -> bool:
        """Check if two contents are duplicates based on hash."""
        return self._compute_hash(content1) == self._compute_hash(content2)

    def get_hash(self, content: str) -> str:
        """Get hash of content."""
        return self._compute_hash(content)

    def clear_cache(self):
        """Clear the hash cache."""
        self._hash_cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {"cached_hashes": len(self._hash_cache)}


# Global instance
_global_deduplicator = ContentDeduplicator()


def deduplicate_items(items: List[str]) -> Tuple[List[str], List[int]]:
    """Module-level deduplication function."""
    return _global_deduplicator.deduplicate(items)


def get_content_deduplicator() -> ContentDeduplicator:
    """Get the global content deduplicator instance."""
    return _global_deduplicator
