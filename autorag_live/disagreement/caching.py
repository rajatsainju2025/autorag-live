"""Smart caching for disagreement computations with dependency tracking."""

import hashlib
from typing import Dict, List, Optional, Tuple


class DisagreementCache:
    """Cache disagreement results with dependency tracking."""

    def __init__(self, max_size: int = 1000):
        """Initialize disagreement cache."""
        self._cache: Dict[str, Dict] = {}
        self._dependencies: Dict[str, set] = {}
        self._reverse_deps: Dict[str, set] = {}  # dependency -> cache keys
        self._max_size = max_size

    @staticmethod
    def _make_key(
        retriever_names: Tuple[str, ...],
        query: str,
        k: int,
    ) -> str:
        """Create cache key from inputs."""
        key_parts = f"{','.join(sorted(retriever_names))}:{query}:{k}"
        return hashlib.md5(key_parts.encode()).hexdigest()

    def get(
        self,
        retriever_names: Tuple[str, ...],
        query: str,
        k: int,
    ) -> Optional[Dict]:
        """Get cached disagreement result."""
        key = self._make_key(retriever_names, query, k)
        return self._cache.get(key)

    def set(
        self,
        retriever_names: Tuple[str, ...],
        query: str,
        k: int,
        result: Dict,
        depends_on: Optional[List[str]] = None,
    ) -> None:
        """Cache disagreement result with dependencies."""
        if len(self._cache) >= self._max_size:
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            self._remove_key(oldest_key)

        key = self._make_key(retriever_names, query, k)
        self._cache[key] = result
        if depends_on:
            deps = set(depends_on)
            self._dependencies[key] = deps
            for dep in deps:
                self._reverse_deps.setdefault(dep, set()).add(key)

    def _remove_key(self, key: str) -> None:
        """Remove a key from cache and all dependency indices."""
        self._cache.pop(key, None)
        deps = self._dependencies.pop(key, None)
        if deps:
            for dep in deps:
                rev = self._reverse_deps.get(dep)
                if rev is not None:
                    rev.discard(key)
                    if not rev:
                        del self._reverse_deps[dep]

    def invalidate_if_depends_on(self, dependency: str) -> int:
        """
        Invalidate cache entries depending on a component.

        Uses a reverse index for O(1) lookup instead of scanning all entries.

        Args:
            dependency: Component name that changed

        Returns:
            Number of entries invalidated
        """
        invalid_keys = list(self._reverse_deps.pop(dependency, set()))
        for key in invalid_keys:
            self._remove_key(key)
        return len(invalid_keys)

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._dependencies.clear()
        self._reverse_deps.clear()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "cached_entries": len(self._cache),
            "max_size": self._max_size,
            "dependency_tracked": len(self._dependencies),
        }
