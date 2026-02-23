"""Prompt Caching Manager.

Manages prompt prefixes to maximize KV cache hits in LLMs (e.g., Anthropic's
prompt caching). Tracks the hash of the prompt prefix and reuses it if it matches.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CachedPrompt:
    prefix_hash: str
    prefix_text: str
    token_count: int
    hit_count: int = 0


class PromptCachingManager:
    """Manages prompt prefixes to maximize KV cache hits."""

    def __init__(self, max_cache_size: int = 100) -> None:
        self.max_cache_size = max_cache_size
        self._cache: Dict[str, CachedPrompt] = {}
        self._total_hits = 0
        self._total_misses = 0

    def _hash_prefix(self, prefix: str) -> str:
        return hashlib.sha256(prefix.encode("utf-8")).hexdigest()

    def get_cached_prefix(self, prefix: str) -> Optional[CachedPrompt]:
        """Check if a prefix is already cached."""
        prefix_hash = self._hash_prefix(prefix)
        if prefix_hash in self._cache:
            self._cache[prefix_hash].hit_count += 1
            self._total_hits += 1
            return self._cache[prefix_hash]
        self._total_misses += 1
        return None

    def cache_prefix(self, prefix: str, token_count: int) -> CachedPrompt:
        """Cache a new prefix."""
        prefix_hash = self._hash_prefix(prefix)
        if prefix_hash in self._cache:
            return self._cache[prefix_hash]

        if len(self._cache) >= self.max_cache_size:
            # Evict least recently used (or least hit)
            lru_hash = min(self._cache, key=lambda k: self._cache[k].hit_count)
            del self._cache[lru_hash]

        cached = CachedPrompt(
            prefix_hash=prefix_hash,
            prefix_text=prefix,
            token_count=token_count,
        )
        self._cache[prefix_hash] = cached
        return cached

    def optimize_prompt(self, system_prompt: str, context: str, query: str) -> str:
        """Optimize prompt by placing static parts first for caching."""
        # In Anthropic's prompt caching, the static parts should be at the beginning.
        # We can structure the prompt as: System Prompt -> Context -> Query
        # The System Prompt and Context can be cached.
        prefix = f"{system_prompt}\n\n{context}"
        cached = self.get_cached_prefix(prefix)
        if not cached:
            # Estimate token count (rough approximation)
            token_count = len(prefix.split())
            self.cache_prefix(prefix, token_count)

        # The actual API call would use the cached prefix hash or text
        # Here we just return the optimized string structure
        return f"{prefix}\n\n{query}"

    def stats(self) -> Dict[str, float]:
        total = self._total_hits + self._total_misses
        hit_rate = self._total_hits / total if total > 0 else 0.0
        return {
            "cache_size": float(len(self._cache)),
            "total_hits": float(self._total_hits),
            "total_misses": float(self._total_misses),
            "hit_rate": round(hit_rate, 4),
        }


def create_prompt_caching_manager(max_cache_size: int = 100) -> PromptCachingManager:
    """Factory for `PromptCachingManager`."""
    return PromptCachingManager(max_cache_size=max_cache_size)
