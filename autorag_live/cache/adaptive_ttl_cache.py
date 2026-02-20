"""
Adaptive-TTL Semantic Cache with query-volatility-aware expiry.

Extends the existing semantic cache with a query-volatility model that
dynamically extends or shortens TTL based on:

1. **Query class volatility** — time-sensitive queries (news, prices, weather)
   get short TTL; stable queries (definitions, historical facts) get long TTL.
2. **Hit-rate feedback** — a cached answer that is repeatedly reused is
   evidence the information is stable → TTL is extended on each cache hit.
3. **Confidence-weighted expiry** — high-confidence answers (verified by
   the self-consistency voter) are cached longer.
4. **Stale-while-revalidate** — expired entries are returned immediately
   while a background refresh is triggered, eliminating cold-start latency.

The module is a drop-in companion to ``EmbeddingSemanticCache``:

    cache = AdaptiveTTLCache(base_ttl_s=300)
    result = await cache.get_or_compute(query, compute_fn, confidence=0.9)

References:
- "Beyond Cache Eviction" (Meta Cachelib team, 2021)
- "Semantic Caching for LLM Serving" (Gim et al., 2024)
  https://arxiv.org/abs/2311.10234
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Query volatility classification
# ---------------------------------------------------------------------------

# Patterns that indicate high-volatility queries (short TTL)
_VOLATILE_PATTERNS = re.compile(
    r"\b(today|now|current|latest|recent|price|stock|weather|live|"
    r"breaking|score|news|update|status|trending)\b",
    re.IGNORECASE,
)

# Patterns that indicate stable queries (long TTL)
_STABLE_PATTERNS = re.compile(
    r"\b(definition|history|what is|who is|when was|founded|invented|"
    r"means|explain|describe|formula|theorem|law of|biography)\b",
    re.IGNORECASE,
)

# Multipliers applied to base_ttl_s per class
_VOLATILITY_MULTIPLIERS: Dict[str, float] = {
    "volatile": 0.1,  # 10% of base TTL
    "stable": 5.0,  # 5× base TTL
    "neutral": 1.0,
}


def classify_query_volatility(query: str) -> str:
    """
    Classify query as 'volatile', 'stable', or 'neutral'.

    Volatile → short TTL (news, prices, current events).
    Stable   → long TTL (definitions, history, science).

    Args:
        query: The user's query string.

    Returns:
        One of 'volatile', 'stable', 'neutral'.
    """
    if _VOLATILE_PATTERNS.search(query):
        return "volatile"
    if _STABLE_PATTERNS.search(query):
        return "stable"
    return "neutral"


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveCacheEntry(Generic[T]):
    """A single cached entry with adaptive TTL metadata."""

    key: str
    value: T
    query: str
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    expires_at: float = 0.0  # absolute monotonic time
    hit_count: int = 0
    volatility_class: str = "neutral"
    confidence: float = 0.5
    embedding: Optional[np.ndarray] = None
    is_refreshing: bool = False  # stale-while-revalidate flag

    @property
    def is_expired(self) -> bool:
        """True if the entry has passed its TTL."""
        return time.monotonic() > self.expires_at

    @property
    def age_s(self) -> float:
        """Seconds since entry was created."""
        return time.monotonic() - self.created_at

    @property
    def ttl_remaining_s(self) -> float:
        """Remaining TTL in seconds (0 if expired)."""
        return max(0.0, self.expires_at - time.monotonic())


# ---------------------------------------------------------------------------
# TTL policy
# ---------------------------------------------------------------------------


@dataclass
class TTLPolicy:
    """
    TTL computation policy.

    TTL = base_ttl_s
          × volatility_multiplier(query)
          × hit_rate_multiplier(hit_count)
          × confidence_multiplier(confidence)
    """

    base_ttl_s: float = 300.0
    """Base TTL in seconds (default 5 minutes)."""

    max_ttl_s: float = 86_400.0
    """Maximum allowed TTL (default 24 hours)."""

    min_ttl_s: float = 5.0
    """Minimum allowed TTL (default 5 seconds)."""

    hit_rate_bonus_per_hit: float = 0.15
    """Fractional TTL increase per cache hit (up to 2×)."""

    confidence_scale: float = 2.0
    """Max multiplier from confidence (applied linearly: confidence × scale)."""

    def compute(
        self,
        query: str,
        hit_count: int = 0,
        confidence: float = 0.5,
    ) -> float:
        """
        Compute the TTL for a cache entry.

        Args:
            query: The query string (used for volatility classification).
            hit_count: How many times this entry has been accessed.
            confidence: Self-consistency or model confidence score ∈ [0, 1].

        Returns:
            TTL in seconds, clamped to [min_ttl_s, max_ttl_s].
        """
        # Volatility adjustment
        vol_class = classify_query_volatility(query)
        vol_mult = _VOLATILITY_MULTIPLIERS[vol_class]

        # Hit-rate adjustment: each hit adds 15% up to a 2× cap
        hit_mult = min(2.0, 1.0 + self.hit_rate_bonus_per_hit * hit_count)

        # Confidence adjustment: scales from 0 → confidence_scale linearly
        conf_mult = max(0.1, min(confidence * self.confidence_scale, self.confidence_scale))

        ttl = self.base_ttl_s * vol_mult * hit_mult * conf_mult
        return float(np.clip(ttl, self.min_ttl_s, self.max_ttl_s))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveCacheStats:
    """Runtime statistics for the adaptive cache."""

    hits: int = 0
    misses: int = 0
    stale_hits: int = 0  # stale-while-revalidate returns
    evictions: int = 0
    refreshes: int = 0
    total_requests: int = 0
    ttl_by_class: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        avg_ttl = {k: (sum(v) / len(v) if v else 0) for k, v in self.ttl_by_class.items()}
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stale_hits": self.stale_hits,
            "evictions": self.evictions,
            "refreshes": self.refreshes,
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
            "avg_ttl_by_class_s": avg_ttl,
        }


# ---------------------------------------------------------------------------
# Adaptive TTL Cache
# ---------------------------------------------------------------------------


class AdaptiveTTLCache(Generic[T]):
    """
    Semantic cache with adaptive TTL and stale-while-revalidate.

    Key behaviours:
    - Fresh hit  → return immediately, extend TTL by hit_rate_bonus.
    - Stale hit  → return stale value, launch background refresh task.
    - Miss       → call compute_fn, store result with computed TTL.
    - Eviction   → LRU when capacity is exceeded.

    Args:
        base_ttl_s: Base cache TTL in seconds.
        max_entries: Maximum cache entries before LRU eviction.
        policy: TTL policy (uses default TTLPolicy if not provided).
        similarity_threshold: Cosine similarity threshold for semantic hits.
        enable_swr: Enable stale-while-revalidate (default True).
    """

    def __init__(
        self,
        base_ttl_s: float = 300.0,
        max_entries: int = 1000,
        policy: Optional[TTLPolicy] = None,
        similarity_threshold: float = 0.92,
        enable_swr: bool = True,
    ) -> None:
        self.policy = policy or TTLPolicy(base_ttl_s=base_ttl_s)
        self.max_entries = max_entries
        self.sim_threshold = similarity_threshold
        self.enable_swr = enable_swr

        # Ordered dict preserves insertion order for LRU eviction
        self._store: Dict[str, AdaptiveCacheEntry[T]] = {}
        self._lock = asyncio.Lock()
        self._stats = AdaptiveCacheStats()
        # Embedding index: list of (key, embedding) for semantic lookup
        self._embedding_index: List[Tuple[str, np.ndarray]] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def get_or_compute(
        self,
        query: str,
        compute_fn: Callable[[], Coroutine[Any, Any, T]],
        query_embedding: Optional[np.ndarray] = None,
        confidence: float = 0.5,
    ) -> T:
        """
        Return a cached result or compute and cache it.

        Args:
            query: The query string.
            compute_fn: Async callable that produces the answer if cache misses.
            query_embedding: Optional embedding for semantic similarity lookup.
            confidence: Confidence score ∈ [0, 1] to scale TTL.

        Returns:
            Cached or freshly computed result.
        """
        self._stats.total_requests += 1
        key = self._make_key(query)

        async with self._lock:
            # ── Exact-key lookup ──────────────────────────────────────────
            entry = self._store.get(key)

            if entry is None and query_embedding is not None:
                # ── Semantic lookup via cosine similarity ─────────────────
                entry = self._semantic_lookup(query_embedding)

            if entry is not None:
                if not entry.is_expired:
                    # Fresh hit — extend TTL
                    entry.hit_count += 1
                    entry.last_accessed = time.monotonic()
                    new_ttl = self.policy.compute(entry.query, entry.hit_count, entry.confidence)
                    entry.expires_at = time.monotonic() + new_ttl
                    self._stats.hits += 1
                    logger.debug("Cache HIT (fresh) for query: %.60s", query)
                    return entry.value

                elif self.enable_swr and not entry.is_refreshing:
                    # Stale-while-revalidate: return stale, refresh in BG
                    entry.is_refreshing = True
                    self._stats.stale_hits += 1
                    logger.debug("Cache HIT (stale, revalidating) for query: %.60s", query)
                    stale_value = entry.value
                    asyncio.create_task(
                        self._background_refresh(key, query, compute_fn, confidence)
                    )
                    return stale_value

            # ── Cache miss ────────────────────────────────────────────────
            self._stats.misses += 1

        # Compute outside lock to avoid blocking other requests
        value = await compute_fn()

        async with self._lock:
            await self._store_entry(key, query, value, confidence, query_embedding)

        return value

    async def invalidate(self, query: str) -> bool:
        """
        Manually invalidate a cache entry by exact key.

        Returns:
            True if an entry was removed.
        """
        key = self._make_key(query)
        async with self._lock:
            if key in self._store:
                del self._store[key]
                self._embedding_index = [(k, e) for k, e in self._embedding_index if k != key]
                return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Return current cache statistics."""
        return self._stats.to_dict()

    async def clear(self) -> None:
        """Purge all cache entries."""
        async with self._lock:
            self._store.clear()
            self._embedding_index.clear()
            self._stats = AdaptiveCacheStats()

    async def evict_expired(self) -> int:
        """
        Proactively remove all expired entries.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            expired_keys = [k for k, e in self._store.items() if e.is_expired]
            for k in expired_keys:
                del self._store[k]
            self._embedding_index = [
                (k, e) for k, e in self._embedding_index if k not in expired_keys
            ]
            self._stats.evictions += len(expired_keys)
        return len(expired_keys)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_key(query: str) -> str:
        """Deterministic cache key from query string."""
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:32]

    def _semantic_lookup(self, query_embedding: np.ndarray) -> Optional[AdaptiveCacheEntry[T]]:
        """
        Find the most similar cached entry above the similarity threshold.

        O(N) scan — fast enough for N < 10,000. For larger caches,
        replace with FAISS/hnswlib from ANNSemanticCache.
        """
        if not self._embedding_index:
            return None

        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        emb_matrix = np.stack([e for _, e in self._embedding_index])  # (N, D)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        normed = emb_matrix / (norms + 1e-9)
        sims: np.ndarray = normed @ q  # (N,)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.sim_threshold:
            best_key = self._embedding_index[best_idx][0]
            return self._store.get(best_key)

        return None

    async def _store_entry(
        self,
        key: str,
        query: str,
        value: T,
        confidence: float,
        embedding: Optional[np.ndarray],
    ) -> None:
        """Store a new cache entry, evicting LRU if at capacity."""
        # Evict LRU if full
        while len(self._store) >= self.max_entries:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
            self._embedding_index = [(k, e) for k, e in self._embedding_index if k != oldest_key]
            self._stats.evictions += 1

        ttl = self.policy.compute(query, hit_count=0, confidence=confidence)
        vol_class = classify_query_volatility(query)
        self._stats.ttl_by_class[vol_class].append(ttl)

        entry: AdaptiveCacheEntry[T] = AdaptiveCacheEntry(
            key=key,
            value=value,
            query=query,
            expires_at=time.monotonic() + ttl,
            confidence=confidence,
            volatility_class=vol_class,
            embedding=embedding,
        )
        self._store[key] = entry

        if embedding is not None:
            self._embedding_index.append((key, embedding))

        logger.debug("Cache STORE: key=%.12s class=%s ttl=%.0fs", key, vol_class, ttl)

    async def _background_refresh(
        self,
        key: str,
        query: str,
        compute_fn: Callable[[], Coroutine[Any, Any, T]],
        confidence: float,
    ) -> None:
        """Recompute and re-store a stale entry in the background."""
        try:
            value = await compute_fn()
            async with self._lock:
                entry = self._store.get(key)
                emb = entry.embedding if entry else None
                await self._store_entry(key, query, value, confidence, emb)
                self._stats.refreshes += 1
        except Exception as exc:
            logger.warning("Background refresh failed for key %.12s: %s", key, exc)
        finally:
            async with self._lock:
                if key in self._store:
                    self._store[key].is_refreshing = False
