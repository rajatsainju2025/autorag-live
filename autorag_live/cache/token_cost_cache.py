"""
Token-Cost-Aware LRU Cache.

Standard LRU caches evict the *least recently used* entry regardless of cost.
For LLM workloads, a short cache entry for a 10-token response should not
be treated the same as one for a 4,000-token response.

This module provides a cache whose eviction policy combines:
  - **Token cost** (larger responses cost more to regenerate → keep them).
  - **Access frequency** (frequently-accessed entries are more valuable).
  - **Recency** (TTL-based expiry + LRU ordering).

Eviction target = argmin( frequency * token_cost / age_seconds )

This is a practical approximation of GD* (Greedy Dual* — Cao & Irani, 1997)
adapted for the token-economy of LLM APIs.

Usage::

    cache = TokenCostAwareCache(max_tokens=100_000, ttl=300)
    cache.set("q:hello", "Hello, world!", token_cost=4)
    response = cache.get("q:hello")   # → "Hello, world!"
    stats = cache.stats()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Generic, Iterator, Optional, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class _Entry(Generic[V]):
    key: object
    value: V
    token_cost: int  # Estimated tokens consumed by this entry
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    access_count: int = 0

    @property
    def age(self) -> float:
        """Seconds since last access."""
        return time.monotonic() - self.last_accessed

    def eviction_priority(self) -> float:
        """
        Lower priority → evicted first.

        priority = frequency * token_cost / age
        """
        freq = max(self.access_count, 1)
        age = max(self.age, 0.001)
        return (freq * self.token_cost) / age


# ---------------------------------------------------------------------------
# Main cache
# ---------------------------------------------------------------------------


class TokenCostAwareCache(Generic[K, V]):
    """
    Thread-safe token-cost-aware cache with TTL and GD*-style eviction.

    Args:
        max_tokens: Total token budget for cached values.
        ttl: Time-to-live in seconds (0 = no expiry).
        max_entries: Hard cap on number of entries regardless of token budget.

    Example::

        cache: TokenCostAwareCache[str, str] = TokenCostAwareCache(max_tokens=50_000, ttl=600)
        cache.set("prompt:abc", llm_response, token_cost=len(llm_response.split()))
        hit = cache.get("prompt:abc")
    """

    def __init__(
        self,
        max_tokens: int = 100_000,
        ttl: float = 300.0,
        max_entries: int = 10_000,
    ) -> None:
        self.max_tokens = max_tokens
        self.ttl = ttl
        self.max_entries = max_entries

        self._store: dict[object, _Entry[V]] = {}
        self._current_tokens: int = 0
        self._lock = threading.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: K) -> Optional[V]:
        """
        Retrieve a value by key.

        Returns None on miss or expiry.
        """
        with self._lock:
            entry = self._store.get(key)  # type: ignore[arg-type]
            if entry is None:
                self._misses += 1
                return None

            if self._is_expired(entry):
                self._remove(key)  # type: ignore[arg-type]
                self._expired += 1
                self._misses += 1
                return None

            entry.last_accessed = time.monotonic()
            entry.access_count += 1
            self._hits += 1
            return entry.value

    def set(self, key: K, value: V, token_cost: int = 1) -> None:
        """
        Insert or update a cache entry.

        Args:
            key: Cache key.
            value: Value to cache.
            token_cost: Estimated token count of the value (used for eviction scoring).
        """
        if token_cost > self.max_tokens:
            logger.warning(
                "Entry token_cost=%d exceeds max_tokens=%d; skipping cache",
                token_cost,
                self.max_tokens,
            )
            return

        with self._lock:
            # Remove existing entry if updating
            if key in self._store:  # type: ignore[operator]
                self._remove(key)  # type: ignore[arg-type]

            # Evict until there's room
            while (
                self._current_tokens + token_cost > self.max_tokens
                or len(self._store) >= self.max_entries
            ):
                if not self._evict_one():
                    break  # Nothing left to evict

            entry: _Entry[V] = _Entry(key=key, value=value, token_cost=token_cost)
            self._store[key] = entry  # type: ignore[index]
            self._current_tokens += token_cost

    def delete(self, key: K) -> bool:
        """Explicitly remove an entry. Returns True if it existed."""
        with self._lock:
            if key in self._store:  # type: ignore[operator]
                self._remove(key)  # type: ignore[arg-type]
                return True
        return False

    def clear(self) -> None:
        """Evict all entries."""
        with self._lock:
            self._store.clear()
            self._current_tokens = 0

    def stats(self) -> dict[str, object]:
        """Return a snapshot of cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "evictions": self._evictions,
                "expired_evictions": self._expired,
                "current_entries": len(self._store),
                "current_tokens": self._current_tokens,
                "max_tokens": self.max_tokens,
                "token_utilization": round(self._current_tokens / self.max_tokens, 4),
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            if self._is_expired(entry):
                return False
            return True

    def keys(self) -> Iterator[object]:
        """Iterate over non-expired keys (snapshot)."""
        with self._lock:
            snapshot = list(self._store.items())
        now = time.monotonic()
        for k, e in snapshot:
            if self.ttl <= 0 or (now - e.last_accessed) < self.ttl:
                yield k

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, entry: _Entry[V]) -> bool:
        if self.ttl <= 0:
            return False
        return (time.monotonic() - entry.last_accessed) >= self.ttl

    def _remove(self, key: object) -> None:
        """Remove an entry without holding a lock (caller must hold lock)."""
        entry = self._store.pop(key, None)
        if entry is not None:
            self._current_tokens -= entry.token_cost

    def _evict_one(self) -> bool:
        """
        Evict the entry with the lowest eviction priority.
        Caller must hold ``_lock``.
        """
        if not self._store:
            return False

        # First, try to evict expired entries cheaply
        for key, entry in list(self._store.items()):
            if self._is_expired(entry):
                self._remove(key)
                self._evictions += 1
                self._expired += 1
                return True

        # GD*-style: evict lowest priority
        victim_key = min(self._store.keys(), key=lambda k: self._store[k].eviction_priority())
        self._remove(victim_key)
        self._evictions += 1
        logger.debug("TokenCostAwareCache: evicted key=%s", victim_key)
        return True
