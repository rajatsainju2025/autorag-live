"""Lightweight text tokenization utilities for repeated lower/split patterns.

Provides cached tokenization and set helpers to avoid repeated allocations and
repeated .lower().split() patterns across the codebase.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Set, Tuple

_WORD_RE = re.compile(r"\b\w+\b")


@lru_cache(maxsize=8192)
def tokenize(text: str) -> Tuple[str, ...]:
    """Return a tuple of normalized tokens for the given text.

    Uses a word regex and lowercasing. Cached to avoid repeated allocations for
    frequently seen strings.
    """
    if not text:
        return tuple()
    return tuple(_WORD_RE.findall(text.casefold()))


def tokenize_list(text: str) -> List[str]:
    """Return tokens as a list."""
    return list(tokenize(text))


def tokenize_set(text: str, min_length: int = 0) -> Set[str]:
    """Return a set of tokens, optionally filtering short tokens."""
    if min_length <= 0:
        return set(tokenize(text))
    return {t for t in tokenize(text) if len(t) >= min_length}
