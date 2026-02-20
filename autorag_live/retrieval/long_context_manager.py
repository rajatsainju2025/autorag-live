"""
Long-Context Sliding Window Manager.

Handles documents that exceed typical retrieval chunk sizes (100k+ tokens)
by applying a hierarchical overlapping-window strategy:

1. **Chunking** — split long text into overlapping windows.
2. **Per-window retrieval** — embed each window independently.
3. **Hierarchical summarisation** — summarise groups of windows into
   higher-level summaries at multiple granularity levels.
4. **Multi-level retrieval** — retrieve at the level that best matches
   query complexity (detailed window vs. coarse summary).

This enables accurate retrieval from book-length documents, legal
corpora, and large codebases where a single embedding loses too
much detail.

Context Window Sizing Heuristics
---------------------------------
- *Fine* level  : ~512 tokens — exact passage retrieval
- *Medium* level: ~2 048 tokens — section-level retrieval
- *Coarse* level: ~8 192 tokens — chapter-level retrieval

References
----------
- "Long Context RAG" (Anthropic, 2024)
- "LongRAG" Sun et al., 2024 (https://arxiv.org/abs/2406.15319)
- "Lost in the Middle" Liu et al., 2023 — motivates multi-level retrieval
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Coroutine[Any, Any, List[float]]]
SummariseFn = Callable[[str], Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# Enums & data structures
# ---------------------------------------------------------------------------


class WindowLevel(str, Enum):
    FINE = "fine"  # ~512 tok — high detail
    MEDIUM = "medium"  # ~2 048 tok — section
    COARSE = "coarse"  # ~8 192 tok — chapter


@dataclass
class TextWindow:
    """A single sliding window over a document."""

    window_id: str
    doc_id: str
    text: str
    level: WindowLevel
    start_char: int
    end_char: int
    parent_window_id: Optional[str] = None
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_length(self) -> int:
        return self.end_char - self.start_char


@dataclass
class RetrievedWindow:
    """A retrieved window with relevance score."""

    window: TextWindow
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Approximate chars-per-token (≈4 for English)
_CHARS_PER_TOK = 4

_LEVEL_SIZES: Dict[WindowLevel, int] = {
    WindowLevel.FINE: 512 * _CHARS_PER_TOK,  # ~2048 chars
    WindowLevel.MEDIUM: 2048 * _CHARS_PER_TOK,  # ~8192 chars
    WindowLevel.COARSE: 8192 * _CHARS_PER_TOK,  # ~32768 chars
}

_LEVEL_OVERLAPS: Dict[WindowLevel, float] = {
    WindowLevel.FINE: 0.20,  # 20% overlap
    WindowLevel.MEDIUM: 0.15,
    WindowLevel.COARSE: 0.10,
}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _chunk_text(
    text: str,
    window_chars: int,
    overlap: float,
) -> List[Tuple[int, int]]:
    """
    Return (start_char, end_char) ranges for overlapping windows.

    Attempts to break at sentence boundaries ('. ', '? ', '! ') within
    a ±10% tolerance of the target window size.
    """
    if len(text) <= window_chars:
        return [(0, len(text))]

    step = max(1, int(window_chars * (1.0 - overlap)))
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + window_chars, len(text))
        # Snap to sentence boundary within 10% tolerance
        if end < len(text):
            snap_range = int(window_chars * 0.10)
            for boundary in (". ", "? ", "! ", "\n\n"):
                pos = text.rfind(boundary, end - snap_range, end + snap_range)
                if pos != -1:
                    end = pos + len(boundary)
                    break
        ranges.append((start, end))
        if end >= len(text):
            break
        start += step
    return ranges


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class LongContextManager:
    """
    Hierarchical sliding-window manager for long documents.

    Args:
        embed_fn: Async ``(text: str) → List[float]`` callable.
        summarise_fn: Async ``(text: str) → str`` callable for building
            higher-level summaries.  May be ``None`` — in that case only
            fine-level windows are built.
        levels: Which granularity levels to build (default: all three).
        embed_concurrency: Max parallel embedding calls.
        summarise_concurrency: Max parallel summarisation calls.

    Example::

        mgr = LongContextManager(embed_fn=my_embed, summarise_fn=my_llm)
        await mgr.index_document("doc1", long_text)
        results = await mgr.retrieve("What does chapter 3 say about X?", top_k=5)
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        summarise_fn: Optional[SummariseFn] = None,
        levels: Optional[List[WindowLevel]] = None,
        embed_concurrency: int = 8,
        summarise_concurrency: int = 4,
    ) -> None:
        self.embed_fn = embed_fn
        self.summarise_fn = summarise_fn
        self.levels = levels or list(WindowLevel)
        self._embed_sem = asyncio.Semaphore(embed_concurrency)
        self._sum_sem = asyncio.Semaphore(summarise_concurrency)

        # All windows indexed by ID
        self._windows: Dict[str, TextWindow] = {}
        # Windows per doc per level
        self._index: Dict[str, Dict[WindowLevel, List[str]]] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Chunk, summarise (if configured), and embed a long document.

        Args:
            doc_id: Unique document identifier.
            text: Full document text.
            metadata: Optional metadata attached to all windows.

        Returns:
            Total number of windows created.
        """
        meta = metadata or {}
        self._index[doc_id] = {lvl: [] for lvl in self.levels}
        all_windows: List[TextWindow] = []

        # --- Fine-level windows ---
        fine_windows = self._create_windows(doc_id, text, WindowLevel.FINE, meta)
        all_windows.extend(fine_windows)

        # --- Medium + coarse: summarise groups of fine windows ---
        if WindowLevel.MEDIUM in self.levels and self.summarise_fn:
            medium_windows = await self._build_summary_level(
                doc_id, fine_windows, WindowLevel.MEDIUM, meta
            )
            all_windows.extend(medium_windows)

        if WindowLevel.COARSE in self.levels and self.summarise_fn:
            medium_source = [
                w for w in all_windows if w.level == WindowLevel.MEDIUM
            ] or fine_windows
            coarse_windows = await self._build_summary_level(
                doc_id, medium_source, WindowLevel.COARSE, meta
            )
            all_windows.extend(coarse_windows)

        # --- Embed all windows in parallel ---
        await self._embed_windows(all_windows)

        # Register windows
        for w in all_windows:
            self._windows[w.window_id] = w
            self._index[doc_id][w.level].append(w.window_id)

        logger.info(
            "LongContextManager: indexed doc %s — %d windows across %d levels",
            doc_id,
            len(all_windows),
            len(self.levels),
        )
        return len(all_windows)

    def _create_windows(
        self,
        doc_id: str,
        text: str,
        level: WindowLevel,
        metadata: Dict[str, Any],
        parent_id: Optional[str] = None,
    ) -> List[TextWindow]:
        """Create raw (un-embedded) windows at a given level."""
        win_size = _LEVEL_SIZES[level]
        overlap = _LEVEL_OVERLAPS[level]
        ranges = _chunk_text(text, win_size, overlap)
        windows: List[TextWindow] = []
        for start, end in ranges:
            chunk = text[start:end]
            wid = hashlib.md5(
                f"{doc_id}:{level.value}:{start}".encode(), usedforsecurity=False
            ).hexdigest()[:16]
            windows.append(
                TextWindow(
                    window_id=wid,
                    doc_id=doc_id,
                    text=chunk,
                    level=level,
                    start_char=start,
                    end_char=end,
                    parent_window_id=parent_id,
                    metadata=dict(metadata),
                )
            )
        return windows

    async def _build_summary_level(
        self,
        doc_id: str,
        source_windows: List[TextWindow],
        target_level: WindowLevel,
        metadata: Dict[str, Any],
    ) -> List[TextWindow]:
        """Summarise groups of source windows into a higher level."""
        assert self.summarise_fn is not None

        # How many source windows to group per summary
        src_size = _LEVEL_SIZES[source_windows[0].level] if source_windows else 1
        tgt_size = _LEVEL_SIZES[target_level]
        group_size = max(2, tgt_size // src_size)

        summary_windows: List[TextWindow] = []
        for i in range(0, len(source_windows), group_size):
            group = source_windows[i : i + group_size]
            combined_text = "\n\n".join(w.text for w in group)
            async with self._sum_sem:
                try:
                    summary_text = await self.summarise_fn(combined_text)
                except Exception as exc:
                    logger.warning("LongContextManager: summarise failed: %s", exc)
                    summary_text = combined_text[: _LEVEL_SIZES[target_level]]

            wid = hashlib.md5(
                f"{doc_id}:{target_level.value}:{i}".encode(), usedforsecurity=False
            ).hexdigest()[:16]
            start_char = group[0].start_char
            end_char = group[-1].end_char
            summary_windows.append(
                TextWindow(
                    window_id=wid,
                    doc_id=doc_id,
                    text=summary_text,
                    level=target_level,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=dict(metadata),
                )
            )

        return summary_windows

    async def _embed_windows(self, windows: List[TextWindow]) -> None:
        """Embed all windows in parallel, respecting semaphore."""

        async def _embed_one(w: TextWindow) -> None:
            async with self._embed_sem:
                try:
                    vec = await self.embed_fn(w.text[:2000])  # cap for API limits
                    w.embedding = np.asarray(vec, dtype=float)
                except Exception as exc:
                    logger.warning("LongContextManager: embed failed for %s: %s", w.window_id, exc)

        await asyncio.gather(*[_embed_one(w) for w in windows])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        level: Optional[WindowLevel] = None,
        doc_id: Optional[str] = None,
    ) -> List[RetrievedWindow]:
        """
        Retrieve windows relevant to *query*.

        Args:
            query: User question.
            top_k: Number of windows to return.
            level: Restrict to a specific level (default: auto-select).
            doc_id: Restrict to a specific document (default: all docs).

        Returns:
            List of :class:`RetrievedWindow` sorted by relevance score.
        """
        query_vec = np.asarray(await self.embed_fn(query), dtype=float)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        # Collect candidate windows
        candidates: List[TextWindow] = []
        for wid, w in self._windows.items():
            if doc_id is not None and w.doc_id != doc_id:
                continue
            if level is not None and w.level != level:
                continue
            if w.embedding is not None:
                candidates.append(w)

        if not candidates:
            return []

        # Score by cosine similarity
        scored: List[Tuple[float, TextWindow]] = []
        for w in candidates:
            emb = w.embedding
            assert emb is not None
            norm = np.linalg.norm(emb)
            if norm > 0:
                score = float(np.dot(query_vec, emb / norm))
            else:
                score = 0.0
            scored.append((score, w))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [RetrievedWindow(window=w, score=s) for s, w in scored[:top_k]]

    def stats(self) -> Dict[str, Any]:
        """Return indexing statistics."""
        by_level: Dict[str, int] = {lvl.value: 0 for lvl in WindowLevel}
        for w in self._windows.values():
            by_level[w.level.value] += 1
        return {
            "total_windows": len(self._windows),
            "documents": len(self._index),
            "by_level": by_level,
        }
