"""Semantic Chunker.

Splits text into chunks based on semantic similarity between adjacent sentences.
Instead of fixed token counts, it groups sentences that are semantically related,
splitting at points where the cosine similarity drops below a threshold or at
local minima.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List

import numpy as np

EmbedFn = Callable[[List[str]], Coroutine[Any, Any, List[List[float]]]]


@dataclass
class SemanticChunk:
    text: str
    start_idx: int
    end_idx: int


class SemanticChunker:
    """Chunks text by finding semantic boundaries between sentences.

    Supports configurable **sentence overlap** so that adjacent chunks share
    context, preventing retrieval misses when an answer spans a boundary.
    """

    def __init__(
        self,
        embed_fn: EmbedFn,
        similarity_threshold: float = 0.5,
        min_chunk_sentences: int = 1,
        max_chunk_sentences: int = 20,
        overlap_sentences: int = 0,
    ) -> None:
        self._embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences
        self.overlap_sentences = max(0, overlap_sentences)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Basic sentence splitter using regex."""
        # Split on punctuation followed by whitespace and a capital letter
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-10:
            return 0.0
        return float(np.dot(a, b) / denom)

    async def chunk(self, text: str) -> List[SemanticChunk]:
        """Split text into semantic chunks with optional sentence overlap."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [SemanticChunk(text=sentences[0], start_idx=0, end_idx=1)]

        # Embed all sentences in a single batch call
        embeddings_list = await self._embed_fn(sentences)
        embeddings = np.array(embeddings_list, dtype=np.float32)

        # Compute similarities between adjacent sentences
        similarities: list[float] = []
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # --- Build raw split points ---
        split_points: list[int] = []  # indices *after which* we split
        current_len = 1  # number of sentences in the current chunk

        for i, sim in enumerate(similarities):
            split = False
            if current_len >= self.max_chunk_sentences:
                split = True
            elif current_len >= self.min_chunk_sentences and sim < self.similarity_threshold:
                split = True

            if split:
                split_points.append(i + 1)  # chunk boundary after sentence i
                current_len = 1
            else:
                current_len += 1

        # --- Materialise chunks with overlap ---
        boundaries: list[tuple[int, int]] = []
        prev = 0
        for sp in split_points:
            boundaries.append((prev, sp))
            prev = sp
        boundaries.append((prev, len(sentences)))

        chunks: list[SemanticChunk] = []
        for idx, (start, end) in enumerate(boundaries):
            # Overlap: prepend the last `overlap_sentences` from the previous chunk
            overlap_start = (
                max(boundaries[idx - 1][1] - self.overlap_sentences, boundaries[idx - 1][0])
                if idx > 0 and self.overlap_sentences
                else start
            )

            effective_start = overlap_start if idx > 0 and self.overlap_sentences else start
            chunk_sents = sentences[effective_start:end]
            chunk_text = " ".join(chunk_sents)
            chunks.append(
                SemanticChunk(
                    text=chunk_text,
                    start_idx=effective_start,
                    end_idx=end,
                )
            )

        return chunks


def create_semantic_chunker(
    embed_fn: EmbedFn,
    similarity_threshold: float = 0.5,
) -> SemanticChunker:
    """Factory for `SemanticChunker`."""
    return SemanticChunker(embed_fn=embed_fn, similarity_threshold=similarity_threshold)
