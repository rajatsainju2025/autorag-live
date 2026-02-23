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
    """Chunks text by finding semantic boundaries between sentences."""

    def __init__(
        self,
        embed_fn: EmbedFn,
        similarity_threshold: float = 0.5,
        min_chunk_sentences: int = 1,
        max_chunk_sentences: int = 20,
    ) -> None:
        self._embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self.max_chunk_sentences = max_chunk_sentences

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
        """Split text into semantic chunks."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [SemanticChunk(text=sentences[0], start_idx=0, end_idx=1)]

        # Embed all sentences
        embeddings_list = await self._embed_fn(sentences)
        embeddings = np.array(embeddings_list, dtype=np.float32)

        # Compute similarities between adjacent sentences
        similarities: list[float] = []
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        chunks: list[SemanticChunk] = []
        current_chunk_sentences: list[str] = [sentences[0]]
        start_idx = 0

        for i, sim in enumerate(similarities):
            # Decide whether to split
            split = False
            if len(current_chunk_sentences) >= self.max_chunk_sentences:
                split = True
            elif len(current_chunk_sentences) >= self.min_chunk_sentences:
                if sim < self.similarity_threshold:
                    split = True

            if split:
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(
                    SemanticChunk(
                        text=chunk_text,
                        start_idx=start_idx,
                        end_idx=i + 1,
                    )
                )
                current_chunk_sentences = [sentences[i + 1]]
                start_idx = i + 1
            else:
                current_chunk_sentences.append(sentences[i + 1])

        # Add the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(
                SemanticChunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=len(sentences),
                )
            )

        return chunks


def create_semantic_chunker(
    embed_fn: EmbedFn,
    similarity_threshold: float = 0.5,
) -> SemanticChunker:
    """Factory for `SemanticChunker`."""
    return SemanticChunker(embed_fn=embed_fn, similarity_threshold=similarity_threshold)
