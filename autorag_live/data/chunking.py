"""
Document Chunking Strategies for RAG Pipeline.

Provides multiple strategies for splitting documents into chunks
optimized for retrieval and context management.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    HIERARCHICAL = "hierarchical"
    RECURSIVE = "recursive"


@dataclass
class DocumentChunk:
    """A chunk of a document."""

    content: str
    chunk_id: str
    document_id: str
    start_index: int
    end_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Get chunk length in characters."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Get word count."""
        return len(self.content.split())


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""

    strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    separator: str = "\n\n"
    preserve_sentences: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseChunker(ABC):
    """Base class for document chunkers."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize chunker."""
        self.config = config or ChunkingConfig()

    @abstractmethod
    def chunk(self, text: str, document_id: str = "") -> list[DocumentChunk]:
        """Split text into chunks."""
        pass

    def _create_chunk(
        self,
        content: str,
        document_id: str,
        chunk_index: int,
        start_index: int,
        end_index: int,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> DocumentChunk:
        """Create a document chunk with metadata."""
        metadata = {
            "chunk_index": chunk_index,
            "strategy": self.config.strategy.value,
            **self.config.metadata,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return DocumentChunk(
            content=content.strip(),
            chunk_id=f"{document_id}_{chunk_index}",
            document_id=document_id,
            start_index=start_index,
            end_index=end_index,
            metadata=metadata,
        )


class FixedSizeChunker(BaseChunker):
    """Chunk documents by fixed character size."""

    def chunk(self, text: str, document_id: str = "") -> list[DocumentChunk]:
        """Split text into fixed-size chunks."""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size

            if self.config.preserve_sentences and end < len(text):
                sentence_end = text.rfind(".", start, end + 50)
                if sentence_end > start + self.config.min_chunk_size:
                    end = sentence_end + 1

            chunk_text = text[start:end]

            if len(chunk_text.strip()) >= self.config.min_chunk_size:
                chunks.append(
                    self._create_chunk(
                        chunk_text, document_id, chunk_index, start, end
                    )
                )
                chunk_index += 1

            start = end - overlap
            if start >= len(text) - self.config.min_chunk_size:
                break

        return chunks


class SentenceChunker(BaseChunker):
    """Chunk documents by sentences."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize sentence chunker."""
        super().__init__(config)
        self._sentence_pattern = re.compile(
            r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$"
        )

    def chunk(self, text: str, document_id: str = "") -> list[DocumentChunk]:
        """Split text into sentence-based chunks."""
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        start_index = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.config.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                end_index = start_index + len(chunk_text)

                chunks.append(
                    self._create_chunk(
                        chunk_text, document_id, chunk_index, start_index, end_index
                    )
                )
                chunk_index += 1

                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.config.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length
                start_index = end_index - overlap_length

            current_chunk.append(sentence)
            current_length += sentence_len

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                self._create_chunk(
                    chunk_text,
                    document_id,
                    chunk_index,
                    start_index,
                    start_index + len(chunk_text),
                )
            )

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = self._sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]


class ParagraphChunker(BaseChunker):
    """Chunk documents by paragraphs."""

    def chunk(self, text: str, document_id: str = "") -> list[DocumentChunk]:
        """Split text into paragraph-based chunks."""
        paragraphs = text.split(self.config.separator)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        current_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            if para_len > self.config.max_chunk_size:
                if current_chunk:
                    chunk_text = self.config.separator.join(current_chunk)
                    chunks.append(
                        self._create_chunk(
                            chunk_text,
                            document_id,
                            chunk_index,
                            current_start,
                            current_start + len(chunk_text),
                        )
                    )
                    chunk_index += 1
                    current_chunk = []
                    current_length = 0

                sub_chunker = FixedSizeChunker(self.config)
                sub_chunks = sub_chunker.chunk(para, document_id)
                for sub_chunk in sub_chunks:
                    sub_chunk.chunk_id = f"{document_id}_{chunk_index}"
                    sub_chunk.metadata["is_split_paragraph"] = True
                    chunks.append(sub_chunk)
                    chunk_index += 1
                continue

            if current_length + para_len > self.config.chunk_size and current_chunk:
                chunk_text = self.config.separator.join(current_chunk)
                chunks.append(
                    self._create_chunk(
                        chunk_text,
                        document_id,
                        chunk_index,
                        current_start,
                        current_start + len(chunk_text),
                    )
                )
                chunk_index += 1
                current_chunk = []
                current_length = 0
                current_start = current_start + len(chunk_text) + len(self.config.separator)

            current_chunk.append(para)
            current_length += para_len

        if current_chunk:
            chunk_text = self.config.separator.join(current_chunk)
            chunks.append(
                self._create_chunk(
                    chunk_text,
                    document_id,
                    chunk_index,
                    current_start,
                    current_start + len(chunk_text),
                )
            )

        return chunks


class SlidingWindowChunker(BaseChunker):
    """Chunk documents using sliding window approach."""

    def chunk(self, text: str, document_id: str = "") -> list[DocumentChunk]:
        """Create overlapping chunks using sliding window."""
        chunks = []
        window_size = self.config.chunk_size
        step_size = window_size - self.config.chunk_overlap

        chunk_index = 0
        for start in range(0, len(text) - self.config.min_chunk_size, step_size):
            end = min(start + window_size, len(text))
            chunk_text = text[start:end]

            if self.config.preserve_sentences:
                if start > 0:
                    first_sentence = chunk_text.find(". ")
                    if first_sentence > 0 and first_sentence < len(chunk_text) // 4:
                        chunk_text = chunk_text[first_sentence + 2 :]

                if end < len(text):
                    last_sentence = chunk_text.rfind(". ")
                    if (
                        last_sentence > 0
                        and last_sentence > len(chunk_text) * 3 // 4
                    ):
                        chunk_text = chunk_text[: last_sentence + 1]

            if len(chunk_text.strip()) >= self.config.min_chunk_size:
                chunks.append(
                    self._create_chunk(
                        chunk_text,
                        document_id,
                        chunk_index,
                        start,
                        end,
                        {"window_start": start, "window_end": end},
                    )
                )
                chunk_index += 1

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursively chunk using multiple separators."""

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        separators: Optional[list[str]] = None,
    ):
        """Initialize recursive chunker."""
        super().__init__(config)
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def chunk(self, text: str, document_id: str = "") -> list[DocumentChunk]:
        """Recursively split text using multiple separators."""
        chunks = self._recursive_split(text, self.separators)
        result = []

        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) >= self.config.min_chunk_size:
                start = text.find(chunk_text)
                result.append(
                    self._create_chunk(
                        chunk_text,
                        document_id,
                        i,
                        start if start >= 0 else 0,
                        (start if start >= 0 else 0) + len(chunk_text),
                    )
                )

        return result

    def _recursive_split(
        self, text: str, separators: list[str], depth: int = 0
    ) -> list[str]:
        """Recursively split text."""
        if not text or not separators:
            return [text] if text else []

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator not in text:
            if remaining_separators:
                return self._recursive_split(text, remaining_separators, depth + 1)
            return [text]

        splits = text.split(separator)
        chunks = []
        current_chunk = ""

        for split in splits:
            potential = current_chunk + separator + split if current_chunk else split

            if len(potential) <= self.config.chunk_size:
                current_chunk = potential
            else:
                if current_chunk:
                    if len(current_chunk) > self.config.max_chunk_size:
                        if remaining_separators:
                            sub_chunks = self._recursive_split(
                                current_chunk, remaining_separators, depth + 1
                            )
                            chunks.extend(sub_chunks)
                        else:
                            chunks.append(current_chunk)
                    else:
                        chunks.append(current_chunk)
                current_chunk = split

        if current_chunk:
            if len(current_chunk) > self.config.max_chunk_size and remaining_separators:
                sub_chunks = self._recursive_split(
                    current_chunk, remaining_separators, depth + 1
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(current_chunk)

        return chunks


class HierarchicalChunker(BaseChunker):
    """Create hierarchical chunks at multiple granularities."""

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        levels: Optional[list[int]] = None,
    ):
        """Initialize hierarchical chunker."""
        super().__init__(config)
        self.levels = levels or [2048, 512, 128]

    def chunk(self, text: str, document_id: str = "") -> list[DocumentChunk]:
        """Create hierarchical chunks at multiple levels."""
        all_chunks = []
        chunk_index = 0

        for level_idx, chunk_size in enumerate(self.levels):
            level_config = ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=min(chunk_size // 4, self.config.chunk_overlap),
                min_chunk_size=max(chunk_size // 4, 50),
                preserve_sentences=self.config.preserve_sentences,
            )
            level_chunker = FixedSizeChunker(level_config)
            level_chunks = level_chunker.chunk(text, document_id)

            for chunk in level_chunks:
                chunk.chunk_id = f"{document_id}_L{level_idx}_{chunk_index}"
                chunk.metadata["level"] = level_idx
                chunk.metadata["level_chunk_size"] = chunk_size
                all_chunks.append(chunk)
                chunk_index += 1

        return all_chunks


class SemanticChunker(BaseChunker):
    """Chunk based on semantic similarity."""

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        embedding_fn: Optional[Callable[[str], list[float]]] = None,
        similarity_threshold: float = 0.5,
    ):
        """Initialize semantic chunker."""
        super().__init__(config)
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str, document_id: str = "") -> list[DocumentChunk]:
        """Split text based on semantic boundaries."""
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        if not self.embedding_fn:
            sentence_chunker = SentenceChunker(self.config)
            return sentence_chunker.chunk(text, document_id)

        embeddings = [self.embedding_fn(s) for s in sentences]

        break_points = [0]
        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if similarity < self.similarity_threshold:
                break_points.append(i)
        break_points.append(len(sentences))

        chunks = []
        for i in range(len(break_points) - 1):
            start_idx = break_points[i]
            end_idx = break_points[i + 1]
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = " ".join(chunk_sentences)

            if len(chunk_text) >= self.config.min_chunk_size:
                start_pos = text.find(chunk_sentences[0])
                chunks.append(
                    self._create_chunk(
                        chunk_text,
                        document_id,
                        i,
                        start_pos if start_pos >= 0 else 0,
                        (start_pos if start_pos >= 0 else 0) + len(chunk_text),
                        {"semantic_boundary": True},
                    )
                )

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        pattern = re.compile(r"(?<=[.!?])\s+")
        sentences = pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity."""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


class DocumentChunkerFactory:
    """Factory for creating document chunkers."""

    _chunkers: dict[ChunkingStrategy, type[BaseChunker]] = {
        ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
        ChunkingStrategy.SENTENCE: SentenceChunker,
        ChunkingStrategy.PARAGRAPH: ParagraphChunker,
        ChunkingStrategy.SLIDING_WINDOW: SlidingWindowChunker,
        ChunkingStrategy.RECURSIVE: RecursiveChunker,
        ChunkingStrategy.HIERARCHICAL: HierarchicalChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
    }

    @classmethod
    def create(
        cls,
        strategy: ChunkingStrategy,
        config: Optional[ChunkingConfig] = None,
        **kwargs,
    ) -> BaseChunker:
        """Create a chunker for the specified strategy."""
        chunker_class = cls._chunkers.get(strategy)
        if not chunker_class:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        return chunker_class(config, **kwargs)

    @classmethod
    def register(
        cls, strategy: ChunkingStrategy, chunker_class: type[BaseChunker]
    ) -> None:
        """Register a custom chunker."""
        cls._chunkers[strategy] = chunker_class


__all__ = [
    "ChunkingStrategy",
    "DocumentChunk",
    "ChunkingConfig",
    "BaseChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "SlidingWindowChunker",
    "RecursiveChunker",
    "HierarchicalChunker",
    "SemanticChunker",
    "DocumentChunkerFactory",
]
