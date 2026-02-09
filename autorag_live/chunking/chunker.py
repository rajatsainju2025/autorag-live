"""
Text chunking module for AutoRAG-Live.

Provides intelligent text splitting strategies for optimal
retrieval performance with various chunking methods.

Features:
- Multiple chunking strategies (fixed, semantic, recursive)
- Overlap support for context preservation
- Sentence-aware splitting
- Code-aware chunking
- Metadata preservation
- Token counting integration

Example usage:
    >>> chunker = TextChunker(strategy="semantic", chunk_size=512)
    >>> chunks = chunker.chunk(document_text)
    >>> for chunk in chunks:
    ...     print(f"Chunk {chunk.index}: {len(chunk.text)} chars")
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies."""

    FIXED = auto()  # Fixed character/token size
    SENTENCE = auto()  # Split by sentences
    PARAGRAPH = auto()  # Split by paragraphs
    SEMANTIC = auto()  # Semantic similarity based
    RECURSIVE = auto()  # Recursive character splitting
    CODE = auto()  # Code-aware splitting
    MARKDOWN = auto()  # Markdown structure aware
    SLIDING_WINDOW = auto()  # Sliding window approach


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""

    # Position info
    start_char: int = 0
    end_char: int = 0
    start_line: int = 0
    end_line: int = 0

    # Source info
    source_id: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None

    # Chunk info
    has_overlap: bool = False
    overlap_chars: int = 0

    # Custom metadata
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextChunk:
    """A chunk of text with metadata."""

    text: str
    index: int

    # Metadata
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)

    # Token count (if computed)
    token_count: Optional[int] = None

    # Embedding (if computed)
    embedding: Optional[List[float]] = None

    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Get word count."""
        return len(self.text.split())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "index": self.index,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "token_count": self.token_count,
            "metadata": {
                "start_char": self.metadata.start_char,
                "end_char": self.metadata.end_char,
                "source_id": self.metadata.source_id,
            },
        }


class BaseChunker(ABC):
    """Base class for text chunkers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Chunker name."""
        pass

    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Split text into chunks."""
        pass


class FixedSizeChunker(BaseChunker):
    """Fixed-size character chunking."""

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        strip_whitespace: bool = True,
    ):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            strip_whitespace: Strip leading/trailing whitespace
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strip_whitespace = strip_whitespace

    @property
    def name(self) -> str:
        return "fixed_size"

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Split text into fixed-size chunks."""
        chunks = []

        if not text:
            return chunks

        start = 0
        index = 0

        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))

            # Extract chunk text
            chunk_text = text[start:end]

            if self.strip_whitespace:
                chunk_text = chunk_text.strip()

            if chunk_text:  # Only add non-empty chunks
                chunk_metadata = ChunkMetadata(
                    start_char=start,
                    end_char=end,
                    has_overlap=index > 0 and self.overlap > 0,
                    overlap_chars=min(self.overlap, start) if index > 0 else 0,
                )

                if metadata:
                    chunk_metadata.custom.update(metadata)

                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        index=index,
                        metadata=chunk_metadata,
                    )
                )

                index += 1

            # Move start position
            start = end - self.overlap if end < len(text) else end

        return chunks


class SentenceChunker(BaseChunker):
    """Sentence-based chunking."""

    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        max_chunk_size: int = 1000,
        min_chunk_size: int = 100,
        overlap_sentences: int = 1,
    ):
        """
        Initialize sentence chunker.

        Args:
            max_chunk_size: Maximum chunk size
            min_chunk_size: Minimum chunk size
            overlap_sentences: Number of sentences to overlap
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_sentences = overlap_sentences

    @property
    def name(self) -> str:
        return "sentence"

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self.SENTENCE_ENDINGS.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Split text by sentences."""
        chunks = []

        sentences = self._split_sentences(text)
        if not sentences:
            return chunks

        current_chunk = []
        current_length = 0
        index = 0
        char_pos = 0

        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence) + 1  # +1 for space

            # Check if adding sentence exceeds max
            if current_length + sentence_len > self.max_chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)

                chunk_metadata = ChunkMetadata(
                    start_char=char_pos - len(chunk_text),
                    end_char=char_pos,
                )
                if metadata:
                    chunk_metadata.custom.update(metadata)

                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        index=index,
                        metadata=chunk_metadata,
                    )
                )

                index += 1

                # Keep overlap sentences
                if self.overlap_sentences > 0:
                    current_chunk = current_chunk[-self.overlap_sentences :]
                    current_length = sum(len(s) + 1 for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_len
            char_pos += sentence_len

        # Add remaining sentences
        if current_chunk:
            chunk_text = " ".join(current_chunk)

            if len(chunk_text) >= self.min_chunk_size or not chunks:
                chunk_metadata = ChunkMetadata(
                    start_char=char_pos - len(chunk_text),
                    end_char=char_pos,
                )
                if metadata:
                    chunk_metadata.custom.update(metadata)

                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        index=index,
                        metadata=chunk_metadata,
                    )
                )

        return chunks


class ParagraphChunker(BaseChunker):
    """Paragraph-based chunking."""

    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")

    def __init__(
        self,
        max_chunk_size: int = 2000,
        combine_small: bool = True,
        min_paragraph_size: int = 50,
    ):
        """
        Initialize paragraph chunker.

        Args:
            max_chunk_size: Maximum chunk size
            combine_small: Combine small paragraphs
            min_paragraph_size: Minimum paragraph size
        """
        self.max_chunk_size = max_chunk_size
        self.combine_small = combine_small
        self.min_paragraph_size = min_paragraph_size

    @property
    def name(self) -> str:
        return "paragraph"

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Split text by paragraphs."""
        chunks = []

        paragraphs = self.PARAGRAPH_PATTERN.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return chunks

        current_chunk = []
        current_length = 0
        index = 0

        for para in paragraphs:
            para_len = len(para)

            # Check if paragraph alone exceeds max
            if para_len > self.max_chunk_size:
                # Save current chunk first
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunk_metadata = ChunkMetadata()
                    if metadata:
                        chunk_metadata.custom.update(metadata)

                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            index=index,
                            metadata=chunk_metadata,
                        )
                    )
                    index += 1
                    current_chunk = []
                    current_length = 0

                # Split large paragraph
                sub_chunker = FixedSizeChunker(
                    chunk_size=self.max_chunk_size,
                    overlap=200,
                )
                for sub_chunk in sub_chunker.chunk(para, metadata):
                    sub_chunk.index = index
                    chunks.append(sub_chunk)
                    index += 1
                continue

            # Check if adding paragraph exceeds max
            if current_length + para_len > self.max_chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_metadata = ChunkMetadata()
                if metadata:
                    chunk_metadata.custom.update(metadata)

                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        index=index,
                        metadata=chunk_metadata,
                    )
                )
                index += 1
                current_chunk = []
                current_length = 0

            current_chunk.append(para)
            current_length += para_len + 2  # +2 for paragraph separator

        # Add remaining
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_metadata = ChunkMetadata()
            if metadata:
                chunk_metadata.custom.update(metadata)

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    index=index,
                    metadata=chunk_metadata,
                )
            )

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursive character text splitting."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize recursive chunker.

        Args:
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            separators: List of separators in order of preference
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    @property
    def name(self) -> str:
        return "recursive"

    def _split_text(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """Recursively split text."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator:
            splits = text.split(separator)
        else:
            # Character-level split
            splits = list(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split) + len(separator)

            if current_length + split_len > self.chunk_size:
                if current_chunk:
                    merged = separator.join(current_chunk)

                    if len(merged) > self.chunk_size and remaining_separators:
                        # Recursively split
                        sub_chunks = self._split_text(merged, remaining_separators)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(merged)

                    current_chunk = []
                    current_length = 0

            current_chunk.append(split)
            current_length += split_len

        if current_chunk:
            merged = separator.join(current_chunk)
            if len(merged) > self.chunk_size and remaining_separators:
                sub_chunks = self._split_text(merged, remaining_separators)
                chunks.extend(sub_chunks)
            else:
                chunks.append(merged)

        return chunks

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Recursively split text into chunks."""
        raw_chunks = self._split_text(text, self.separators)

        chunks = []
        char_pos = 0

        for i, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            chunk_metadata = ChunkMetadata(
                start_char=char_pos,
                end_char=char_pos + len(chunk_text),
            )
            if metadata:
                chunk_metadata.custom.update(metadata)

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    index=len(chunks),
                    metadata=chunk_metadata,
                )
            )

            char_pos += len(chunk_text)

        return chunks


class CodeChunker(BaseChunker):
    """Code-aware chunking."""

    # Patterns for code structure
    FUNCTION_PATTERNS = {
        "python": re.compile(r"^(async\s+)?(def|class)\s+\w+", re.MULTILINE),
        "javascript": re.compile(r"^(async\s+)?(function|class|const|let|var)\s+\w+", re.MULTILINE),
        "generic": re.compile(r"^[a-zA-Z_]\w*\s*[({]", re.MULTILINE),
    }

    def __init__(
        self,
        chunk_size: int = 1500,
        language: str = "python",
        preserve_functions: bool = True,
    ):
        """
        Initialize code chunker.

        Args:
            chunk_size: Target chunk size
            language: Programming language
            preserve_functions: Keep functions together
        """
        self.chunk_size = chunk_size
        self.language = language
        self.preserve_functions = preserve_functions
        self._pattern = self.FUNCTION_PATTERNS.get(language, self.FUNCTION_PATTERNS["generic"])

    @property
    def name(self) -> str:
        return "code"

    def _find_code_blocks(self, text: str) -> List[Tuple[int, int]]:
        """Find code block boundaries."""
        blocks = []

        for match in self._pattern.finditer(text):
            start = match.start()

            # Find end of block (next block or end of text)
            end = len(text)

            # Look for next function/class
            next_match = self._pattern.search(text, match.end())
            if next_match:
                end = next_match.start()

            blocks.append((start, end))

        return blocks

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Split code into chunks."""
        chunks = []

        if self.preserve_functions:
            blocks = self._find_code_blocks(text)

            if blocks:
                current_chunk = []
                current_length = 0
                index = 0

                for start, end in blocks:
                    block_text = text[start:end].strip()
                    block_len = len(block_text)

                    if current_length + block_len > self.chunk_size and current_chunk:
                        chunk_text = "\n\n".join(current_chunk)
                        chunk_metadata = ChunkMetadata(
                            custom={"language": self.language},
                        )
                        if metadata:
                            chunk_metadata.custom.update(metadata)

                        chunks.append(
                            TextChunk(
                                text=chunk_text,
                                index=index,
                                metadata=chunk_metadata,
                            )
                        )
                        index += 1
                        current_chunk = []
                        current_length = 0

                    current_chunk.append(block_text)
                    current_length += block_len

                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunk_metadata = ChunkMetadata(
                        custom={"language": self.language},
                    )
                    if metadata:
                        chunk_metadata.custom.update(metadata)

                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            index=index,
                            metadata=chunk_metadata,
                        )
                    )

                return chunks

        # Fall back to line-based chunking
        lines = text.split("\n")
        current_chunk = []
        current_length = 0
        index = 0

        for line in lines:
            line_len = len(line) + 1

            if current_length + line_len > self.chunk_size and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunk_metadata = ChunkMetadata(
                    custom={"language": self.language},
                )
                if metadata:
                    chunk_metadata.custom.update(metadata)

                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        index=index,
                        metadata=chunk_metadata,
                    )
                )
                index += 1
                current_chunk = []
                current_length = 0

            current_chunk.append(line)
            current_length += line_len

        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunk_metadata = ChunkMetadata(
                custom={"language": self.language},
            )
            if metadata:
                chunk_metadata.custom.update(metadata)

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    index=index,
                    metadata=chunk_metadata,
                )
            )

        return chunks


class MarkdownChunker(BaseChunker):
    """Markdown-aware chunking."""

    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)

    def __init__(
        self,
        chunk_size: int = 1500,
        respect_headers: bool = True,
        preserve_code_blocks: bool = True,
    ):
        """
        Initialize markdown chunker.

        Args:
            chunk_size: Target chunk size
            respect_headers: Split on headers
            preserve_code_blocks: Keep code blocks together
        """
        self.chunk_size = chunk_size
        self.respect_headers = respect_headers
        self.preserve_code_blocks = preserve_code_blocks

    @property
    def name(self) -> str:
        return "markdown"

    def _extract_sections(self, text: str) -> List[Tuple[str, str, int]]:
        """Extract sections with headers."""
        sections = []

        # Find all headers
        headers = list(self.HEADER_PATTERN.finditer(text))

        if not headers:
            return [("", text, 0)]

        for i, match in enumerate(headers):
            header_level = len(match.group(1))
            header_text = match.group(2)

            start = match.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)

            section_text = text[start:end].strip()
            sections.append((header_text, section_text, header_level))

        # Add any text before first header
        if headers[0].start() > 0:
            preamble = text[: headers[0].start()].strip()
            if preamble:
                sections.insert(0, ("", preamble, 0))

        return sections

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Split markdown into chunks."""
        chunks = []

        if self.respect_headers:
            sections = self._extract_sections(text)

            current_chunk = []
            current_length = 0
            index = 0

            for header, content, level in sections:
                content_len = len(content)

                # Large section - split it
                if content_len > self.chunk_size:
                    # Save current chunk
                    if current_chunk:
                        chunk_text = "\n\n".join(current_chunk)
                        chunk_metadata = ChunkMetadata()
                        if metadata:
                            chunk_metadata.custom.update(metadata)

                        chunks.append(
                            TextChunk(
                                text=chunk_text,
                                index=index,
                                metadata=chunk_metadata,
                            )
                        )
                        index += 1
                        current_chunk = []
                        current_length = 0

                    # Split large section
                    sub_chunker = ParagraphChunker(
                        max_chunk_size=self.chunk_size,
                    )
                    for sub_chunk in sub_chunker.chunk(content, metadata):
                        sub_chunk.index = index
                        sub_chunk.metadata.section = header
                        chunks.append(sub_chunk)
                        index += 1
                    continue

                # Check size
                if current_length + content_len > self.chunk_size and current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunk_metadata = ChunkMetadata()
                    if metadata:
                        chunk_metadata.custom.update(metadata)

                    chunks.append(
                        TextChunk(
                            text=chunk_text,
                            index=index,
                            metadata=chunk_metadata,
                        )
                    )
                    index += 1
                    current_chunk = []
                    current_length = 0

                current_chunk.append(content)
                current_length += content_len

            # Add remaining
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_metadata = ChunkMetadata()
                if metadata:
                    chunk_metadata.custom.update(metadata)

                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        index=index,
                        metadata=chunk_metadata,
                    )
                )
        else:
            # Fall back to paragraph chunking
            para_chunker = ParagraphChunker(max_chunk_size=self.chunk_size)
            chunks = para_chunker.chunk(text, metadata)

        return chunks


class SlidingWindowChunker(BaseChunker):
    """Sliding window chunking."""

    def __init__(
        self,
        window_size: int = 512,
        step_size: int = 256,
    ):
        """
        Initialize sliding window chunker.

        Args:
            window_size: Size of sliding window
            step_size: Step size between windows
        """
        self.window_size = window_size
        self.step_size = step_size

    @property
    def name(self) -> str:
        return "sliding_window"

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Create sliding window chunks."""
        chunks = []

        if len(text) <= self.window_size:
            chunk_metadata = ChunkMetadata(
                start_char=0,
                end_char=len(text),
            )
            if metadata:
                chunk_metadata.custom.update(metadata)

            return [
                TextChunk(
                    text=text,
                    index=0,
                    metadata=chunk_metadata,
                )
            ]

        position = 0
        index = 0

        while position < len(text):
            end = min(position + self.window_size, len(text))
            chunk_text = text[position:end]

            chunk_metadata = ChunkMetadata(
                start_char=position,
                end_char=end,
                has_overlap=position > 0,
                overlap_chars=self.window_size - self.step_size if position > 0 else 0,
            )
            if metadata:
                chunk_metadata.custom.update(metadata)

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    index=index,
                    metadata=chunk_metadata,
                )
            )

            index += 1
            position += self.step_size

            if end >= len(text):
                break

        return chunks


class TextChunker:
    """
    Main text chunking interface.

    Example:
        >>> # Basic usage
        >>> chunker = TextChunker(strategy="sentence", chunk_size=500)
        >>> chunks = chunker.chunk("Long document text...")
        >>>
        >>> # With custom strategy
        >>> chunker = TextChunker(
        ...     strategy="recursive",
        ...     chunk_size=1000,
        ...     overlap=200,
        ... )
        >>> chunks = chunker.chunk(text, metadata={"source": "doc.pdf"})
    """

    CHUNKERS = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceChunker,
        "paragraph": ParagraphChunker,
        "recursive": RecursiveChunker,
        "code": CodeChunker,
        "markdown": MarkdownChunker,
        "sliding_window": SlidingWindowChunker,
    }

    def __init__(
        self,
        strategy: Union[str, ChunkingStrategy] = "recursive",
        chunk_size: int = 1000,
        overlap: int = 200,
        **kwargs,
    ):
        """
        Initialize text chunker.

        Args:
            strategy: Chunking strategy
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            **kwargs: Strategy-specific parameters
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.kwargs = kwargs

        self._chunker = self._create_chunker()

    def _create_chunker(self) -> BaseChunker:
        """Create chunker instance."""
        strategy_name = (
            self.strategy.name.lower()
            if isinstance(self.strategy, ChunkingStrategy)
            else self.strategy.lower()
        )

        if strategy_name not in self.CHUNKERS:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        chunker_class = self.CHUNKERS[strategy_name]

        # Map common params
        params = {"chunk_size": self.chunk_size}

        if strategy_name in ("fixed", "recursive", "sliding_window"):
            params["overlap"] = self.overlap

        params.update(self.kwargs)

        # Filter valid params for chunker
        import inspect

        sig = inspect.signature(chunker_class.__init__)
        valid_params = {k: v for k, v in params.items() if k in sig.parameters}

        return chunker_class(**valid_params)

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """
        Chunk text using configured strategy.

        Args:
            text: Text to chunk
            metadata: Optional metadata

        Returns:
            List of TextChunk objects
        """
        return self._chunker.chunk(text, metadata)

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
    ) -> List[TextChunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dicts
            text_field: Field containing text

        Returns:
            List of all chunks
        """
        all_chunks = []
        chunk_index = 0

        for doc in documents:
            text = doc.get(text_field, "")
            if not text:
                continue

            # Create metadata from doc
            metadata = {k: v for k, v in doc.items() if k != text_field}

            chunks = self.chunk(text, metadata)

            # Update global indices
            for chunk in chunks:
                chunk.index = chunk_index
                chunk_index += 1
                all_chunks.append(chunk)

        return all_chunks


# Convenience functions


def chunk_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 1000,
) -> List[TextChunk]:
    """
    Quick text chunking.

    Args:
        text: Text to chunk
        strategy: Chunking strategy
        chunk_size: Target chunk size

    Returns:
        List of chunks
    """
    chunker = TextChunker(strategy=strategy, chunk_size=chunk_size)
    return chunker.chunk(text)


def chunk_code(
    code: str,
    language: str = "python",
    chunk_size: int = 1500,
) -> List[TextChunk]:
    """
    Chunk source code.

    Args:
        code: Source code
        language: Programming language
        chunk_size: Target chunk size

    Returns:
        List of chunks
    """
    chunker = TextChunker(
        strategy="code",
        chunk_size=chunk_size,
        language=language,
    )
    return chunker.chunk(code)


def chunk_markdown(
    text: str,
    chunk_size: int = 1500,
) -> List[TextChunk]:
    """
    Chunk markdown text.

    Args:
        text: Markdown text
        chunk_size: Target chunk size

    Returns:
        List of chunks
    """
    chunker = TextChunker(strategy="markdown", chunk_size=chunk_size)
    return chunker.chunk(text)


# =============================================================================
# OPTIMIZATION 1: Proposition-Level Chunking (Dense X Retrieval)
# Based on: "Dense X Retrieval: What Retrieval Granularity Should We Use?"
# (Chen et al., 2024) - Microsoft Research
#
# Decomposes documents into atomic propositions (single facts) for
# fine-grained retrieval with 15-25% recall improvement on multi-hop QA.
# =============================================================================


class LLMProtocol(Protocol):
    """Protocol for LLM interactions."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response from prompt."""
        ...


@dataclass
class Proposition:
    """An atomic factual proposition extracted from text."""

    text: str
    source_sentence: str
    confidence: float = 1.0
    entities: List[str] = field(default_factory=list)
    source_chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PropositionChunkResult:
    """Result of proposition-level chunking."""

    propositions: List[Proposition]
    original_chunks: List[TextChunk]
    extraction_stats: Dict[str, Any] = field(default_factory=dict)


class PropositionChunker(BaseChunker):
    """
    Proposition-level chunker using LLM decomposition.

    Decomposes text into atomic factual propositions where each
    proposition contains exactly one fact and is self-contained.

    This enables fine-grained retrieval that can match specific
    facts rather than entire paragraphs.

    Example:
        >>> chunker = PropositionChunker(llm=my_llm)
        >>> result = await chunker.chunk_to_propositions(
        ...     "Albert Einstein developed the theory of relativity. "
        ...     "He was born in Germany in 1879."
        ... )
        >>> for prop in result.propositions:
        ...     print(prop.text)
        # "Albert Einstein developed the theory of relativity."
        # "Albert Einstein was born in Germany."
        # "Albert Einstein was born in 1879."

    References:
        - Dense X Retrieval (Chen et al., 2024)
        - Propositionizer (Chen et al., 2023)
    """

    PROPOSITION_PROMPT = '''Decompose the following text into atomic propositions.

Rules for each proposition:
1. Contains exactly ONE factual statement
2. Is self-contained (understandable without external context)
3. Replaces pronouns with explicit entity names
4. Splits compound statements into separate propositions
5. Preserves the original meaning accurately

Text:
"""
{text}
"""

Output each proposition on a new line, prefixed with "- ":
'''

    DECONTEXTUALIZE_PROMPT = """Make the following sentence self-contained by:
1. Replacing pronouns (he, she, it, they, etc.) with the actual entity names
2. Adding necessary context that was in previous sentences
3. Keeping the factual content unchanged

Previous context: {context}

Sentence to decontextualize: {sentence}

Self-contained sentence:"""

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        base_chunker: Optional[BaseChunker] = None,
        max_propositions_per_chunk: int = 20,
        min_proposition_length: int = 10,
        batch_size: int = 5,
        decontextualize: bool = True,
    ):
        """
        Initialize proposition chunker.

        Args:
            llm: Language model for proposition extraction
            base_chunker: Base chunker for initial text splitting
            max_propositions_per_chunk: Max propositions to extract per chunk
            min_proposition_length: Minimum proposition length in chars
            batch_size: Batch size for LLM calls
            decontextualize: Whether to decontextualize pronouns
        """
        self.llm = llm
        self.base_chunker = base_chunker or SentenceChunker(
            max_chunk_size=2000,
            overlap_sentences=1,
        )
        self.max_propositions_per_chunk = max_propositions_per_chunk
        self.min_proposition_length = min_proposition_length
        self.batch_size = batch_size
        self.decontextualize = decontextualize

        # Cache for decontextualized sentences
        self._decontext_cache: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "proposition"

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """
        Synchronous chunking - falls back to base chunker.

        For proposition extraction, use chunk_to_propositions() instead.
        """
        return self.base_chunker.chunk(text, metadata)

    async def chunk_to_propositions(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PropositionChunkResult:
        """
        Extract atomic propositions from text.

        Args:
            text: Text to decompose
            metadata: Optional metadata

        Returns:
            PropositionChunkResult with extracted propositions
        """
        import asyncio

        # First, chunk text into manageable pieces
        base_chunks = self.base_chunker.chunk(text, metadata)

        all_propositions: List[Proposition] = []
        extraction_stats = {
            "total_chunks": len(base_chunks),
            "total_propositions": 0,
            "avg_propositions_per_chunk": 0.0,
            "llm_calls": 0,
        }

        # Process chunks in batches
        for i in range(0, len(base_chunks), self.batch_size):
            batch = base_chunks[i : i + self.batch_size]

            # Extract propositions from each chunk in parallel
            tasks = [self._extract_propositions(chunk, chunk.index) for chunk in batch]
            batch_results = await asyncio.gather(*tasks)

            for props in batch_results:
                all_propositions.extend(props)
                extraction_stats["llm_calls"] += 1

        extraction_stats["total_propositions"] = len(all_propositions)
        extraction_stats["avg_propositions_per_chunk"] = len(all_propositions) / max(
            1, len(base_chunks)
        )

        return PropositionChunkResult(
            propositions=all_propositions,
            original_chunks=base_chunks,
            extraction_stats=extraction_stats,
        )

    async def _extract_propositions(
        self,
        chunk: TextChunk,
        chunk_index: int,
    ) -> List[Proposition]:
        """Extract propositions from a single chunk."""
        if not self.llm:
            # Fallback: split by sentences
            return self._fallback_extraction(chunk, chunk_index)

        prompt = self.PROPOSITION_PROMPT.format(text=chunk.text)
        response = await self.llm.generate(prompt, temperature=0.0)

        propositions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                prop_text = line[2:].strip()
            elif line.startswith("-"):
                prop_text = line[1:].strip()
            else:
                prop_text = line

            if len(prop_text) >= self.min_proposition_length:
                prop = Proposition(
                    text=prop_text,
                    source_sentence=chunk.text[:200],
                    source_chunk_index=chunk_index,
                    entities=self._extract_entities(prop_text),
                )
                propositions.append(prop)

            if len(propositions) >= self.max_propositions_per_chunk:
                break

        return propositions

    def _fallback_extraction(
        self,
        chunk: TextChunk,
        chunk_index: int,
    ) -> List[Proposition]:
        """Fallback extraction using sentence splitting."""
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        propositions = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= self.min_proposition_length:
                prop = Proposition(
                    text=sentence,
                    source_sentence=sentence,
                    source_chunk_index=chunk_index,
                    confidence=0.7,  # Lower confidence for fallback
                )
                propositions.append(prop)

        return propositions

    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential named entities from text."""
        # Simple heuristic: capitalized words that aren't sentence starters
        words = text.split()
        entities = []

        for i, word in enumerate(words):
            # Skip first word (might be capitalized due to sentence start)
            if i == 0:
                continue
            # Check if word is capitalized and not a common word
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                entities.append(clean_word)

        return list(set(entities))

    async def decontextualize_proposition(
        self,
        proposition: str,
        context: str,
    ) -> str:
        """
        Make a proposition self-contained by resolving pronouns.

        Args:
            proposition: The proposition to decontextualize
            context: Previous context for reference resolution

        Returns:
            Self-contained proposition
        """
        cache_key = f"{proposition[:50]}:{context[:50]}"
        if cache_key in self._decontext_cache:
            return self._decontext_cache[cache_key]

        if not self.llm:
            return proposition

        prompt = self.DECONTEXTUALIZE_PROMPT.format(
            context=context[:500],
            sentence=proposition,
        )
        result = await self.llm.generate(prompt, temperature=0.0)
        result = result.strip()

        self._decontext_cache[cache_key] = result
        return result

    def propositions_to_chunks(
        self,
        propositions: List[Proposition],
    ) -> List[TextChunk]:
        """
        Convert propositions back to TextChunk format for retrieval.

        Args:
            propositions: List of propositions

        Returns:
            List of TextChunks (one per proposition)
        """
        chunks = []
        for i, prop in enumerate(propositions):
            chunk = TextChunk(
                text=prop.text,
                index=i,
                metadata=ChunkMetadata(
                    custom={
                        "type": "proposition",
                        "source_chunk": prop.source_chunk_index,
                        "entities": prop.entities,
                        "confidence": prop.confidence,
                    }
                ),
            )
            chunks.append(chunk)
        return chunks


class HierarchicalPropositionChunker(BaseChunker):
    """
    Hierarchical chunking with propositions at leaf level.

    Creates a tree structure:
    - Document → Sections → Paragraphs → Propositions

    Enables both coarse and fine-grained retrieval.
    """

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        section_chunker: Optional[BaseChunker] = None,
        paragraph_chunker: Optional[BaseChunker] = None,
        proposition_chunker: Optional[PropositionChunker] = None,
    ):
        """Initialize hierarchical chunker."""
        self.llm = llm
        self.section_chunker = section_chunker or MarkdownChunker(
            chunk_size=5000, preserve_hierarchy=True
        )
        self.paragraph_chunker = paragraph_chunker or ParagraphChunker(max_chunk_size=1000)
        self.proposition_chunker = proposition_chunker or PropositionChunker(llm=llm)

    @property
    def name(self) -> str:
        return "hierarchical_proposition"

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Chunk text hierarchically (sync version)."""
        # For sync, just use paragraph chunking
        return self.paragraph_chunker.chunk(text, metadata)

    async def chunk_hierarchical(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[TextChunk]]:
        """
        Create hierarchical chunks at multiple granularities.

        Returns:
            Dict with keys 'sections', 'paragraphs', 'propositions'
        """
        result: Dict[str, List[TextChunk]] = {
            "sections": [],
            "paragraphs": [],
            "propositions": [],
        }

        # Level 1: Sections
        sections = self.section_chunker.chunk(text, metadata)
        result["sections"] = sections

        # Level 2: Paragraphs within sections
        all_paragraphs = []
        for section in sections:
            paragraphs = self.paragraph_chunker.chunk(
                section.text,
                {"parent_section": section.index},
            )
            all_paragraphs.extend(paragraphs)
        result["paragraphs"] = all_paragraphs

        # Level 3: Propositions from paragraphs
        prop_result = await self.proposition_chunker.chunk_to_propositions(text, metadata)
        result["propositions"] = self.proposition_chunker.propositions_to_chunks(
            prop_result.propositions
        )

        return result


# Reference existing chunkers for HierarchicalPropositionChunker
class MarkdownChunker(BaseChunker):
    """Placeholder reference to existing MarkdownChunker."""

    def __init__(self, chunk_size: int = 5000, preserve_hierarchy: bool = True):
        self.chunk_size = chunk_size
        self.preserve_hierarchy = preserve_hierarchy

    @property
    def name(self) -> str:
        return "markdown"

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        # Use recursive chunker as fallback
        chunker = RecursiveChunker(chunk_size=self.chunk_size)
        return chunker.chunk(text, metadata)


class ParagraphChunker(BaseChunker):
    """Paragraph-based chunker."""

    PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    @property
    def name(self) -> str:
        return "paragraph"

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        paragraphs = self.PARAGRAPH_SPLIT.split(text)
        chunks = []

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            chunk_metadata = ChunkMetadata(custom=metadata or {})
            chunks.append(
                TextChunk(
                    text=para[: self.max_chunk_size],
                    index=i,
                    metadata=chunk_metadata,
                )
            )

        return chunks


class RecursiveChunker(BaseChunker):
    """Placeholder for recursive chunker reference."""

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    @property
    def name(self) -> str:
        return "recursive"

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        # Simple fixed-size fallback
        chunker = FixedSizeChunker(chunk_size=self.chunk_size, overlap=100)
        return chunker.chunk(text, metadata)


class ContextualChunker(BaseChunker):
    """
    Contextual retrieval chunking (Anthropic, 2024).

    Prepends document-level context to each chunk before embedding,
    dramatically improving retrieval recall by providing each chunk
    with the surrounding context it needs for disambiguation.

    Standard chunking loses document-level context, causing chunks like
    "The company's revenue grew 20%" to be unattributable. Contextual
    chunking prepends "This chunk is from Apple's Q3 2024 earnings report.
    It discusses revenue growth in the services segment."

    Based on: "Introducing Contextual Retrieval" (Anthropic, Sept 2024)
    - Reduces retrieval failure rate by 49% (from 5.7% to 2.9%)
    - Combined with BM25 hybrid search: 67% reduction in failures

    Example:
        >>> chunker = ContextualChunker(
        ...     base_chunker=SentenceChunker(max_chunk_size=512),
        ...     context_generator=my_llm_context_fn,
        ... )
        >>> chunks = chunker.chunk(document_text, metadata={"title": "Q3 Report"})
        >>> # Each chunk now has contextual prefix for better retrieval
    """

    # Default template for generating contextual descriptions
    CONTEXT_PROMPT_TEMPLATE = (
        "Here is the full document:\n<document>\n{document}\n</document>\n\n"
        "Here is the chunk we want to situate within the document:\n"
        "<chunk>\n{chunk}\n</chunk>\n\n"
        "Please give a short succinct context (2-3 sentences) to situate this "
        "chunk within the overall document for the purposes of improving search "
        "retrieval. Answer only with the context, nothing else."
    )

    def __init__(
        self,
        base_chunker: Optional[BaseChunker] = None,
        context_generator: Optional[Any] = None,
        context_template: Optional[str] = None,
        max_context_length: int = 200,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize contextual chunker.

        Args:
            base_chunker: Underlying chunker for initial splitting (default: SentenceChunker)
            context_generator: Callable(prompt: str) -> str for LLM-based context generation.
                               If None, uses extractive heuristic context.
            context_template: Custom prompt template with {document} and {chunk} placeholders
            max_context_length: Max characters for contextual prefix
            chunk_size: Chunk size if using default base chunker
            chunk_overlap: Overlap for default base chunker
        """
        self.base_chunker = base_chunker or SentenceChunker(max_chunk_size=chunk_size)
        self.context_generator = context_generator
        self.context_template = context_template or self.CONTEXT_PROMPT_TEMPLATE
        self.max_context_length = max_context_length
        self.chunk_size = chunk_size

    @property
    def name(self) -> str:
        return "contextual"

    def _extract_heuristic_context(
        self,
        full_document: str,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate contextual prefix using extractive heuristics (no LLM needed).

        Extracts document title, section headers, and positional context.
        This is a fast fallback when no LLM context_generator is provided.

        Args:
            full_document: Complete document text
            chunk_text: The specific chunk text
            chunk_index: Position of chunk in document
            total_chunks: Total number of chunks
            doc_metadata: Document-level metadata

        Returns:
            Contextual prefix string
        """
        parts = []

        # Add document title if available
        if doc_metadata:
            title = doc_metadata.get("title") or doc_metadata.get("source_id", "")
            if title:
                parts.append(f"From: {title}.")

        # Add positional context
        if total_chunks > 1:
            position = (
                "beginning"
                if chunk_index == 0
                else ("end" if chunk_index == total_chunks - 1 else "middle")
            )
            parts.append(
                f"This is from the {position} of the document"
                f" (section {chunk_index + 1}/{total_chunks})."
            )

        # Extract nearest section header (look backwards in document)
        chunk_start = full_document.find(chunk_text[:50]) if len(chunk_text) >= 50 else -1
        if chunk_start > 0:
            preceding = full_document[:chunk_start]
            # Find last header-like line (lines starting with # or all caps)
            lines = preceding.split("\n")
            for line in reversed(lines):
                stripped = line.strip()
                if stripped and (
                    stripped.startswith("#")
                    or (stripped.isupper() and len(stripped) < 100)
                    or stripped.endswith(":")
                ):
                    parts.append(f"Section: {stripped.lstrip('#').strip()}.")
                    break

        context = " ".join(parts)
        return context[: self.max_context_length] if context else ""

    def _generate_llm_context(self, full_document: str, chunk_text: str) -> str:
        """
        Generate contextual prefix using LLM.

        Args:
            full_document: Complete document text (may be truncated for token limits)
            chunk_text: The specific chunk text

        Returns:
            LLM-generated contextual prefix
        """
        # Truncate document if too long (keep first/last parts)
        max_doc_chars = 6000
        if len(full_document) > max_doc_chars:
            half = max_doc_chars // 2
            full_document = (
                full_document[:half]
                + "\n\n[... middle of document omitted ...]\n\n"
                + full_document[-half:]
            )

        prompt = self.context_template.format(
            document=full_document,
            chunk=chunk_text,
        )

        try:
            context = self.context_generator(prompt)
            return context[: self.max_context_length]
        except Exception as e:
            logger.warning(f"LLM context generation failed: {e}")
            return ""

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """
        Split text into contextually-enriched chunks.

        Each chunk gets a contextual prefix prepended that describes
        where it fits within the overall document.

        Args:
            text: Full document text
            metadata: Document-level metadata (title, source, etc.)

        Returns:
            List of TextChunks with contextual prefixes
        """
        if not text:
            return []

        # Step 1: Split with base chunker
        base_chunks = self.base_chunker.chunk(text, metadata)
        total_chunks = len(base_chunks)

        if total_chunks == 0:
            return []

        # Step 2: Add contextual prefix to each chunk
        contextual_chunks = []
        for chunk in base_chunks:
            # Generate context
            if self.context_generator is not None:
                context_prefix = self._generate_llm_context(text, chunk.text)
            else:
                context_prefix = self._extract_heuristic_context(
                    full_document=text,
                    chunk_text=chunk.text,
                    chunk_index=chunk.index,
                    total_chunks=total_chunks,
                    doc_metadata=metadata,
                )

            # Prepend context to chunk text
            if context_prefix:
                enriched_text = f"{context_prefix}\n\n{chunk.text}"
            else:
                enriched_text = chunk.text

            # Preserve original metadata and add context info
            chunk.metadata.custom["contextual_prefix"] = context_prefix
            chunk.metadata.custom["original_text"] = chunk.text

            contextual_chunks.append(
                TextChunk(
                    text=enriched_text,
                    index=chunk.index,
                    metadata=chunk.metadata,
                    token_count=chunk.token_count,
                )
            )

        return contextual_chunks
