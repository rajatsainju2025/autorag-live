"""Document Loaders for AutoRAG-Live.

Comprehensive document loading system supporting:
- PDF documents
- JSON files
- CSV files
- Markdown files
- Web pages
- Text files
- Directory scanning
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a loaded document."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str | None = None
    source: str | None = None
    doc_type: str = "text"

    def __post_init__(self) -> None:
        """Generate document ID if not provided."""
        if not self.doc_id:
            content_hash = hashlib.md5(self.content[:500].encode()).hexdigest()[:12]
            self.doc_id = f"doc_{content_hash}"

    @property
    def word_count(self) -> int:
        """Get word count."""
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.content)


@dataclass
class LoaderConfig:
    """Configuration for document loaders."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    encoding: str = "utf-8"
    max_file_size_mb: int = 100
    extract_metadata: bool = True
    clean_text: bool = True
    skip_empty: bool = True


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    def __init__(self, config: LoaderConfig | None = None) -> None:
        """Initialize loader.

        Args:
            config: Loader configuration
        """
        self.config = config or LoaderConfig()

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """Load documents from source.

        Args:
            source: Source path or URL

        Returns:
            List of loaded documents
        """
        pass

    @abstractmethod
    def supports(self, source: str) -> bool:
        """Check if loader supports the source.

        Args:
            source: Source path or URL

        Returns:
            True if supported
        """
        pass

    def _clean_text(self, text: str) -> str:
        """Clean text content."""
        if not self.config.clean_text:
            return text

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        return text.strip()

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if len(text) <= self.config.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + self.config.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 20% of chunk
                search_start = end - int(self.config.chunk_size * 0.2)
                search_region = text[search_start:end]

                # Find last sentence boundary
                for sep in [". ", "! ", "? ", "\n\n", "\n"]:
                    last_sep = search_region.rfind(sep)
                    if last_sep != -1:
                        end = search_start + last_sep + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.config.chunk_overlap
            if start < 0:
                start = end

        return chunks


class TextLoader(BaseLoader):
    """Load plain text files."""

    SUPPORTED_EXTENSIONS = {".txt", ".text", ".log"}

    def supports(self, source: str) -> bool:
        """Check if source is a text file."""
        path = Path(source)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, source: str) -> list[Document]:
        """Load text file."""
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        try:
            content = path.read_text(encoding=self.config.encoding)
            content = self._clean_text(content)

            if self.config.skip_empty and not content.strip():
                return []

            metadata = {
                "source": str(path.absolute()),
                "filename": path.name,
                "file_size": path.stat().st_size,
                "created": datetime.fromtimestamp(path.stat().st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            }

            # Chunk if needed
            chunks = self._chunk_text(content)

            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata.copy()
                doc_metadata["chunk_index"] = i
                doc_metadata["total_chunks"] = len(chunks)

                documents.append(
                    Document(
                        content=chunk,
                        metadata=doc_metadata,
                        source=str(path),
                        doc_type="text",
                    )
                )

            return documents

        except Exception as e:
            logger.error(f"Failed to load text file {source}: {e}")
            raise


class JSONLoader(BaseLoader):
    """Load JSON files."""

    SUPPORTED_EXTENSIONS = {".json", ".jsonl"}

    def __init__(
        self,
        config: LoaderConfig | None = None,
        content_key: str | None = None,
        metadata_keys: list[str] | None = None,
    ) -> None:
        """Initialize JSON loader.

        Args:
            config: Loader configuration
            content_key: Key containing document content
            metadata_keys: Keys to extract as metadata
        """
        super().__init__(config)
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []

    def supports(self, source: str) -> bool:
        """Check if source is a JSON file."""
        path = Path(source)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, source: str) -> list[Document]:
        """Load JSON file."""
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        documents: list[Document] = []

        try:
            if path.suffix.lower() == ".jsonl":
                # Load JSONL (one JSON object per line)
                with open(path, "r", encoding=self.config.encoding) as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            obj = json.loads(line)
                            doc = self._json_to_document(obj, path, i)
                            if doc:
                                documents.append(doc)
            else:
                # Load regular JSON
                with open(path, "r", encoding=self.config.encoding) as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for i, item in enumerate(data):
                        doc = self._json_to_document(item, path, i)
                        if doc:
                            documents.append(doc)
                else:
                    doc = self._json_to_document(data, path, 0)
                    if doc:
                        documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Failed to load JSON file {source}: {e}")
            raise

    def _json_to_document(
        self,
        obj: dict[str, Any],
        path: Path,
        index: int,
    ) -> Document | None:
        """Convert JSON object to Document."""
        if not isinstance(obj, dict):
            content = json.dumps(obj)
        elif self.content_key and self.content_key in obj:
            content = str(obj[self.content_key])
        else:
            # Try common content keys
            for key in ["content", "text", "body", "document", "data"]:
                if key in obj:
                    content = str(obj[key])
                    break
            else:
                content = json.dumps(obj, indent=2)

        content = self._clean_text(content)

        if self.config.skip_empty and not content.strip():
            return None

        metadata = {
            "source": str(path.absolute()),
            "filename": path.name,
            "index": index,
        }

        # Extract specified metadata keys
        for key in self.metadata_keys:
            if key in obj:
                metadata[key] = obj[key]

        return Document(
            content=content,
            metadata=metadata,
            source=str(path),
            doc_type="json",
        )


class CSVLoader(BaseLoader):
    """Load CSV files."""

    SUPPORTED_EXTENSIONS = {".csv", ".tsv"}

    def __init__(
        self,
        config: LoaderConfig | None = None,
        content_columns: list[str] | None = None,
        metadata_columns: list[str] | None = None,
        delimiter: str | None = None,
    ) -> None:
        """Initialize CSV loader.

        Args:
            config: Loader configuration
            content_columns: Columns to use as content
            metadata_columns: Columns to use as metadata
            delimiter: CSV delimiter
        """
        super().__init__(config)
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.delimiter = delimiter

    def supports(self, source: str) -> bool:
        """Check if source is a CSV file."""
        path = Path(source)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, source: str) -> list[Document]:
        """Load CSV file."""
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        documents: list[Document] = []

        # Determine delimiter
        delimiter = self.delimiter
        if delimiter is None:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

        try:
            with open(path, "r", encoding=self.config.encoding, newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)

                for i, row in enumerate(reader):
                    doc = self._row_to_document(row, path, i)
                    if doc:
                        documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Failed to load CSV file {source}: {e}")
            raise

    def _row_to_document(
        self,
        row: dict[str, str],
        path: Path,
        index: int,
    ) -> Document | None:
        """Convert CSV row to Document."""
        # Build content
        if self.content_columns:
            content_parts = [str(row.get(col, "")) for col in self.content_columns if row.get(col)]
            content = " ".join(content_parts)
        else:
            # Use all columns
            content = " ".join(str(v) for v in row.values() if v)

        content = self._clean_text(content)

        if self.config.skip_empty and not content.strip():
            return None

        metadata = {
            "source": str(path.absolute()),
            "filename": path.name,
            "row_index": index,
        }

        # Extract metadata columns
        if self.metadata_columns:
            for col in self.metadata_columns:
                if col in row:
                    metadata[col] = row[col]

        return Document(
            content=content,
            metadata=metadata,
            source=str(path),
            doc_type="csv",
        )


class MarkdownLoader(BaseLoader):
    """Load Markdown files with section awareness."""

    SUPPORTED_EXTENSIONS = {".md", ".markdown", ".mdown"}

    def __init__(
        self,
        config: LoaderConfig | None = None,
        split_by_headers: bool = True,
        min_header_level: int = 1,
        max_header_level: int = 3,
    ) -> None:
        """Initialize Markdown loader.

        Args:
            config: Loader configuration
            split_by_headers: Split document by headers
            min_header_level: Minimum header level to split at
            max_header_level: Maximum header level to split at
        """
        super().__init__(config)
        self.split_by_headers = split_by_headers
        self.min_header_level = min_header_level
        self.max_header_level = max_header_level

    def supports(self, source: str) -> bool:
        """Check if source is a Markdown file."""
        path = Path(source)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, source: str) -> list[Document]:
        """Load Markdown file."""
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        try:
            content = path.read_text(encoding=self.config.encoding)

            base_metadata = {
                "source": str(path.absolute()),
                "filename": path.name,
                "file_size": path.stat().st_size,
            }

            if self.split_by_headers:
                return self._split_by_headers(content, path, base_metadata)
            else:
                content = self._clean_text(content)
                if self.config.skip_empty and not content.strip():
                    return []

                return [
                    Document(
                        content=content,
                        metadata=base_metadata,
                        source=str(path),
                        doc_type="markdown",
                    )
                ]

        except Exception as e:
            logger.error(f"Failed to load Markdown file {source}: {e}")
            raise

    def _split_by_headers(
        self,
        content: str,
        path: Path,
        base_metadata: dict[str, Any],
    ) -> list[Document]:
        """Split markdown by headers."""
        documents: list[Document] = []

        # Pattern for headers
        header_pattern = re.compile(
            r"^(#{" + str(self.min_header_level) + "," + str(self.max_header_level) + r"})\s+(.+)$",
            re.MULTILINE,
        )

        sections: list[tuple[str | None, str]] = []
        current_header = None
        current_content: list[str] = []

        lines = content.split("\n")

        for line in lines:
            match = header_pattern.match(line)
            if match:
                # Save previous section
                if current_content:
                    sections.append((current_header, "\n".join(current_content)))

                current_header = match.group(2).strip()
                current_content = [line]
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections.append((current_header, "\n".join(current_content)))

        # Create documents
        for i, (header, section_content) in enumerate(sections):
            section_content = self._clean_text(section_content)

            if self.config.skip_empty and not section_content.strip():
                continue

            metadata = base_metadata.copy()
            metadata["section_index"] = i
            metadata["section_header"] = header
            metadata["total_sections"] = len(sections)

            documents.append(
                Document(
                    content=section_content,
                    metadata=metadata,
                    source=str(path),
                    doc_type="markdown",
                )
            )

        return documents


class PDFLoader(BaseLoader):
    """Load PDF files."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    def supports(self, source: str) -> bool:
        """Check if source is a PDF file."""
        path = Path(source)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, source: str) -> list[Document]:
        """Load PDF file."""
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        documents: list[Document] = []

        try:
            # Try PyPDF2 first
            try:
                import importlib.util

                if importlib.util.find_spec("PyPDF2") is not None:
                    documents = self._load_with_pypdf2(path)
                elif importlib.util.find_spec("pdfplumber") is not None:
                    documents = self._load_with_pdfplumber(path)
                else:
                    raise ImportError(
                        "PDF loading requires PyPDF2 or pdfplumber. "
                        "Install with: pip install PyPDF2 or pip install pdfplumber"
                    )
            except ImportError:
                raise ImportError(
                    "PDF loading requires PyPDF2 or pdfplumber. "
                    "Install with: pip install PyPDF2 or pip install pdfplumber"
                )

            return documents

        except Exception as e:
            logger.error(f"Failed to load PDF file {source}: {e}")
            raise

    def _load_with_pypdf2(self, path: Path) -> list[Document]:
        """Load PDF using PyPDF2."""
        import PyPDF2

        documents: list[Document] = []

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)

            for i, page in enumerate(reader.pages):
                content = page.extract_text() or ""
                content = self._clean_text(content)

                if self.config.skip_empty and not content.strip():
                    continue

                metadata = {
                    "source": str(path.absolute()),
                    "filename": path.name,
                    "page": i + 1,
                    "total_pages": num_pages,
                }

                # Try to extract PDF metadata
                if reader.metadata:
                    if reader.metadata.title:
                        metadata["title"] = reader.metadata.title
                    if reader.metadata.author:
                        metadata["author"] = reader.metadata.author

                documents.append(
                    Document(
                        content=content,
                        metadata=metadata,
                        source=str(path),
                        doc_type="pdf",
                    )
                )

        return documents

    def _load_with_pdfplumber(self, path: Path) -> list[Document]:
        """Load PDF using pdfplumber."""
        import pdfplumber

        documents: list[Document] = []

        with pdfplumber.open(path) as pdf:
            num_pages = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                content = page.extract_text() or ""
                content = self._clean_text(content)

                if self.config.skip_empty and not content.strip():
                    continue

                metadata = {
                    "source": str(path.absolute()),
                    "filename": path.name,
                    "page": i + 1,
                    "total_pages": num_pages,
                    "page_width": page.width,
                    "page_height": page.height,
                }

                documents.append(
                    Document(
                        content=content,
                        metadata=metadata,
                        source=str(path),
                        doc_type="pdf",
                    )
                )

        return documents


class WebLoader(BaseLoader):
    """Load web pages."""

    def __init__(
        self,
        config: LoaderConfig | None = None,
        extract_links: bool = False,
        user_agent: str | None = None,
    ) -> None:
        """Initialize web loader.

        Args:
            config: Loader configuration
            extract_links: Extract and store links from page
            user_agent: Custom user agent string
        """
        super().__init__(config)
        self.extract_links = extract_links
        self.user_agent = user_agent or "AutoRAG-Live WebLoader/1.0"

    def supports(self, source: str) -> bool:
        """Check if source is a URL."""
        try:
            parsed = urlparse(source)
            return parsed.scheme in ("http", "https")
        except Exception:
            return False

    def load(self, source: str) -> list[Document]:
        """Load web page."""
        try:
            import urllib.request

            # Fetch page
            request = urllib.request.Request(
                source,
                headers={"User-Agent": self.user_agent},
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                html = response.read().decode(self.config.encoding, errors="ignore")
                final_url = response.geturl()

            # Parse HTML
            content, links, title = self._parse_html(html)
            content = self._clean_text(content)

            if self.config.skip_empty and not content.strip():
                return []

            metadata = {
                "source": final_url,
                "url": source,
                "title": title,
            }

            if self.extract_links:
                metadata["links"] = links[:50]  # Limit links

            return [
                Document(
                    content=content,
                    metadata=metadata,
                    source=source,
                    doc_type="web",
                )
            ]

        except Exception as e:
            logger.error(f"Failed to load web page {source}: {e}")
            raise

    def _parse_html(self, html: str) -> tuple[str, list[str], str | None]:
        """Parse HTML and extract text content."""
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.text_parts: list[str] = []
                self.links: list[str] = []
                self.title: str | None = None
                self.in_title = False
                self.skip_tags = {"script", "style", "noscript", "nav", "footer", "header"}
                self.current_skip_depth = 0

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag in self.skip_tags:
                    self.current_skip_depth += 1
                if tag == "title":
                    self.in_title = True
                if tag == "a":
                    for attr, value in attrs:
                        if attr == "href" and value:
                            self.links.append(value)

            def handle_endtag(self, tag: str) -> None:
                if tag in self.skip_tags:
                    self.current_skip_depth = max(0, self.current_skip_depth - 1)
                if tag == "title":
                    self.in_title = False

            def handle_data(self, data: str) -> None:
                if self.in_title:
                    self.title = data.strip()
                elif self.current_skip_depth == 0:
                    text = data.strip()
                    if text:
                        self.text_parts.append(text)

        parser = TextExtractor()
        parser.feed(html)

        content = " ".join(parser.text_parts)
        return content, parser.links, parser.title


class DirectoryLoader(BaseLoader):
    """Load all documents from a directory."""

    def __init__(
        self,
        config: LoaderConfig | None = None,
        recursive: bool = True,
        glob_pattern: str = "*",
        exclude_patterns: list[str] | None = None,
    ) -> None:
        """Initialize directory loader.

        Args:
            config: Loader configuration
            recursive: Recursively scan subdirectories
            glob_pattern: File pattern to match
            exclude_patterns: Patterns to exclude
        """
        super().__init__(config)
        self.recursive = recursive
        self.glob_pattern = glob_pattern
        self.exclude_patterns = exclude_patterns or []

        # Initialize sub-loaders
        self._loaders = [
            TextLoader(config),
            JSONLoader(config),
            CSVLoader(config),
            MarkdownLoader(config),
            PDFLoader(config),
        ]

    def supports(self, source: str) -> bool:
        """Check if source is a directory."""
        return Path(source).is_dir()

    def load(self, source: str) -> list[Document]:
        """Load all documents from directory."""
        path = Path(source)

        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {source}")

        documents: list[Document] = []

        # Find files
        if self.recursive:
            files = path.rglob(self.glob_pattern)
        else:
            files = path.glob(self.glob_pattern)

        for file_path in files:
            if not file_path.is_file():
                continue

            # Check exclusions
            if self._should_exclude(file_path):
                continue

            # Find appropriate loader
            for loader in self._loaders:
                if loader.supports(str(file_path)):
                    try:
                        docs = loader.load(str(file_path))
                        documents.extend(docs)
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
                    break

        return documents

    def _should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded."""
        path_str = str(path)
        for pattern in self.exclude_patterns:
            if re.search(pattern, path_str):
                return True
        return False


class DocumentLoaderRegistry:
    """Registry for document loaders."""

    def __init__(self) -> None:
        """Initialize registry with default loaders."""
        self._loaders: list[BaseLoader] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default loaders."""
        config = LoaderConfig()
        self._loaders = [
            TextLoader(config),
            JSONLoader(config),
            CSVLoader(config),
            MarkdownLoader(config),
            PDFLoader(config),
            WebLoader(config),
            DirectoryLoader(config),
        ]

    def register(self, loader: BaseLoader) -> None:
        """Register a custom loader.

        Args:
            loader: Loader to register
        """
        self._loaders.insert(0, loader)  # Priority to custom loaders

    def load(self, source: str) -> list[Document]:
        """Load documents using appropriate loader.

        Args:
            source: Source path or URL

        Returns:
            List of documents
        """
        for loader in self._loaders:
            if loader.supports(source):
                return loader.load(source)

        raise ValueError(f"No loader found for source: {source}")

    def load_multiple(self, sources: list[str]) -> list[Document]:
        """Load documents from multiple sources.

        Args:
            sources: List of source paths or URLs

        Returns:
            Combined list of documents
        """
        documents: list[Document] = []

        for source in sources:
            try:
                docs = self.load(source)
                documents.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load {source}: {e}")

        return documents


# Global registry
_registry: DocumentLoaderRegistry | None = None


def get_loader_registry() -> DocumentLoaderRegistry:
    """Get global loader registry."""
    global _registry
    if _registry is None:
        _registry = DocumentLoaderRegistry()
    return _registry


def load_documents(source: str) -> list[Document]:
    """Convenience function to load documents.

    Args:
        source: Source path or URL

    Returns:
        List of documents
    """
    return get_loader_registry().load(source)


def load_documents_from_directory(
    directory: str,
    recursive: bool = True,
    glob_pattern: str = "*",
) -> list[Document]:
    """Load all documents from a directory.

    Args:
        directory: Directory path
        recursive: Recursively scan
        glob_pattern: File pattern

    Returns:
        List of documents
    """
    loader = DirectoryLoader(
        recursive=recursive,
        glob_pattern=glob_pattern,
    )
    return loader.load(directory)
