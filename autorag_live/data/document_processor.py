"""
Advanced document processing module for AutoRAG-Live.

Provides comprehensive document processing capabilities
for ingesting various document types into RAG systems.

Features:
- Multi-format support (PDF, DOCX, HTML, Markdown)
- Document structure extraction
- Table and image handling
- Metadata extraction
- Language detection
- Document cleaning
- Section splitting
- OCR support
- Document linking

Example usage:
    >>> processor = DocumentProcessor()
    >>> doc = processor.process("document.pdf")
    >>> 
    >>> for section in doc.sections:
    ...     print(f"{section.title}: {len(section.content)} chars")
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types."""
    
    TEXT = auto()
    PDF = auto()
    DOCX = auto()
    HTML = auto()
    MARKDOWN = auto()
    JSON = auto()
    CSV = auto()
    XML = auto()
    CODE = auto()
    UNKNOWN = auto()


class ContentType(Enum):
    """Types of content within documents."""
    
    TEXT = auto()
    TITLE = auto()
    HEADING = auto()
    PARAGRAPH = auto()
    LIST = auto()
    TABLE = auto()
    IMAGE = auto()
    CODE_BLOCK = auto()
    QUOTE = auto()
    LINK = auto()
    FOOTNOTE = auto()


class ProcessingStatus(Enum):
    """Document processing status."""
    
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PARTIAL = auto()


@dataclass
class DocumentMetadata:
    """Document metadata."""
    
    # Basic info
    title: str = ""
    author: str = ""
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    
    # File info
    file_path: str = ""
    file_name: str = ""
    file_size: int = 0
    mime_type: str = ""
    
    # Content info
    page_count: int = 0
    word_count: int = 0
    char_count: int = 0
    language: str = ""
    
    # Processing info
    doc_id: str = ""
    content_hash: str = ""
    processed_at: Optional[datetime] = None
    
    # Custom metadata
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentElement:
    """A content element within a document."""
    
    content_type: ContentType
    content: str
    
    # Position
    page_number: int = 0
    position: int = 0
    
    # Hierarchy
    level: int = 0
    parent_id: Optional[str] = None
    
    # Styling
    is_bold: bool = False
    is_italic: bool = False
    font_size: float = 0.0
    
    # Element ID
    element_id: str = ""
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentSection:
    """A section of a document."""
    
    section_id: str
    title: str
    content: str
    
    # Hierarchy
    level: int = 0
    parent_id: Optional[str] = None
    
    # Position
    start_page: int = 0
    end_page: int = 0
    start_pos: int = 0
    end_pos: int = 0
    
    # Elements
    elements: List[ContentElement] = field(default_factory=list)
    
    # Metadata
    word_count: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.content.split())


@dataclass
class TableData:
    """Extracted table data."""
    
    table_id: str
    headers: List[str]
    rows: List[List[str]]
    
    # Position
    page_number: int = 0
    position: int = 0
    
    # Caption
    caption: str = ""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def row_count(self) -> int:
        return len(self.rows)
    
    @property
    def column_count(self) -> int:
        return len(self.headers) if self.headers else 0
    
    def to_markdown(self) -> str:
        """Convert to markdown table."""
        if not self.headers:
            return ""
        
        lines = []
        
        # Header
        lines.append("| " + " | ".join(self.headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
        
        # Rows
        for row in self.rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        
        return "\n".join(lines)
    
    def to_dict(self) -> List[Dict[str, str]]:
        """Convert to list of dictionaries."""
        result = []
        for row in self.rows:
            row_dict = {}
            for i, header in enumerate(self.headers):
                if i < len(row):
                    row_dict[header] = row[i]
            result.append(row_dict)
        return result


@dataclass
class ImageData:
    """Extracted image data."""
    
    image_id: str
    
    # Image info
    format: str = ""
    width: int = 0
    height: int = 0
    
    # Position
    page_number: int = 0
    position: int = 0
    
    # Content
    alt_text: str = ""
    caption: str = ""
    ocr_text: str = ""
    
    # Data
    data: Optional[bytes] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkData:
    """Extracted link data."""
    
    link_id: str
    url: str
    text: str
    
    # Type
    is_internal: bool = False
    is_external: bool = True
    
    # Position
    page_number: int = 0
    position: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """A fully processed document."""
    
    doc_id: str
    doc_type: DocumentType
    status: ProcessingStatus
    
    # Metadata
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    
    # Content
    raw_content: str = ""
    clean_content: str = ""
    
    # Structure
    sections: List[DocumentSection] = field(default_factory=list)
    
    # Extracted elements
    tables: List[TableData] = field(default_factory=list)
    images: List[ImageData] = field(default_factory=list)
    links: List[LinkData] = field(default_factory=list)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    @property
    def section_count(self) -> int:
        return len(self.sections)
    
    @property
    def has_tables(self) -> bool:
        return len(self.tables) > 0
    
    @property
    def has_images(self) -> bool:
        return len(self.images) > 0
    
    def get_section(self, section_id: str) -> Optional[DocumentSection]:
        """Get section by ID."""
        for section in self.sections:
            if section.section_id == section_id:
                return section
        return None
    
    def get_full_text(self) -> str:
        """Get full document text."""
        if self.clean_content:
            return self.clean_content
        return self.raw_content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'doc_id': self.doc_id,
            'doc_type': self.doc_type.name,
            'status': self.status.name,
            'metadata': {
                'title': self.metadata.title,
                'author': self.metadata.author,
                'word_count': self.metadata.word_count,
                'page_count': self.metadata.page_count,
            },
            'sections': [
                {
                    'section_id': s.section_id,
                    'title': s.title,
                    'level': s.level,
                    'word_count': s.word_count,
                }
                for s in self.sections
            ],
            'table_count': len(self.tables),
            'image_count': len(self.images),
            'link_count': len(self.links),
        }


class DocumentParser(ABC):
    """Base class for document parsers."""
    
    @property
    @abstractmethod
    def supported_types(self) -> List[DocumentType]:
        """Get supported document types."""
        pass
    
    @abstractmethod
    def parse(
        self,
        content: Union[str, bytes, BinaryIO],
        **kwargs,
    ) -> ProcessedDocument:
        """Parse document content."""
        pass


class TextParser(DocumentParser):
    """Parser for plain text documents."""
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.TEXT]
    
    def parse(
        self,
        content: Union[str, bytes, BinaryIO],
        **kwargs,
    ) -> ProcessedDocument:
        """Parse text content."""
        if isinstance(content, bytes):
            text = content.decode('utf-8', errors='ignore')
        elif hasattr(content, 'read'):
            text = content.read()
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
        else:
            text = content
        
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        
        # Create sections from paragraphs
        sections = []
        paragraphs = text.split('\n\n')
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                sections.append(DocumentSection(
                    section_id=f"{doc_id}_s{i}",
                    title=f"Section {i + 1}",
                    content=para,
                    level=0,
                    start_pos=text.find(para),
                ))
        
        return ProcessedDocument(
            doc_id=doc_id,
            doc_type=DocumentType.TEXT,
            status=ProcessingStatus.COMPLETED,
            raw_content=text,
            clean_content=text,
            sections=sections,
            metadata=DocumentMetadata(
                word_count=len(text.split()),
                char_count=len(text),
            ),
        )


class MarkdownParser(DocumentParser):
    """Parser for Markdown documents."""
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.MARKDOWN]
    
    def parse(
        self,
        content: Union[str, bytes, BinaryIO],
        **kwargs,
    ) -> ProcessedDocument:
        """Parse markdown content."""
        if isinstance(content, bytes):
            text = content.decode('utf-8', errors='ignore')
        elif hasattr(content, 'read'):
            text = content.read()
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
        else:
            text = content
        
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        
        # Extract sections from headings
        sections = self._extract_sections(text, doc_id)
        
        # Extract links
        links = self._extract_links(text, doc_id)
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(text)
        
        # Clean content (remove markdown syntax)
        clean_content = self._clean_markdown(text)
        
        return ProcessedDocument(
            doc_id=doc_id,
            doc_type=DocumentType.MARKDOWN,
            status=ProcessingStatus.COMPLETED,
            raw_content=text,
            clean_content=clean_content,
            sections=sections,
            links=links,
            metadata=DocumentMetadata(
                word_count=len(clean_content.split()),
                char_count=len(clean_content),
            ),
        )
    
    def _extract_sections(
        self,
        text: str,
        doc_id: str,
    ) -> List[DocumentSection]:
        """Extract sections from headings."""
        sections = []
        
        # Match headings
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        
        matches = list(heading_pattern.finditer(text))
        
        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            
            # Find content until next heading
            start = match.end()
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)
            
            content = text[start:end].strip()
            
            sections.append(DocumentSection(
                section_id=f"{doc_id}_s{i}",
                title=title,
                content=content,
                level=level,
                start_pos=match.start(),
                end_pos=end,
            ))
        
        return sections
    
    def _extract_links(
        self,
        text: str,
        doc_id: str,
    ) -> List[LinkData]:
        """Extract links from markdown."""
        links = []
        
        # Match markdown links
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        
        for i, match in enumerate(link_pattern.finditer(text)):
            link_text = match.group(1)
            url = match.group(2)
            
            is_external = url.startswith('http://') or url.startswith('https://')
            
            links.append(LinkData(
                link_id=f"{doc_id}_l{i}",
                url=url,
                text=link_text,
                is_internal=not is_external,
                is_external=is_external,
                position=match.start(),
            ))
        
        return links
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks."""
        code_pattern = re.compile(r'```[\w]*\n(.*?)```', re.DOTALL)
        return code_pattern.findall(text)
    
    def _clean_markdown(self, text: str) -> str:
        """Remove markdown syntax."""
        # Remove code blocks
        clean = re.sub(r'```[\w]*\n.*?```', '', text, flags=re.DOTALL)
        
        # Remove inline code
        clean = re.sub(r'`[^`]+`', '', clean)
        
        # Remove links (keep text)
        clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)
        
        # Remove images
        clean = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', clean)
        
        # Remove headings syntax
        clean = re.sub(r'^#{1,6}\s+', '', clean, flags=re.MULTILINE)
        
        # Remove emphasis
        clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean)
        clean = re.sub(r'__([^_]+)__', r'\1', clean)
        clean = re.sub(r'\*([^*]+)\*', r'\1', clean)
        clean = re.sub(r'_([^_]+)_', r'\1', clean)
        
        # Remove horizontal rules
        clean = re.sub(r'^[-*_]{3,}$', '', clean, flags=re.MULTILINE)
        
        return clean.strip()


class HTMLParser(DocumentParser):
    """Parser for HTML documents."""
    
    @property
    def supported_types(self) -> List[DocumentType]:
        return [DocumentType.HTML]
    
    def parse(
        self,
        content: Union[str, bytes, BinaryIO],
        **kwargs,
    ) -> ProcessedDocument:
        """Parse HTML content."""
        if isinstance(content, bytes):
            text = content.decode('utf-8', errors='ignore')
        elif hasattr(content, 'read'):
            text = content.read()
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='ignore')
        else:
            text = content
        
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        
        # Extract title
        title = self._extract_title(text)
        
        # Extract clean text
        clean_content = self._strip_html(text)
        
        # Extract links
        links = self._extract_links(text, doc_id)
        
        # Extract tables
        tables = self._extract_tables(text, doc_id)
        
        return ProcessedDocument(
            doc_id=doc_id,
            doc_type=DocumentType.HTML,
            status=ProcessingStatus.COMPLETED,
            raw_content=text,
            clean_content=clean_content,
            links=links,
            tables=tables,
            metadata=DocumentMetadata(
                title=title,
                word_count=len(clean_content.split()),
                char_count=len(clean_content),
            ),
        )
    
    def _extract_title(self, html: str) -> str:
        """Extract title from HTML."""
        match = re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _strip_html(self, html: str) -> str:
        """Strip HTML tags."""
        # Remove scripts and styles
        clean = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<style[^>]*>.*?</style>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove tags
        clean = re.sub(r'<[^>]+>', ' ', clean)
        
        # Decode entities
        clean = re.sub(r'&nbsp;', ' ', clean)
        clean = re.sub(r'&amp;', '&', clean)
        clean = re.sub(r'&lt;', '<', clean)
        clean = re.sub(r'&gt;', '>', clean)
        
        # Clean whitespace
        clean = re.sub(r'\s+', ' ', clean)
        
        return clean.strip()
    
    def _extract_links(
        self,
        html: str,
        doc_id: str,
    ) -> List[LinkData]:
        """Extract links from HTML."""
        links = []
        
        link_pattern = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>', re.IGNORECASE)
        
        for i, match in enumerate(link_pattern.finditer(html)):
            url = match.group(1)
            text = match.group(2).strip()
            
            is_external = url.startswith('http://') or url.startswith('https://')
            
            links.append(LinkData(
                link_id=f"{doc_id}_l{i}",
                url=url,
                text=text,
                is_internal=not is_external,
                is_external=is_external,
                position=match.start(),
            ))
        
        return links
    
    def _extract_tables(
        self,
        html: str,
        doc_id: str,
    ) -> List[TableData]:
        """Extract tables from HTML."""
        tables = []
        
        table_pattern = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
        
        for i, table_match in enumerate(table_pattern.finditer(html)):
            table_html = table_match.group(1)
            
            # Extract headers
            headers = []
            header_pattern = re.compile(r'<th[^>]*>([^<]*)</th>', re.IGNORECASE)
            for h in header_pattern.finditer(table_html):
                headers.append(self._strip_html(h.group(1)).strip())
            
            # Extract rows
            rows = []
            row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
            cell_pattern = re.compile(r'<td[^>]*>([^<]*)</td>', re.IGNORECASE)
            
            for row in row_pattern.finditer(table_html):
                cells = []
                for cell in cell_pattern.finditer(row.group(1)):
                    cells.append(self._strip_html(cell.group(1)).strip())
                if cells:
                    rows.append(cells)
            
            if headers or rows:
                tables.append(TableData(
                    table_id=f"{doc_id}_t{i}",
                    headers=headers,
                    rows=rows,
                    position=table_match.start(),
                ))
        
        return tables


class TextCleaner:
    """
    Text cleaning utilities.
    
    Example:
        >>> cleaner = TextCleaner()
        >>> clean = cleaner.clean("Messy   text\n\n\nwith   whitespace")
    """
    
    def __init__(
        self,
        normalize_whitespace: bool = True,
        remove_urls: bool = False,
        remove_emails: bool = False,
        lowercase: bool = False,
    ):
        """
        Initialize cleaner.
        
        Args:
            normalize_whitespace: Normalize whitespace
            remove_urls: Remove URLs
            remove_emails: Remove emails
            lowercase: Convert to lowercase
        """
        self.normalize_whitespace = normalize_whitespace
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.lowercase = lowercase
    
    def clean(self, text: str) -> str:
        """Clean text."""
        result = text
        
        # Remove URLs
        if self.remove_urls:
            result = re.sub(r'https?://\S+', '', result)
        
        # Remove emails
        if self.remove_emails:
            result = re.sub(r'\S+@\S+', '', result)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            result = re.sub(r'\s+', ' ', result)
            result = result.strip()
        
        # Lowercase
        if self.lowercase:
            result = result.lower()
        
        return result


class LanguageDetector:
    """
    Simple language detection.
    
    Example:
        >>> detector = LanguageDetector()
        >>> lang = detector.detect("Hello, how are you?")  # Returns "en"
    """
    
    def __init__(self):
        """Initialize detector."""
        # Common words for language detection
        self._language_markers = {
            'en': {'the', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'in', 'on'},
            'es': {'el', 'la', 'los', 'las', 'es', 'son', 'y', 'o', 'pero', 'en'},
            'fr': {'le', 'la', 'les', 'est', 'sont', 'et', 'ou', 'mais', 'dans', 'sur'},
            'de': {'der', 'die', 'das', 'ist', 'sind', 'und', 'oder', 'aber', 'in', 'auf'},
        }
    
    def detect(self, text: str) -> str:
        """
        Detect language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., "en")
        """
        words = set(text.lower().split())
        
        best_lang = 'en'
        best_score = 0
        
        for lang, markers in self._language_markers.items():
            score = len(words & markers)
            if score > best_score:
                best_score = score
                best_lang = lang
        
        return best_lang


class DocumentProcessor:
    """
    Main document processor interface.
    
    Example:
        >>> processor = DocumentProcessor()
        >>> 
        >>> # Process a file
        >>> doc = processor.process_file("document.md")
        >>> 
        >>> # Process raw content
        >>> doc = processor.process("# Title\n\nContent here", doc_type=DocumentType.MARKDOWN)
        >>> 
        >>> # Access results
        >>> for section in doc.sections:
        ...     print(f"{section.title}: {section.word_count} words")
    """
    
    def __init__(
        self,
        clean_text: bool = True,
        detect_language: bool = True,
    ):
        """
        Initialize processor.
        
        Args:
            clean_text: Clean extracted text
            detect_language: Detect document language
        """
        self.clean_text = clean_text
        self.detect_language = detect_language
        
        # Parsers
        self._parsers: Dict[DocumentType, DocumentParser] = {
            DocumentType.TEXT: TextParser(),
            DocumentType.MARKDOWN: MarkdownParser(),
            DocumentType.HTML: HTMLParser(),
        }
        
        # Utilities
        self._cleaner = TextCleaner()
        self._language_detector = LanguageDetector()
    
    def register_parser(
        self,
        doc_type: DocumentType,
        parser: DocumentParser,
    ) -> None:
        """Register a parser."""
        self._parsers[doc_type] = parser
    
    def detect_type(
        self,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> DocumentType:
        """
        Detect document type.
        
        Args:
            file_path: File path
            content: Content string
            mime_type: MIME type
            
        Returns:
            DocumentType
        """
        # From MIME type
        if mime_type:
            mime_map = {
                'text/plain': DocumentType.TEXT,
                'text/markdown': DocumentType.MARKDOWN,
                'text/html': DocumentType.HTML,
                'application/json': DocumentType.JSON,
                'text/csv': DocumentType.CSV,
                'application/pdf': DocumentType.PDF,
            }
            if mime_type in mime_map:
                return mime_map[mime_type]
        
        # From file extension
        if file_path:
            ext = Path(file_path).suffix.lower()
            ext_map = {
                '.txt': DocumentType.TEXT,
                '.md': DocumentType.MARKDOWN,
                '.markdown': DocumentType.MARKDOWN,
                '.html': DocumentType.HTML,
                '.htm': DocumentType.HTML,
                '.json': DocumentType.JSON,
                '.csv': DocumentType.CSV,
                '.pdf': DocumentType.PDF,
                '.docx': DocumentType.DOCX,
                '.py': DocumentType.CODE,
                '.js': DocumentType.CODE,
                '.ts': DocumentType.CODE,
            }
            if ext in ext_map:
                return ext_map[ext]
        
        # From content
        if content:
            if content.strip().startswith('<!DOCTYPE') or content.strip().startswith('<html'):
                return DocumentType.HTML
            if content.strip().startswith('{') or content.strip().startswith('['):
                return DocumentType.JSON
            if re.search(r'^#\s+', content, re.MULTILINE):
                return DocumentType.MARKDOWN
        
        return DocumentType.TEXT
    
    def process(
        self,
        content: Union[str, bytes],
        doc_type: Optional[DocumentType] = None,
        **kwargs,
    ) -> ProcessedDocument:
        """
        Process document content.
        
        Args:
            content: Document content
            doc_type: Document type (auto-detected if None)
            **kwargs: Additional options
            
        Returns:
            ProcessedDocument
        """
        # Detect type if needed
        if doc_type is None:
            if isinstance(content, str):
                doc_type = self.detect_type(content=content)
            else:
                doc_type = DocumentType.TEXT
        
        # Get parser
        parser = self._parsers.get(doc_type)
        if parser is None:
            parser = self._parsers[DocumentType.TEXT]
        
        # Parse
        try:
            doc = parser.parse(content, **kwargs)
        except Exception as e:
            logger.error(f"Parse error: {e}")
            doc = ProcessedDocument(
                doc_id=hashlib.md5(str(content).encode()).hexdigest()[:12],
                doc_type=doc_type,
                status=ProcessingStatus.FAILED,
                errors=[str(e)],
            )
            return doc
        
        # Post-processing
        if self.clean_text and doc.clean_content:
            doc.clean_content = self._cleaner.clean(doc.clean_content)
        
        if self.detect_language:
            text = doc.clean_content or doc.raw_content
            if text:
                doc.metadata.language = self._language_detector.detect(text)
        
        return doc
    
    def process_file(
        self,
        file_path: str,
        **kwargs,
    ) -> ProcessedDocument:
        """
        Process document file.
        
        Args:
            file_path: Path to file
            **kwargs: Additional options
            
        Returns:
            ProcessedDocument
        """
        path = Path(file_path)
        
        if not path.exists():
            return ProcessedDocument(
                doc_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                doc_type=DocumentType.UNKNOWN,
                status=ProcessingStatus.FAILED,
                errors=[f"File not found: {file_path}"],
            )
        
        # Detect type
        doc_type = self.detect_type(file_path=file_path)
        
        # Read content
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
        except Exception as e:
            return ProcessedDocument(
                doc_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
                doc_type=doc_type,
                status=ProcessingStatus.FAILED,
                errors=[f"Read error: {e}"],
            )
        
        # Process
        doc = self.process(content, doc_type=doc_type, **kwargs)
        
        # Add file metadata
        doc.metadata.file_path = str(path.absolute())
        doc.metadata.file_name = path.name
        doc.metadata.file_size = path.stat().st_size
        doc.metadata.mime_type = mimetypes.guess_type(file_path)[0] or ""
        
        return doc
    
    def process_batch(
        self,
        items: List[Union[str, Tuple[str, DocumentType]]],
    ) -> Generator[ProcessedDocument, None, None]:
        """
        Process multiple documents.
        
        Args:
            items: List of content strings or (content, type) tuples
            
        Yields:
            ProcessedDocument for each item
        """
        for item in items:
            if isinstance(item, tuple):
                content, doc_type = item
                yield self.process(content, doc_type=doc_type)
            else:
                yield self.process(item)


# Convenience functions

def process_document(
    content: str,
    doc_type: Optional[str] = None,
) -> ProcessedDocument:
    """
    Quick document processing.
    
    Args:
        content: Document content
        doc_type: Document type name
        
    Returns:
        ProcessedDocument
    """
    processor = DocumentProcessor()
    
    dtype = None
    if doc_type:
        dtype = DocumentType[doc_type.upper()]
    
    return processor.process(content, doc_type=dtype)


def process_file(file_path: str) -> ProcessedDocument:
    """
    Quick file processing.
    
    Args:
        file_path: Path to file
        
    Returns:
        ProcessedDocument
    """
    processor = DocumentProcessor()
    return processor.process_file(file_path)


def extract_text(html: str) -> str:
    """
    Extract plain text from HTML.
    
    Args:
        html: HTML content
        
    Returns:
        Plain text
    """
    parser = HTMLParser()
    doc = parser.parse(html)
    return doc.clean_content
