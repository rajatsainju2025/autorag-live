"""
Document preprocessing utilities for AutoRAG-Live.

Provides comprehensive document preprocessing including cleaning,
normalization, language detection, and text extraction.

Features:
- Text cleaning and normalization
- Language detection
- Encoding handling
- HTML/Markdown parsing
- Whitespace normalization
- Special character handling

Example usage:
    >>> preprocessor = DocumentPreprocessor()
    >>> cleaned = preprocessor.preprocess("Raw document text...")
    >>> print(cleaned.text)
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class Language(str, Enum):
    """Supported languages."""
    
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    RUSSIAN = "ru"
    ARABIC = "ar"
    UNKNOWN = "unknown"


class ContentType(str, Enum):
    """Document content types."""
    
    PLAIN_TEXT = "plain_text"
    HTML = "html"
    MARKDOWN = "markdown"
    CODE = "code"
    MIXED = "mixed"


@dataclass
class PreprocessedDocument:
    """Result of document preprocessing."""
    
    text: str
    original_text: str
    
    # Detected info
    language: Language = Language.UNKNOWN
    content_type: ContentType = ContentType.PLAIN_TEXT
    encoding: str = "utf-8"
    
    # Statistics
    original_length: int = 0
    processed_length: int = 0
    word_count: int = 0
    sentence_count: int = 0
    
    # Extracted elements
    urls: List[str] = field(default_factory=list)
    emails: List[str] = field(default_factory=list)
    code_blocks: List[str] = field(default_factory=list)
    
    # Quality indicators
    quality_score: float = 1.0
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def compression_ratio(self) -> float:
        """Get text compression ratio."""
        if self.original_length > 0:
            return self.processed_length / self.original_length
        return 1.0
    
    @property
    def is_empty(self) -> bool:
        """Check if processed text is empty."""
        return len(self.text.strip()) == 0


class BasePreprocessingStep(ABC):
    """Base class for preprocessing steps."""
    
    @abstractmethod
    def process(self, text: str) -> str:
        """Process text."""
        pass
    
    @property
    def name(self) -> str:
        """Get step name."""
        return self.__class__.__name__


class WhitespaceNormalizer(BasePreprocessingStep):
    """Normalize whitespace in text."""
    
    def __init__(
        self,
        collapse_spaces: bool = True,
        normalize_newlines: bool = True,
        strip_lines: bool = True,
        max_consecutive_newlines: int = 2,
    ):
        """
        Initialize whitespace normalizer.
        
        Args:
            collapse_spaces: Collapse multiple spaces to one
            normalize_newlines: Normalize different newline styles
            strip_lines: Strip whitespace from line ends
            max_consecutive_newlines: Maximum consecutive newlines
        """
        self.collapse_spaces = collapse_spaces
        self.normalize_newlines = normalize_newlines
        self.strip_lines = strip_lines
        self.max_consecutive_newlines = max_consecutive_newlines
    
    def process(self, text: str) -> str:
        """Normalize whitespace."""
        # Normalize newlines
        if self.normalize_newlines:
            text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Strip lines
        if self.strip_lines:
            lines = text.split('\n')
            lines = [line.rstrip() for line in lines]
            text = '\n'.join(lines)
        
        # Collapse multiple spaces
        if self.collapse_spaces:
            text = re.sub(r'[ \t]+', ' ', text)
        
        # Limit consecutive newlines
        if self.max_consecutive_newlines > 0:
            pattern = r'\n{' + str(self.max_consecutive_newlines + 1) + r',}'
            replacement = '\n' * self.max_consecutive_newlines
            text = re.sub(pattern, replacement, text)
        
        return text.strip()


class UnicodeNormalizer(BasePreprocessingStep):
    """Normalize Unicode characters."""
    
    def __init__(
        self,
        form: str = "NFKC",
        remove_control_chars: bool = True,
        replace_smart_quotes: bool = True,
    ):
        """
        Initialize Unicode normalizer.
        
        Args:
            form: Unicode normalization form (NFC, NFKC, NFD, NFKD)
            remove_control_chars: Remove control characters
            replace_smart_quotes: Replace smart quotes with ASCII
        """
        self.form = form
        self.remove_control_chars = remove_control_chars
        self.replace_smart_quotes = replace_smart_quotes
    
    def process(self, text: str) -> str:
        """Normalize Unicode."""
        # Apply Unicode normalization
        text = unicodedata.normalize(self.form, text)
        
        # Remove control characters
        if self.remove_control_chars:
            text = ''.join(
                char for char in text
                if not unicodedata.category(char).startswith('C')
                or char in '\n\t'
            )
        
        # Replace smart quotes
        if self.replace_smart_quotes:
            replacements = {
                '\u2018': "'",  # Left single quote
                '\u2019': "'",  # Right single quote
                '\u201C': '"',  # Left double quote
                '\u201D': '"',  # Right double quote
                '\u2013': '-',  # En dash
                '\u2014': '--', # Em dash
                '\u2026': '...', # Ellipsis
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
        
        return text


class HTMLCleaner(BasePreprocessingStep):
    """Clean HTML content."""
    
    def __init__(
        self,
        remove_tags: bool = True,
        decode_entities: bool = True,
        preserve_structure: bool = False,
        keep_links: bool = False,
    ):
        """
        Initialize HTML cleaner.
        
        Args:
            remove_tags: Remove HTML tags
            decode_entities: Decode HTML entities
            preserve_structure: Preserve some structural elements
            keep_links: Keep link URLs
        """
        self.remove_tags = remove_tags
        self.decode_entities = decode_entities
        self.preserve_structure = preserve_structure
        self.keep_links = keep_links
    
    def process(self, text: str) -> str:
        """Clean HTML."""
        # Decode HTML entities first
        if self.decode_entities:
            text = html.unescape(text)
        
        if not self.remove_tags:
            return text
        
        # Extract links if needed
        links = []
        if self.keep_links:
            link_pattern = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>', re.IGNORECASE)
            for match in link_pattern.finditer(text):
                links.append((match.group(2), match.group(1)))
        
        # Remove script and style content
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Handle structural elements
        if self.preserve_structure:
            # Add newlines for block elements
            block_tags = ['p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr']
            for tag in block_tags:
                text = re.sub(rf'</?{tag}[^>]*>', '\n', text, flags=re.IGNORECASE)
        
        # Remove remaining tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Re-add links if keeping them
        for link_text, url in links:
            text = text.replace(link_text, f"{link_text} ({url})")
        
        return text


class MarkdownCleaner(BasePreprocessingStep):
    """Clean Markdown content."""
    
    def __init__(
        self,
        remove_formatting: bool = True,
        preserve_code_blocks: bool = True,
        preserve_links: bool = False,
    ):
        """
        Initialize Markdown cleaner.
        
        Args:
            remove_formatting: Remove Markdown formatting
            preserve_code_blocks: Keep code block content
            preserve_links: Keep link text and URLs
        """
        self.remove_formatting = remove_formatting
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_links = preserve_links
    
    def process(self, text: str) -> str:
        """Clean Markdown."""
        if not self.remove_formatting:
            return text
        
        # Extract and preserve code blocks
        code_blocks = []
        if self.preserve_code_blocks:
            code_pattern = re.compile(r'```[\w]*\n(.*?)```', re.DOTALL)
            for i, match in enumerate(code_pattern.finditer(text)):
                placeholder = f"__CODE_BLOCK_{i}__"
                code_blocks.append((placeholder, match.group(1)))
                text = text.replace(match.group(0), placeholder)
        
        # Remove headers (#)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # Handle links
        if self.preserve_links:
            text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
        else:
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove emphasis
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'__([^_]+)__', r'\1', text)      # Bold
        text = re.sub(r'_([^_]+)_', r'\1', text)        # Italic
        
        # Remove inline code
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove blockquotes
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
        
        # Remove horizontal rules
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        
        # Remove list markers
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Restore code blocks
        for placeholder, code in code_blocks:
            text = text.replace(placeholder, f"\n{code}\n")
        
        return text


class SpecialCharacterHandler(BasePreprocessingStep):
    """Handle special characters."""
    
    def __init__(
        self,
        remove_emojis: bool = False,
        normalize_punctuation: bool = True,
        remove_zero_width: bool = True,
    ):
        """
        Initialize special character handler.
        
        Args:
            remove_emojis: Remove emoji characters
            normalize_punctuation: Normalize punctuation
            remove_zero_width: Remove zero-width characters
        """
        self.remove_emojis = remove_emojis
        self.normalize_punctuation = normalize_punctuation
        self.remove_zero_width = remove_zero_width
    
    def process(self, text: str) -> str:
        """Handle special characters."""
        # Remove zero-width characters
        if self.remove_zero_width:
            zero_width = ['\u200b', '\u200c', '\u200d', '\ufeff', '\u00ad']
            for char in zero_width:
                text = text.replace(char, '')
        
        # Remove emojis
        if self.remove_emojis:
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # Emoticons
                "\U0001F300-\U0001F5FF"  # Symbols & pictographs
                "\U0001F680-\U0001F6FF"  # Transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # Flags
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+",
                flags=re.UNICODE
            )
            text = emoji_pattern.sub('', text)
        
        # Normalize punctuation
        if self.normalize_punctuation:
            # Normalize quotes
            text = re.sub(r'[""„]', '"', text)
            text = re.sub(r"[''‚]", "'", text)
            
            # Normalize dashes
            text = re.sub(r'[–—]', '-', text)
            
            # Remove duplicate punctuation
            text = re.sub(r'([.!?])\1+', r'\1', text)
        
        return text


class LanguageDetector:
    """Detect document language."""
    
    # Common words by language for simple detection
    LANGUAGE_WORDS = {
        Language.ENGLISH: {'the', 'is', 'and', 'to', 'of', 'a', 'in', 'that', 'it', 'for'},
        Language.SPANISH: {'el', 'la', 'de', 'en', 'y', 'que', 'los', 'del', 'las', 'un'},
        Language.FRENCH: {'le', 'la', 'de', 'et', 'les', 'des', 'en', 'du', 'une', 'est'},
        Language.GERMAN: {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'ist'},
        Language.PORTUGUESE: {'de', 'da', 'do', 'em', 'para', 'os', 'uma', 'com', 'não', 'por'},
        Language.ITALIAN: {'di', 'che', 'il', 'la', 'per', 'un', 'del', 'sono', 'alla', 'con'},
    }
    
    def detect(self, text: str) -> Tuple[Language, float]:
        """
        Detect language of text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (Language, confidence)
        """
        # Try using external library first
        try:
            import langdetect
            detected = langdetect.detect(text)
            lang_map = {
                'en': Language.ENGLISH,
                'es': Language.SPANISH,
                'fr': Language.FRENCH,
                'de': Language.GERMAN,
                'pt': Language.PORTUGUESE,
                'it': Language.ITALIAN,
                'zh-cn': Language.CHINESE,
                'zh-tw': Language.CHINESE,
                'ja': Language.JAPANESE,
                'ko': Language.KOREAN,
                'nl': Language.DUTCH,
                'ru': Language.RUSSIAN,
                'ar': Language.ARABIC,
            }
            return lang_map.get(detected, Language.UNKNOWN), 0.8
        except (ImportError, Exception):
            pass
        
        # Fallback to simple word matching
        return self._simple_detect(text)
    
    def _simple_detect(self, text: str) -> Tuple[Language, float]:
        """Simple language detection using common words."""
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        best_lang = Language.UNKNOWN
        best_score = 0.0
        
        for lang, lang_words in self.LANGUAGE_WORDS.items():
            matches = len(words & lang_words)
            score = matches / len(lang_words) if lang_words else 0
            
            if score > best_score:
                best_score = score
                best_lang = lang
        
        confidence = min(best_score * 2, 1.0)
        return best_lang, confidence


class ContentTypeDetector:
    """Detect document content type."""
    
    def detect(self, text: str) -> ContentType:
        """
        Detect content type.
        
        Args:
            text: Input text
            
        Returns:
            ContentType
        """
        # Check for HTML
        if re.search(r'<(?:html|head|body|div|p|span)[^>]*>', text, re.IGNORECASE):
            return ContentType.HTML
        
        # Check for Markdown
        markdown_patterns = [
            r'^#+\s',              # Headers
            r'\[.*\]\(.*\)',       # Links
            r'```\w*\n',           # Code blocks
            r'^\s*[-*]\s+',        # Lists
            r'\*\*.*\*\*',         # Bold
        ]
        markdown_matches = sum(
            1 for p in markdown_patterns
            if re.search(p, text, re.MULTILINE)
        )
        if markdown_matches >= 2:
            return ContentType.MARKDOWN
        
        # Check for code
        code_patterns = [
            r'def\s+\w+\s*\(',     # Python functions
            r'function\s+\w+\s*\(', # JS functions
            r'class\s+\w+',         # Classes
            r'import\s+[\w.]+',     # Imports
            r'\{\s*\n',             # Braces
        ]
        code_matches = sum(
            1 for p in code_patterns
            if re.search(p, text)
        )
        if code_matches >= 2:
            return ContentType.CODE
        
        return ContentType.PLAIN_TEXT


class TextExtractor:
    """Extract text from various formats."""
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        url_pattern = re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
        )
        return url_pattern.findall(text)
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses."""
        email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        return email_pattern.findall(text)
    
    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks."""
        # Fenced code blocks
        fenced = re.findall(r'```[\w]*\n(.*?)```', text, re.DOTALL)
        
        # Indented code blocks (4 spaces)
        indented = re.findall(r'(?:^    .+\n)+', text, re.MULTILINE)
        
        return fenced + indented
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class QualityScorer:
    """Score document quality."""
    
    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 100000,
        min_word_count: int = 10,
    ):
        """
        Initialize quality scorer.
        
        Args:
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length
            min_word_count: Minimum word count
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_word_count = min_word_count
    
    def score(self, text: str) -> Tuple[float, List[str]]:
        """
        Score text quality.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (score, warnings)
        """
        warnings = []
        score = 1.0
        
        # Check length
        length = len(text)
        if length < self.min_length:
            warnings.append(f"Text too short ({length} < {self.min_length})")
            score *= 0.5
        elif length > self.max_length:
            warnings.append(f"Text too long ({length} > {self.max_length})")
            score *= 0.8
        
        # Check word count
        words = len(text.split())
        if words < self.min_word_count:
            warnings.append(f"Too few words ({words} < {self.min_word_count})")
            score *= 0.5
        
        # Check for excessive special characters
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(length, 1)
        if alpha_ratio < 0.5:
            warnings.append(f"Low alphabetic ratio ({alpha_ratio:.2f})")
            score *= 0.7
        
        # Check for repetition
        if self._has_repetition(text):
            warnings.append("Contains repetitive content")
            score *= 0.8
        
        return max(score, 0.0), warnings
    
    def _has_repetition(self, text: str) -> bool:
        """Check for repetitive content."""
        # Check for repeated lines
        lines = text.split('\n')
        if len(lines) > 5:
            unique_lines = set(lines)
            if len(unique_lines) / len(lines) < 0.5:
                return True
        
        # Check for repeated phrases
        words = text.lower().split()
        if len(words) > 20:
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            if len(set(trigrams)) / len(trigrams) < 0.7:
                return True
        
        return False


class DocumentPreprocessor:
    """
    Main document preprocessing interface.
    
    Example:
        >>> preprocessor = DocumentPreprocessor()
        >>> 
        >>> # Basic preprocessing
        >>> result = preprocessor.preprocess("Raw text...")
        >>> print(result.text)
        >>> 
        >>> # With specific options
        >>> preprocessor = DocumentPreprocessor(
        ...     remove_html=True,
        ...     detect_language=True,
        ...     min_quality=0.5
        ... )
        >>> result = preprocessor.preprocess(html_content)
    """
    
    def __init__(
        self,
        remove_html: bool = True,
        remove_markdown: bool = False,
        normalize_unicode: bool = True,
        normalize_whitespace: bool = True,
        detect_language: bool = True,
        detect_content_type: bool = True,
        compute_quality: bool = True,
        min_quality: float = 0.0,
        custom_steps: Optional[List[BasePreprocessingStep]] = None,
    ):
        """
        Initialize document preprocessor.
        
        Args:
            remove_html: Remove HTML tags
            remove_markdown: Remove Markdown formatting
            normalize_unicode: Normalize Unicode
            normalize_whitespace: Normalize whitespace
            detect_language: Detect document language
            detect_content_type: Detect content type
            compute_quality: Compute quality score
            min_quality: Minimum quality threshold
            custom_steps: Additional preprocessing steps
        """
        self.remove_html = remove_html
        self.remove_markdown = remove_markdown
        self.normalize_unicode = normalize_unicode
        self.normalize_whitespace = normalize_whitespace
        self.detect_language = detect_language
        self.detect_content_type = detect_content_type
        self.compute_quality = compute_quality
        self.min_quality = min_quality
        
        # Build processing pipeline
        self._steps = self._build_pipeline(custom_steps)
        
        # Detectors and scorers
        self._language_detector = LanguageDetector()
        self._content_detector = ContentTypeDetector()
        self._quality_scorer = QualityScorer()
        self._text_extractor = TextExtractor()
    
    def _build_pipeline(
        self,
        custom_steps: Optional[List[BasePreprocessingStep]],
    ) -> List[BasePreprocessingStep]:
        """Build preprocessing pipeline."""
        steps = []
        
        if self.normalize_unicode:
            steps.append(UnicodeNormalizer())
        
        if self.remove_html:
            steps.append(HTMLCleaner())
        
        if self.remove_markdown:
            steps.append(MarkdownCleaner())
        
        steps.append(SpecialCharacterHandler())
        
        if self.normalize_whitespace:
            steps.append(WhitespaceNormalizer())
        
        if custom_steps:
            steps.extend(custom_steps)
        
        return steps
    
    def preprocess(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PreprocessedDocument:
        """
        Preprocess a document.
        
        Args:
            text: Raw document text
            metadata: Optional metadata
            
        Returns:
            PreprocessedDocument
        """
        original_text = text
        original_length = len(text)
        
        # Detect content type
        content_type = ContentType.PLAIN_TEXT
        if self.detect_content_type:
            content_type = self._content_detector.detect(text)
        
        # Extract elements before cleaning
        urls = self._text_extractor.extract_urls(text)
        emails = self._text_extractor.extract_emails(text)
        code_blocks = self._text_extractor.extract_code_blocks(text)
        
        # Apply preprocessing steps
        for step in self._steps:
            try:
                text = step.process(text)
            except Exception as e:
                logger.warning(f"Step {step.name} failed: {e}")
        
        # Detect language
        language = Language.UNKNOWN
        if self.detect_language and text.strip():
            language, _ = self._language_detector.detect(text)
        
        # Compute quality
        quality_score = 1.0
        warnings = []
        if self.compute_quality:
            quality_score, warnings = self._quality_scorer.score(text)
        
        # Count words and sentences
        word_count = len(text.split())
        sentence_count = len(self._text_extractor.extract_sentences(text))
        
        return PreprocessedDocument(
            text=text,
            original_text=original_text,
            language=language,
            content_type=content_type,
            original_length=original_length,
            processed_length=len(text),
            word_count=word_count,
            sentence_count=sentence_count,
            urls=urls,
            emails=emails,
            code_blocks=code_blocks,
            quality_score=quality_score,
            warnings=warnings,
            metadata=metadata or {},
        )
    
    def preprocess_batch(
        self,
        texts: List[str],
        filter_low_quality: bool = True,
    ) -> List[PreprocessedDocument]:
        """
        Preprocess multiple documents.
        
        Args:
            texts: List of raw texts
            filter_low_quality: Filter out low quality documents
            
        Returns:
            List of PreprocessedDocument
        """
        results = []
        
        for text in texts:
            doc = self.preprocess(text)
            
            if filter_low_quality and doc.quality_score < self.min_quality:
                continue
            
            results.append(doc)
        
        return results


# Convenience functions

def preprocess_text(
    text: str,
    remove_html: bool = True,
    normalize: bool = True,
) -> str:
    """
    Quick text preprocessing.
    
    Args:
        text: Raw text
        remove_html: Remove HTML tags
        normalize: Normalize whitespace
        
    Returns:
        Cleaned text
    """
    preprocessor = DocumentPreprocessor(
        remove_html=remove_html,
        normalize_whitespace=normalize,
        detect_language=False,
        compute_quality=False,
    )
    result = preprocessor.preprocess(text)
    return result.text


def detect_language(text: str) -> Language:
    """
    Detect text language.
    
    Args:
        text: Input text
        
    Returns:
        Detected language
    """
    detector = LanguageDetector()
    lang, _ = detector.detect(text)
    return lang


def clean_html(text: str) -> str:
    """
    Clean HTML content.
    
    Args:
        text: HTML text
        
    Returns:
        Cleaned text
    """
    cleaner = HTMLCleaner()
    return cleaner.process(text)


def normalize_text(text: str) -> str:
    """
    Normalize text (Unicode and whitespace).
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    text = UnicodeNormalizer().process(text)
    text = WhitespaceNormalizer().process(text)
    return text
