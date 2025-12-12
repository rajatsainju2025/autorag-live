"""Output Formatting Module for AutoRAG-Live.

Format RAG outputs in various styles:
- Markdown formatting
- JSON/structured output
- HTML rendering
- Plain text
- Citation formatting
"""

from __future__ import annotations

import html
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutputFormat(Enum):
    """Available output formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    PLAIN = "plain"
    CITATIONS = "citations"
    STRUCTURED = "structured"


@dataclass
class Citation:
    """Represents a citation/reference."""

    source_id: str
    source_name: str
    url: str | None = None
    page: int | None = None
    excerpt: str = ""
    relevance: float = 1.0


@dataclass
class RAGOutput:
    """Represents a RAG output to be formatted."""

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional components
    thinking: str = ""
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    related_questions: list[str] = field(default_factory=list)


@dataclass
class FormatterConfig:
    """Configuration for output formatting."""

    # General settings
    include_sources: bool = True
    include_confidence: bool = True
    include_metadata: bool = False

    # Citation settings
    citation_style: str = "numbered"  # numbered, footnote, inline, apa
    max_citations: int = 10
    show_relevance: bool = False

    # Markdown settings
    heading_level: int = 2
    use_collapsible: bool = False

    # JSON settings
    pretty_print: bool = True
    indent: int = 2

    # HTML settings
    css_classes: dict[str, str] = field(default_factory=dict)
    sanitize_html: bool = True


class BaseFormatter(ABC):
    """Abstract base class for output formatters."""

    output_format: OutputFormat

    @abstractmethod
    def format(
        self,
        output: RAGOutput,
        config: FormatterConfig,
    ) -> str:
        """Format RAG output.

        Args:
            output: RAG output to format
            config: Formatter configuration

        Returns:
            Formatted string
        """
        pass


class MarkdownFormatter(BaseFormatter):
    """Format output as Markdown."""

    output_format = OutputFormat.MARKDOWN

    def format(
        self,
        output: RAGOutput,
        config: FormatterConfig,
    ) -> str:
        """Format as Markdown."""
        parts: list[str] = []

        # Main answer
        parts.append(output.answer)

        # Key points
        if output.key_points:
            parts.append("")
            heading = "#" * config.heading_level
            parts.append(f"{heading} Key Points")
            parts.append("")
            for point in output.key_points:
                parts.append(f"- {point}")

        # Confidence indicator
        if config.include_confidence and output.confidence < 1.0:
            parts.append("")
            confidence_pct = int(output.confidence * 100)
            parts.append(f"*Confidence: {confidence_pct}%*")

        # Sources/Citations
        if config.include_sources and (output.sources or output.citations):
            parts.append("")
            parts.append(self._format_sources(output, config))

        # Related questions
        if output.related_questions:
            parts.append("")
            heading = "#" * config.heading_level
            parts.append(f"{heading} Related Questions")
            parts.append("")
            for q in output.related_questions:
                parts.append(f"- {q}")

        return "\n".join(parts)

    def _format_sources(self, output: RAGOutput, config: FormatterConfig) -> str:
        """Format sources section."""
        parts: list[str] = []
        heading = "#" * config.heading_level
        parts.append(f"{heading} Sources")
        parts.append("")

        # Use citations if available, otherwise sources
        items = output.citations or [
            Citation(
                source_id=str(i + 1),
                source_name=s.get("name", s.get("source", f"Source {i + 1}")),
                url=s.get("url"),
                excerpt=s.get("excerpt", s.get("content", "")[:200]),
                relevance=s.get("score", s.get("relevance", 1.0)),
            )
            for i, s in enumerate(output.sources)
        ]

        for i, citation in enumerate(items[: config.max_citations], 1):
            if config.citation_style == "numbered":
                if citation.url:
                    parts.append(f"{i}. [{citation.source_name}]({citation.url})")
                else:
                    parts.append(f"{i}. {citation.source_name}")
            elif config.citation_style == "footnote":
                parts.append(f"[^{i}]: {citation.source_name}")
            else:
                parts.append(f"- {citation.source_name}")

            if citation.excerpt and config.use_collapsible:
                parts.append(f"   > {citation.excerpt[:150]}...")

        return "\n".join(parts)


class JSONFormatter(BaseFormatter):
    """Format output as JSON."""

    output_format = OutputFormat.JSON

    def format(
        self,
        output: RAGOutput,
        config: FormatterConfig,
    ) -> str:
        """Format as JSON."""
        result: dict[str, Any] = {
            "answer": output.answer,
        }

        if config.include_confidence:
            result["confidence"] = output.confidence

        if output.key_points:
            result["key_points"] = output.key_points

        if config.include_sources and output.citations:
            result["citations"] = [
                {
                    "id": c.source_id,
                    "name": c.source_name,
                    "url": c.url,
                    "excerpt": c.excerpt,
                    "relevance": c.relevance if config.show_relevance else None,
                }
                for c in output.citations[: config.max_citations]
            ]

        if output.related_questions:
            result["related_questions"] = output.related_questions

        if config.include_metadata and output.metadata:
            result["metadata"] = output.metadata

        if config.pretty_print:
            return json.dumps(result, indent=config.indent, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)


class HTMLFormatter(BaseFormatter):
    """Format output as HTML."""

    output_format = OutputFormat.HTML

    DEFAULT_CLASSES = {
        "container": "rag-output",
        "answer": "rag-answer",
        "sources": "rag-sources",
        "source_item": "rag-source-item",
        "confidence": "rag-confidence",
        "key_points": "rag-key-points",
        "related": "rag-related",
    }

    def format(
        self,
        output: RAGOutput,
        config: FormatterConfig,
    ) -> str:
        """Format as HTML."""
        classes = {**self.DEFAULT_CLASSES, **config.css_classes}
        parts: list[str] = []

        parts.append(f'<div class="{classes["container"]}">')

        # Main answer
        answer_html = self._format_text_as_html(output.answer, config)
        parts.append(f'  <div class="{classes["answer"]}">')
        parts.append(f"    {answer_html}")
        parts.append("  </div>")

        # Key points
        if output.key_points:
            parts.append(f'  <div class="{classes["key_points"]}">')
            parts.append("    <h3>Key Points</h3>")
            parts.append("    <ul>")
            for point in output.key_points:
                safe_point = html.escape(point) if config.sanitize_html else point
                parts.append(f"      <li>{safe_point}</li>")
            parts.append("    </ul>")
            parts.append("  </div>")

        # Confidence
        if config.include_confidence and output.confidence < 1.0:
            confidence_pct = int(output.confidence * 100)
            parts.append(f'  <div class="{classes["confidence"]}">')
            parts.append(f"    Confidence: {confidence_pct}%")
            parts.append("  </div>")

        # Sources
        if config.include_sources and (output.sources or output.citations):
            parts.append(f'  <div class="{classes["sources"]}">')
            parts.append("    <h3>Sources</h3>")
            parts.append("    <ol>")

            citations = output.citations or self._sources_to_citations(output.sources)

            for citation in citations[: config.max_citations]:
                safe_name = html.escape(citation.source_name) if config.sanitize_html else citation.source_name

                parts.append(f'      <li class="{classes["source_item"]}">')
                if citation.url:
                    parts.append(f'        <a href="{citation.url}">{safe_name}</a>')
                else:
                    parts.append(f"        {safe_name}")
                parts.append("      </li>")

            parts.append("    </ol>")
            parts.append("  </div>")

        # Related questions
        if output.related_questions:
            parts.append(f'  <div class="{classes["related"]}">')
            parts.append("    <h3>Related Questions</h3>")
            parts.append("    <ul>")
            for q in output.related_questions:
                safe_q = html.escape(q) if config.sanitize_html else q
                parts.append(f"      <li>{safe_q}</li>")
            parts.append("    </ul>")
            parts.append("  </div>")

        parts.append("</div>")

        return "\n".join(parts)

    def _format_text_as_html(self, text: str, config: FormatterConfig) -> str:
        """Convert text to HTML paragraphs."""
        if config.sanitize_html:
            text = html.escape(text)

        # Split into paragraphs
        paragraphs = text.split("\n\n")
        html_parts = [f"<p>{p.strip()}</p>" for p in paragraphs if p.strip()]

        return "\n    ".join(html_parts)

    def _sources_to_citations(self, sources: list[dict[str, Any]]) -> list[Citation]:
        """Convert source dicts to Citation objects."""
        return [
            Citation(
                source_id=str(i + 1),
                source_name=s.get("name", s.get("source", f"Source {i + 1}")),
                url=s.get("url"),
                excerpt=s.get("excerpt", ""),
            )
            for i, s in enumerate(sources)
        ]


class PlainTextFormatter(BaseFormatter):
    """Format output as plain text."""

    output_format = OutputFormat.PLAIN

    def format(
        self,
        output: RAGOutput,
        config: FormatterConfig,
    ) -> str:
        """Format as plain text."""
        parts: list[str] = []

        # Main answer
        parts.append(output.answer)

        # Key points
        if output.key_points:
            parts.append("")
            parts.append("Key Points:")
            for i, point in enumerate(output.key_points, 1):
                parts.append(f"  {i}. {point}")

        # Confidence
        if config.include_confidence and output.confidence < 1.0:
            parts.append("")
            confidence_pct = int(output.confidence * 100)
            parts.append(f"Confidence: {confidence_pct}%")

        # Sources
        if config.include_sources and (output.sources or output.citations):
            parts.append("")
            parts.append("Sources:")
            citations = output.citations or self._sources_to_citations(output.sources)
            for i, c in enumerate(citations[: config.max_citations], 1):
                if c.url:
                    parts.append(f"  {i}. {c.source_name} ({c.url})")
                else:
                    parts.append(f"  {i}. {c.source_name}")

        # Related questions
        if output.related_questions:
            parts.append("")
            parts.append("Related Questions:")
            for q in output.related_questions:
                parts.append(f"  - {q}")

        return "\n".join(parts)

    def _sources_to_citations(self, sources: list[dict[str, Any]]) -> list[Citation]:
        """Convert source dicts to Citation objects."""
        return [
            Citation(
                source_id=str(i + 1),
                source_name=s.get("name", s.get("source", f"Source {i + 1}")),
                url=s.get("url"),
            )
            for i, s in enumerate(sources)
        ]


class CitationFormatter(BaseFormatter):
    """Format citations in various academic styles."""

    output_format = OutputFormat.CITATIONS

    def format(
        self,
        output: RAGOutput,
        config: FormatterConfig,
    ) -> str:
        """Format citations."""
        citations = output.citations or self._sources_to_citations(output.sources)

        if config.citation_style == "apa":
            return self._format_apa(citations, config)
        elif config.citation_style == "mla":
            return self._format_mla(citations, config)
        elif config.citation_style == "numbered":
            return self._format_numbered(citations, config)
        else:
            return self._format_numbered(citations, config)

    def _format_numbered(
        self,
        citations: list[Citation],
        config: FormatterConfig,
    ) -> str:
        """Format as numbered references."""
        parts: list[str] = []
        for i, c in enumerate(citations[: config.max_citations], 1):
            line = f"[{i}] {c.source_name}"
            if c.url:
                line += f". {c.url}"
            if c.page:
                line += f", p. {c.page}"
            parts.append(line)
        return "\n".join(parts)

    def _format_apa(
        self,
        citations: list[Citation],
        config: FormatterConfig,
    ) -> str:
        """Format in APA style (simplified)."""
        parts: list[str] = []
        for c in citations[: config.max_citations]:
            # Simplified APA format
            author = c.metadata.get("author", "Unknown") if hasattr(c, "metadata") else "Unknown"
            year = c.metadata.get("year", "n.d.") if hasattr(c, "metadata") else "n.d."
            line = f"{author} ({year}). {c.source_name}."
            if c.url:
                line += f" Retrieved from {c.url}"
            parts.append(line)
        return "\n".join(parts)

    def _format_mla(
        self,
        citations: list[Citation],
        config: FormatterConfig,
    ) -> str:
        """Format in MLA style (simplified)."""
        parts: list[str] = []
        for c in citations[: config.max_citations]:
            line = f'"{c.source_name}."'
            if c.url:
                line += f" {c.url}."
            parts.append(line)
        return "\n".join(parts)

    def _sources_to_citations(self, sources: list[dict[str, Any]]) -> list[Citation]:
        """Convert source dicts to Citation objects."""
        return [
            Citation(
                source_id=str(i + 1),
                source_name=s.get("name", s.get("source", f"Source {i + 1}")),
                url=s.get("url"),
            )
            for i, s in enumerate(sources)
        ]


class OutputFormatterRegistry:
    """Registry for output formatters."""

    def __init__(self) -> None:
        """Initialize with default formatters."""
        self._formatters: dict[OutputFormat, BaseFormatter] = {
            OutputFormat.MARKDOWN: MarkdownFormatter(),
            OutputFormat.JSON: JSONFormatter(),
            OutputFormat.HTML: HTMLFormatter(),
            OutputFormat.PLAIN: PlainTextFormatter(),
            OutputFormat.CITATIONS: CitationFormatter(),
        }

    def register(
        self,
        format_type: OutputFormat,
        formatter: BaseFormatter,
    ) -> None:
        """Register a custom formatter.

        Args:
            format_type: Output format
            formatter: Formatter implementation
        """
        self._formatters[format_type] = formatter

    def format(
        self,
        output: RAGOutput,
        format_type: OutputFormat = OutputFormat.MARKDOWN,
        config: FormatterConfig | None = None,
    ) -> str:
        """Format output using specified formatter.

        Args:
            output: RAG output to format
            format_type: Output format
            config: Formatter configuration

        Returns:
            Formatted string
        """
        config = config or FormatterConfig()

        if format_type not in self._formatters:
            raise ValueError(f"Unknown format type: {format_type}")

        return self._formatters[format_type].format(output, config)


# Global registry
_registry: OutputFormatterRegistry | None = None


def get_formatter_registry() -> OutputFormatterRegistry:
    """Get global formatter registry."""
    global _registry
    if _registry is None:
        _registry = OutputFormatterRegistry()
    return _registry


# Convenience functions


def format_output(
    answer: str,
    sources: list[dict[str, Any]] | None = None,
    format_type: str = "markdown",
    include_sources: bool = True,
) -> str:
    """Format a RAG output.

    Args:
        answer: Answer text
        sources: Source documents
        format_type: Output format (markdown, json, html, plain)
        include_sources: Include sources in output

    Returns:
        Formatted string
    """
    output = RAGOutput(answer=answer, sources=sources or [])
    config = FormatterConfig(include_sources=include_sources)

    format_map = {
        "markdown": OutputFormat.MARKDOWN,
        "md": OutputFormat.MARKDOWN,
        "json": OutputFormat.JSON,
        "html": OutputFormat.HTML,
        "plain": OutputFormat.PLAIN,
        "text": OutputFormat.PLAIN,
    }

    output_format = format_map.get(format_type.lower(), OutputFormat.MARKDOWN)

    return get_formatter_registry().format(output, output_format, config)


def format_with_citations(
    answer: str,
    citations: list[dict[str, Any]],
    style: str = "numbered",
) -> str:
    """Format answer with citations.

    Args:
        answer: Answer text
        citations: Citation information
        style: Citation style (numbered, apa, mla)

    Returns:
        Formatted string with citations
    """
    citation_objects = [
        Citation(
            source_id=str(i + 1),
            source_name=c.get("name", c.get("title", f"Source {i + 1}")),
            url=c.get("url"),
            excerpt=c.get("excerpt", ""),
        )
        for i, c in enumerate(citations)
    ]

    output = RAGOutput(answer=answer, citations=citation_objects)
    config = FormatterConfig(citation_style=style)

    return get_formatter_registry().format(output, OutputFormat.MARKDOWN, config)


def format_json_output(
    answer: str,
    sources: list[dict[str, Any]] | None = None,
    confidence: float = 1.0,
    pretty: bool = True,
) -> str:
    """Format output as JSON.

    Args:
        answer: Answer text
        sources: Source documents
        confidence: Confidence score
        pretty: Pretty print JSON

    Returns:
        JSON string
    """
    output = RAGOutput(
        answer=answer,
        sources=sources or [],
        confidence=confidence,
    )
    config = FormatterConfig(pretty_print=pretty)

    return get_formatter_registry().format(output, OutputFormat.JSON, config)
