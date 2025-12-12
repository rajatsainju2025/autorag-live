"""
Response generation for AutoRAG-Live.

Generates well-structured responses with citations,
source attribution, and multiple formatting options.

Features:
- Citation injection and formatting
- Multiple output formats (markdown, plain, HTML)
- Confidence scoring
- Response streaming support
- Source attribution

Example usage:
    >>> generator = ResponseGenerator()
    >>> response = generator.generate(
    ...     query="What is Python?",
    ...     context=[{"text": "Python is a programming language...", "source": "wiki"}],
    ...     include_citations=True
    ... )
    >>> print(response.text)
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class ResponseFormat(str, Enum):
    """Response output formats."""
    
    PLAIN = "plain"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


class CitationStyle(str, Enum):
    """Citation formatting styles."""
    
    NUMERIC = "numeric"  # [1], [2]
    AUTHOR_YEAR = "author_year"  # (Smith, 2023)
    INLINE = "inline"  # (source: Wikipedia)
    FOOTNOTE = "footnote"  # Text¹
    SUPERSCRIPT = "superscript"  # Text^[1]


@dataclass
class Source:
    """Represents a source/citation."""
    
    id: str
    text: str
    
    # Source metadata
    title: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None
    date: Optional[str] = None
    
    # Relevance
    score: float = 0.0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_citation(self, style: CitationStyle, index: int = 1) -> str:
        """Format as citation string."""
        if style == CitationStyle.NUMERIC:
            return f"[{index}]"
        
        elif style == CitationStyle.AUTHOR_YEAR:
            author = self.author or "Unknown"
            year = self.date[:4] if self.date else "n.d."
            return f"({author}, {year})"
        
        elif style == CitationStyle.INLINE:
            source = self.title or self.id
            return f"(source: {source})"
        
        elif style == CitationStyle.FOOTNOTE:
            return f"^{index}"
        
        elif style == CitationStyle.SUPERSCRIPT:
            return f"^[{index}]"
        
        return f"[{index}]"
    
    def to_reference(self, style: CitationStyle, index: int = 1) -> str:
        """Format as full reference."""
        parts = []
        
        if style == CitationStyle.NUMERIC:
            parts.append(f"[{index}]")
        
        if self.author:
            parts.append(self.author)
        
        if self.title:
            parts.append(f'"{self.title}"')
        
        if self.date:
            parts.append(f"({self.date})")
        
        if self.url:
            parts.append(self.url)
        
        return " ".join(parts) if parts else f"[{index}] {self.id}"


@dataclass
class GeneratedResponse:
    """Generated response with metadata."""
    
    text: str
    sources: List[Source] = field(default_factory=list)
    
    # Generation metadata
    confidence: float = 0.0
    generation_time: float = 0.0
    token_count: int = 0
    
    # Citations
    citations_used: List[int] = field(default_factory=list)
    
    # Format info
    format: ResponseFormat = ResponseFormat.PLAIN
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_references_section(
        self,
        style: CitationStyle = CitationStyle.NUMERIC,
    ) -> str:
        """Get formatted references section."""
        if not self.sources:
            return ""
        
        lines = ["", "**References:**", ""]
        
        for i, source in enumerate(self.sources, 1):
            if i in self.citations_used or not self.citations_used:
                lines.append(source.to_reference(style, i))
        
        return "\n".join(lines)
    
    def with_references(self) -> str:
        """Get response with references section."""
        return self.text + self.get_references_section()


class ResponseBuilder:
    """Build responses with formatting."""
    
    def __init__(
        self,
        format: ResponseFormat = ResponseFormat.MARKDOWN,
        citation_style: CitationStyle = CitationStyle.NUMERIC,
    ):
        """
        Initialize response builder.
        
        Args:
            format: Output format
            citation_style: Citation style
        """
        self.format = format
        self.citation_style = citation_style
        
        self._parts: List[str] = []
        self._sources: List[Source] = []
        self._citations_used: List[int] = []
    
    def add_text(self, text: str) -> "ResponseBuilder":
        """Add plain text."""
        self._parts.append(text)
        return self
    
    def add_paragraph(self, text: str) -> "ResponseBuilder":
        """Add paragraph with formatting."""
        if self.format == ResponseFormat.HTML:
            self._parts.append(f"<p>{text}</p>")
        else:
            self._parts.append(f"\n{text}\n")
        return self
    
    def add_heading(self, text: str, level: int = 2) -> "ResponseBuilder":
        """Add heading."""
        if self.format == ResponseFormat.MARKDOWN:
            prefix = "#" * level
            self._parts.append(f"\n{prefix} {text}\n")
        elif self.format == ResponseFormat.HTML:
            self._parts.append(f"<h{level}>{text}</h{level}>")
        else:
            self._parts.append(f"\n{text.upper()}\n")
        return self
    
    def add_list(self, items: List[str], ordered: bool = False) -> "ResponseBuilder":
        """Add list."""
        if self.format == ResponseFormat.MARKDOWN:
            for i, item in enumerate(items, 1):
                prefix = f"{i}." if ordered else "-"
                self._parts.append(f"{prefix} {item}")
        elif self.format == ResponseFormat.HTML:
            tag = "ol" if ordered else "ul"
            list_html = f"<{tag}>"
            for item in items:
                list_html += f"<li>{item}</li>"
            list_html += f"</{tag}>"
            self._parts.append(list_html)
        else:
            for i, item in enumerate(items, 1):
                prefix = f"{i}." if ordered else "*"
                self._parts.append(f"{prefix} {item}")
        return self
    
    def add_citation(
        self,
        text: str,
        source_index: int,
    ) -> "ResponseBuilder":
        """Add text with citation."""
        if source_index <= len(self._sources):
            source = self._sources[source_index - 1]
            citation = source.to_citation(self.citation_style, source_index)
            self._parts.append(f"{text}{citation}")
            
            if source_index not in self._citations_used:
                self._citations_used.append(source_index)
        else:
            self._parts.append(text)
        return self
    
    def add_source(self, source: Source) -> int:
        """Add source and return its index."""
        self._sources.append(source)
        return len(self._sources)
    
    def add_sources(self, sources: List[Source]) -> "ResponseBuilder":
        """Add multiple sources."""
        self._sources.extend(sources)
        return self
    
    def build(self) -> GeneratedResponse:
        """Build the final response."""
        text = " ".join(self._parts)
        
        # Clean up spacing
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return GeneratedResponse(
            text=text,
            sources=self._sources,
            citations_used=self._citations_used,
            format=self.format,
        )


class CitationInjector:
    """Inject citations into generated text."""
    
    def __init__(
        self,
        style: CitationStyle = CitationStyle.NUMERIC,
    ):
        """Initialize citation injector."""
        self.style = style
    
    def inject_citations(
        self,
        text: str,
        sources: List[Source],
        threshold: float = 0.5,
    ) -> tuple[str, List[int]]:
        """
        Inject citations into text based on source matching.
        
        Args:
            text: Generated text
            sources: Available sources
            threshold: Minimum similarity for citation
            
        Returns:
            Tuple of (cited text, list of used source indices)
        """
        sentences = self._split_sentences(text)
        cited_sentences = []
        used_indices = []
        
        for sentence in sentences:
            # Find best matching source
            best_source_idx = self._find_best_source(sentence, sources, threshold)
            
            if best_source_idx is not None:
                source = sources[best_source_idx]
                citation = source.to_citation(self.style, best_source_idx + 1)
                
                # Remove trailing punctuation, add citation, restore punctuation
                if sentence and sentence[-1] in '.!?':
                    cited_sentences.append(f"{sentence[:-1]}{citation}{sentence[-1]}")
                else:
                    cited_sentences.append(f"{sentence}{citation}")
                
                if best_source_idx + 1 not in used_indices:
                    used_indices.append(best_source_idx + 1)
            else:
                cited_sentences.append(sentence)
        
        return " ".join(cited_sentences), used_indices
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_best_source(
        self,
        sentence: str,
        sources: List[Source],
        threshold: float,
    ) -> Optional[int]:
        """Find the best matching source for a sentence."""
        if not sources:
            return None
        
        best_idx = None
        best_score = threshold
        
        sentence_words = set(sentence.lower().split())
        
        for i, source in enumerate(sources):
            # Simple word overlap scoring
            source_words = set(source.text.lower().split())
            overlap = len(sentence_words & source_words)
            
            if overlap > 0:
                score = overlap / max(len(sentence_words), 1)
                
                if score > best_score:
                    best_score = score
                    best_idx = i
        
        return best_idx


class ResponseTemplate:
    """Response template for consistent formatting."""
    
    def __init__(
        self,
        template: str,
        format: ResponseFormat = ResponseFormat.MARKDOWN,
    ):
        """
        Initialize template.
        
        Args:
            template: Template string with {placeholders}
            format: Output format
        """
        self.template = template
        self.format = format
    
    def render(self, **kwargs: Any) -> str:
        """Render template with values."""
        return self.template.format(**kwargs)


# Common templates
TEMPLATES = {
    "qa": ResponseTemplate(
        "## Answer\n\n{answer}\n\n{sources_section}",
        ResponseFormat.MARKDOWN,
    ),
    "summary": ResponseTemplate(
        "## Summary\n\n{summary}\n\n### Key Points\n\n{key_points}",
        ResponseFormat.MARKDOWN,
    ),
    "comparison": ResponseTemplate(
        "## Comparison: {topic}\n\n{comparison}\n\n### Conclusion\n\n{conclusion}",
        ResponseFormat.MARKDOWN,
    ),
}


class BaseResponseGenerator(ABC):
    """Base class for response generators."""
    
    @abstractmethod
    def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> GeneratedResponse:
        """Generate a response."""
        pass
    
    @abstractmethod
    async def agenerate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> GeneratedResponse:
        """Generate a response asynchronously."""
        pass


class TemplateResponseGenerator(BaseResponseGenerator):
    """Generate responses using templates."""
    
    def __init__(
        self,
        format: ResponseFormat = ResponseFormat.MARKDOWN,
        citation_style: CitationStyle = CitationStyle.NUMERIC,
    ):
        """Initialize generator."""
        self.format = format
        self.citation_style = citation_style
        self.injector = CitationInjector(citation_style)
    
    def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        template: Optional[str] = None,
        include_citations: bool = True,
        **kwargs: Any,
    ) -> GeneratedResponse:
        """
        Generate response from template.
        
        Args:
            query: User query
            context: Retrieved context
            template: Template name or custom template
            include_citations: Include citations
            **kwargs: Additional template variables
            
        Returns:
            GeneratedResponse
        """
        start_time = time.time()
        
        # Convert context to sources
        sources = []
        for i, ctx in enumerate(context):
            sources.append(Source(
                id=f"source_{i}",
                text=ctx.get("text", ""),
                title=ctx.get("title"),
                author=ctx.get("author"),
                url=ctx.get("url"),
                score=ctx.get("score", 0.0),
                metadata=ctx.get("metadata", {}),
            ))
        
        # Build response
        builder = ResponseBuilder(self.format, self.citation_style)
        builder.add_sources(sources)
        
        # Create answer from context
        if sources:
            combined_text = " ".join(s.text for s in sources[:3])  # Use top 3
            
            if include_citations:
                combined_text, used = self.injector.inject_citations(
                    combined_text, sources
                )
                builder._citations_used = used
            
            builder.add_paragraph(combined_text)
        else:
            builder.add_paragraph("No relevant information found.")
        
        response = builder.build()
        response.generation_time = time.time() - start_time
        
        return response
    
    async def agenerate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> GeneratedResponse:
        """Generate asynchronously."""
        return self.generate(query, context, **kwargs)


class LLMResponseGenerator(BaseResponseGenerator):
    """Generate responses using LLM."""
    
    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
        format: ResponseFormat = ResponseFormat.MARKDOWN,
        citation_style: CitationStyle = CitationStyle.NUMERIC,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize LLM generator.
        
        Args:
            llm_func: Function to call LLM
            format: Output format
            citation_style: Citation style
            system_prompt: System prompt for LLM
        """
        self.llm_func = llm_func
        self.format = format
        self.citation_style = citation_style
        self.injector = CitationInjector(citation_style)
        
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a helpful assistant that answers questions based on the provided context.
        
Guidelines:
- Answer based only on the provided context
- Be concise but comprehensive
- If the context doesn't contain enough information, say so
- Use clear, well-structured responses
"""
    
    def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        include_citations: bool = True,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> GeneratedResponse:
        """
        Generate response using LLM.
        
        Args:
            query: User query
            context: Retrieved context
            include_citations: Include citations
            max_tokens: Maximum tokens
            **kwargs: Additional arguments
            
        Returns:
            GeneratedResponse
        """
        start_time = time.time()
        
        # Convert context to sources
        sources = []
        for i, ctx in enumerate(context):
            sources.append(Source(
                id=f"source_{i}",
                text=ctx.get("text", ""),
                title=ctx.get("title"),
                author=ctx.get("author"),
                url=ctx.get("url"),
                score=ctx.get("score", 0.0),
                metadata=ctx.get("metadata", {}),
            ))
        
        # Build prompt
        prompt = self._build_prompt(query, sources)
        
        # Generate with LLM
        if self.llm_func:
            generated_text = self.llm_func(prompt)
        else:
            # Fallback: use context directly
            generated_text = self._fallback_generate(query, sources)
        
        # Inject citations if requested
        citations_used = []
        if include_citations and sources:
            generated_text, citations_used = self.injector.inject_citations(
                generated_text, sources
            )
        
        response = GeneratedResponse(
            text=generated_text,
            sources=sources,
            citations_used=citations_used,
            format=self.format,
            generation_time=time.time() - start_time,
        )
        
        return response
    
    async def agenerate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> GeneratedResponse:
        """Generate asynchronously."""
        # For async LLM calls, override this method
        return self.generate(query, context, **kwargs)
    
    def _build_prompt(self, query: str, sources: List[Source]) -> str:
        """Build prompt for LLM."""
        context_text = "\n\n".join(
            f"[{i+1}] {s.text}" for i, s in enumerate(sources)
        )
        
        return f"""Context:
{context_text}

Question: {query}

Please answer the question based on the provided context."""
    
    def _fallback_generate(self, query: str, sources: List[Source]) -> str:
        """Fallback generation without LLM."""
        if not sources:
            return f"No information available for: {query}"
        
        # Simple extractive response
        return sources[0].text[:500]


class StreamingResponseGenerator:
    """Generate streaming responses."""
    
    def __init__(
        self,
        generator: BaseResponseGenerator,
        chunk_size: int = 50,
    ):
        """
        Initialize streaming generator.
        
        Args:
            generator: Base generator
            chunk_size: Characters per chunk
        """
        self.generator = generator
        self.chunk_size = chunk_size
    
    def stream(
        self,
        query: str,
        context: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Generate response as stream.
        
        Args:
            query: User query
            context: Retrieved context
            **kwargs: Additional arguments
            
        Yields:
            Response chunks
        """
        # Generate full response
        response = self.generator.generate(query, context, **kwargs)
        
        # Stream in chunks
        text = response.text
        for i in range(0, len(text), self.chunk_size):
            yield text[i:i + self.chunk_size]
    
    async def astream(
        self,
        query: str,
        context: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate response as async stream.
        
        Args:
            query: User query
            context: Retrieved context
            **kwargs: Additional arguments
            
        Yields:
            Response chunks
        """
        import asyncio
        
        response = await self.generator.agenerate(query, context, **kwargs)
        
        text = response.text
        for i in range(0, len(text), self.chunk_size):
            yield text[i:i + self.chunk_size]
            await asyncio.sleep(0)  # Allow other tasks


class ResponseGenerator:
    """
    Main response generation interface.
    
    Example:
        >>> generator = ResponseGenerator()
        >>> 
        >>> # Generate with context
        >>> response = generator.generate(
        ...     query="What is machine learning?",
        ...     context=[
        ...         {"text": "Machine learning is a subset of AI...", "title": "ML Intro"},
        ...     ],
        ...     include_citations=True,
        ... )
        >>> 
        >>> print(response.text)
        >>> print(response.with_references())
    """
    
    def __init__(
        self,
        format: ResponseFormat = ResponseFormat.MARKDOWN,
        citation_style: CitationStyle = CitationStyle.NUMERIC,
        llm_func: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize response generator.
        
        Args:
            format: Output format
            citation_style: Citation style
            llm_func: Optional LLM function
        """
        self.format = format
        self.citation_style = citation_style
        
        if llm_func:
            self._generator = LLMResponseGenerator(
                llm_func=llm_func,
                format=format,
                citation_style=citation_style,
            )
        else:
            self._generator = TemplateResponseGenerator(
                format=format,
                citation_style=citation_style,
            )
        
        self.injector = CitationInjector(citation_style)
    
    def generate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        include_citations: bool = True,
        include_references: bool = False,
        **kwargs: Any,
    ) -> GeneratedResponse:
        """
        Generate a response.
        
        Args:
            query: User query
            context: Retrieved context
            include_citations: Include inline citations
            include_references: Include references section
            **kwargs: Additional arguments
            
        Returns:
            GeneratedResponse
        """
        response = self._generator.generate(
            query, context, include_citations=include_citations, **kwargs
        )
        
        if include_references:
            response.text = response.with_references()
        
        return response
    
    async def agenerate(
        self,
        query: str,
        context: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> GeneratedResponse:
        """Generate asynchronously."""
        return await self._generator.agenerate(query, context, **kwargs)
    
    def stream(
        self,
        query: str,
        context: List[Dict[str, Any]],
        chunk_size: int = 50,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream response chunks."""
        streamer = StreamingResponseGenerator(self._generator, chunk_size)
        yield from streamer.stream(query, context, **kwargs)
    
    async def astream(
        self,
        query: str,
        context: List[Dict[str, Any]],
        chunk_size: int = 50,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response chunks asynchronously."""
        streamer = StreamingResponseGenerator(self._generator, chunk_size)
        async for chunk in streamer.astream(query, context, **kwargs):
            yield chunk


class ConfidenceScorer:
    """Score response confidence."""
    
    def __init__(
        self,
        min_sources: int = 1,
        min_relevance: float = 0.3,
    ):
        """
        Initialize scorer.
        
        Args:
            min_sources: Minimum sources for high confidence
            min_relevance: Minimum relevance score
        """
        self.min_sources = min_sources
        self.min_relevance = min_relevance
    
    def score(
        self,
        response: GeneratedResponse,
    ) -> float:
        """
        Score response confidence.
        
        Args:
            response: Generated response
            
        Returns:
            Confidence score 0-1
        """
        if not response.sources:
            return 0.0
        
        # Factor 1: Number of sources
        source_factor = min(len(response.sources) / self.min_sources, 1.0)
        
        # Factor 2: Average relevance
        avg_relevance = sum(s.score for s in response.sources) / len(response.sources)
        relevance_factor = min(avg_relevance / self.min_relevance, 1.0)
        
        # Factor 3: Citation coverage
        if response.citations_used:
            citation_factor = len(response.citations_used) / len(response.sources)
        else:
            citation_factor = 0.5
        
        # Weighted average
        confidence = (
            source_factor * 0.3 +
            relevance_factor * 0.5 +
            citation_factor * 0.2
        )
        
        return min(max(confidence, 0.0), 1.0)


# Global generator instance
_default_generator: Optional[ResponseGenerator] = None


def get_response_generator(
    format: ResponseFormat = ResponseFormat.MARKDOWN,
    citation_style: CitationStyle = CitationStyle.NUMERIC,
) -> ResponseGenerator:
    """Get or create the default response generator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = ResponseGenerator(format, citation_style)
    return _default_generator


def generate_response(
    query: str,
    context: List[Dict[str, Any]],
    **kwargs: Any,
) -> GeneratedResponse:
    """
    Convenience function to generate a response.
    
    Args:
        query: User query
        context: Retrieved context
        **kwargs: Additional arguments
        
    Returns:
        GeneratedResponse
    """
    return get_response_generator().generate(query, context, **kwargs)
