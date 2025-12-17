"""
Answer generation utilities for AutoRAG-Live.

Provides structured answer generation with citations,
source attribution, and confidence scoring.

Features:
- Structured answer generation
- Automatic citation insertion
- Source attribution
- Confidence scoring
- Multi-format output (text, JSON, markdown)
- Answer validation

Example usage:
    >>> generator = AnswerGenerator(llm_client)
    >>> answer = await generator.generate(
    ...     query="What is RAG?",
    ...     contexts=retrieved_docs
    ... )
    >>> print(answer.text)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

logger = logging.getLogger(__name__)


class AnswerFormat(str, Enum):
    """Output format for answers."""
    
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


class CitationStyle(str, Enum):
    """Citation formatting style."""
    
    NUMERIC = "numeric"        # [1], [2], [3]
    AUTHOR_YEAR = "author_year"  # (Smith, 2023)
    INLINE = "inline"          # "according to Source X"
    FOOTNOTE = "footnote"      # Superscript numbers
    NONE = "none"              # No citations


@dataclass
class Source:
    """A source document for citation."""
    
    id: str
    title: str
    content: str
    
    # Optional metadata
    author: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    page: Optional[int] = None
    
    # Relevance info
    relevance_score: float = 0.0
    chunk_index: Optional[int] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_citation(self, style: CitationStyle, index: int) -> str:
        """Format source as citation."""
        if style == CitationStyle.NUMERIC:
            return f"[{index}]"
        elif style == CitationStyle.AUTHOR_YEAR:
            author = self.author or "Unknown"
            year = self.date[:4] if self.date else "n.d."
            return f"({author}, {year})"
        elif style == CitationStyle.INLINE:
            return f'(Source: {self.title})'
        elif style == CitationStyle.FOOTNOTE:
            return f"^{index}"
        return ""


@dataclass
class Citation:
    """A citation reference in the answer."""
    
    source_id: str
    source_index: int
    text_span: Tuple[int, int]  # Start and end positions
    
    # The text being cited
    cited_text: str = ""
    
    # Citation metadata
    confidence: float = 1.0
    is_direct_quote: bool = False


@dataclass
class GeneratedAnswer:
    """A generated answer with metadata."""
    
    text: str
    query: str
    
    # Citations and sources
    citations: List[Citation] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    
    # Quality metrics
    confidence: float = 0.0
    relevance_score: float = 0.0
    groundedness_score: float = 0.0
    
    # Generation metadata
    model: str = ""
    generation_time: float = 0.0
    tokens_used: int = 0
    
    # Formatting
    format: AnswerFormat = AnswerFormat.PLAIN_TEXT
    citation_style: CitationStyle = CitationStyle.NUMERIC
    
    # Additional info
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_citations(self) -> bool:
        """Check if answer has citations."""
        return len(self.citations) > 0
    
    @property
    def citation_count(self) -> int:
        """Get number of citations."""
        return len(self.citations)
    
    @property
    def source_count(self) -> int:
        """Get number of unique sources."""
        return len(set(c.source_id for c in self.citations))
    
    def get_formatted_answer(self) -> str:
        """Get answer with formatting applied."""
        if self.format == AnswerFormat.MARKDOWN:
            return self._format_markdown()
        elif self.format == AnswerFormat.JSON:
            return self._format_json()
        elif self.format == AnswerFormat.HTML:
            return self._format_html()
        return self.text
    
    def _format_markdown(self) -> str:
        """Format as Markdown."""
        output = [self.text, "", "---", "**Sources:**"]
        
        for i, source in enumerate(self.sources, 1):
            ref = f"{i}. **{source.title}**"
            if source.author:
                ref += f" - {source.author}"
            if source.url:
                ref += f" ([link]({source.url}))"
            output.append(ref)
        
        return "\n".join(output)
    
    def _format_json(self) -> str:
        """Format as JSON."""
        return json.dumps({
            "answer": self.text,
            "query": self.query,
            "confidence": self.confidence,
            "sources": [
                {
                    "id": s.id,
                    "title": s.title,
                    "author": s.author,
                    "url": s.url,
                }
                for s in self.sources
            ],
            "citations": [
                {
                    "source_id": c.source_id,
                    "text": c.cited_text,
                }
                for c in self.citations
            ],
        }, indent=2)
    
    def _format_html(self) -> str:
        """Format as HTML."""
        html = f'<div class="answer">\n<p>{self.text}</p>\n'
        
        if self.sources:
            html += '<div class="sources">\n<h4>Sources:</h4>\n<ul>\n'
            for source in self.sources:
                html += f'<li><strong>{source.title}</strong>'
                if source.url:
                    html += f' - <a href="{source.url}">Link</a>'
                html += '</li>\n'
            html += '</ul>\n</div>\n'
        
        html += '</div>'
        return html


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients."""
    
    async def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """Generate text from prompt."""
        ...


class CitationInserter:
    """Insert citations into generated text."""
    
    def __init__(
        self,
        style: CitationStyle = CitationStyle.NUMERIC,
    ):
        """
        Initialize citation inserter.
        
        Args:
            style: Citation style to use
        """
        self.style = style
    
    def insert_citations(
        self,
        text: str,
        sources: List[Source],
        min_similarity: float = 0.5,
    ) -> Tuple[str, List[Citation]]:
        """
        Insert citations into text.
        
        Args:
            text: Generated text
            sources: Source documents
            min_similarity: Minimum similarity for citation
            
        Returns:
            Tuple of (cited_text, citations)
        """
        if self.style == CitationStyle.NONE:
            return text, []
        
        citations = []
        cited_text = text
        
        # Find sentences that need citations
        sentences = self._split_sentences(text)
        offset = 0
        
        for sentence in sentences:
            # Find best matching source
            best_source, best_score = self._find_best_source(
                sentence, sources
            )
            
            if best_source and best_score >= min_similarity:
                # Get source index
                source_idx = next(
                    i for i, s in enumerate(sources)
                    if s.id == best_source.id
                ) + 1
                
                # Create citation marker
                citation_marker = best_source.to_citation(
                    self.style, source_idx
                )
                
                # Find sentence position in text
                sentence_pos = cited_text.find(sentence, offset)
                if sentence_pos >= 0:
                    # Insert citation at end of sentence
                    insert_pos = sentence_pos + len(sentence)
                    
                    # Handle punctuation
                    if cited_text[insert_pos - 1] in '.!?':
                        insert_pos -= 1
                        cited_text = (
                            cited_text[:insert_pos] +
                            citation_marker +
                            cited_text[insert_pos:]
                        )
                    else:
                        cited_text = (
                            cited_text[:insert_pos] +
                            citation_marker +
                            cited_text[insert_pos:]
                        )
                    
                    # Record citation
                    citations.append(Citation(
                        source_id=best_source.id,
                        source_index=source_idx,
                        text_span=(sentence_pos, insert_pos),
                        cited_text=sentence,
                        confidence=best_score,
                    ))
                    
                    offset = insert_pos + len(citation_marker)
        
        return cited_text, citations
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_best_source(
        self,
        sentence: str,
        sources: List[Source],
    ) -> Tuple[Optional[Source], float]:
        """Find best matching source for sentence."""
        best_source = None
        best_score = 0.0
        
        sentence_words = set(sentence.lower().split())
        
        for source in sources:
            # Simple word overlap similarity
            source_words = set(source.content.lower().split())
            
            if not sentence_words or not source_words:
                continue
            
            overlap = len(sentence_words & source_words)
            score = overlap / len(sentence_words)
            
            if score > best_score:
                best_score = score
                best_source = source
        
        return best_source, best_score


class ConfidenceScorer:
    """Score answer confidence."""
    
    def score(
        self,
        answer: str,
        query: str,
        sources: List[Source],
    ) -> Dict[str, float]:
        """
        Score answer quality.
        
        Args:
            answer: Generated answer
            query: Original query
            sources: Source documents
            
        Returns:
            Dict with confidence scores
        """
        return {
            "overall": self._compute_overall(answer, query, sources),
            "relevance": self._compute_relevance(answer, query),
            "groundedness": self._compute_groundedness(answer, sources),
            "completeness": self._compute_completeness(answer, query),
        }
    
    def _compute_overall(
        self,
        answer: str,
        query: str,
        sources: List[Source],
    ) -> float:
        """Compute overall confidence."""
        relevance = self._compute_relevance(answer, query)
        groundedness = self._compute_groundedness(answer, sources)
        completeness = self._compute_completeness(answer, query)
        
        return (relevance + groundedness + completeness) / 3
    
    def _compute_relevance(self, answer: str, query: str) -> float:
        """Compute relevance to query."""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 0.5
        
        overlap = len(query_words & answer_words)
        return min(overlap / len(query_words), 1.0)
    
    def _compute_groundedness(
        self,
        answer: str,
        sources: List[Source],
    ) -> float:
        """Compute how grounded answer is in sources."""
        if not sources:
            return 0.0
        
        answer_words = set(answer.lower().split())
        all_source_words = set()
        
        for source in sources:
            all_source_words.update(source.content.lower().split())
        
        if not answer_words:
            return 0.5
        
        overlap = len(answer_words & all_source_words)
        return overlap / len(answer_words)
    
    def _compute_completeness(self, answer: str, query: str) -> float:
        """Estimate answer completeness."""
        # Simple heuristic based on answer length
        min_length = 50
        ideal_length = 200
        
        length = len(answer)
        
        if length < min_length:
            return length / min_length * 0.5
        elif length < ideal_length:
            return 0.5 + (length - min_length) / (ideal_length - min_length) * 0.5
        else:
            return 1.0


class PromptBuilder:
    """Build prompts for answer generation."""
    
    def __init__(
        self,
        system_template: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        """
        Initialize prompt builder.
        
        Args:
            system_template: System prompt template
            user_template: User prompt template
        """
        self.system_template = system_template or self._default_system()
        self.user_template = user_template or self._default_user()
    
    def _default_system(self) -> str:
        """Get default system prompt."""
        return """You are a helpful assistant that answers questions based on provided context.

Guidelines:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so
3. Be concise but comprehensive
4. Use specific facts from the context
5. Do not make up information"""
    
    def _default_user(self) -> str:
        """Get default user prompt."""
        return """Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
    
    def build_prompt(
        self,
        query: str,
        sources: List[Source],
        format_instructions: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Build system and user prompts.
        
        Args:
            query: User query
            sources: Source documents
            format_instructions: Optional format instructions
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Build context from sources
        context_parts = []
        for i, source in enumerate(sources, 1):
            part = f"[Source {i}]: {source.title}\n{source.content}"
            context_parts.append(part)
        
        context = "\n\n".join(context_parts)
        
        # Build prompts
        system = self.system_template
        if format_instructions:
            system += f"\n\n{format_instructions}"
        
        user = self.user_template.format(
            context=context,
            query=query,
        )
        
        return system, user


class AnswerValidator:
    """Validate generated answers."""
    
    def __init__(
        self,
        min_length: int = 20,
        max_length: int = 2000,
        check_hallucination: bool = True,
    ):
        """
        Initialize validator.
        
        Args:
            min_length: Minimum answer length
            max_length: Maximum answer length
            check_hallucination: Check for hallucination
        """
        self.min_length = min_length
        self.max_length = max_length
        self.check_hallucination = check_hallucination
    
    def validate(
        self,
        answer: str,
        query: str,
        sources: List[Source],
    ) -> Tuple[bool, List[str]]:
        """
        Validate answer.
        
        Args:
            answer: Generated answer
            query: Original query
            sources: Source documents
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings = []
        is_valid = True
        
        # Check length
        if len(answer) < self.min_length:
            warnings.append(f"Answer too short ({len(answer)} chars)")
            is_valid = False
        elif len(answer) > self.max_length:
            warnings.append(f"Answer too long ({len(answer)} chars)")
        
        # Check for common issues
        if self._is_refusal(answer):
            warnings.append("Answer appears to be a refusal")
        
        if self._is_generic(answer):
            warnings.append("Answer appears generic")
        
        # Check hallucination
        if self.check_hallucination:
            hallucination_score = self._check_hallucination(answer, sources)
            if hallucination_score > 0.5:
                warnings.append(f"Possible hallucination (score: {hallucination_score:.2f})")
        
        return is_valid, warnings
    
    def _is_refusal(self, answer: str) -> bool:
        """Check if answer is a refusal."""
        refusal_phrases = [
            "i cannot",
            "i can't",
            "i am unable",
            "i'm unable",
            "i don't have",
            "no information",
            "not able to",
        ]
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in refusal_phrases)
    
    def _is_generic(self, answer: str) -> bool:
        """Check if answer is generic."""
        generic_phrases = [
            "it depends",
            "there are many",
            "various factors",
            "in general",
        ]
        answer_lower = answer.lower()
        matches = sum(1 for p in generic_phrases if p in answer_lower)
        return matches >= 2
    
    def _check_hallucination(
        self,
        answer: str,
        sources: List[Source],
    ) -> float:
        """Check for hallucination."""
        if not sources:
            return 0.5
        
        # Get all words from sources
        source_words = set()
        for source in sources:
            source_words.update(source.content.lower().split())
        
        # Get answer words (excluding common words)
        common_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'it', 'its', 'and', 'or',
            'but', 'if', 'of', 'to', 'in', 'for', 'on', 'with', 'as',
            'by', 'at', 'from',
        }
        
        answer_words = set(answer.lower().split()) - common_words
        
        if not answer_words:
            return 0.0
        
        # Check overlap
        overlap = len(answer_words & source_words)
        return 1.0 - (overlap / len(answer_words))


class AnswerGenerator:
    """
    Main answer generation interface.
    
    Example:
        >>> from autorag_live.llm import LLMClient
        >>> 
        >>> llm = LLMClient()
        >>> generator = AnswerGenerator(llm)
        >>> 
        >>> sources = [
        ...     Source(id="1", title="Doc 1", content="RAG is..."),
        ... ]
        >>> 
        >>> answer = await generator.generate(
        ...     query="What is RAG?",
        ...     sources=sources,
        ...     citation_style=CitationStyle.NUMERIC
        ... )
        >>> print(answer.text)
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClientProtocol] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        citation_style: CitationStyle = CitationStyle.NUMERIC,
        output_format: AnswerFormat = AnswerFormat.PLAIN_TEXT,
        validate_answers: bool = True,
    ):
        """
        Initialize answer generator.
        
        Args:
            llm_client: LLM client for generation
            prompt_builder: Prompt builder
            citation_style: Citation style
            output_format: Output format
            validate_answers: Whether to validate answers
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.citation_style = citation_style
        self.output_format = output_format
        self.validate_answers = validate_answers
        
        # Components
        self.citation_inserter = CitationInserter(citation_style)
        self.confidence_scorer = ConfidenceScorer()
        self.validator = AnswerValidator()
    
    async def generate(
        self,
        query: str,
        sources: List[Source],
        format_instructions: Optional[str] = None,
        **kwargs,
    ) -> GeneratedAnswer:
        """
        Generate an answer.
        
        Args:
            query: User query
            sources: Source documents
            format_instructions: Optional format instructions
            **kwargs: Additional LLM parameters
            
        Returns:
            GeneratedAnswer
        """
        import time
        start_time = time.time()
        
        # Build prompts
        system_prompt, user_prompt = self.prompt_builder.build_prompt(
            query=query,
            sources=sources,
            format_instructions=format_instructions,
        )
        
        # Generate answer
        if self.llm_client:
            raw_answer = await self.llm_client.generate(
                prompt=user_prompt,
                system=system_prompt,
                **kwargs,
            )
        else:
            # Mock response for testing
            raw_answer = self._mock_generate(query, sources)
        
        # Insert citations
        cited_answer, citations = self.citation_inserter.insert_citations(
            raw_answer,
            sources,
        )
        
        # Score confidence
        scores = self.confidence_scorer.score(raw_answer, query, sources)
        
        # Validate
        warnings = []
        if self.validate_answers:
            _, warnings = self.validator.validate(raw_answer, query, sources)
        
        generation_time = time.time() - start_time
        
        return GeneratedAnswer(
            text=cited_answer,
            query=query,
            citations=citations,
            sources=sources,
            confidence=scores["overall"],
            relevance_score=scores["relevance"],
            groundedness_score=scores["groundedness"],
            generation_time=generation_time,
            format=self.output_format,
            citation_style=self.citation_style,
            warnings=warnings,
        )
    
    def _mock_generate(self, query: str, sources: List[Source]) -> str:
        """Generate mock answer for testing."""
        if not sources:
            return "I don't have enough information to answer this question."
        
        # Create simple answer from sources
        answer_parts = [f"Based on the available information:"]
        for source in sources[:3]:
            snippet = source.content[:200]
            answer_parts.append(snippet)
        
        return " ".join(answer_parts)
    
    async def generate_with_retry(
        self,
        query: str,
        sources: List[Source],
        max_retries: int = 3,
        **kwargs,
    ) -> GeneratedAnswer:
        """
        Generate answer with retry on failure.
        
        Args:
            query: User query
            sources: Source documents
            max_retries: Maximum retry attempts
            **kwargs: Additional parameters
            
        Returns:
            GeneratedAnswer
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return await self.generate(query, sources, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 ** attempt)
        
        # Return error answer
        return GeneratedAnswer(
            text=f"Failed to generate answer: {last_error}",
            query=query,
            sources=sources,
            confidence=0.0,
            warnings=[f"Generation failed after {max_retries} attempts"],
        )


class StreamingAnswerGenerator:
    """Generate answers with streaming output."""
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ):
        """
        Initialize streaming generator.
        
        Args:
            llm_client: Streaming LLM client
            prompt_builder: Prompt builder
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()
    
    async def generate_stream(
        self,
        query: str,
        sources: List[Source],
        on_chunk: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        """
        Generate answer with streaming.
        
        Args:
            query: User query
            sources: Source documents
            on_chunk: Callback for each chunk
            **kwargs: Additional parameters
            
        Yields:
            Answer chunks
        """
        system_prompt, user_prompt = self.prompt_builder.build_prompt(
            query=query,
            sources=sources,
        )
        
        full_answer = ""
        
        if self.llm_client and hasattr(self.llm_client, 'generate_stream'):
            async for chunk in self.llm_client.generate_stream(
                prompt=user_prompt,
                system=system_prompt,
                **kwargs,
            ):
                full_answer += chunk
                if on_chunk:
                    on_chunk(chunk)
                yield chunk
        else:
            # Mock streaming
            mock_answer = "This is a mock streaming response."
            for word in mock_answer.split():
                chunk = word + " "
                full_answer += chunk
                if on_chunk:
                    on_chunk(chunk)
                yield chunk
                await asyncio.sleep(0.1)


# Convenience functions

def create_source(
    content: str,
    title: Optional[str] = None,
    **kwargs,
) -> Source:
    """
    Create a source document.
    
    Args:
        content: Source content
        title: Source title
        **kwargs: Additional source attributes
        
    Returns:
        Source instance
    """
    import hashlib
    
    source_id = kwargs.pop('id', None) or hashlib.md5(
        content.encode()
    ).hexdigest()[:8]
    
    return Source(
        id=source_id,
        title=title or f"Source {source_id}",
        content=content,
        **kwargs,
    )


async def generate_answer(
    query: str,
    contexts: List[str],
    llm_client: Optional[LLMClientProtocol] = None,
    **kwargs,
) -> str:
    """
    Quick answer generation.
    
    Args:
        query: User query
        contexts: Context strings
        llm_client: Optional LLM client
        **kwargs: Additional parameters
        
    Returns:
        Generated answer text
    """
    sources = [
        create_source(ctx, title=f"Context {i}")
        for i, ctx in enumerate(contexts, 1)
    ]
    
    generator = AnswerGenerator(llm_client)
    answer = await generator.generate(query, sources, **kwargs)
    return answer.text
