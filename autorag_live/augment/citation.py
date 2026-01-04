"""
Citation and Attribution System Module.

Implements inline citation generation and source attribution
for RAG-generated responses, following ALCE (Gao et al., 2023).

Key Features:
1. Inline citation insertion with [1], [2], etc.
2. Source span extraction and linking
3. Citation verification against sources
4. Attribution score calculation
5. Citation format customization

Example:
    >>> citator = CitationGenerator(llm)
    >>> result = await citator.generate_with_citations(answer, sources)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from autorag_live.core.protocols import BaseLLM, Document, Message

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


class CitationStyle(str, Enum):
    """Available citation styles."""

    NUMERIC = "numeric"  # [1], [2], [3]
    SUPERSCRIPT = "superscript"  # ¹, ², ³
    AUTHOR_DATE = "author_date"  # (Author, 2024)
    FOOTNOTE = "footnote"  # Footnote markers


@dataclass
class Citation:
    """
    A single citation reference.

    Attributes:
        citation_id: Unique citation ID
        source_id: Reference to source document
        source_title: Title of the source
        source_url: URL if available
        span_start: Character position in answer
        span_end: End position in answer
        cited_text: The text being cited
        source_text: Supporting text from source
        confidence: Confidence in citation (0-1)
    """

    citation_id: int
    source_id: str
    source_title: str = ""
    source_url: str = ""
    span_start: int = 0
    span_end: int = 0
    cited_text: str = ""
    source_text: str = ""
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "citation_id": self.citation_id,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "cited_text": self.cited_text,
            "source_text": self.source_text[:200],
            "confidence": self.confidence,
        }


@dataclass
class SourceSpan:
    """
    A span of text that maps to a source.

    Attributes:
        text: The span text
        source_idx: Index of the source document
        match_score: How well it matches the source
        source_excerpt: The matching excerpt from source
    """

    text: str
    source_idx: int
    match_score: float = 0.8
    source_excerpt: str = ""


@dataclass
class CitationResult:
    """
    Result of citation generation.

    Attributes:
        cited_text: Text with inline citations
        citations: List of citation objects
        bibliography: Formatted bibliography
        attribution_score: Overall attribution score
        uncited_claims: Claims without citations
    """

    cited_text: str
    citations: List[Citation] = field(default_factory=list)
    bibliography: str = ""
    attribution_score: float = 0.8
    uncited_claims: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cited_text": self.cited_text,
            "citations": [c.to_dict() for c in self.citations],
            "bibliography": self.bibliography,
            "attribution_score": self.attribution_score,
            "uncited_claims": self.uncited_claims,
        }


@dataclass
class VerificationResult:
    """Result of citation verification."""

    is_valid: bool
    support_score: float
    issues: List[str] = field(default_factory=list)


# =============================================================================
# Claim Extractor
# =============================================================================


class ClaimExtractor:
    """
    Extracts factual claims from text for citation.

    Identifies statements that should be attributed to sources.
    """

    # Patterns indicating factual claims
    CLAIM_PATTERNS = [
        r"(?:is|are|was|were|has|have|had)\s+[a-z]+(?:ly)?\s+[a-z]+",
        r"\d+[\d,\.]*\s*(?:%|percent|million|billion|thousand)",
        r"(?:in|during|since|from)\s+\d{4}",
        r"(?:according to|research shows|studies indicate)",
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:is|was|are|were)",
    ]

    def extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.

        Args:
            text: Input text

        Returns:
            List of claim strings
        """
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if sentence contains factual content
            if self._is_factual_claim(sentence):
                claims.append(sentence)

        return claims

    def _is_factual_claim(self, sentence: str) -> bool:
        """Check if sentence is a factual claim."""
        sentence_lower = sentence.lower()

        # Skip questions and opinions
        if sentence.endswith("?"):
            return False
        if any(w in sentence_lower for w in ["i think", "i believe", "in my opinion"]):
            return False

        # Check for factual indicators
        for pattern in self.CLAIM_PATTERNS:
            if re.search(pattern, sentence_lower):
                return True

        # Check for named entities (capitalized words)
        entities = re.findall(r"\b[A-Z][a-z]+\b", sentence)
        if len(entities) >= 2:
            return True

        # Check for numbers
        if re.search(r"\b\d+\b", sentence):
            return True

        return False


# =============================================================================
# Source Matcher
# =============================================================================


class SourceMatcher:
    """
    Matches claims to source documents.

    Uses text overlap and semantic similarity for matching.
    """

    def __init__(self, min_overlap: float = 0.3):
        """
        Initialize matcher.

        Args:
            min_overlap: Minimum word overlap for match
        """
        self.min_overlap = min_overlap

    def match_claim_to_sources(
        self,
        claim: str,
        sources: List[Document],
    ) -> List[Tuple[int, float, str]]:
        """
        Match a claim to source documents.

        Args:
            claim: Claim text
            sources: List of source documents

        Returns:
            List of (source_idx, score, matching_excerpt)
        """
        matches = []
        claim_words = set(self._tokenize(claim))

        for idx, source in enumerate(sources):
            # Find best matching excerpt
            best_score = 0.0
            best_excerpt = ""

            # Check against source sentences
            source_sentences = re.split(r"(?<=[.!?])\s+", source.content)

            for sentence in source_sentences:
                sentence_words = set(self._tokenize(sentence))
                overlap = len(claim_words & sentence_words) / len(claim_words) if claim_words else 0
                if overlap > best_score:
                    best_score = overlap
                    best_excerpt = sentence

            if best_score >= self.min_overlap:
                matches.append((idx, best_score, best_excerpt))

        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def find_source_spans(
        self,
        text: str,
        sources: List[Document],
    ) -> List[SourceSpan]:
        """
        Find spans in text that map to sources.

        Args:
            text: Response text
            sources: Source documents

        Returns:
            List of source spans
        """
        spans = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            matches = self.match_claim_to_sources(sentence, sources)
            if matches:
                best_match = matches[0]
                spans.append(
                    SourceSpan(
                        text=sentence,
                        source_idx=best_match[0],
                        match_score=best_match[1],
                        source_excerpt=best_match[2],
                    )
                )

        return spans

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and extract words
        return re.findall(r"\b\w+\b", text.lower())


# =============================================================================
# Citation Generator
# =============================================================================


class CitationGenerator:
    """
    Generates inline citations for RAG responses.

    Supports multiple citation styles and automatic source attribution.

    Example:
        >>> generator = CitationGenerator(llm)
        >>> result = await generator.generate_with_citations(
        ...     "Python was created by Guido van Rossum.",
        ...     [Document(content="Python created by Guido...")]
        ... )
        >>> print(result.cited_text)
        "Python was created by Guido van Rossum [1]."
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        *,
        style: CitationStyle = CitationStyle.NUMERIC,
        min_confidence: float = 0.5,
    ):
        """
        Initialize generator.

        Args:
            llm: LLM for advanced citation (optional)
            style: Citation style to use
            min_confidence: Minimum confidence for citation
        """
        self.llm = llm
        self.style = style
        self.min_confidence = min_confidence

        self.claim_extractor = ClaimExtractor()
        self.source_matcher = SourceMatcher()

    def generate_citations(
        self,
        text: str,
        sources: List[Document],
    ) -> CitationResult:
        """
        Generate citations for text using heuristics.

        Args:
            text: Response text
            sources: Source documents

        Returns:
            CitationResult with cited text
        """
        if not sources:
            return CitationResult(
                cited_text=text,
                attribution_score=0.0,
                uncited_claims=self.claim_extractor.extract_claims(text),
            )

        # Find source spans
        spans = self.source_matcher.find_source_spans(text, sources)

        # Build citations
        citations = []
        cited_source_idxs = set()

        for span in spans:
            if span.match_score >= self.min_confidence:
                citation = Citation(
                    citation_id=len(citations) + 1,
                    source_id=sources[span.source_idx].id,
                    source_title=sources[span.source_idx].metadata.get(
                        "title", f"Source {span.source_idx + 1}"
                    ),
                    cited_text=span.text,
                    source_text=span.source_excerpt,
                    confidence=span.match_score,
                )
                citations.append(citation)
                cited_source_idxs.add(span.source_idx)

        # Insert citations into text
        cited_text = self._insert_citations(text, spans, citations)

        # Generate bibliography
        bibliography = self._generate_bibliography(sources, cited_source_idxs)

        # Find uncited claims
        all_claims = self.claim_extractor.extract_claims(text)
        cited_texts = {c.cited_text for c in citations}
        uncited = [c for c in all_claims if c not in cited_texts]

        # Calculate attribution score
        attribution_score = len(citations) / len(all_claims) if all_claims else 1.0

        return CitationResult(
            cited_text=cited_text,
            citations=citations,
            bibliography=bibliography,
            attribution_score=attribution_score,
            uncited_claims=uncited,
        )

    async def generate_with_llm(
        self,
        text: str,
        sources: List[Document],
    ) -> CitationResult:
        """
        Generate citations using LLM for better accuracy.

        Args:
            text: Response text
            sources: Source documents

        Returns:
            CitationResult
        """
        if not self.llm:
            return self.generate_citations(text, sources)

        # Format sources for prompt
        source_list = "\n".join(f"[{i+1}] {s.content[:500]}" for i, s in enumerate(sources))

        prompt = f"""Add inline citations to this text based on the sources provided.

Text to cite:
{text}

Sources:
{source_list}

Rules:
- Add [1], [2], etc. after statements supported by sources
- Only cite when the source directly supports the claim
- A statement can have multiple citations if supported by multiple sources
- Don't add citations to opinions or hedged statements

Return the text with citations added:"""

        try:
            result = await self.llm.generate(
                [Message.user(prompt)],
                temperature=0.3,
            )

            cited_text = result.content.strip()

            # Extract citations from the text
            citations = self._extract_citations_from_text(cited_text, sources)

            # Generate bibliography
            bibliography = self._generate_bibliography(
                sources,
                {c.citation_id - 1 for c in citations},
            )

            return CitationResult(
                cited_text=cited_text,
                citations=citations,
                bibliography=bibliography,
                attribution_score=len(citations)
                / max(1, len(self.claim_extractor.extract_claims(text))),
            )

        except Exception as e:
            logger.warning(f"LLM citation failed: {e}")
            return self.generate_citations(text, sources)

    def _insert_citations(
        self,
        text: str,
        spans: List[SourceSpan],
        citations: List[Citation],
    ) -> str:
        """Insert citation markers into text."""
        # Build mapping of sentences to citations
        sentence_to_citation = {}
        for citation in citations:
            sentence_to_citation[citation.cited_text] = citation.citation_id

        # Process text sentence by sentence
        result_parts = []
        sentences = re.split(r"(?<=[.!?])(\s+)", text)

        for part in sentences:
            stripped = part.strip()
            if stripped in sentence_to_citation:
                # Add citation marker
                citation_id = sentence_to_citation[stripped]
                marker = self._format_citation_marker(citation_id)

                # Insert before final punctuation
                if stripped and stripped[-1] in ".!?":
                    result_parts.append(stripped[:-1] + f" {marker}" + stripped[-1])
                else:
                    result_parts.append(stripped + f" {marker}")
            else:
                result_parts.append(part)

        return "".join(result_parts)

    def _format_citation_marker(self, citation_id: int) -> str:
        """Format citation marker based on style."""
        if self.style == CitationStyle.NUMERIC:
            return f"[{citation_id}]"
        elif self.style == CitationStyle.SUPERSCRIPT:
            superscripts = "⁰¹²³⁴⁵⁶⁷⁸⁹"
            return "".join(superscripts[int(d)] for d in str(citation_id))
        elif self.style == CitationStyle.FOOTNOTE:
            return f"†{citation_id}"
        else:
            return f"[{citation_id}]"

    def _generate_bibliography(
        self,
        sources: List[Document],
        cited_idxs: set,
    ) -> str:
        """Generate bibliography from cited sources."""
        lines = ["\n---\nReferences:\n"]

        for idx in sorted(cited_idxs):
            if idx < len(sources):
                source = sources[idx]
                title = source.metadata.get("title", f"Source {idx + 1}")
                url = source.metadata.get("url", "")

                line = f"[{idx + 1}] {title}"
                if url:
                    line += f" - {url}"
                lines.append(line)

        return "\n".join(lines)

    def _extract_citations_from_text(
        self,
        text: str,
        sources: List[Document],
    ) -> List[Citation]:
        """Extract citations from LLM-generated text."""
        citations = []

        # Find all citation markers
        pattern = r"\[(\d+)\]"
        matches = list(re.finditer(pattern, text))

        for match in matches:
            citation_id = int(match.group(1))
            if 1 <= citation_id <= len(sources):
                # Find the sentence containing this citation
                start = text.rfind(".", 0, match.start())
                start = start + 1 if start >= 0 else 0
                cited_text = text[start : match.end()].strip()

                source = sources[citation_id - 1]
                citation = Citation(
                    citation_id=citation_id,
                    source_id=source.id,
                    source_title=source.metadata.get("title", f"Source {citation_id}"),
                    cited_text=cited_text,
                    source_text=source.content[:200],
                    confidence=0.8,
                )
                citations.append(citation)

        return citations


# =============================================================================
# Citation Verifier
# =============================================================================


class CitationVerifier:
    """
    Verifies citations against source documents.

    Checks that citations accurately reflect the source content.
    """

    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize verifier."""
        self.llm = llm
        self.matcher = SourceMatcher()

    def verify(
        self,
        citation: Citation,
        source: Document,
    ) -> VerificationResult:
        """
        Verify a single citation.

        Args:
            citation: Citation to verify
            source: Source document

        Returns:
            VerificationResult
        """
        issues = []

        # Check text overlap
        matches = self.matcher.match_claim_to_sources(
            citation.cited_text,
            [source],
        )

        if not matches:
            return VerificationResult(
                is_valid=False,
                support_score=0.0,
                issues=["No matching content found in source"],
            )

        support_score = matches[0][1]

        if support_score < 0.3:
            issues.append("Low text overlap with source")

        # Check for contradictions (simple heuristic)
        if self._check_contradiction(citation.cited_text, source.content):
            issues.append("Potential contradiction with source")
            support_score *= 0.5

        return VerificationResult(
            is_valid=support_score >= 0.5 and len(issues) == 0,
            support_score=support_score,
            issues=issues,
        )

    def _check_contradiction(self, claim: str, source: str) -> bool:
        """Check for obvious contradictions."""
        claim_lower = claim.lower()
        source_lower = source.lower()

        # Check for negation patterns
        negation_words = ["not", "never", "no", "without"]

        for neg in negation_words:
            # Claim has negation, source doesn't (or vice versa)
            claim_has_neg = neg in claim_lower
            source_has_neg = neg in source_lower

            if claim_has_neg != source_has_neg:
                # Check if they're talking about same thing
                claim_keywords = set(re.findall(r"\b\w{4,}\b", claim_lower))
                source_keywords = set(re.findall(r"\b\w{4,}\b", source_lower))
                overlap = (
                    len(claim_keywords & source_keywords) / len(claim_keywords)
                    if claim_keywords
                    else 0
                )

                if overlap > 0.5:
                    return True

        return False


# =============================================================================
# Convenience Functions
# =============================================================================


def cite_response(
    response: str,
    sources: List[Document],
    style: CitationStyle = CitationStyle.NUMERIC,
) -> CitationResult:
    """
    Add citations to a response.

    Args:
        response: Generated response
        sources: Source documents
        style: Citation style

    Returns:
        CitationResult
    """
    generator = CitationGenerator(style=style)
    return generator.generate_citations(response, sources)


async def cite_with_llm(
    response: str,
    sources: List[Document],
    llm: BaseLLM,
) -> CitationResult:
    """
    Add citations using LLM.

    Args:
        response: Generated response
        sources: Source documents
        llm: Language model

    Returns:
        CitationResult
    """
    generator = CitationGenerator(llm=llm)
    return await generator.generate_with_llm(response, sources)
