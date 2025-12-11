"""Source Attribution System for AutoRAG-Live.

Track and attribute sources in generated responses:
- Citation extraction and validation
- Source mapping and linking
- Attribution scoring
- Quote verification
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AttributionType(Enum):
    """Types of attribution."""
    
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    SUMMARY = "summary"
    INFERENCE = "inference"
    NONE = "none"


class CitationStyle(Enum):
    """Citation format styles."""
    
    INLINE_NUMBER = "inline_number"  # [1], [2]
    INLINE_AUTHOR = "inline_author"  # (Smith, 2023)
    SUPERSCRIPT = "superscript"  # ¹, ²
    FOOTNOTE = "footnote"
    HYPERLINK = "hyperlink"


@dataclass
class Source:
    """Represents a source document/passage."""
    
    id: str
    content: str
    title: str | None = None
    url: str | None = None
    author: str | None = None
    date: str | None = None
    page: int | None = None
    relevance_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if not self.id:
            self.id = hashlib.md5(self.content[:100].encode()).hexdigest()[:8]


@dataclass
class Attribution:
    """Attribution of a response segment to a source."""
    
    source_id: str
    attribution_type: AttributionType
    response_segment: str
    source_segment: str
    similarity_score: float
    start_pos: int
    end_pos: int
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """A citation in the response."""
    
    citation_id: str
    source_id: str
    citation_text: str
    position: int
    style: CitationStyle
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributionResult:
    """Complete attribution analysis result."""
    
    response: str
    sources: list[Source]
    attributions: list[Attribution]
    citations: list[Citation]
    overall_attribution_score: float
    attributed_ratio: float
    unattributed_segments: list[str]
    analysis_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TextMatcher:
    """Match text segments between response and sources."""
    
    def __init__(
        self,
        min_match_length: int = 5,
        similarity_threshold: float = 0.6,
    ) -> None:
        """Initialize text matcher.
        
        Args:
            min_match_length: Minimum word count for match
            similarity_threshold: Minimum similarity for match
        """
        self.min_match_length = min_match_length
        self.similarity_threshold = similarity_threshold
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def find_matches(
        self,
        response: str,
        source_content: str,
        window_size: int = 50,
        step_size: int = 10,
    ) -> list[tuple[str, str, float, int, int]]:
        """Find matching segments between response and source.
        
        Args:
            response: Response text
            source_content: Source content
            window_size: Word window size
            step_size: Window step size
            
        Returns:
            List of (response_segment, source_segment, similarity, start, end)
        """
        matches: list[tuple[str, str, float, int, int]] = []
        
        response_words = response.split()
        source_words = source_content.split()
        
        # Slide window over response
        for i in range(0, len(response_words) - window_size + 1, step_size):
            response_segment = " ".join(response_words[i:i + window_size])
            
            # Find best match in source
            best_match = None
            best_similarity = 0.0
            
            for j in range(0, len(source_words) - window_size + 1, step_size):
                source_segment = " ".join(source_words[j:j + window_size])
                similarity = self.compute_similarity(response_segment, source_segment)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = source_segment
            
            if best_match and best_similarity >= self.similarity_threshold:
                # Calculate character positions
                start_pos = len(" ".join(response_words[:i]))
                if i > 0:
                    start_pos += 1  # Space before segment
                end_pos = start_pos + len(response_segment)
                
                matches.append((
                    response_segment,
                    best_match,
                    best_similarity,
                    start_pos,
                    end_pos,
                ))
        
        # Merge overlapping matches
        return self._merge_overlapping_matches(matches)
    
    def _merge_overlapping_matches(
        self,
        matches: list[tuple[str, str, float, int, int]],
    ) -> list[tuple[str, str, float, int, int]]:
        """Merge overlapping match segments."""
        if not matches:
            return []
        
        # Sort by start position
        sorted_matches = sorted(matches, key=lambda x: x[3])
        
        merged: list[tuple[str, str, float, int, int]] = []
        current = sorted_matches[0]
        
        for next_match in sorted_matches[1:]:
            # Check for overlap
            if next_match[3] <= current[4]:
                # Merge: keep the one with higher similarity
                if next_match[2] > current[2]:
                    current = (
                        current[0] + " " + next_match[0].split()[-1],
                        current[1],
                        max(current[2], next_match[2]),
                        current[3],
                        next_match[4],
                    )
            else:
                merged.append(current)
                current = next_match
        
        merged.append(current)
        return merged
    
    def detect_direct_quote(
        self,
        response_segment: str,
        source_content: str,
    ) -> tuple[bool, str | None, float]:
        """Detect if segment is a direct quote.
        
        Args:
            response_segment: Response segment
            source_content: Source content
            
        Returns:
            (is_quote, matched_quote, similarity)
        """
        # Check for quote markers
        if response_segment.startswith('"') or response_segment.startswith("'"):
            clean_segment = response_segment.strip("\"'")
        else:
            clean_segment = response_segment
        
        # Find exact or near-exact match
        similarity = self.compute_similarity(clean_segment, source_content)
        
        if similarity >= 0.9:
            return True, clean_segment, similarity
        
        # Try to find the segment within source
        if clean_segment.lower() in source_content.lower():
            return True, clean_segment, 1.0
        
        return False, None, similarity


class CitationExtractor:
    """Extract and parse citations from text."""
    
    CITATION_PATTERNS = {
        CitationStyle.INLINE_NUMBER: r"\[(\d+)\]",
        CitationStyle.INLINE_AUTHOR: r"\(([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4})\)",
        CitationStyle.SUPERSCRIPT: r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+",
    }
    
    def extract_citations(
        self,
        text: str,
    ) -> list[tuple[str, int, CitationStyle]]:
        """Extract citations from text.
        
        Args:
            text: Text to extract citations from
            
        Returns:
            List of (citation_text, position, style)
        """
        citations: list[tuple[str, int, CitationStyle]] = []
        
        for style, pattern in self.CITATION_PATTERNS.items():
            for match in re.finditer(pattern, text):
                citations.append((
                    match.group(),
                    match.start(),
                    style,
                ))
        
        return sorted(citations, key=lambda x: x[1])
    
    def parse_citation(
        self,
        citation_text: str,
        style: CitationStyle,
    ) -> dict[str, Any]:
        """Parse citation into components.
        
        Args:
            citation_text: Citation text
            style: Citation style
            
        Returns:
            Parsed citation components
        """
        result: dict[str, Any] = {"raw": citation_text, "style": style.value}
        
        if style == CitationStyle.INLINE_NUMBER:
            match = re.search(r"\[(\d+)\]", citation_text)
            if match:
                result["number"] = int(match.group(1))
        
        elif style == CitationStyle.INLINE_AUTHOR:
            # Parse author and year
            match = re.search(r"\(([^,]+),?\s*(\d{4})\)", citation_text)
            if match:
                result["author"] = match.group(1).strip()
                result["year"] = int(match.group(2))
        
        return result


class CitationGenerator:
    """Generate citations for sources."""
    
    def __init__(self, style: CitationStyle = CitationStyle.INLINE_NUMBER) -> None:
        """Initialize citation generator.
        
        Args:
            style: Default citation style
        """
        self.style = style
        self._citation_counter = 0
        self._source_to_citation: dict[str, str] = {}
    
    def reset(self) -> None:
        """Reset citation counter and mappings."""
        self._citation_counter = 0
        self._source_to_citation.clear()
    
    def generate_citation(
        self,
        source: Source,
        style: CitationStyle | None = None,
    ) -> str:
        """Generate citation for a source.
        
        Args:
            source: Source to cite
            style: Citation style (uses default if None)
            
        Returns:
            Citation text
        """
        style = style or self.style
        
        # Check if already cited
        if source.id in self._source_to_citation:
            return self._source_to_citation[source.id]
        
        self._citation_counter += 1
        
        if style == CitationStyle.INLINE_NUMBER:
            citation = f"[{self._citation_counter}]"
        
        elif style == CitationStyle.INLINE_AUTHOR:
            author = source.author or "Unknown"
            year = source.date[:4] if source.date else "n.d."
            citation = f"({author}, {year})"
        
        elif style == CitationStyle.SUPERSCRIPT:
            superscripts = "⁰¹²³⁴⁵⁶⁷⁸⁹"
            citation = "".join(
                superscripts[int(d)] for d in str(self._citation_counter)
            )
        
        elif style == CitationStyle.HYPERLINK:
            url = source.url or "#"
            citation = f"[{self._citation_counter}]({url})"
        
        else:
            citation = f"[{self._citation_counter}]"
        
        self._source_to_citation[source.id] = citation
        return citation
    
    def generate_reference_list(
        self,
        sources: list[Source],
    ) -> str:
        """Generate reference list for sources.
        
        Args:
            sources: Sources to include
            
        Returns:
            Formatted reference list
        """
        lines: list[str] = ["## References", ""]
        
        for i, source in enumerate(sources, 1):
            author = source.author or "Unknown Author"
            title = source.title or "Untitled"
            date = source.date or "n.d."
            url = source.url or ""
            
            if url:
                line = f"{i}. {author} ({date}). *{title}*. Retrieved from {url}"
            else:
                line = f"{i}. {author} ({date}). *{title}*."
            
            lines.append(line)
        
        return "\n".join(lines)


class AttributionScorer:
    """Score attribution quality."""
    
    def __init__(
        self,
        quote_weight: float = 1.0,
        paraphrase_weight: float = 0.8,
        summary_weight: float = 0.6,
        inference_weight: float = 0.3,
    ) -> None:
        """Initialize attribution scorer.
        
        Args:
            quote_weight: Weight for direct quotes
            paraphrase_weight: Weight for paraphrases
            summary_weight: Weight for summaries
            inference_weight: Weight for inferences
        """
        self.weights = {
            AttributionType.DIRECT_QUOTE: quote_weight,
            AttributionType.PARAPHRASE: paraphrase_weight,
            AttributionType.SUMMARY: summary_weight,
            AttributionType.INFERENCE: inference_weight,
            AttributionType.NONE: 0.0,
        }
    
    def score_attribution(self, attribution: Attribution) -> float:
        """Score a single attribution.
        
        Args:
            attribution: Attribution to score
            
        Returns:
            Attribution score
        """
        type_weight = self.weights[attribution.attribution_type]
        return attribution.similarity_score * type_weight * attribution.confidence
    
    def score_response(
        self,
        response: str,
        attributions: list[Attribution],
    ) -> tuple[float, float]:
        """Score overall response attribution.
        
        Args:
            response: Response text
            attributions: List of attributions
            
        Returns:
            (overall_score, attributed_ratio)
        """
        if not attributions or not response:
            return 0.0, 0.0
        
        # Calculate attributed characters
        attributed_chars = 0
        for attr in attributions:
            segment_len = attr.end_pos - attr.start_pos
            attributed_chars += segment_len
        
        response_len = len(response)
        attributed_ratio = min(1.0, attributed_chars / response_len)
        
        # Calculate weighted score
        scores = [self.score_attribution(attr) for attr in attributions]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Overall score combines coverage and quality
        overall_score = (attributed_ratio * 0.5 + avg_score * 0.5)
        
        return overall_score, attributed_ratio


class SourceAttributionEngine:
    """Main engine for source attribution analysis."""
    
    def __init__(
        self,
        similarity_threshold: float = 0.6,
        min_segment_words: int = 5,
        citation_style: CitationStyle = CitationStyle.INLINE_NUMBER,
    ) -> None:
        """Initialize attribution engine.
        
        Args:
            similarity_threshold: Minimum similarity for attribution
            min_segment_words: Minimum words per segment
            citation_style: Default citation style
        """
        self.matcher = TextMatcher(
            min_match_length=min_segment_words,
            similarity_threshold=similarity_threshold,
        )
        self.citation_extractor = CitationExtractor()
        self.citation_generator = CitationGenerator(citation_style)
        self.scorer = AttributionScorer()
        
        # Statistics
        self._stats: dict[str, Any] = {
            "total_analyses": 0,
            "total_attributions": 0,
            "avg_attribution_score": 0.0,
        }
    
    def analyze(
        self,
        response: str,
        sources: list[Source],
    ) -> AttributionResult:
        """Analyze attribution in a response.
        
        Args:
            response: Generated response text
            sources: Source documents
            
        Returns:
            Complete attribution result
        """
        start_time = time.time()
        
        attributions: list[Attribution] = []
        attributed_positions: set[tuple[int, int]] = set()
        
        # Find attributions for each source
        for source in sources:
            matches = self.matcher.find_matches(
                response,
                source.content,
            )
            
            for resp_seg, src_seg, similarity, start, end in matches:
                # Determine attribution type
                is_quote, _, quote_sim = self.matcher.detect_direct_quote(
                    resp_seg, source.content
                )
                
                if is_quote:
                    attr_type = AttributionType.DIRECT_QUOTE
                    confidence = quote_sim
                elif similarity >= 0.8:
                    attr_type = AttributionType.PARAPHRASE
                    confidence = similarity
                elif similarity >= 0.6:
                    attr_type = AttributionType.SUMMARY
                    confidence = similarity * 0.9
                else:
                    attr_type = AttributionType.INFERENCE
                    confidence = similarity * 0.7
                
                attribution = Attribution(
                    source_id=source.id,
                    attribution_type=attr_type,
                    response_segment=resp_seg,
                    source_segment=src_seg,
                    similarity_score=similarity,
                    start_pos=start,
                    end_pos=end,
                    confidence=confidence,
                )
                
                attributions.append(attribution)
                attributed_positions.add((start, end))
        
        # Extract existing citations
        citation_tuples = self.citation_extractor.extract_citations(response)
        citations = [
            Citation(
                citation_id=f"c{i}",
                source_id="",  # Would need mapping
                citation_text=text,
                position=pos,
                style=style,
            )
            for i, (text, pos, style) in enumerate(citation_tuples)
        ]
        
        # Find unattributed segments
        unattributed = self._find_unattributed_segments(
            response, attributed_positions
        )
        
        # Score attribution
        overall_score, attributed_ratio = self.scorer.score_response(
            response, attributions
        )
        
        analysis_time_ms = (time.time() - start_time) * 1000
        
        # Update stats
        self._stats["total_analyses"] += 1
        self._stats["total_attributions"] += len(attributions)
        self._stats["avg_attribution_score"] = (
            (self._stats["avg_attribution_score"] * (self._stats["total_analyses"] - 1) +
             overall_score) / self._stats["total_analyses"]
        )
        
        return AttributionResult(
            response=response,
            sources=sources,
            attributions=attributions,
            citations=citations,
            overall_attribution_score=overall_score,
            attributed_ratio=attributed_ratio,
            unattributed_segments=unattributed,
            analysis_time_ms=analysis_time_ms,
        )
    
    def _find_unattributed_segments(
        self,
        response: str,
        attributed_positions: set[tuple[int, int]],
        min_length: int = 20,
    ) -> list[str]:
        """Find segments without attribution.
        
        Args:
            response: Response text
            attributed_positions: Set of (start, end) positions with attribution
            min_length: Minimum length for unattributed segment
            
        Returns:
            List of unattributed segments
        """
        unattributed: list[str] = []
        
        # Sort positions
        sorted_positions = sorted(attributed_positions)
        
        current_pos = 0
        for start, end in sorted_positions:
            if start > current_pos:
                segment = response[current_pos:start].strip()
                if len(segment) >= min_length:
                    unattributed.append(segment)
            current_pos = max(current_pos, end)
        
        # Check remainder
        if current_pos < len(response):
            segment = response[current_pos:].strip()
            if len(segment) >= min_length:
                unattributed.append(segment)
        
        return unattributed
    
    def add_citations(
        self,
        response: str,
        attributions: list[Attribution],
        sources: list[Source],
    ) -> str:
        """Add citations to response based on attributions.
        
        Args:
            response: Original response
            attributions: Attribution results
            sources: Source documents
            
        Returns:
            Response with citations added
        """
        self.citation_generator.reset()
        
        # Create source ID to source mapping
        source_map = {s.id: s for s in sources}
        
        # Sort attributions by position (reverse to add from end)
        sorted_attrs = sorted(
            attributions,
            key=lambda a: a.end_pos,
            reverse=True,
        )
        
        result = response
        
        for attr in sorted_attrs:
            source = source_map.get(attr.source_id)
            if source:
                citation = self.citation_generator.generate_citation(source)
                # Insert citation at end of attributed segment
                result = result[:attr.end_pos] + citation + result[attr.end_pos:]
        
        return result
    
    def generate_attribution_report(
        self,
        result: AttributionResult,
    ) -> str:
        """Generate human-readable attribution report.
        
        Args:
            result: Attribution result
            
        Returns:
            Formatted report
        """
        lines = [
            "# Attribution Report",
            "",
            f"**Overall Score:** {result.overall_attribution_score:.2%}",
            f"**Attributed Ratio:** {result.attributed_ratio:.2%}",
            f"**Total Attributions:** {len(result.attributions)}",
            f"**Analysis Time:** {result.analysis_time_ms:.2f}ms",
            "",
            "## Attributions by Type",
        ]
        
        # Group by type
        by_type: dict[AttributionType, list[Attribution]] = {}
        for attr in result.attributions:
            if attr.attribution_type not in by_type:
                by_type[attr.attribution_type] = []
            by_type[attr.attribution_type].append(attr)
        
        for attr_type, attrs in by_type.items():
            lines.append(f"\n### {attr_type.value.title()} ({len(attrs)})")
            for attr in attrs[:3]:  # Show top 3
                lines.append(f"- \"{attr.response_segment[:50]}...\" → Source {attr.source_id}")
        
        # Unattributed segments
        if result.unattributed_segments:
            lines.append("\n## Unattributed Segments")
            for seg in result.unattributed_segments[:3]:
                lines.append(f"- \"{seg[:50]}...\"")
        
        return "\n".join(lines)
    
    def get_stats(self) -> dict[str, Any]:
        """Get attribution statistics."""
        return self._stats.copy()


# Convenience function
def attribute_sources(
    response: str,
    sources: list[dict[str, Any]],
) -> AttributionResult:
    """Convenience function to attribute sources.
    
    Args:
        response: Generated response
        sources: List of source dicts with 'content' key
        
    Returns:
        Attribution result
    """
    engine = SourceAttributionEngine()
    source_objects = [
        Source(
            id=s.get("id", ""),
            content=s.get("content", ""),
            title=s.get("title"),
            url=s.get("url"),
            author=s.get("author"),
            date=s.get("date"),
        )
        for s in sources
    ]
    return engine.analyze(response, source_objects)
