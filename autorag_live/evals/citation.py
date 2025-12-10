"""
Citation Extraction and Verification for Agentic RAG.

Extracts citations from generated responses and verifies them against
source documents for grounding and factual accuracy.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from autorag_live.llm.providers import LLMProvider


class CitationStatus(str, Enum):
    """Status of citation verification."""

    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    NOT_FOUND = "not_found"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


class CitationStyle(str, Enum):
    """Citation formatting styles."""

    NUMERIC = "numeric"
    INLINE = "inline"
    FOOTNOTE = "footnote"
    AUTHOR_YEAR = "author_year"


@dataclass
class Citation:
    """A citation extracted from text."""

    text: str
    source_id: str
    source_title: str = ""
    page_or_section: str = ""
    confidence: float = 1.0
    start_pos: int = 0
    end_pos: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceDocument:
    """A source document for citation verification."""

    doc_id: str
    title: str
    content: str
    url: str = ""
    author: str = ""
    date: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_snippet(self, query: str, context_chars: int = 200) -> str:
        """Get a snippet of content around a query match."""
        query_lower = query.lower()
        content_lower = self.content.lower()

        pos = content_lower.find(query_lower)
        if pos == -1:
            return self.content[:context_chars] + "..."

        start = max(0, pos - context_chars // 2)
        end = min(len(self.content), pos + len(query) + context_chars // 2)

        snippet = self.content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(self.content):
            snippet = snippet + "..."

        return snippet


@dataclass
class VerificationResult:
    """Result of citation verification."""

    citation: Citation
    status: CitationStatus
    source_snippet: str = ""
    similarity_score: float = 0.0
    explanation: str = ""
    verified_at: datetime = field(default_factory=datetime.now)


@dataclass
class CitationReport:
    """Complete citation report for a response."""

    response_text: str
    citations: list[Citation]
    verification_results: list[VerificationResult]
    overall_grounding_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def num_verified(self) -> int:
        """Count verified citations."""
        return sum(1 for v in self.verification_results if v.status == CitationStatus.VERIFIED)

    @property
    def num_total(self) -> int:
        """Total number of citations."""
        return len(self.citations)

    @property
    def verification_rate(self) -> float:
        """Percentage of verified citations."""
        if not self.citations:
            return 1.0
        return self.num_verified / self.num_total


class CitationExtractor:
    """Extracts citations from generated text."""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """Initialize citation extractor."""
        self.llm_provider = llm_provider
        self._patterns = {
            CitationStyle.NUMERIC: r"\[(\d+)\]",
            CitationStyle.INLINE: r"\(([^)]+,\s*\d{4})\)",
            CitationStyle.FOOTNOTE: r"\[\^(\d+)\]",
            CitationStyle.AUTHOR_YEAR: r"([A-Z][a-z]+(?:\s+et\s+al\.?)?\s*\(\d{4}\))",
        }

    def extract(
        self,
        text: str,
        source_docs: list[SourceDocument],
        style: CitationStyle = CitationStyle.NUMERIC,
    ) -> list[Citation]:
        """Extract citations from text."""
        if self.llm_provider:
            return self._llm_extract(text, source_docs)
        return self._pattern_extract(text, source_docs, style)

    def _pattern_extract(
        self,
        text: str,
        source_docs: list[SourceDocument],
        style: CitationStyle,
    ) -> list[Citation]:
        """Extract citations using regex patterns."""
        pattern = self._patterns.get(style, self._patterns[CitationStyle.NUMERIC])
        citations = []

        doc_map = {str(i + 1): doc for i, doc in enumerate(source_docs)}

        for match in re.finditer(pattern, text):
            ref_id = match.group(1)
            doc = doc_map.get(ref_id)

            if doc:
                citation = Citation(
                    text=match.group(0),
                    source_id=doc.doc_id,
                    source_title=doc.title,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
                citations.append(citation)

        return citations

    def _llm_extract(self, text: str, source_docs: list[SourceDocument]) -> list[Citation]:
        """Extract citations using LLM."""
        source_list = "\n".join(
            [f"{i+1}. {doc.title} (ID: {doc.doc_id})" for i, doc in enumerate(source_docs)]
        )

        prompt = f"""Analyze this text and identify all claims that need citations.
For each claim, determine which source document supports it.

Text:
{text}

Available Sources:
{source_list}

For each citation needed, output in format:
CLAIM: [the claim text]
SOURCE_ID: [the source document ID]
CONFIDENCE: [0.0-1.0]

List all citations:"""

        response = self.llm_provider.generate(prompt)  # type: ignore
        response_text = response.content if hasattr(response, "content") else str(response)

        citations = []
        current_claim = None
        current_source = None
        current_confidence = 0.8

        for line in response_text.strip().split("\n"):
            line = line.strip()
            if line.startswith("CLAIM:"):
                current_claim = line[6:].strip()
            elif line.startswith("SOURCE_ID:"):
                current_source = line[10:].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    current_confidence = float(line[11:].strip())
                except ValueError:
                    current_confidence = 0.8

                if current_claim and current_source:
                    source_doc = next((d for d in source_docs if d.doc_id == current_source), None)
                    citations.append(
                        Citation(
                            text=current_claim,
                            source_id=current_source,
                            source_title=source_doc.title if source_doc else "",
                            confidence=current_confidence,
                        )
                    )
                    current_claim = None
                    current_source = None

        return citations


class CitationVerifier:
    """Verifies extracted citations against source documents."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        similarity_threshold: float = 0.7,
    ):
        """Initialize citation verifier."""
        self.llm_provider = llm_provider
        self.similarity_threshold = similarity_threshold

    def verify(self, citation: Citation, source_docs: list[SourceDocument]) -> VerificationResult:
        """Verify a single citation."""
        source_doc = next((d for d in source_docs if d.doc_id == citation.source_id), None)

        if not source_doc:
            return VerificationResult(
                citation=citation,
                status=CitationStatus.NOT_FOUND,
                explanation="Source document not found",
            )

        if self.llm_provider:
            return self._llm_verify(citation, source_doc)
        return self._text_verify(citation, source_doc)

    def _text_verify(self, citation: Citation, source_doc: SourceDocument) -> VerificationResult:
        """Verify citation using text matching."""
        claim_words = set(citation.text.lower().split())
        source_words = set(source_doc.content.lower().split())

        if not claim_words:
            return VerificationResult(
                citation=citation,
                status=CitationStatus.UNVERIFIABLE,
                explanation="Empty claim",
            )

        overlap = len(claim_words & source_words) / len(claim_words)

        snippet = source_doc.get_snippet(citation.text)

        if overlap >= self.similarity_threshold:
            return VerificationResult(
                citation=citation,
                status=CitationStatus.VERIFIED,
                source_snippet=snippet,
                similarity_score=overlap,
                explanation="High word overlap with source",
            )
        elif overlap >= self.similarity_threshold * 0.7:
            return VerificationResult(
                citation=citation,
                status=CitationStatus.PARTIALLY_VERIFIED,
                source_snippet=snippet,
                similarity_score=overlap,
                explanation="Partial word overlap with source",
            )
        else:
            return VerificationResult(
                citation=citation,
                status=CitationStatus.NOT_FOUND,
                source_snippet=snippet,
                similarity_score=overlap,
                explanation="Low word overlap with source",
            )

    def _llm_verify(self, citation: Citation, source_doc: SourceDocument) -> VerificationResult:
        """Verify citation using LLM."""
        prompt = f"""Verify if the following claim is supported by the source document.

Claim: {citation.text}

Source Document ({source_doc.title}):
{source_doc.content[:2000]}

Analyze the claim and respond with:
STATUS: [VERIFIED/PARTIALLY_VERIFIED/NOT_FOUND/CONTRADICTED]
SCORE: [0.0-1.0 confidence score]
EXPLANATION: [Brief explanation]
SNIPPET: [Relevant snippet from source if found]"""

        response = self.llm_provider.generate(prompt)  # type: ignore
        response_text = response.content if hasattr(response, "content") else str(response)

        status = CitationStatus.UNVERIFIABLE
        score = 0.0
        explanation = ""
        snippet = ""

        for line in response_text.strip().split("\n"):
            line = line.strip()
            if line.startswith("STATUS:"):
                status_str = line[7:].strip().upper()
                status_map = {
                    "VERIFIED": CitationStatus.VERIFIED,
                    "PARTIALLY_VERIFIED": CitationStatus.PARTIALLY_VERIFIED,
                    "NOT_FOUND": CitationStatus.NOT_FOUND,
                    "CONTRADICTED": CitationStatus.CONTRADICTED,
                }
                status = status_map.get(status_str, CitationStatus.UNVERIFIABLE)
            elif line.startswith("SCORE:"):
                try:
                    score = float(line[6:].strip())
                except ValueError:
                    score = 0.5
            elif line.startswith("EXPLANATION:"):
                explanation = line[12:].strip()
            elif line.startswith("SNIPPET:"):
                snippet = line[8:].strip()

        return VerificationResult(
            citation=citation,
            status=status,
            source_snippet=snippet,
            similarity_score=score,
            explanation=explanation,
        )

    def verify_all(
        self, citations: list[Citation], source_docs: list[SourceDocument]
    ) -> list[VerificationResult]:
        """Verify all citations."""
        return [self.verify(c, source_docs) for c in citations]


class CitationFormatter:
    """Formats citations in generated text."""

    def __init__(self, style: CitationStyle = CitationStyle.NUMERIC):
        """Initialize citation formatter."""
        self.style = style

    def add_citations(
        self,
        text: str,
        citations: list[Citation],
        source_docs: list[SourceDocument],
    ) -> str:
        """Add citations to text where appropriate."""
        doc_index = {doc.doc_id: i + 1 for i, doc in enumerate(source_docs)}

        sentences = re.split(r"(?<=[.!?])\s+", text)
        result_sentences = []

        for sentence in sentences:
            matching_citations = [
                c
                for c in citations
                if c.text.lower() in sentence.lower()
                or any(w in sentence.lower() for w in c.text.lower().split()[:3])
            ]

            if matching_citations:
                citation_refs = sorted(
                    set(doc_index.get(c.source_id, 0) for c in matching_citations)
                )
                if citation_refs and citation_refs[0] > 0:
                    refs_str = self._format_refs(citation_refs)
                    sentence = sentence.rstrip(".!?") + refs_str + sentence[-1]

            result_sentences.append(sentence)

        return " ".join(result_sentences)

    def _format_refs(self, refs: list[int]) -> str:
        """Format reference numbers based on style."""
        if self.style == CitationStyle.NUMERIC:
            return " " + ", ".join(f"[{r}]" for r in refs)
        elif self.style == CitationStyle.FOOTNOTE:
            return " " + ", ".join(f"[^{r}]" for r in refs)
        else:
            return " " + ", ".join(f"[{r}]" for r in refs)

    def generate_bibliography(
        self, source_docs: list[SourceDocument], used_ids: Optional[set[str]] = None
    ) -> str:
        """Generate bibliography from source documents."""
        if used_ids:
            docs = [d for d in source_docs if d.doc_id in used_ids]
        else:
            docs = source_docs

        lines = ["\n## References\n"]
        for i, doc in enumerate(docs, 1):
            line = f"[{i}] {doc.title}"
            if doc.author:
                line += f" - {doc.author}"
            if doc.date:
                line += f" ({doc.date})"
            if doc.url:
                line += f". {doc.url}"
            lines.append(line)

        return "\n".join(lines)


class CitationPipeline:
    """Complete pipeline for citation extraction, verification, and formatting."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        citation_style: CitationStyle = CitationStyle.NUMERIC,
        similarity_threshold: float = 0.7,
        auto_add_citations: bool = True,
    ):
        """Initialize citation pipeline."""
        self.extractor = CitationExtractor(llm_provider)
        self.verifier = CitationVerifier(llm_provider, similarity_threshold)
        self.formatter = CitationFormatter(citation_style)
        self.auto_add_citations = auto_add_citations
        self._reports: list[CitationReport] = []

    def process(
        self,
        response_text: str,
        source_docs: list[SourceDocument],
        add_bibliography: bool = True,
    ) -> tuple[str, CitationReport]:
        """Process response text with full citation pipeline."""
        citations = self.extractor.extract(response_text, source_docs)

        verification_results = self.verifier.verify_all(citations, source_docs)

        verified_count = sum(1 for v in verification_results if v.status == CitationStatus.VERIFIED)
        partially_count = sum(
            1 for v in verification_results if v.status == CitationStatus.PARTIALLY_VERIFIED
        )
        total = len(verification_results) if verification_results else 1

        grounding_score = (verified_count + 0.5 * partially_count) / total

        report = CitationReport(
            response_text=response_text,
            citations=citations,
            verification_results=verification_results,
            overall_grounding_score=grounding_score,
        )
        self._reports.append(report)

        final_text = response_text
        if self.auto_add_citations and citations:
            final_text = self.formatter.add_citations(response_text, citations, source_docs)

        if add_bibliography:
            used_ids = {c.source_id for c in citations}
            bibliography = self.formatter.generate_bibliography(source_docs, used_ids)
            final_text += bibliography

        return final_text, report

    def get_grounding_summary(self) -> dict[str, Any]:
        """Get summary of all processed citations."""
        if not self._reports:
            return {"total_responses": 0, "average_grounding_score": 0.0}

        avg_score = sum(r.overall_grounding_score for r in self._reports) / len(self._reports)
        total_citations = sum(len(r.citations) for r in self._reports)
        total_verified = sum(r.num_verified for r in self._reports)

        return {
            "total_responses": len(self._reports),
            "total_citations": total_citations,
            "total_verified": total_verified,
            "average_grounding_score": avg_score,
            "verification_rate": total_verified / max(total_citations, 1),
        }

    @property
    def reports(self) -> list[CitationReport]:
        """Get all citation reports."""
        return self._reports.copy()


__all__ = [
    "CitationStatus",
    "CitationStyle",
    "Citation",
    "SourceDocument",
    "VerificationResult",
    "CitationReport",
    "CitationExtractor",
    "CitationVerifier",
    "CitationFormatter",
    "CitationPipeline",
]
