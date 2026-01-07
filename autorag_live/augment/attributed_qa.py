"""
Attributed Question Answering Implementation.

Implements inline citation generation for RAG responses,
ensuring every factual claim is grounded in retrieved evidence.

Key features:
1. Inline citation generation
2. Claim extraction and verification
3. Citation styles (numeric, author-year, footnote)
4. Attribution scoring and confidence
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# ============================================================================
# Protocols
# ============================================================================


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM providers."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""
    title: str = ""
    url: str = ""
    author: str = ""
    date: str = ""


# ============================================================================
# Citation Types
# ============================================================================


class CitationStyle(Enum):
    """Citation formatting styles."""

    NUMERIC = "numeric"  # [1], [2], [3]
    AUTHOR_YEAR = "author_year"  # (Smith, 2023)
    SUPERSCRIPT = "superscript"  # ^1, ^2
    FOOTNOTE = "footnote"  # [^1], [^2]
    INLINE = "inline"  # According to [Source Title]...
    NONE = "none"  # No visible citations


@dataclass
class Citation:
    """A citation reference."""

    doc_id: str
    doc_index: int  # Index in source list
    text_span: tuple[int, int]  # Start, end in response
    claim: str
    confidence: float
    document: RetrievedDocument


@dataclass
class AttributedClaim:
    """A factual claim with attribution."""

    claim: str
    citations: list[Citation]
    support_level: str  # "strong", "weak", "unsupported"
    confidence: float


@dataclass
class AttributedResponse:
    """A response with inline citations."""

    raw_response: str
    formatted_response: str
    claims: list[AttributedClaim]
    citations: list[Citation]
    sources: list[RetrievedDocument]
    citation_style: CitationStyle
    overall_attribution_score: float


# ============================================================================
# Claim Extractors
# ============================================================================


class ClaimExtractor(ABC):
    """Abstract base for claim extraction."""

    @abstractmethod
    async def extract(self, text: str) -> list[str]:
        """Extract factual claims from text."""
        ...


class LLMClaimExtractor(ClaimExtractor):
    """LLM-based claim extraction."""

    EXTRACTION_PROMPT = """Extract all factual claims from this text. List each claim on a separate line.
A factual claim is a statement that can be verified as true or false.
Do not include opinions, questions, or hedged statements.

Text: {text}

Factual claims (one per line):"""

    def __init__(self, llm: LLMProtocol):
        """Initialize with LLM."""
        self.llm = llm

    async def extract(self, text: str) -> list[str]:
        """Extract factual claims using LLM."""
        prompt = self.EXTRACTION_PROMPT.format(text=text)
        response = await self.llm.generate(prompt, max_tokens=500, temperature=0.0)

        # Parse claims
        claims = []
        for line in response.split("\n"):
            line = line.strip()
            if line and not line.startswith("-"):
                # Remove numbering
                if line[0].isdigit():
                    line = line.split(".", 1)[-1].strip()
                    line = line.split(")", 1)[-1].strip()
                if line:
                    claims.append(line)

        return claims


class SentenceClaimExtractor(ClaimExtractor):
    """Simple sentence-based claim extraction."""

    async def extract(self, text: str) -> list[str]:
        """Extract sentences as claims."""
        import re

        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        claims = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
        return claims


# ============================================================================
# Attribution Verifier
# ============================================================================


class AttributionVerifier:
    """Verifies claims against source documents."""

    VERIFICATION_PROMPT = """Does this source document support the claim?

Claim: {claim}

Source document:
{source}

Answer with:
- SUPPORTED: The claim is directly stated or clearly implied
- PARTIALLY: Some aspects are supported but not all
- NOT_SUPPORTED: The source doesn't address this claim
- CONTRADICTED: The source contradicts this claim

Answer:"""

    def __init__(self, llm: LLMProtocol):
        """Initialize verifier."""
        self.llm = llm

    async def verify(
        self,
        claim: str,
        documents: list[RetrievedDocument],
    ) -> list[tuple[RetrievedDocument, str, float]]:
        """
        Verify claim against documents.

        Returns:
            List of (document, support_level, confidence) tuples
        """
        results = []

        for doc in documents:
            support, confidence = await self._verify_single(claim, doc)
            results.append((doc, support, confidence))

        return results

    async def _verify_single(
        self,
        claim: str,
        document: RetrievedDocument,
    ) -> tuple[str, float]:
        """Verify claim against single document."""
        prompt = self.VERIFICATION_PROMPT.format(
            claim=claim,
            source=document.content[:1000],
        )

        response = await self.llm.generate(prompt, max_tokens=50, temperature=0.0)
        response_upper = response.upper()

        if "SUPPORTED" in response_upper and "NOT" not in response_upper:
            return "supported", 0.9
        elif "PARTIALLY" in response_upper:
            return "partial", 0.6
        elif "CONTRADICTED" in response_upper:
            return "contradicted", 0.1
        else:
            return "not_supported", 0.3


# ============================================================================
# Citation Formatters
# ============================================================================


class CitationFormatter(ABC):
    """Abstract base for citation formatting."""

    @abstractmethod
    def format_inline(self, doc_index: int, document: RetrievedDocument) -> str:
        """Format inline citation marker."""
        ...

    @abstractmethod
    def format_reference(self, doc_index: int, document: RetrievedDocument) -> str:
        """Format reference list entry."""
        ...


class NumericCitationFormatter(CitationFormatter):
    """Numeric citation style [1], [2], etc."""

    def format_inline(self, doc_index: int, document: RetrievedDocument) -> str:
        """Format as [1], [2], etc."""
        return f"[{doc_index + 1}]"

    def format_reference(self, doc_index: int, document: RetrievedDocument) -> str:
        """Format reference entry."""
        ref = f"[{doc_index + 1}] "
        if document.title:
            ref += f'"{document.title}". '
        if document.author:
            ref += f"{document.author}. "
        if document.date:
            ref += f"({document.date}). "
        if document.url:
            ref += f"URL: {document.url}"
        return ref.strip()


class AuthorYearFormatter(CitationFormatter):
    """Author-year citation style (Smith, 2023)."""

    def format_inline(self, doc_index: int, document: RetrievedDocument) -> str:
        """Format as (Author, Year)."""
        author = document.author or f"Source {doc_index + 1}"
        year = document.date[:4] if document.date else "n.d."

        # Extract last name
        if "," in author:
            author = author.split(",")[0]
        elif " " in author:
            author = author.split()[-1]

        return f"({author}, {year})"

    def format_reference(self, doc_index: int, document: RetrievedDocument) -> str:
        """Format reference entry."""
        author = document.author or f"Source {doc_index + 1}"
        year = document.date[:4] if document.date else "n.d."

        ref = f"{author} ({year}). "
        if document.title:
            ref += f'"{document.title}". '
        if document.url:
            ref += f"Retrieved from {document.url}"
        return ref.strip()


class FootnoteCitationFormatter(CitationFormatter):
    """Footnote citation style [^1], [^2]."""

    def format_inline(self, doc_index: int, document: RetrievedDocument) -> str:
        """Format as [^1]."""
        return f"[^{doc_index + 1}]"

    def format_reference(self, doc_index: int, document: RetrievedDocument) -> str:
        """Format footnote entry."""
        ref = f"[^{doc_index + 1}]: "
        if document.title:
            ref += f"{document.title}"
        if document.url:
            ref += f" - {document.url}"
        return ref


class InlineCitationFormatter(CitationFormatter):
    """Inline citation style "According to [Source]..."."""

    def format_inline(self, doc_index: int, document: RetrievedDocument) -> str:
        """Format inline reference."""
        source = document.title or f"Source {doc_index + 1}"
        return f"[{source}]"

    def format_reference(self, doc_index: int, document: RetrievedDocument) -> str:
        """Format reference entry."""
        source = document.title or f"Source {doc_index + 1}"
        ref = f"- {source}"
        if document.url:
            ref += f": {document.url}"
        return ref


# ============================================================================
# Attributed QA Generator
# ============================================================================


@dataclass
class AttributedQAConfig:
    """Configuration for attributed QA."""

    citation_style: CitationStyle = CitationStyle.NUMERIC
    min_confidence: float = 0.5
    include_unsupported: bool = False
    add_reference_list: bool = True
    verify_all_claims: bool = True


class AttributedQAGenerator:
    """
    Generate answers with inline citations.

    Extracts claims, verifies against sources, and adds citations.
    """

    GENERATION_PROMPT = """Answer the question using the provided sources. Include citation markers after each factual claim.

Question: {question}

Sources:
{sources}

Use [1], [2], etc. to cite sources after each factual claim.
Answer:"""

    def __init__(
        self,
        llm: LLMProtocol,
        config: AttributedQAConfig | None = None,
    ):
        """
        Initialize attributed QA generator.

        Args:
            llm: LLM provider
            config: Configuration options
        """
        self.llm = llm
        self.config = config or AttributedQAConfig()

        # Initialize components
        self.claim_extractor = LLMClaimExtractor(llm)
        self.verifier = AttributionVerifier(llm)
        self.formatter = self._get_formatter()

    def _get_formatter(self) -> CitationFormatter:
        """Get citation formatter based on style."""
        formatters = {
            CitationStyle.NUMERIC: NumericCitationFormatter(),
            CitationStyle.AUTHOR_YEAR: AuthorYearFormatter(),
            CitationStyle.FOOTNOTE: FootnoteCitationFormatter(),
            CitationStyle.INLINE: InlineCitationFormatter(),
        }
        return formatters.get(self.config.citation_style, NumericCitationFormatter())

    async def generate(
        self,
        question: str,
        documents: list[RetrievedDocument],
    ) -> AttributedResponse:
        """
        Generate attributed response.

        Args:
            question: User question
            documents: Retrieved source documents

        Returns:
            AttributedResponse with citations
        """
        # Format sources for prompt
        sources_text = self._format_sources(documents)

        # Generate initial response
        prompt = self.GENERATION_PROMPT.format(
            question=question,
            sources=sources_text,
        )

        raw_response = await self.llm.generate(prompt, max_tokens=500, temperature=0.3)

        # Process response
        claims = await self._extract_and_verify_claims(raw_response, documents)
        citations = self._extract_citations(raw_response, documents)

        # Calculate attribution score
        attribution_score = self._calculate_attribution_score(claims)

        # Format response with citations
        formatted_response = self._format_response(raw_response, documents)

        return AttributedResponse(
            raw_response=raw_response,
            formatted_response=formatted_response,
            claims=claims,
            citations=citations,
            sources=documents,
            citation_style=self.config.citation_style,
            overall_attribution_score=attribution_score,
        )

    def _format_sources(self, documents: list[RetrievedDocument]) -> str:
        """Format documents for prompt."""
        formatted = []
        for i, doc in enumerate(documents):
            formatted.append(f"[{i + 1}] {doc.content[:500]}")
        return "\n\n".join(formatted)

    async def _extract_and_verify_claims(
        self,
        response: str,
        documents: list[RetrievedDocument],
    ) -> list[AttributedClaim]:
        """Extract claims and verify attribution."""
        # Extract claims
        claim_texts = await self.claim_extractor.extract(response)

        attributed_claims = []
        for claim_text in claim_texts:
            # Verify against sources
            if self.config.verify_all_claims:
                verifications = await self.verifier.verify(claim_text, documents)
            else:
                verifications = []

            # Find supporting documents
            citations = []
            best_support = "unsupported"
            best_confidence = 0.0

            for doc, support, confidence in verifications:
                if support == "supported" and confidence >= self.config.min_confidence:
                    doc_idx = documents.index(doc)
                    citations.append(
                        Citation(
                            doc_id=doc.doc_id or str(doc_idx),
                            doc_index=doc_idx,
                            text_span=(0, 0),
                            claim=claim_text,
                            confidence=confidence,
                            document=doc,
                        )
                    )
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_support = "strong"
                elif support == "partial":
                    best_support = "weak"
                    best_confidence = max(best_confidence, confidence)

            attributed_claims.append(
                AttributedClaim(
                    claim=claim_text,
                    citations=citations,
                    support_level=best_support,
                    confidence=best_confidence,
                )
            )

        return attributed_claims

    def _extract_citations(
        self,
        response: str,
        documents: list[RetrievedDocument],
    ) -> list[Citation]:
        """Extract existing citations from response."""
        import re

        citations = []

        # Find citation markers like [1], [2], etc.
        pattern = r"\[(\d+)\]"
        for match in re.finditer(pattern, response):
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(documents):
                citations.append(
                    Citation(
                        doc_id=documents[idx].doc_id or str(idx),
                        doc_index=idx,
                        text_span=(match.start(), match.end()),
                        claim="",
                        confidence=1.0,
                        document=documents[idx],
                    )
                )

        return citations

    def _calculate_attribution_score(
        self,
        claims: list[AttributedClaim],
    ) -> float:
        """Calculate overall attribution score."""
        if not claims:
            return 1.0

        total_score = 0.0
        for claim in claims:
            if claim.support_level == "strong":
                total_score += 1.0
            elif claim.support_level == "weak":
                total_score += 0.5
            # unsupported = 0

        return total_score / len(claims)

    def _format_response(
        self,
        response: str,
        documents: list[RetrievedDocument],
    ) -> str:
        """Format response with citation style and reference list."""
        formatted = response

        # Add reference list if configured
        if self.config.add_reference_list and documents:
            references = "\n\n**References:**\n"
            for i, doc in enumerate(documents):
                ref = self.formatter.format_reference(i, doc)
                references += f"{ref}\n"
            formatted += references

        return formatted


# ============================================================================
# Post-hoc Attribution
# ============================================================================


class PostHocAttributor:
    """
    Add citations to existing responses.

    Takes a response without citations and adds attribution.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        config: AttributedQAConfig | None = None,
    ):
        """Initialize post-hoc attributor."""
        self.llm = llm
        self.config = config or AttributedQAConfig()
        self.claim_extractor = LLMClaimExtractor(llm)
        self.verifier = AttributionVerifier(llm)
        self.formatter = NumericCitationFormatter()

    async def attribute(
        self,
        response: str,
        documents: list[RetrievedDocument],
    ) -> AttributedResponse:
        """
        Add citations to response.

        Args:
            response: Response without citations
            documents: Source documents

        Returns:
            AttributedResponse with added citations
        """
        # Extract claims
        claims = await self.claim_extractor.extract(response)

        # Verify each claim and find best supporting document
        attributed_claims = []
        citation_map: dict[str, int] = {}  # Claim text -> doc index

        for claim in claims:
            verifications = await self.verifier.verify(claim, documents)

            best_doc_idx = -1
            best_confidence = 0.0

            for i, (_, support, confidence) in enumerate(verifications):
                if support == "supported" and confidence > best_confidence:
                    best_confidence = confidence
                    best_doc_idx = i

            if best_doc_idx >= 0:
                citation_map[claim] = best_doc_idx
                attributed_claims.append(
                    AttributedClaim(
                        claim=claim,
                        citations=[
                            Citation(
                                doc_id=str(best_doc_idx),
                                doc_index=best_doc_idx,
                                text_span=(0, 0),
                                claim=claim,
                                confidence=best_confidence,
                                document=documents[best_doc_idx],
                            )
                        ],
                        support_level="strong",
                        confidence=best_confidence,
                    )
                )

        # Insert citations into response
        formatted = self._insert_citations(response, citation_map)

        # Add reference list
        if self.config.add_reference_list:
            formatted += self._format_references(documents, citation_map)

        attribution_score = len(citation_map) / max(len(claims), 1)

        return AttributedResponse(
            raw_response=response,
            formatted_response=formatted,
            claims=attributed_claims,
            citations=[c for ac in attributed_claims for c in ac.citations],
            sources=documents,
            citation_style=self.config.citation_style,
            overall_attribution_score=attribution_score,
        )

    def _insert_citations(
        self,
        response: str,
        citation_map: dict[str, int],
    ) -> str:
        """Insert citation markers into response."""
        formatted = response

        # Sort claims by length (longest first) to avoid partial replacements
        sorted_claims = sorted(citation_map.keys(), key=len, reverse=True)

        for claim in sorted_claims:
            doc_idx = citation_map[claim]
            citation = f" [{doc_idx + 1}]"

            # Find claim in response and add citation
            if claim in formatted:
                # Add citation at end of claim
                formatted = formatted.replace(claim, f"{claim}{citation}", 1)

        return formatted

    def _format_references(
        self,
        documents: list[RetrievedDocument],
        citation_map: dict[str, int],
    ) -> str:
        """Format reference list for cited documents."""
        cited_indices = set(citation_map.values())

        if not cited_indices:
            return ""

        refs = "\n\n**References:**\n"
        for idx in sorted(cited_indices):
            if idx < len(documents):
                ref = self.formatter.format_reference(idx, documents[idx])
                refs += f"{ref}\n"

        return refs


# ============================================================================
# Factory Functions
# ============================================================================


def create_attributed_qa(
    llm: LLMProtocol,
    citation_style: CitationStyle = CitationStyle.NUMERIC,
    verify_all: bool = True,
) -> AttributedQAGenerator:
    """
    Create attributed QA generator.

    Args:
        llm: LLM provider
        citation_style: Citation formatting style
        verify_all: Verify all claims

    Returns:
        Configured AttributedQAGenerator
    """
    config = AttributedQAConfig(
        citation_style=citation_style,
        verify_all_claims=verify_all,
    )
    return AttributedQAGenerator(llm, config)


def create_post_hoc_attributor(
    llm: LLMProtocol,
    citation_style: CitationStyle = CitationStyle.NUMERIC,
) -> PostHocAttributor:
    """
    Create post-hoc attributor.

    Args:
        llm: LLM provider
        citation_style: Citation formatting style

    Returns:
        Configured PostHocAttributor
    """
    config = AttributedQAConfig(citation_style=citation_style)
    return PostHocAttributor(llm, config)


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example demonstrating attributed QA."""
    print("Attributed QA Implementation Ready")
    print("=" * 50)
    print("\nFeatures:")
    print("- LLM-based claim extraction")
    print("- Claim verification against sources")
    print("- Multiple citation styles:")
    print("  - Numeric: [1], [2]")
    print("  - Author-year: (Smith, 2023)")
    print("  - Footnote: [^1]")
    print("  - Inline: [Source Title]")
    print("- Post-hoc attribution for existing responses")
    print("- Attribution scoring")
    print("- Reference list generation")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
