"""
Grounding and Hallucination Mitigation Module.

Implements FActScore and RARR-inspired patterns for verifying LLM outputs
against source documents to detect and mitigate hallucinations.

Key Features:
1. Claim decomposition from LLM responses
2. Evidence-based claim verification
3. Grounding score computation
4. Hallucination detection and flagging
5. Source attribution validation

References:
- FActScore: Min et al., 2023 "FActScore: Fine-grained Atomic Evaluation"
- RARR: Gao et al., 2022 "RARR: Researching and Revising"

Example:
    >>> grounding = GroundingVerifier(llm, retriever)
    >>> result = await grounding.verify_response(response, sources)
    >>> print(f"Grounding score: {result.grounding_score}")
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols and Interfaces
# =============================================================================


class LLMProtocol(Protocol):
    """Protocol for LLM interface."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response from prompt."""
        ...


class RetrieverProtocol(Protocol):
    """Protocol for retriever interface."""

    async def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant documents."""
        ...


# =============================================================================
# Data Structures
# =============================================================================


class VerificationStatus(str, Enum):
    """Status of claim verification."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_FOUND = "not_found"
    PARTIALLY_SUPPORTED = "partially_supported"


@dataclass
class AtomicClaim:
    """
    An atomic claim extracted from a response.

    Based on FActScore's atomic fact decomposition.

    Attributes:
        text: The claim text
        source_sentence: Original sentence containing claim
        claim_type: Type of claim (factual, subjective, etc.)
        confidence: Extraction confidence
    """

    text: str
    source_sentence: str
    claim_type: str = "factual"
    confidence: float = 1.0

    def __hash__(self) -> int:
        return hash(self.text)


@dataclass
class ClaimVerification:
    """
    Result of verifying a single claim.

    Attributes:
        claim: The atomic claim being verified
        status: Verification status
        evidence: Supporting or contradicting evidence
        confidence: Verification confidence
        source_ids: IDs of source documents used
    """

    claim: AtomicClaim
    status: VerificationStatus
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source_ids: List[str] = field(default_factory=list)

    @property
    def is_supported(self) -> bool:
        """Check if claim is supported."""
        return self.status in (
            VerificationStatus.SUPPORTED,
            VerificationStatus.PARTIALLY_SUPPORTED,
        )

    @property
    def is_hallucination(self) -> bool:
        """Check if claim is likely a hallucination."""
        return self.status in (
            VerificationStatus.CONTRADICTED,
            VerificationStatus.NOT_FOUND,
        )


@dataclass
class GroundingResult:
    """
    Result of grounding verification for a response.

    Attributes:
        response: Original response text
        claims: Extracted atomic claims
        verifications: Verification results per claim
        grounding_score: Overall grounding score (0-1)
        hallucination_rate: Rate of unsupported claims
        supported_claims: Claims supported by evidence
        unsupported_claims: Claims not supported
        contradicted_claims: Claims contradicted by evidence
    """

    response: str
    claims: List[AtomicClaim]
    verifications: List[ClaimVerification]
    grounding_score: float = 0.0
    hallucination_rate: float = 0.0
    supported_claims: List[AtomicClaim] = field(default_factory=list)
    unsupported_claims: List[AtomicClaim] = field(default_factory=list)
    contradicted_claims: List[AtomicClaim] = field(default_factory=list)

    def __post_init__(self):
        """Compute derived fields."""
        if self.verifications:
            self.supported_claims = [v.claim for v in self.verifications if v.is_supported]
            self.unsupported_claims = [
                v.claim for v in self.verifications if v.status == VerificationStatus.NOT_FOUND
            ]
            self.contradicted_claims = [
                v.claim for v in self.verifications if v.status == VerificationStatus.CONTRADICTED
            ]

            # Compute scores
            if len(self.verifications) > 0:
                supported = sum(1 for v in self.verifications if v.is_supported)
                self.grounding_score = supported / len(self.verifications)
                self.hallucination_rate = 1 - self.grounding_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "grounding_score": self.grounding_score,
            "hallucination_rate": self.hallucination_rate,
            "total_claims": len(self.claims),
            "supported_claims": len(self.supported_claims),
            "unsupported_claims": len(self.unsupported_claims),
            "contradicted_claims": len(self.contradicted_claims),
        }


# =============================================================================
# Claim Extraction
# =============================================================================


class ClaimExtractor(ABC):
    """Abstract base class for claim extraction."""

    @abstractmethod
    async def extract_claims(self, text: str) -> List[AtomicClaim]:
        """Extract atomic claims from text."""
        pass


class LLMClaimExtractor(ClaimExtractor):
    """
    LLM-based claim extractor.

    Uses an LLM to decompose text into atomic claims.
    Based on FActScore's decomposition approach.
    """

    EXTRACTION_PROMPT = """Break down the following text into atomic, self-contained factual claims.
Each claim should be:
1. A single, verifiable statement
2. Self-contained (understandable without context)
3. Factual (not opinion or speculation)

Text:
{text}

Output each claim on a new line, starting with a dash (-).
Only include factual claims, not opinions or questions.

Claims:"""

    def __init__(self, llm: Optional[LLMProtocol] = None):
        """
        Initialize extractor.

        Args:
            llm: LLM for claim extraction
        """
        self.llm = llm

    async def extract_claims(self, text: str) -> List[AtomicClaim]:
        """
        Extract atomic claims from text.

        Args:
            text: Text to decompose

        Returns:
            List of atomic claims
        """
        if self.llm is None:
            # Fall back to heuristic extraction
            return self._heuristic_extract(text)

        prompt = self.EXTRACTION_PROMPT.format(text=text)

        try:
            response = await self.llm.generate(prompt)
            return self._parse_claims(response, text)
        except Exception as e:
            logger.warning(f"LLM claim extraction failed: {e}")
            return self._heuristic_extract(text)

    def _parse_claims(self, response: str, source_text: str) -> List[AtomicClaim]:
        """Parse LLM response into claims."""
        claims = []
        lines = response.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("-"):
                claim_text = line[1:].strip()
                if claim_text:
                    claims.append(
                        AtomicClaim(
                            text=claim_text,
                            source_sentence=source_text,
                            claim_type="factual",
                        )
                    )

        return claims

    def _heuristic_extract(self, text: str) -> List[AtomicClaim]:
        """
        Heuristic claim extraction.

        Splits text into sentences and treats each as a claim.
        """
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Skip very short fragments
                claims.append(
                    AtomicClaim(
                        text=sentence,
                        source_sentence=sentence,
                        claim_type="factual",
                    )
                )

        return claims


class RuleBasedExtractor(ClaimExtractor):
    """
    Rule-based claim extractor.

    Uses NLP patterns to extract claims without LLM.
    """

    # Patterns that indicate factual claims
    FACTUAL_PATTERNS = [
        r"is\s+(?:a|an|the)\s+",
        r"are\s+(?:a|an|the)?\s*",
        r"was\s+(?:a|an|the)\s+",
        r"were\s+(?:a|an|the)?\s*",
        r"has\s+",
        r"have\s+",
        r"contains?\s+",
        r"consists?\s+of",
        r"means?\s+",
        r"equals?\s+",
        r"\d+\s*%",
        r"\$[\d,]+",
        r"in\s+\d{4}",
    ]

    def __init__(self):
        """Initialize extractor."""
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.FACTUAL_PATTERNS]

    async def extract_claims(self, text: str) -> List[AtomicClaim]:
        """Extract claims using rules."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if self._is_factual(sentence):
                claims.append(
                    AtomicClaim(
                        text=sentence,
                        source_sentence=sentence,
                        claim_type="factual",
                        confidence=self._compute_confidence(sentence),
                    )
                )

        return claims

    def _is_factual(self, sentence: str) -> bool:
        """Check if sentence contains factual patterns."""
        return any(p.search(sentence) for p in self.patterns)

    def _compute_confidence(self, sentence: str) -> float:
        """Compute extraction confidence."""
        matches = sum(1 for p in self.patterns if p.search(sentence))
        return min(0.5 + matches * 0.1, 1.0)


# =============================================================================
# Claim Verification
# =============================================================================


class ClaimVerifier(ABC):
    """Abstract base class for claim verification."""

    @abstractmethod
    async def verify_claim(
        self,
        claim: AtomicClaim,
        sources: List[str],
    ) -> ClaimVerification:
        """Verify a single claim against sources."""
        pass


class LLMClaimVerifier(ClaimVerifier):
    """
    LLM-based claim verifier.

    Uses an LLM to check if claims are supported by evidence.
    """

    VERIFICATION_PROMPT = """Given the claim and evidence below, determine if the claim is:
1. SUPPORTED - The evidence directly supports the claim
2. CONTRADICTED - The evidence contradicts the claim
3. NOT_FOUND - The evidence neither supports nor contradicts

Claim: {claim}

Evidence:
{evidence}

Analysis:
1. What does the evidence say about the claim?
2. Is there direct support, contradiction, or neither?

Verdict (one of: SUPPORTED, CONTRADICTED, NOT_FOUND):"""

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize verifier.

        Args:
            llm: LLM for verification
            confidence_threshold: Threshold for confident verification
        """
        self.llm = llm
        self.confidence_threshold = confidence_threshold

    async def verify_claim(
        self,
        claim: AtomicClaim,
        sources: List[str],
    ) -> ClaimVerification:
        """
        Verify claim against sources.

        Args:
            claim: Claim to verify
            sources: Source documents

        Returns:
            ClaimVerification result
        """
        if not sources:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.NOT_FOUND,
                confidence=1.0,
            )

        if self.llm is None:
            return self._heuristic_verify(claim, sources)

        evidence = "\n".join(f"- {s[:500]}" for s in sources[:5])
        prompt = self.VERIFICATION_PROMPT.format(
            claim=claim.text,
            evidence=evidence,
        )

        try:
            response = await self.llm.generate(prompt)
            return self._parse_verification(claim, response, sources)
        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return self._heuristic_verify(claim, sources)

    def _parse_verification(
        self,
        claim: AtomicClaim,
        response: str,
        sources: List[str],
    ) -> ClaimVerification:
        """Parse LLM verification response."""
        response_lower = response.lower()

        if (
            "supported" in response_lower
            and "not" not in response_lower.split("supported")[0][-10:]
        ):
            status = VerificationStatus.SUPPORTED
            confidence = 0.9
        elif "contradicted" in response_lower:
            status = VerificationStatus.CONTRADICTED
            confidence = 0.85
        elif "not_found" in response_lower or "not found" in response_lower:
            status = VerificationStatus.NOT_FOUND
            confidence = 0.8
        else:
            # Default to not found if unclear
            status = VerificationStatus.NOT_FOUND
            confidence = 0.5

        return ClaimVerification(
            claim=claim,
            status=status,
            evidence=sources[:3],
            confidence=confidence,
        )

    def _heuristic_verify(
        self,
        claim: AtomicClaim,
        sources: List[str],
    ) -> ClaimVerification:
        """Heuristic verification using text matching."""
        claim_words = set(claim.text.lower().split())
        best_overlap = 0.0
        best_source = None

        for source in sources:
            source_words = set(source.lower().split())
            if len(claim_words) > 0:
                overlap = len(claim_words & source_words) / len(claim_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_source = source

        if best_overlap > 0.6:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.SUPPORTED,
                evidence=[best_source] if best_source else [],
                confidence=min(best_overlap, 0.9),
            )
        elif best_overlap > 0.3:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.PARTIALLY_SUPPORTED,
                evidence=[best_source] if best_source else [],
                confidence=best_overlap,
            )
        else:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.NOT_FOUND,
                evidence=[],
                confidence=0.7,
            )


class NLIClaimVerifier(ClaimVerifier):
    """
    NLI-based claim verifier.

    Uses Natural Language Inference to verify claims.
    """

    def __init__(
        self,
        nli_model: Optional[Callable[[str, str], Tuple[str, float]]] = None,
    ):
        """
        Initialize NLI verifier.

        Args:
            nli_model: Function (premise, hypothesis) -> (label, score)
        """
        self.nli_model = nli_model

    async def verify_claim(
        self,
        claim: AtomicClaim,
        sources: List[str],
    ) -> ClaimVerification:
        """Verify claim using NLI."""
        if self.nli_model is None or not sources:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.NOT_FOUND,
                confidence=0.5,
            )

        best_status = VerificationStatus.NOT_FOUND
        best_confidence = 0.0
        best_evidence = []

        for source in sources:
            try:
                label, score = self.nli_model(source, claim.text)

                if label == "entailment" and score > best_confidence:
                    best_status = VerificationStatus.SUPPORTED
                    best_confidence = score
                    best_evidence = [source]
                elif label == "contradiction" and score > best_confidence:
                    best_status = VerificationStatus.CONTRADICTED
                    best_confidence = score
                    best_evidence = [source]
            except Exception as e:
                logger.warning(f"NLI inference failed: {e}")
                continue

        return ClaimVerification(
            claim=claim,
            status=best_status,
            evidence=best_evidence,
            confidence=best_confidence,
        )


# =============================================================================
# Grounding Verifier (Main Interface)
# =============================================================================


class GroundingVerifier:
    """
    Main grounding verification interface.

    Combines claim extraction and verification to assess response grounding.

    Example:
        >>> grounding = GroundingVerifier(llm=my_llm)
        >>> result = await grounding.verify_response(response, sources)
        >>> if result.grounding_score < 0.7:
        ...     print("Warning: Low grounding score")
    """

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        retriever: Optional[RetrieverProtocol] = None,
        extractor: Optional[ClaimExtractor] = None,
        verifier: Optional[ClaimVerifier] = None,
        min_grounding_score: float = 0.7,
    ):
        """
        Initialize grounding verifier.

        Args:
            llm: LLM for extraction/verification
            retriever: Retriever for additional evidence
            extractor: Custom claim extractor
            verifier: Custom claim verifier
            min_grounding_score: Minimum acceptable grounding score
        """
        self.llm = llm
        self.retriever = retriever
        self.extractor = extractor or LLMClaimExtractor(llm)
        self.verifier = verifier or LLMClaimVerifier(llm)
        self.min_grounding_score = min_grounding_score

    async def verify_response(
        self,
        response: str,
        sources: List[str],
        *,
        retrieve_additional: bool = False,
    ) -> GroundingResult:
        """
        Verify response grounding.

        Args:
            response: LLM response to verify
            sources: Source documents
            retrieve_additional: Whether to retrieve additional evidence

        Returns:
            GroundingResult with verification details
        """
        # Extract claims
        claims = await self.extractor.extract_claims(response)

        if not claims:
            return GroundingResult(
                response=response,
                claims=[],
                verifications=[],
                grounding_score=1.0,  # No claims = fully grounded
            )

        # Optionally retrieve additional evidence
        all_sources = list(sources)
        if retrieve_additional and self.retriever:
            for claim in claims[:3]:  # Limit to avoid too many calls
                try:
                    additional = await self.retriever.retrieve(claim.text, k=3)
                    all_sources.extend(additional)
                except Exception as e:
                    logger.warning(f"Additional retrieval failed: {e}")

        # Verify claims in parallel
        verifications = await asyncio.gather(
            *[self.verifier.verify_claim(claim, all_sources) for claim in claims]
        )

        return GroundingResult(
            response=response,
            claims=claims,
            verifications=list(verifications),
        )

    async def is_grounded(
        self,
        response: str,
        sources: List[str],
    ) -> bool:
        """
        Check if response meets minimum grounding threshold.

        Args:
            response: Response to check
            sources: Source documents

        Returns:
            True if grounding score >= threshold
        """
        result = await self.verify_response(response, sources)
        return result.grounding_score >= self.min_grounding_score

    async def get_hallucinations(
        self,
        response: str,
        sources: List[str],
    ) -> List[AtomicClaim]:
        """
        Get list of likely hallucinated claims.

        Args:
            response: Response to check
            sources: Source documents

        Returns:
            List of unsupported/contradicted claims
        """
        result = await self.verify_response(response, sources)
        return result.unsupported_claims + result.contradicted_claims


# =============================================================================
# RARR-Style Response Revision
# =============================================================================


class ResponseReviser:
    """
    RARR-style response revision.

    Revises responses to remove or fix hallucinated content.

    Example:
        >>> reviser = ResponseReviser(llm, grounding)
        >>> revised = await reviser.revise(response, sources)
    """

    REVISION_PROMPT = """The following response contains some unsupported claims.
Please revise the response to remove or correct the unsupported parts while
keeping the supported content intact.

Original Response:
{response}

Unsupported Claims:
{unsupported}

Supported Claims:
{supported}

Please provide a revised response that:
1. Removes unsupported claims
2. Keeps all supported content
3. Maintains readability and coherence

Revised Response:"""

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        grounding: Optional[GroundingVerifier] = None,
    ):
        """
        Initialize reviser.

        Args:
            llm: LLM for revision
            grounding: Grounding verifier
        """
        self.llm = llm
        self.grounding = grounding or GroundingVerifier(llm)

    async def revise(
        self,
        response: str,
        sources: List[str],
        *,
        max_iterations: int = 2,
    ) -> str:
        """
        Revise response to fix hallucinations.

        Args:
            response: Original response
            sources: Source documents
            max_iterations: Maximum revision iterations

        Returns:
            Revised response
        """
        if self.llm is None:
            return response

        current = response

        for _ in range(max_iterations):
            result = await self.grounding.verify_response(current, sources)

            if result.grounding_score >= 0.9:
                break

            if not result.unsupported_claims:
                break

            # Revise
            unsupported = "\n".join(f"- {c.text}" for c in result.unsupported_claims)
            supported = "\n".join(f"- {c.text}" for c in result.supported_claims)

            prompt = self.REVISION_PROMPT.format(
                response=current,
                unsupported=unsupported or "None",
                supported=supported or "None",
            )

            try:
                current = await self.llm.generate(prompt)
            except Exception as e:
                logger.warning(f"Revision failed: {e}")
                break

        return current

    async def revise_with_citations(
        self,
        response: str,
        sources: List[str],
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Revise response and add source citations.

        Args:
            response: Original response
            sources: Source documents

        Returns:
            Tuple of (revised response, citations)
        """
        revised = await self.revise(response, sources)
        result = await self.grounding.verify_response(revised, sources)

        citations = []
        for v in result.verifications:
            if v.is_supported and v.evidence:
                citations.append(
                    {
                        "claim": v.claim.text,
                        "evidence": v.evidence[0][:200],
                    }
                )

        return revised, citations


# =============================================================================
# Convenience Functions
# =============================================================================


def create_grounding_verifier(
    llm: Optional[LLMProtocol] = None,
    min_score: float = 0.7,
) -> GroundingVerifier:
    """
    Create a grounding verifier.

    Args:
        llm: LLM for verification
        min_score: Minimum grounding score

    Returns:
        GroundingVerifier instance
    """
    return GroundingVerifier(llm=llm, min_grounding_score=min_score)


async def verify_and_revise(
    response: str,
    sources: List[str],
    llm: Optional[LLMProtocol] = None,
) -> Tuple[str, GroundingResult]:
    """
    Verify response and revise if needed.

    Args:
        response: Response to verify
        sources: Source documents
        llm: LLM for verification/revision

    Returns:
        Tuple of (revised response, grounding result)
    """
    grounding = GroundingVerifier(llm=llm)
    result = await grounding.verify_response(response, sources)

    if result.grounding_score < 0.7:
        reviser = ResponseReviser(llm, grounding)
        revised = await reviser.revise(response, sources)
        return revised, result

    return response, result


# =============================================================================
# OPTIMIZATION 4: Self-RAG Verification Token Generation
# Based on: "Self-RAG: Learning to Retrieve, Generate, and Critique
# through Self-Reflection" (Asai et al., 2023)
#
# Implements special verification tokens that the LLM generates during
# response generation to self-assess retrieval necessity, relevance,
# and factual support.
# =============================================================================


class SelfRAGToken(str, Enum):
    """
    Special tokens for Self-RAG verification.

    These tokens are generated inline during LLM response to signal
    self-assessment of retrieval needs and factual grounding.
    """

    # Retrieval necessity tokens
    RETRIEVE_YES = "[Retrieve]"
    RETRIEVE_NO = "[No Retrieve]"

    # Document relevance tokens
    RELEVANT = "[Relevant]"
    IRRELEVANT = "[Irrelevant]"
    PARTIALLY_RELEVANT = "[Partially Relevant]"

    # Support level tokens
    FULLY_SUPPORTED = "[Fully Supported]"
    PARTIALLY_SUPPORTED = "[Partially Supported]"
    NO_SUPPORT = "[No Support]"

    # Utility tokens
    USEFUL = "[Useful]"
    NOT_USEFUL = "[Not Useful]"

    # Critique tokens
    IS_RELEVANT = "[IsRel]"
    IS_SUPPORTED = "[IsSup]"
    IS_USEFUL = "[IsUse]"


@dataclass
class SelfRAGAssessment:
    """Assessment result from Self-RAG verification tokens."""

    needs_retrieval: Optional[bool] = None
    relevance_score: float = 0.0
    support_score: float = 0.0
    utility_score: float = 0.0
    tokens_found: List[str] = field(default_factory=list)
    segment_text: str = ""

    @property
    def overall_score(self) -> float:
        """Compute overall quality score."""
        weights = {"relevance": 0.3, "support": 0.5, "utility": 0.2}
        return (
            weights["relevance"] * self.relevance_score
            + weights["support"] * self.support_score
            + weights["utility"] * self.utility_score
        )


@dataclass
class SelfRAGResult:
    """Complete result from Self-RAG verification."""

    response: str
    segments: List[str] = field(default_factory=list)
    assessments: List[SelfRAGAssessment] = field(default_factory=list)
    overall_support: float = 0.0
    retrieval_decisions: List[bool] = field(default_factory=list)
    revised_response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SelfRAGVerifier:
    """
    Self-RAG verification with inline reflection tokens.

    Implements the Self-RAG paradigm where the model learns to:
    1. Decide when retrieval is needed ([Retrieve]/[No Retrieve])
    2. Assess document relevance ([Relevant]/[Irrelevant])
    3. Evaluate factual support ([Fully Supported]/[No Support])
    4. Judge overall utility ([Useful]/[Not Useful])

    Example:
        >>> verifier = SelfRAGVerifier(llm, retriever)
        >>> result = await verifier.generate_with_verification(
        ...     query="What is quantum computing?",
        ...     passages=retrieved_docs
        ... )
        >>> print(f"Support score: {result.overall_support}")
    """

    # Token patterns for extraction
    TOKEN_PATTERNS = {
        "retrieve": re.compile(r"\[(Retrieve|No Retrieve)\]", re.I),
        "relevance": re.compile(r"\[(Relevant|Irrelevant|Partially Relevant)\]", re.I),
        "support": re.compile(r"\[(Fully Supported|Partially Supported|No Support)\]", re.I),
        "utility": re.compile(r"\[(Useful|Not Useful)\]", re.I),
    }

    # Token to score mapping
    TOKEN_SCORES = {
        "relevant": 1.0,
        "partially relevant": 0.5,
        "irrelevant": 0.0,
        "fully supported": 1.0,
        "partially supported": 0.5,
        "no support": 0.0,
        "useful": 1.0,
        "not useful": 0.0,
    }

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        retriever: Optional[RetrieverProtocol] = None,
        support_threshold: float = 0.5,
        auto_retrieve_threshold: float = 0.3,
        max_retrievals_per_response: int = 3,
    ):
        """
        Initialize Self-RAG verifier.

        Args:
            llm: Language model for generation/verification
            retriever: Document retriever for on-demand retrieval
            support_threshold: Minimum support score to accept segment
            auto_retrieve_threshold: Score below which to auto-retrieve
            max_retrievals_per_response: Maximum retrieval operations
        """
        self.llm = llm
        self.retriever = retriever
        self.support_threshold = support_threshold
        self.auto_retrieve_threshold = auto_retrieve_threshold
        self.max_retrievals = max_retrievals_per_response

        # Statistics
        self._stats = {
            "generations": 0,
            "retrievals_triggered": 0,
            "segments_revised": 0,
            "avg_support_score": 0.0,
        }

    async def generate_with_verification(
        self,
        query: str,
        passages: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> SelfRAGResult:
        """
        Generate response with Self-RAG verification tokens.

        Args:
            query: User query
            passages: Pre-retrieved passages
            context: Additional context

        Returns:
            SelfRAGResult with verification assessment
        """
        if not self.llm:
            return SelfRAGResult(response="")

        # Build prompt with Self-RAG instructions
        prompt = self._build_selfrag_prompt(query, passages, context)

        # Generate response with tokens
        raw_response = await self.llm.generate(prompt, temperature=0.3)

        # Parse and assess
        return await self._parse_and_assess(raw_response, query, passages)

    def _build_selfrag_prompt(
        self,
        query: str,
        passages: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> str:
        """Build prompt with Self-RAG token instructions."""
        passage_text = ""
        if passages:
            passage_text = "\n\nProvided Passages:\n" + "\n---\n".join(passages[:5])

        context_text = f"\n\nContext: {context}" if context else ""

        return f"""You are a Self-RAG assistant that generates responses with inline verification tokens.

INSTRUCTIONS:
As you generate your response, include special tokens to self-assess:

1. Before generating content, decide if retrieval is needed:
   [Retrieve] - need additional information
   [No Retrieve] - can answer from passages/knowledge

2. After using a passage, assess its relevance:
   [Relevant] - directly addresses the query
   [Partially Relevant] - somewhat related
   [Irrelevant] - not useful for this query

3. For each factual claim, assess support level:
   [Fully Supported] - clearly supported by passages
   [Partially Supported] - some support but not complete
   [No Support] - cannot verify from passages

4. Assess overall utility of your response:
   [Useful] - directly answers the user's need
   [Not Useful] - doesn't adequately address query

EXAMPLE:
Query: What causes climate change?
[No Retrieve] Based on the passages provided, [Relevant] greenhouse gas emissions
from human activities [Fully Supported] are the primary cause of climate change.
[Useful]

Now respond to this query:
Query: {query}
{passage_text}
{context_text}

Response with verification tokens:"""

    async def _parse_and_assess(
        self,
        raw_response: str,
        query: str,
        passages: Optional[List[str]] = None,
    ) -> SelfRAGResult:
        """Parse response and extract assessments."""
        # Split into segments by sentences or token boundaries
        segments = self._segment_response(raw_response)

        assessments = []
        retrieval_decisions = []
        total_support = 0.0

        retrieval_count = 0

        for segment in segments:
            assessment = self._assess_segment(segment)
            assessments.append(assessment)

            # Track retrieval decisions
            if assessment.needs_retrieval is not None:
                retrieval_decisions.append(assessment.needs_retrieval)

            total_support += assessment.support_score

            # Auto-retrieve if support is low
            if (
                assessment.support_score < self.auto_retrieve_threshold
                and retrieval_count < self.max_retrievals
                and self.retriever
            ):
                # Trigger retrieval for more context
                new_passages = await self.retriever.retrieve(query, k=3)
                if new_passages:
                    retrieval_count += 1
                    self._stats["retrievals_triggered"] += 1

        # Compute overall support
        overall_support = total_support / max(1, len(segments))

        # Clean response (remove tokens for final output)
        clean_response = self._clean_response(raw_response)

        self._stats["generations"] += 1
        self._stats["avg_support_score"] = (
            self._stats["avg_support_score"] * 0.9 + overall_support * 0.1
        )

        return SelfRAGResult(
            response=clean_response,
            segments=segments,
            assessments=assessments,
            overall_support=overall_support,
            retrieval_decisions=retrieval_decisions,
            metadata={"raw_response": raw_response},
        )

    def _segment_response(self, response: str) -> List[str]:
        """Segment response by sentences and tokens."""
        # Split by sentence boundaries while keeping tokens
        sentences = re.split(r"(?<=[.!?])\s+", response)
        return [s.strip() for s in sentences if s.strip()]

    def _assess_segment(self, segment: str) -> SelfRAGAssessment:
        """Assess a single segment for Self-RAG tokens."""
        assessment = SelfRAGAssessment(segment_text=segment)

        # Check retrieval tokens
        retrieve_match = self.TOKEN_PATTERNS["retrieve"].search(segment)
        if retrieve_match:
            token = retrieve_match.group(1).lower()
            assessment.needs_retrieval = token == "retrieve"
            assessment.tokens_found.append(retrieve_match.group(0))

        # Check relevance tokens
        relevance_match = self.TOKEN_PATTERNS["relevance"].search(segment)
        if relevance_match:
            token = relevance_match.group(1).lower()
            assessment.relevance_score = self.TOKEN_SCORES.get(token, 0.5)
            assessment.tokens_found.append(relevance_match.group(0))

        # Check support tokens
        support_match = self.TOKEN_PATTERNS["support"].search(segment)
        if support_match:
            token = support_match.group(1).lower()
            assessment.support_score = self.TOKEN_SCORES.get(token, 0.5)
            assessment.tokens_found.append(support_match.group(0))

        # Check utility tokens
        utility_match = self.TOKEN_PATTERNS["utility"].search(segment)
        if utility_match:
            token = utility_match.group(1).lower()
            assessment.utility_score = self.TOKEN_SCORES.get(token, 0.5)
            assessment.tokens_found.append(utility_match.group(0))

        return assessment

    def _clean_response(self, response: str) -> str:
        """Remove Self-RAG tokens from response."""
        clean = response
        for pattern in self.TOKEN_PATTERNS.values():
            clean = pattern.sub("", clean)
        # Clean up extra whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    async def verify_with_critique(
        self,
        response: str,
        query: str,
        passages: Optional[List[str]] = None,
    ) -> SelfRAGResult:
        """
        Verify existing response using Self-RAG critique.

        Args:
            response: Response to verify
            query: Original query
            passages: Source passages

        Returns:
            SelfRAGResult with critique
        """
        if not self.llm:
            return SelfRAGResult(response=response)

        prompt = f"""Analyze this response using Self-RAG verification tokens.

For each sentence, add verification tokens to assess:
- [Relevant]/[Partially Relevant]/[Irrelevant] for query relevance
- [Fully Supported]/[Partially Supported]/[No Support] for factual support
- [Useful]/[Not Useful] for overall utility

Query: {query}

Passages:
{chr(10).join(passages[:3]) if passages else 'No passages provided'}

Response to analyze:
{response}

Analyzed response with tokens:"""

        analyzed = await self.llm.generate(prompt, temperature=0.1)
        return await self._parse_and_assess(analyzed, query, passages)

    async def revise_low_support_segments(
        self,
        result: SelfRAGResult,
        query: str,
        passages: Optional[List[str]] = None,
    ) -> SelfRAGResult:
        """
        Revise segments with low support scores.

        Args:
            result: Previous SelfRAGResult
            query: Original query
            passages: Source passages

        Returns:
            SelfRAGResult with revised segments
        """
        if not self.llm:
            return result

        revised_segments = []

        for segment, assessment in zip(result.segments, result.assessments):
            if assessment.support_score < self.support_threshold:
                # Revise this segment
                revision_prompt = f"""Revise this response segment to be more accurate and supported.

Query: {query}
Original segment: {segment}

Available passages:
{chr(10).join(passages[:3]) if passages else 'None'}

Provide a revised segment that is factually supported:"""

                revised = await self.llm.generate(revision_prompt, temperature=0.3)
                revised_segments.append(revised.strip())
                self._stats["segments_revised"] += 1
            else:
                revised_segments.append(self._clean_response(segment))

        # Combine revised segments
        revised_response = " ".join(revised_segments)

        # Create new result with revision
        new_result = SelfRAGResult(
            response=result.response,
            segments=result.segments,
            assessments=result.assessments,
            overall_support=result.overall_support,
            retrieval_decisions=result.retrieval_decisions,
            revised_response=revised_response,
            metadata=result.metadata,
        )

        return new_result

    def get_stats(self) -> Dict[str, Any]:
        """Get Self-RAG verification statistics."""
        return dict(self._stats)


class AdaptiveSelfRAG:
    """
    Adaptive Self-RAG that learns retrieval patterns.

    Tracks when retrieval improves response quality and adapts
    the retrieval threshold based on historical performance.
    """

    def __init__(
        self,
        base_verifier: SelfRAGVerifier,
        learning_rate: float = 0.1,
        min_samples_for_adaptation: int = 10,
    ):
        """
        Initialize adaptive Self-RAG.

        Args:
            base_verifier: Base SelfRAGVerifier
            learning_rate: Rate of threshold adaptation
            min_samples_for_adaptation: Samples before adapting
        """
        self.verifier = base_verifier
        self.learning_rate = learning_rate
        self.min_samples = min_samples_for_adaptation

        # Track retrieval outcomes
        self._retrieval_outcomes: List[Dict[str, float]] = []
        self._current_threshold = base_verifier.auto_retrieve_threshold

    async def generate_adaptive(
        self,
        query: str,
        passages: Optional[List[str]] = None,
    ) -> SelfRAGResult:
        """Generate with adaptive retrieval thresholds."""
        # Update verifier threshold
        self.verifier.auto_retrieve_threshold = self._current_threshold

        result = await self.verifier.generate_with_verification(query, passages)

        # Track outcome for adaptation
        self._retrieval_outcomes.append(
            {
                "support_score": result.overall_support,
                "retrieval_triggered": any(result.retrieval_decisions),
                "threshold_used": self._current_threshold,
            }
        )

        # Adapt threshold if enough samples
        if len(self._retrieval_outcomes) >= self.min_samples:
            self._adapt_threshold()

        return result

    def _adapt_threshold(self) -> None:
        """Adapt retrieval threshold based on outcomes."""
        recent = self._retrieval_outcomes[-self.min_samples :]

        # Compare outcomes with vs without retrieval
        with_retrieval = [o for o in recent if o["retrieval_triggered"]]
        without_retrieval = [o for o in recent if not o["retrieval_triggered"]]

        if with_retrieval and without_retrieval:
            avg_with = sum(o["support_score"] for o in with_retrieval) / len(with_retrieval)
            avg_without = sum(o["support_score"] for o in without_retrieval) / len(
                without_retrieval
            )

            # If retrieval helps, lower threshold (retrieve more often)
            # If retrieval doesn't help, raise threshold (retrieve less)
            if avg_with > avg_without + 0.1:
                self._current_threshold = max(0.1, self._current_threshold - self.learning_rate)
            elif avg_without > avg_with + 0.1:
                self._current_threshold = min(0.9, self._current_threshold + self.learning_rate)


def create_selfrag_verifier(
    llm: Optional[LLMProtocol] = None,
    retriever: Optional[RetrieverProtocol] = None,
) -> SelfRAGVerifier:
    """
    Create a Self-RAG verifier.

    Args:
        llm: Language model
        retriever: Document retriever

    Returns:
        SelfRAGVerifier instance
    """
    return SelfRAGVerifier(llm=llm, retriever=retriever)
