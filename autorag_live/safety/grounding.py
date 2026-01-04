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
