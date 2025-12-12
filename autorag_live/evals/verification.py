"""Answer Verification System for AutoRAG-Live.

Verify generated answers against source documents:
- Factual consistency checking
- Citation verification
- Hallucination detection
- Confidence scoring
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Verification result status."""

    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ClaimType(Enum):
    """Types of claims in answers."""

    FACTUAL = "factual"
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    ATTRIBUTIVE = "attributive"


@dataclass
class Claim:
    """Represents an extracted claim from an answer."""

    text: str
    claim_type: ClaimType = ClaimType.FACTUAL
    source_sentence: str = ""
    confidence: float = 1.0
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """Represents evidence supporting or refuting a claim."""

    text: str
    source_id: str
    relevance_score: float = 0.0
    supports_claim: bool | None = None
    match_type: str = "semantic"  # exact, partial, semantic


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""

    claim: Claim
    status: VerificationStatus
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)
    explanation: str = ""
    hallucination_risk: float = 0.0


@dataclass
class AnswerVerification:
    """Complete verification result for an answer."""

    answer: str
    status: VerificationStatus
    overall_confidence: float
    claim_verifications: list[ClaimVerification] = field(default_factory=list)
    verified_percentage: float = 0.0
    hallucination_score: float = 0.0
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ClaimExtractor:
    """Extracts claims from answer text."""

    # Patterns for different claim types
    NUMERICAL_PATTERN = re.compile(
        r"\b(\d+(?:\.\d+)?(?:\s*%|\s*percent)?)\b",
        re.IGNORECASE,
    )
    TEMPORAL_PATTERN = re.compile(
        r"\b(in \d{4}|since \d{4}|\d{4}s?|"
        r"january|february|march|april|may|june|"
        r"july|august|september|october|november|december)\b",
        re.IGNORECASE,
    )
    CAUSAL_MARKERS = [
        "because",
        "therefore",
        "thus",
        "consequently",
        "as a result",
        "due to",
        "caused by",
        "leads to",
    ]
    COMPARATIVE_MARKERS = [
        "more than",
        "less than",
        "greater than",
        "better than",
        "worse than",
        "higher than",
        "lower than",
        "compared to",
    ]

    def extract_claims(self, answer: str) -> list[Claim]:
        """Extract claims from answer text.

        Args:
            answer: Answer text

        Returns:
            List of extracted claims
        """
        claims: list[Claim] = []

        # Split into sentences
        sentences = self._split_sentences(answer)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue

            claim_type = self._determine_claim_type(sentence)
            entities = self._extract_entities(sentence)

            claims.append(
                Claim(
                    text=sentence,
                    claim_type=claim_type,
                    source_sentence=sentence,
                    entities=entities,
                )
            )

        return claims

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in sentences if s.strip()]

    def _determine_claim_type(self, sentence: str) -> ClaimType:
        """Determine the type of claim."""
        sentence_lower = sentence.lower()

        if self.NUMERICAL_PATTERN.search(sentence):
            return ClaimType.NUMERICAL

        if self.TEMPORAL_PATTERN.search(sentence):
            return ClaimType.TEMPORAL

        for marker in self.CAUSAL_MARKERS:
            if marker in sentence_lower:
                return ClaimType.CAUSAL

        for marker in self.COMPARATIVE_MARKERS:
            if marker in sentence_lower:
                return ClaimType.COMPARATIVE

        return ClaimType.FACTUAL

    def _extract_entities(self, sentence: str) -> list[str]:
        """Extract potential entities from sentence."""
        entities: list[str] = []

        # Find capitalized words (simple NER)
        words = sentence.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 1:
                # Clean punctuation
                clean_word = re.sub(r"[^a-zA-Z0-9]", "", word)
                if clean_word and clean_word not in entities:
                    entities.append(clean_word)

        return entities


class EvidenceFinder:
    """Finds evidence for claims in source documents."""

    def __init__(
        self,
        similarity_threshold: float = 0.5,
    ) -> None:
        """Initialize evidence finder.

        Args:
            similarity_threshold: Minimum similarity for relevance
        """
        self.similarity_threshold = similarity_threshold

    def find_evidence(
        self,
        claim: Claim,
        documents: list[str],
        doc_ids: list[str] | None = None,
    ) -> list[Evidence]:
        """Find evidence for a claim in documents.

        Args:
            claim: Claim to find evidence for
            documents: Source documents
            doc_ids: Optional document identifiers

        Returns:
            List of evidence items
        """
        doc_ids = doc_ids or [f"doc_{i}" for i in range(len(documents))]
        evidence_list: list[Evidence] = []

        claim_words = set(claim.text.lower().split())

        for doc, doc_id in zip(documents, doc_ids):
            # Split document into sentences
            sentences = re.split(r"(?<=[.!?])\s+", doc)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Calculate relevance
                relevance = self._calculate_relevance(claim, sentence)

                if relevance >= self.similarity_threshold:
                    # Determine match type
                    match_type = self._determine_match_type(claim.text, sentence)

                    evidence_list.append(
                        Evidence(
                            text=sentence,
                            source_id=doc_id,
                            relevance_score=relevance,
                            match_type=match_type,
                        )
                    )

        # Sort by relevance
        evidence_list.sort(key=lambda e: e.relevance_score, reverse=True)

        return evidence_list[:5]  # Return top 5 evidence items

    def _calculate_relevance(self, claim: Claim, sentence: str) -> float:
        """Calculate relevance between claim and sentence."""
        # Simple word overlap similarity
        claim_words = set(claim.text.lower().split())
        sentence_words = set(sentence.lower().split())

        # Filter common words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may",
            "might", "must", "shall", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "or",
            "and", "but", "if", "then", "than", "so", "that",
            "this", "it", "its",
        }

        claim_words = claim_words - stop_words
        sentence_words = sentence_words - stop_words

        if not claim_words or not sentence_words:
            return 0.0

        intersection = len(claim_words & sentence_words)
        union = len(claim_words | sentence_words)

        base_similarity = intersection / union if union > 0 else 0.0

        # Boost for entity matches
        entity_boost = 0.0
        for entity in claim.entities:
            if entity.lower() in sentence.lower():
                entity_boost += 0.1

        return min(1.0, base_similarity + entity_boost)

    def _determine_match_type(self, claim_text: str, evidence_text: str) -> str:
        """Determine the type of match between claim and evidence."""
        claim_lower = claim_text.lower()
        evidence_lower = evidence_text.lower()

        # Check for exact match
        if claim_lower in evidence_lower or evidence_lower in claim_lower:
            return "exact"

        # Check for high word overlap
        claim_words = set(claim_lower.split())
        evidence_words = set(evidence_lower.split())

        overlap = len(claim_words & evidence_words)
        total = len(claim_words)

        if total > 0 and overlap / total > 0.7:
            return "partial"

        return "semantic"


class BaseVerifier(ABC):
    """Abstract base class for claim verification."""

    @abstractmethod
    def verify_claim(
        self,
        claim: Claim,
        evidence: list[Evidence],
    ) -> ClaimVerification:
        """Verify a single claim against evidence.

        Args:
            claim: Claim to verify
            evidence: Available evidence

        Returns:
            Verification result
        """
        pass


class RuleBasedVerifier(BaseVerifier):
    """Rule-based claim verification."""

    def __init__(
        self,
        require_evidence: bool = True,
        min_evidence_score: float = 0.5,
    ) -> None:
        """Initialize verifier.

        Args:
            require_evidence: Require evidence for verification
            min_evidence_score: Minimum evidence relevance score
        """
        self.require_evidence = require_evidence
        self.min_evidence_score = min_evidence_score

    def verify_claim(
        self,
        claim: Claim,
        evidence: list[Evidence],
    ) -> ClaimVerification:
        """Verify claim using rule-based approach."""
        # Filter relevant evidence
        relevant_evidence = [
            e for e in evidence if e.relevance_score >= self.min_evidence_score
        ]

        if not relevant_evidence:
            if self.require_evidence:
                return ClaimVerification(
                    claim=claim,
                    status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                    confidence=0.3,
                    evidence=[],
                    explanation="No relevant evidence found in source documents.",
                    hallucination_risk=0.7,
                )
            else:
                return ClaimVerification(
                    claim=claim,
                    status=VerificationStatus.UNVERIFIED,
                    confidence=0.5,
                    evidence=[],
                    explanation="Claim could not be verified without evidence.",
                    hallucination_risk=0.5,
                )

        # Check evidence support
        supporting_evidence: list[Evidence] = []
        contradicting_evidence: list[Evidence] = []

        for ev in relevant_evidence:
            support_score = self._assess_support(claim, ev)

            if support_score > 0.6:
                ev.supports_claim = True
                supporting_evidence.append(ev)
            elif support_score < 0.3:
                ev.supports_claim = False
                contradicting_evidence.append(ev)

        # Determine verification status
        if supporting_evidence and not contradicting_evidence:
            avg_confidence = sum(e.relevance_score for e in supporting_evidence) / len(
                supporting_evidence
            )

            # Adjust for match type
            for ev in supporting_evidence:
                if ev.match_type == "exact":
                    avg_confidence = min(1.0, avg_confidence + 0.2)
                elif ev.match_type == "partial":
                    avg_confidence = min(1.0, avg_confidence + 0.1)

            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.VERIFIED,
                confidence=avg_confidence,
                evidence=supporting_evidence,
                explanation=f"Claim supported by {len(supporting_evidence)} evidence item(s).",
                hallucination_risk=max(0.0, 1.0 - avg_confidence),
            )

        elif contradicting_evidence and not supporting_evidence:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.CONTRADICTED,
                confidence=0.7,
                evidence=contradicting_evidence,
                explanation="Evidence contradicts this claim.",
                hallucination_risk=0.9,
            )

        elif supporting_evidence and contradicting_evidence:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.PARTIALLY_VERIFIED,
                confidence=0.5,
                evidence=supporting_evidence + contradicting_evidence,
                explanation="Mixed evidence - some supports, some contradicts.",
                hallucination_risk=0.5,
            )

        else:
            return ClaimVerification(
                claim=claim,
                status=VerificationStatus.UNVERIFIED,
                confidence=0.4,
                evidence=relevant_evidence,
                explanation="Evidence found but cannot confirm or deny claim.",
                hallucination_risk=0.6,
            )

    def _assess_support(self, claim: Claim, evidence: Evidence) -> float:
        """Assess how well evidence supports the claim."""
        # Start with relevance score
        support_score = evidence.relevance_score

        # Adjust based on match type
        if evidence.match_type == "exact":
            support_score = min(1.0, support_score + 0.3)
        elif evidence.match_type == "partial":
            support_score = min(1.0, support_score + 0.1)

        # Check for negation in evidence
        negation_words = ["not", "never", "no", "none", "neither", "nobody", "nowhere"]
        evidence_lower = evidence.text.lower()
        claim_lower = claim.text.lower()

        evidence_has_negation = any(w in evidence_lower for w in negation_words)
        claim_has_negation = any(w in claim_lower for w in negation_words)

        # If negation mismatch, likely contradiction
        if evidence_has_negation != claim_has_negation:
            support_score *= 0.5

        return support_score


class AnswerVerifier:
    """Complete answer verification system."""

    def __init__(
        self,
        claim_extractor: ClaimExtractor | None = None,
        evidence_finder: EvidenceFinder | None = None,
        verifier: BaseVerifier | None = None,
    ) -> None:
        """Initialize answer verifier.

        Args:
            claim_extractor: Claim extraction component
            evidence_finder: Evidence finding component
            verifier: Claim verification component
        """
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.evidence_finder = evidence_finder or EvidenceFinder()
        self.verifier = verifier or RuleBasedVerifier()

    def verify(
        self,
        answer: str,
        source_documents: list[str],
        doc_ids: list[str] | None = None,
    ) -> AnswerVerification:
        """Verify an answer against source documents.

        Args:
            answer: Generated answer to verify
            source_documents: Source documents for verification
            doc_ids: Optional document identifiers

        Returns:
            Complete verification result
        """
        # Extract claims
        claims = self.claim_extractor.extract_claims(answer)

        if not claims:
            return AnswerVerification(
                answer=answer,
                status=VerificationStatus.UNVERIFIED,
                overall_confidence=0.5,
                claim_verifications=[],
                verified_percentage=0.0,
                hallucination_score=0.0,
                summary="No verifiable claims found in answer.",
            )

        # Verify each claim
        claim_verifications: list[ClaimVerification] = []

        for claim in claims:
            # Find evidence
            evidence = self.evidence_finder.find_evidence(
                claim, source_documents, doc_ids
            )

            # Verify claim
            verification = self.verifier.verify_claim(claim, evidence)
            claim_verifications.append(verification)

        # Calculate overall metrics
        verified_count = sum(
            1
            for v in claim_verifications
            if v.status in (VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED)
        )
        verified_percentage = verified_count / len(claim_verifications) if claim_verifications else 0

        avg_confidence = (
            sum(v.confidence for v in claim_verifications) / len(claim_verifications)
            if claim_verifications
            else 0
        )

        avg_hallucination = (
            sum(v.hallucination_risk for v in claim_verifications)
            / len(claim_verifications)
            if claim_verifications
            else 0
        )

        # Determine overall status
        if verified_percentage >= 0.8 and avg_confidence >= 0.7:
            overall_status = VerificationStatus.VERIFIED
        elif verified_percentage >= 0.5:
            overall_status = VerificationStatus.PARTIALLY_VERIFIED
        elif any(v.status == VerificationStatus.CONTRADICTED for v in claim_verifications):
            overall_status = VerificationStatus.CONTRADICTED
        else:
            overall_status = VerificationStatus.UNVERIFIED

        # Generate summary
        summary = self._generate_summary(claim_verifications, verified_percentage)

        return AnswerVerification(
            answer=answer,
            status=overall_status,
            overall_confidence=avg_confidence,
            claim_verifications=claim_verifications,
            verified_percentage=verified_percentage,
            hallucination_score=avg_hallucination,
            summary=summary,
            metadata={
                "total_claims": len(claims),
                "verified_claims": verified_count,
            },
        )

    def _generate_summary(
        self,
        verifications: list[ClaimVerification],
        verified_percentage: float,
    ) -> str:
        """Generate human-readable summary."""
        total = len(verifications)
        verified = sum(1 for v in verifications if v.status == VerificationStatus.VERIFIED)
        partial = sum(
            1 for v in verifications if v.status == VerificationStatus.PARTIALLY_VERIFIED
        )
        contradicted = sum(
            1 for v in verifications if v.status == VerificationStatus.CONTRADICTED
        )
        unverified = total - verified - partial - contradicted

        parts = [f"Analyzed {total} claims:"]

        if verified > 0:
            parts.append(f"- {verified} fully verified")
        if partial > 0:
            parts.append(f"- {partial} partially verified")
        if contradicted > 0:
            parts.append(f"- {contradicted} contradicted by sources")
        if unverified > 0:
            parts.append(f"- {unverified} could not be verified")

        parts.append(f"\nOverall verification rate: {verified_percentage:.0%}")

        return "\n".join(parts)

    def quick_check(
        self,
        answer: str,
        source_documents: list[str],
    ) -> tuple[bool, float, str]:
        """Quick verification check.

        Args:
            answer: Answer to check
            source_documents: Source documents

        Returns:
            Tuple of (is_verified, confidence, summary)
        """
        result = self.verify(answer, source_documents)

        is_verified = result.status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.PARTIALLY_VERIFIED,
        )

        return is_verified, result.overall_confidence, result.summary


# Convenience functions


def verify_answer(
    answer: str,
    sources: list[str],
) -> AnswerVerification:
    """Verify an answer against source documents.

    Args:
        answer: Generated answer
        sources: Source documents

    Returns:
        Verification result
    """
    verifier = AnswerVerifier()
    return verifier.verify(answer, sources)


def check_hallucination(
    answer: str,
    sources: list[str],
    threshold: float = 0.5,
) -> tuple[bool, float]:
    """Check for potential hallucination.

    Args:
        answer: Generated answer
        sources: Source documents
        threshold: Hallucination score threshold

    Returns:
        Tuple of (has_hallucination, score)
    """
    verifier = AnswerVerifier()
    result = verifier.verify(answer, sources)

    has_hallucination = result.hallucination_score >= threshold
    return has_hallucination, result.hallucination_score


def extract_claims(answer: str) -> list[str]:
    """Extract claims from an answer.

    Args:
        answer: Answer text

    Returns:
        List of claim texts
    """
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(answer)
    return [c.text for c in claims]
