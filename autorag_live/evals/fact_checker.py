"""
Fact checking and answer verification for AutoRAG-Live.

Provides utilities to verify generated answers against
source documents and detect potential hallucinations.

Features:
- Claim extraction from answers
- Claim verification against sources
- Hallucination detection
- Confidence scoring
- Citation verification

Example usage:
    >>> checker = FactChecker()
    >>> result = checker.verify(
    ...     answer="Python was created in 1991.",
    ...     sources=["Python was released in 1991 by Guido van Rossum."]
    ... )
    >>> print(f"Verified: {result.is_verified}, Score: {result.confidence}")
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    """Claim verification status."""
    
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNVERIFIABLE = "unverifiable"
    PARTIAL = "partial"


class ClaimType(str, Enum):
    """Types of claims."""
    
    FACTUAL = "factual"
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    OPINION = "opinion"


@dataclass
class Claim:
    """Represents an extracted claim."""
    
    text: str
    claim_type: ClaimType = ClaimType.FACTUAL
    
    # Entities and values
    entities: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    
    # Source reference
    source_sentence: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of verifying a claim."""
    
    claim: Claim
    status: VerificationStatus
    confidence: float
    
    # Supporting evidence
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    
    # Reasoning
    reasoning: Optional[str] = None
    
    # Source references
    source_indices: List[int] = field(default_factory=list)
    
    @property
    def is_verified(self) -> bool:
        """Check if claim is verified."""
        return self.status == VerificationStatus.SUPPORTED
    
    @property
    def is_hallucination(self) -> bool:
        """Check if claim appears to be hallucinated."""
        return self.status in (VerificationStatus.REFUTED, VerificationStatus.UNVERIFIABLE)


@dataclass
class FactCheckResult:
    """Complete fact-checking result for an answer."""
    
    answer: str
    claims: List[Claim]
    verifications: List[VerificationResult]
    
    # Overall scores
    overall_confidence: float = 0.0
    hallucination_score: float = 0.0
    
    # Summary
    supported_count: int = 0
    refuted_count: int = 0
    unverifiable_count: int = 0
    
    @property
    def is_verified(self) -> bool:
        """Check if answer is verified overall."""
        return self.overall_confidence >= 0.7 and self.refuted_count == 0
    
    @property
    def verification_rate(self) -> float:
        """Get percentage of verified claims."""
        if not self.claims:
            return 1.0
        return self.supported_count / len(self.claims)
    
    def get_problematic_claims(self) -> List[VerificationResult]:
        """Get claims that failed verification."""
        return [
            v for v in self.verifications
            if v.status in (VerificationStatus.REFUTED, VerificationStatus.UNVERIFIABLE)
        ]


class ClaimExtractor:
    """Extract claims from text."""
    
    # Patterns for different claim types
    NUMERICAL_PATTERN = re.compile(r'\b\d+(?:\.\d+)?(?:\s*%|\s*percent)?\b')
    DATE_PATTERN = re.compile(
        r'\b(?:\d{4}|'
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}|'
        r'\d{1,2}/\d{1,2}/\d{2,4})\b',
        re.IGNORECASE
    )
    COMPARATIVE_PATTERN = re.compile(
        r'\b(?:better|worse|more|less|higher|lower|faster|slower|larger|smaller|'
        r'greater|fewer|best|worst|most|least)\b',
        re.IGNORECASE
    )
    CAUSAL_PATTERN = re.compile(
        r'\b(?:because|therefore|thus|hence|causes?|results?\s+in|leads?\s+to|'
        r'due\s+to|as\s+a\s+result)\b',
        re.IGNORECASE
    )
    
    def __init__(
        self,
        min_claim_length: int = 10,
        max_claim_length: int = 500,
    ):
        """
        Initialize claim extractor.
        
        Args:
            min_claim_length: Minimum claim length
            max_claim_length: Maximum claim length
        """
        self.min_claim_length = min_claim_length
        self.max_claim_length = max_claim_length
    
    def extract(self, text: str) -> List[Claim]:
        """
        Extract claims from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted claims
        """
        sentences = self._split_sentences(text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            if not self._is_valid_claim(sentence):
                continue
            
            claim = self._create_claim(sentence)
            claims.append(claim)
        
        return claims
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_valid_claim(self, sentence: str) -> bool:
        """Check if sentence is a valid claim."""
        # Length check
        if len(sentence) < self.min_claim_length:
            return False
        if len(sentence) > self.max_claim_length:
            return False
        
        # Must contain some alphanumeric content
        if not re.search(r'\w+', sentence):
            return False
        
        # Skip questions
        if sentence.strip().endswith("?"):
            return False
        
        # Skip commands/imperatives (simple heuristic)
        imperative_starters = [
            "please", "note", "remember", "consider", "see", "check",
            "click", "go to", "visit", "refer to",
        ]
        lower = sentence.lower()
        if any(lower.startswith(s) for s in imperative_starters):
            return False
        
        return True
    
    def _create_claim(self, sentence: str) -> Claim:
        """Create a Claim object from sentence."""
        claim_type = self._classify_claim(sentence)
        
        claim = Claim(
            text=sentence,
            claim_type=claim_type,
            entities=self._extract_entities(sentence),
            numbers=self._extract_numbers(sentence),
            dates=self._extract_dates(sentence),
            source_sentence=sentence,
        )
        
        return claim
    
    def _classify_claim(self, sentence: str) -> ClaimType:
        """Classify the type of claim."""
        if self.DATE_PATTERN.search(sentence):
            return ClaimType.TEMPORAL
        
        if self.NUMERICAL_PATTERN.search(sentence):
            return ClaimType.NUMERICAL
        
        if self.COMPARATIVE_PATTERN.search(sentence):
            return ClaimType.COMPARATIVE
        
        if self.CAUSAL_PATTERN.search(sentence):
            return ClaimType.CAUSAL
        
        # Check for opinion markers
        opinion_markers = [
            "i think", "i believe", "in my opinion", "arguably",
            "probably", "possibly", "might", "may", "could",
        ]
        lower = sentence.lower()
        if any(marker in lower for marker in opinion_markers):
            return ClaimType.OPINION
        
        return ClaimType.FACTUAL
    
    def _extract_entities(self, sentence: str) -> List[str]:
        """Extract named entities (simple approach)."""
        # Extract capitalized phrases
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
        
        # Filter common words
        common = {"The", "This", "That", "These", "Those", "It", "They", "We", "He", "She"}
        entities = [e for e in entities if e not in common]
        
        return entities
    
    def _extract_numbers(self, sentence: str) -> List[str]:
        """Extract numerical values."""
        return self.NUMERICAL_PATTERN.findall(sentence)
    
    def _extract_dates(self, sentence: str) -> List[str]:
        """Extract dates."""
        return self.DATE_PATTERN.findall(sentence)


class BaseVerifier(ABC):
    """Base class for claim verifiers."""
    
    @abstractmethod
    def verify(
        self,
        claim: Claim,
        sources: List[str],
    ) -> VerificationResult:
        """Verify a claim against sources."""
        pass


class TextOverlapVerifier(BaseVerifier):
    """Verify claims using text overlap."""
    
    def __init__(
        self,
        min_overlap: float = 0.3,
        entity_weight: float = 0.4,
    ):
        """
        Initialize text overlap verifier.
        
        Args:
            min_overlap: Minimum overlap for verification
            entity_weight: Weight for entity matching
        """
        self.min_overlap = min_overlap
        self.entity_weight = entity_weight
    
    def verify(
        self,
        claim: Claim,
        sources: List[str],
    ) -> VerificationResult:
        """Verify claim using text overlap."""
        best_score = 0.0
        best_source_idx = -1
        supporting = []
        
        claim_words = self._tokenize(claim.text)
        
        for idx, source in enumerate(sources):
            source_words = self._tokenize(source)
            
            # Word overlap
            overlap = len(claim_words & source_words)
            overlap_score = overlap / len(claim_words) if claim_words else 0
            
            # Entity overlap
            if claim.entities:
                source_lower = source.lower()
                entity_matches = sum(
                    1 for e in claim.entities
                    if e.lower() in source_lower
                )
                entity_score = entity_matches / len(claim.entities)
            else:
                entity_score = 0.5
            
            # Combined score
            score = (
                (1 - self.entity_weight) * overlap_score +
                self.entity_weight * entity_score
            )
            
            if score > self.min_overlap:
                supporting.append(source)
            
            if score > best_score:
                best_score = score
                best_source_idx = idx
        
        # Determine status
        if best_score >= 0.7:
            status = VerificationStatus.SUPPORTED
        elif best_score >= 0.3:
            status = VerificationStatus.PARTIAL
        else:
            status = VerificationStatus.UNVERIFIABLE
        
        return VerificationResult(
            claim=claim,
            status=status,
            confidence=best_score,
            supporting_evidence=supporting,
            source_indices=[best_source_idx] if best_source_idx >= 0 else [],
        )
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into word set."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords
        stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does",
            "this", "that", "it", "its", "they", "them", "he", "she",
        }
        
        return {w for w in words if w not in stopwords and len(w) > 1}


class NLIVerifier(BaseVerifier):
    """Verify claims using Natural Language Inference."""
    
    def __init__(
        self,
        nli_func: Optional[Callable[[str, str], Tuple[str, float]]] = None,
    ):
        """
        Initialize NLI verifier.
        
        Args:
            nli_func: Function that returns (label, confidence) for premise-hypothesis pair
        """
        self.nli_func = nli_func
    
    def verify(
        self,
        claim: Claim,
        sources: List[str],
    ) -> VerificationResult:
        """Verify claim using NLI."""
        if not self.nli_func:
            # Fallback to overlap
            overlap_verifier = TextOverlapVerifier()
            return overlap_verifier.verify(claim, sources)
        
        supporting = []
        contradicting = []
        best_entailment = 0.0
        max_contradiction = 0.0
        source_indices = []
        
        for idx, source in enumerate(sources):
            label, confidence = self.nli_func(source, claim.text)
            
            if label == "entailment":
                supporting.append(source)
                source_indices.append(idx)
                best_entailment = max(best_entailment, confidence)
            elif label == "contradiction":
                contradicting.append(source)
                max_contradiction = max(max_contradiction, confidence)
        
        # Determine status
        if contradicting and max_contradiction > 0.7:
            status = VerificationStatus.REFUTED
            confidence = 1 - max_contradiction
        elif supporting and best_entailment > 0.7:
            status = VerificationStatus.SUPPORTED
            confidence = best_entailment
        elif supporting:
            status = VerificationStatus.PARTIAL
            confidence = best_entailment
        else:
            status = VerificationStatus.UNVERIFIABLE
            confidence = 0.0
        
        return VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            source_indices=source_indices,
        )


class LLMVerifier(BaseVerifier):
    """Verify claims using LLM."""
    
    def __init__(
        self,
        llm_func: Callable[[str], str],
    ):
        """
        Initialize LLM verifier.
        
        Args:
            llm_func: Function to call LLM
        """
        self.llm_func = llm_func
    
    def verify(
        self,
        claim: Claim,
        sources: List[str],
    ) -> VerificationResult:
        """Verify claim using LLM."""
        prompt = self._build_prompt(claim, sources)
        
        try:
            response = self.llm_func(prompt)
            return self._parse_response(claim, response, sources)
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            # Fallback to text overlap
            fallback = TextOverlapVerifier()
            return fallback.verify(claim, sources)
    
    def _build_prompt(self, claim: Claim, sources: List[str]) -> str:
        """Build verification prompt."""
        source_text = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sources))
        
        return f"""Verify the following claim against the provided sources.

Claim: {claim.text}

Sources:
{source_text}

Analyze whether the claim is SUPPORTED, REFUTED, or UNVERIFIABLE based on the sources.
Respond in the format:
STATUS: [SUPPORTED/REFUTED/UNVERIFIABLE]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
"""
    
    def _parse_response(
        self,
        claim: Claim,
        response: str,
        sources: List[str],
    ) -> VerificationResult:
        """Parse LLM response."""
        # Extract status
        status_match = re.search(r'STATUS:\s*(\w+)', response, re.IGNORECASE)
        status_str = status_match.group(1).upper() if status_match else "UNVERIFIABLE"
        
        status_map = {
            "SUPPORTED": VerificationStatus.SUPPORTED,
            "REFUTED": VerificationStatus.REFUTED,
            "UNVERIFIABLE": VerificationStatus.UNVERIFIABLE,
            "PARTIAL": VerificationStatus.PARTIAL,
        }
        status = status_map.get(status_str, VerificationStatus.UNVERIFIABLE)
        
        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        
        # Extract reasoning
        reason_match = re.search(r'REASONING:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        reasoning = reason_match.group(1).strip() if reason_match else None
        
        return VerificationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            reasoning=reasoning,
        )


class FactChecker:
    """
    Main fact-checking interface.
    
    Example:
        >>> checker = FactChecker()
        >>> 
        >>> # Verify an answer
        >>> result = checker.verify(
        ...     answer="Python was created by Guido van Rossum in 1991.",
        ...     sources=["Python is a programming language released in 1991 by Guido van Rossum."]
        ... )
        >>> 
        >>> print(f"Verified: {result.is_verified}")
        >>> print(f"Confidence: {result.overall_confidence:.2f}")
        >>> 
        >>> # Check for hallucinations
        >>> if result.hallucination_score > 0.3:
        ...     print("Warning: Possible hallucinations detected")
    """
    
    def __init__(
        self,
        verifier: Optional[BaseVerifier] = None,
        extractor: Optional[ClaimExtractor] = None,
        min_confidence: float = 0.5,
    ):
        """
        Initialize fact checker.
        
        Args:
            verifier: Claim verifier (default: TextOverlapVerifier)
            extractor: Claim extractor
            min_confidence: Minimum confidence for verification
        """
        self.verifier = verifier or TextOverlapVerifier()
        self.extractor = extractor or ClaimExtractor()
        self.min_confidence = min_confidence
    
    def verify(
        self,
        answer: str,
        sources: List[str],
    ) -> FactCheckResult:
        """
        Verify an answer against sources.
        
        Args:
            answer: Generated answer to verify
            sources: Source documents/passages
            
        Returns:
            FactCheckResult
        """
        # Extract claims
        claims = self.extractor.extract(answer)
        
        if not claims:
            return FactCheckResult(
                answer=answer,
                claims=[],
                verifications=[],
                overall_confidence=1.0,
                hallucination_score=0.0,
            )
        
        # Verify each claim
        verifications = []
        for claim in claims:
            result = self.verifier.verify(claim, sources)
            verifications.append(result)
        
        # Calculate statistics
        supported = sum(1 for v in verifications if v.status == VerificationStatus.SUPPORTED)
        refuted = sum(1 for v in verifications if v.status == VerificationStatus.REFUTED)
        unverifiable = sum(1 for v in verifications if v.status == VerificationStatus.UNVERIFIABLE)
        
        # Calculate overall confidence
        if verifications:
            avg_confidence = sum(v.confidence for v in verifications) / len(verifications)
        else:
            avg_confidence = 1.0
        
        # Calculate hallucination score
        if claims:
            hallucination_score = (refuted + 0.5 * unverifiable) / len(claims)
        else:
            hallucination_score = 0.0
        
        return FactCheckResult(
            answer=answer,
            claims=claims,
            verifications=verifications,
            overall_confidence=avg_confidence,
            hallucination_score=hallucination_score,
            supported_count=supported,
            refuted_count=refuted,
            unverifiable_count=unverifiable,
        )
    
    def check_hallucination(
        self,
        answer: str,
        sources: List[str],
        threshold: float = 0.3,
    ) -> Tuple[bool, float]:
        """
        Check if answer contains hallucinations.
        
        Args:
            answer: Answer to check
            sources: Source documents
            threshold: Hallucination threshold
            
        Returns:
            Tuple of (has_hallucination, hallucination_score)
        """
        result = self.verify(answer, sources)
        return result.hallucination_score > threshold, result.hallucination_score
    
    def get_unsupported_claims(
        self,
        answer: str,
        sources: List[str],
    ) -> List[Claim]:
        """
        Get claims that are not supported by sources.
        
        Args:
            answer: Answer to check
            sources: Source documents
            
        Returns:
            List of unsupported claims
        """
        result = self.verify(answer, sources)
        return [
            v.claim for v in result.verifications
            if v.status != VerificationStatus.SUPPORTED
        ]


class CitationVerifier:
    """
    Verify citations in generated text.
    
    Example:
        >>> verifier = CitationVerifier()
        >>> result = verifier.verify(
        ...     text="Python is popular [1]. It was created in 1991 [2].",
        ...     sources=["Python is widely used.", "Python was released in 1991."]
        ... )
    """
    
    def __init__(
        self,
        citation_pattern: str = r'\[(\d+)\]',
    ):
        """
        Initialize citation verifier.
        
        Args:
            citation_pattern: Regex pattern for citations
        """
        self.citation_pattern = re.compile(citation_pattern)
        self.fact_checker = FactChecker()
    
    def verify(
        self,
        text: str,
        sources: List[str],
    ) -> Dict[str, Any]:
        """
        Verify citations in text.
        
        Args:
            text: Text with citations
            sources: Source documents
            
        Returns:
            Verification result dict
        """
        # Find all citations
        citations = self.citation_pattern.findall(text)
        citation_indices = [int(c) - 1 for c in citations]  # Convert to 0-indexed
        
        # Split text by citations
        segments = self.citation_pattern.split(text)
        
        results = {
            "total_citations": len(citations),
            "valid_citations": 0,
            "invalid_citations": 0,
            "citation_details": [],
        }
        
        # Verify each cited segment
        current_text = ""
        for i, segment in enumerate(segments):
            if segment.isdigit():
                # This is a citation number
                idx = int(segment) - 1
                
                if 0 <= idx < len(sources):
                    # Verify the preceding text against the cited source
                    if current_text.strip():
                        verification = self.fact_checker.verify(
                            current_text.strip(),
                            [sources[idx]]
                        )
                        
                        is_valid = verification.overall_confidence >= 0.5
                        
                        results["citation_details"].append({
                            "citation": int(segment),
                            "text": current_text.strip(),
                            "valid": is_valid,
                            "confidence": verification.overall_confidence,
                        })
                        
                        if is_valid:
                            results["valid_citations"] += 1
                        else:
                            results["invalid_citations"] += 1
                    
                    current_text = ""
                else:
                    results["invalid_citations"] += 1
                    results["citation_details"].append({
                        "citation": int(segment),
                        "text": current_text.strip(),
                        "valid": False,
                        "error": "Citation index out of range",
                    })
            else:
                current_text = segment
        
        return results


class ConsistencyChecker:
    """
    Check internal consistency of answers.
    
    Example:
        >>> checker = ConsistencyChecker()
        >>> is_consistent = checker.check("Python was created in 1991. Python was released in 1989.")
    """
    
    def __init__(self):
        """Initialize consistency checker."""
        self.extractor = ClaimExtractor()
    
    def check(self, text: str) -> Tuple[bool, List[str]]:
        """
        Check internal consistency.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_consistent, contradictions)
        """
        claims = self.extractor.extract(text)
        contradictions = []
        
        # Check for numerical contradictions
        numerical_claims = [c for c in claims if c.claim_type == ClaimType.NUMERICAL]
        for i, claim1 in enumerate(numerical_claims):
            for claim2 in numerical_claims[i+1:]:
                if self._check_numerical_contradiction(claim1, claim2):
                    contradictions.append(f"'{claim1.text}' vs '{claim2.text}'")
        
        # Check for temporal contradictions
        temporal_claims = [c for c in claims if c.claim_type == ClaimType.TEMPORAL]
        for i, claim1 in enumerate(temporal_claims):
            for claim2 in temporal_claims[i+1:]:
                if self._check_temporal_contradiction(claim1, claim2):
                    contradictions.append(f"'{claim1.text}' vs '{claim2.text}'")
        
        return len(contradictions) == 0, contradictions
    
    def _check_numerical_contradiction(
        self,
        claim1: Claim,
        claim2: Claim,
    ) -> bool:
        """Check for numerical contradiction."""
        # Simple check: if same entity has different numbers
        common_entities = set(e.lower() for e in claim1.entities) & set(e.lower() for e in claim2.entities)
        
        if common_entities and claim1.numbers and claim2.numbers:
            return claim1.numbers != claim2.numbers
        
        return False
    
    def _check_temporal_contradiction(
        self,
        claim1: Claim,
        claim2: Claim,
    ) -> bool:
        """Check for temporal contradiction."""
        # Simple check: if same entity has different dates
        common_entities = set(e.lower() for e in claim1.entities) & set(e.lower() for e in claim2.entities)
        
        if common_entities and claim1.dates and claim2.dates:
            return claim1.dates != claim2.dates
        
        return False


# Convenience functions

def verify_answer(
    answer: str,
    sources: List[str],
) -> FactCheckResult:
    """
    Verify an answer against sources.
    
    Args:
        answer: Answer to verify
        sources: Source documents
        
    Returns:
        FactCheckResult
    """
    checker = FactChecker()
    return checker.verify(answer, sources)


def check_hallucination(
    answer: str,
    sources: List[str],
    threshold: float = 0.3,
) -> Tuple[bool, float]:
    """
    Check for hallucinations in answer.
    
    Args:
        answer: Answer to check
        sources: Source documents
        threshold: Hallucination threshold
        
    Returns:
        Tuple of (has_hallucination, score)
    """
    checker = FactChecker()
    return checker.check_hallucination(answer, sources, threshold)


def extract_claims(text: str) -> List[Claim]:
    """
    Extract claims from text.
    
    Args:
        text: Input text
        
    Returns:
        List of claims
    """
    extractor = ClaimExtractor()
    return extractor.extract(text)
