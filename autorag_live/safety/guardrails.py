"""
Safety guardrails for responsible agentic RAG.

Detects and prevents harmful outputs, hallucinations, and ensures
answer grounding and factual accuracy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""

    is_safe: bool
    risk_level: float  # 0-1 scale
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_safe": self.is_safe,
            "risk_level": self.risk_level,
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


class HallucinationDetector:
    """Detects potential hallucinations in generated responses."""

    def __init__(self):
        """Initialize hallucination detector."""
        self.logger = logging.getLogger("HallucinationDetector")

    def detect(
        self,
        response: str,
        sources: List[str],
        confidence_threshold: float = 0.7,
    ) -> SafetyCheckResult:
        """
        Detect hallucinations in response.

        Args:
            response: Generated response text
            sources: Retrieved source documents
            confidence_threshold: Threshold for flagging claims

        Returns:
            SafetyCheckResult with hallucination analysis
        """
        issues = []
        risk_level = 0.0

        if not sources:
            issues.append("No sources available for verification")
            risk_level = 0.8
        else:
            # Check claim grounding
            grounded_claims = self._verify_claims(response, sources)

            ungrounded_ratio = (
                sum(1 for g in grounded_claims.values() if not g) / len(grounded_claims)
                if grounded_claims
                else 0
            )

            if ungrounded_ratio > 0.3:
                issues.append(f"~{int(ungrounded_ratio * 100)}% of claims lack source support")
                risk_level += 0.4

            # Check for overconfidence
            confidence_matches = re.findall(
                r"(certainly|definitely|absolutely|always|never)",
                response.lower(),
            )

            if len(confidence_matches) > 3:
                issues.append(f"High confidence language ({len(confidence_matches)} instances)")
                risk_level += 0.2

            # Check for contradictions
            contradictions = self._detect_contradictions(response)
            if contradictions:
                issues.append(f"Potential contradictions detected: {contradictions[0]}")
                risk_level += 0.3

        # Make safety determination
        is_safe = risk_level < confidence_threshold

        suggestions = self._generate_suggestions(issues)

        return SafetyCheckResult(
            is_safe=is_safe,
            risk_level=min(1.0, risk_level),
            issues=issues,
            suggestions=suggestions,
        )

    def _verify_claims(self, response: str, sources: List[str]) -> Dict[str, bool]:
        """Verify individual claims against sources."""
        # Extract claims (simple heuristic: sentences)
        claims = response.split(".")
        source_text = " ".join(sources).lower()

        grounded = {}

        for claim in claims:
            claim = claim.strip()
            if len(claim) < 10:
                continue

            # Check if key words from claim appear in sources
            words = claim.lower().split()
            key_words = [
                w
                for w in words
                if len(w) > 4 and w not in {"that", "this", "which", "where", "when"}
            ]

            matches = sum(1 for w in key_words if w in source_text)
            grounded[claim[:50]] = matches >= len(key_words) * 0.5

        return grounded

    def _detect_contradictions(self, response: str) -> List[str]:
        """Detect contradictions within response."""
        sentences = response.split(".")
        contradictions = []

        # Simple heuristic: check for "but" statements
        for i, sent in enumerate(sentences):
            if "but" in sent.lower() and i > 0:
                # This might be contradicting previous statement
                prev_sent = sentences[i - 1]

                if prev_sent and sent:
                    contradiction_strength = self._similarity(prev_sent, sent)

                    if contradiction_strength > 0.5:
                        contradictions.append(f"'{prev_sent[:30]}...' vs '{sent[:30]}...'")

        return contradictions

    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """Generate suggestions based on detected issues."""
        suggestions = []

        for issue in issues:
            if "sources" in issue.lower():
                suggestions.append("Retrieve additional documents to support claims")
            elif "confidence" in issue.lower():
                suggestions.append("Use more measured language (e.g., 'may', 'likely')")
            elif "contradiction" in issue.lower():
                suggestions.append("Clarify conflicting statements or add context")

        return suggestions

    @staticmethod
    def _similarity(text1: str, text2: str) -> float:
        """Compute text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        return overlap / union if union > 0 else 0.0


class ToxicityFilter:
    """Filters toxic, harmful, or biased content."""

    def __init__(self):
        """Initialize toxicity filter."""
        self.logger = logging.getLogger("ToxicityFilter")

        # Simple keyword-based filtering
        self.harmful_patterns = {
            "violent": ["kill", "murder", "attack", "violence", "harm", "destroy"],
            "hateful": ["hate", "detest", "offensive", "discriminat", "racist", "sexist", "biased"],
            "explicit": [
                "adult",
                "sexual",
                "explicit",
                "pornograph",
            ],
        }

        self.bias_indicators = {
            "gender_bias": [
                "women are",
                "men are",
                "girls are",
                "boys are",
                "female",
                "male",
            ],
            "racial_bias": [
                "people are",
                "race",
                "ethnic",
                "national",
            ],
        }

    def check(self, text: str) -> SafetyCheckResult:
        """
        Check for toxic or harmful content.

        Args:
            text: Text to check

        Returns:
            SafetyCheckResult with toxicity analysis
        """
        issues = []
        risk_level = 0.0
        text_lower = text.lower()

        # Check for harmful content
        for category, patterns in self.harmful_patterns.items():
            matches = sum(1 for p in patterns if p in text_lower)

            if matches > 0:
                issues.append(f"Potential {category} content detected")
                risk_level += 0.3

        # Check for bias
        for bias_type, patterns in self.bias_indicators.items():
            matches = sum(1 for p in patterns if p in text_lower)

            if matches > 2:
                issues.append(f"Potential {bias_type} detected")
                risk_level += 0.2

        is_safe = risk_level < 0.5

        suggestions = (
            [
                "Review content for fairness and inclusivity",
                "Use neutral language where possible",
                "Avoid generalizations about groups",
            ]
            if issues
            else []
        )

        return SafetyCheckResult(
            is_safe=is_safe,
            risk_level=min(1.0, risk_level),
            issues=issues,
            suggestions=suggestions,
        )


class GroundingValidator:
    """Validates that answers are grounded in sources."""

    def __init__(self, threshold: float = 0.6):
        """Initialize grounding validator."""
        self.logger = logging.getLogger("GroundingValidator")
        self.threshold = threshold

    def validate(self, response: str, sources: List[str]) -> SafetyCheckResult:
        """
        Validate answer grounding in sources.

        Args:
            response: Generated response
            sources: Retrieved sources

        Returns:
            SafetyCheckResult with grounding analysis
        """
        if not sources:
            return SafetyCheckResult(
                is_safe=False,
                risk_level=1.0,
                issues=["No sources provided"],
                suggestions=["Retrieve relevant documents"],
            )

        # Calculate grounding score
        grounding_score = self._calculate_grounding(response, sources)

        is_safe = grounding_score >= self.threshold
        issues = []

        if grounding_score < self.threshold:
            issues.append(
                f"Low grounding score ({grounding_score:.1%}). " "Answer may not be well-supported."
            )

        suggestions = []

        if not is_safe:
            suggestions.append("Retrieve more relevant sources")
            suggestions.append("Emphasize uncertainty with hedging language")
            suggestions.append("Add explicit citations")

        return SafetyCheckResult(
            is_safe=is_safe,
            risk_level=1.0 - grounding_score,
            issues=issues,
            suggestions=suggestions,
        )

    def _calculate_grounding(self, response: str, sources: List[str]) -> float:
        """Calculate grounding score."""
        if not sources or not response:
            return 0.0

        response_words = set(w.lower() for w in response.split() if len(w) > 3)
        source_words = set()

        for source in sources:
            source_words.update(w.lower() for w in source.split() if len(w) > 3)

        if not response_words:
            return 0.0

        overlap = len(response_words & source_words)
        return overlap / len(response_words)


class SafetyGuardrails:
    """
    Comprehensive safety checking system.

    Combines multiple safety checks for robust content validation.
    """

    def __init__(self):
        """Initialize safety guardrails."""
        self.logger = logging.getLogger("SafetyGuardrails")
        self.hallucination_detector = HallucinationDetector()
        self.toxicity_filter = ToxicityFilter()
        self.grounding_validator = GroundingValidator()

        self.check_history: List[Dict[str, Any]] = []

    def check_response(
        self,
        response: str,
        sources: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> SafetyCheckResult:
        """
        Perform comprehensive safety check on response.

        Args:
            response: Generated response
            sources: Retrieved source documents
            query: Original query (for context)

        Returns:
            SafetyCheckResult with combined analysis
        """
        sources = sources or []
        results = []

        # Check for hallucinations
        hallucination_result = self.hallucination_detector.detect(response, sources)
        results.append(("hallucination", hallucination_result))

        # Check for toxicity
        toxicity_result = self.toxicity_filter.check(response)
        results.append(("toxicity", toxicity_result))

        # Check grounding
        grounding_result = self.grounding_validator.validate(response, sources)
        results.append(("grounding", grounding_result))

        # Combine results
        combined_result = self._combine_results(results)

        # Record in history
        self.check_history.append(
            {
                "response": response[:100],
                "query": query,
                "is_safe": combined_result.is_safe,
                "risk_level": combined_result.risk_level,
            }
        )

        return combined_result

    def _combine_results(self, results: List[Tuple[str, SafetyCheckResult]]) -> SafetyCheckResult:
        """Combine individual safety check results."""
        all_issues = []
        all_suggestions = []
        avg_risk = 0.0

        for check_name, result in results:
            all_issues.extend([f"[{check_name}] {issue}" for issue in result.issues])
            all_suggestions.extend(result.suggestions)
            avg_risk += result.risk_level

        avg_risk = avg_risk / len(results) if results else 0.0
        is_safe = avg_risk < 0.5

        return SafetyCheckResult(
            is_safe=is_safe,
            risk_level=avg_risk,
            issues=all_issues,
            suggestions=list(set(all_suggestions)),
        )

    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety check statistics."""
        if not self.check_history:
            return {}

        total = len(self.check_history)
        safe_count = sum(1 for r in self.check_history if r["is_safe"])
        avg_risk = sum(r["risk_level"] for r in self.check_history) / total

        return {
            "total_checks": total,
            "safe_rate": safe_count / total,
            "avg_risk_level": avg_risk,
        }
