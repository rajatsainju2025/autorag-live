"""
Response Synthesis Module for Multi-Source Answer Fusion.

Implements advanced response synthesis techniques for combining
information from multiple sources into coherent, accurate answers.

Key Features:
1. Multi-source evidence fusion
2. Answer aggregation strategies
3. Conflict resolution between sources
4. Confidence-weighted synthesis
5. Summary generation

References:
- FiD (Fusion-in-Decoder): Izacard & Grave, 2021
- RAG-Sequence/RAG-Token: Lewis et al., 2020

Example:
    >>> synthesizer = ResponseSynthesizer(llm)
    >>> answer = await synthesizer.synthesize(query, documents)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols and Interfaces
# =============================================================================


class LLMProtocol(Protocol):
    """Protocol for LLM interface."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response from prompt."""
        ...


class EmbedderProtocol(Protocol):
    """Protocol for embedder interface."""

    async def embed(self, text: str) -> List[float]:
        """Get embedding for text."""
        ...


# =============================================================================
# Data Structures
# =============================================================================


class SynthesisStrategy(str, Enum):
    """Strategy for synthesizing responses."""

    CONCAT = "concat"  # Concatenate all sources
    SUMMARIZE = "summarize"  # Summarize sources
    AGGREGATE = "aggregate"  # Aggregate key points
    FUSION = "fusion"  # Fusion-in-decoder style
    WEIGHTED = "weighted"  # Confidence-weighted fusion
    CONSENSUS = "consensus"  # Find consensus across sources


@dataclass
class SourceDocument:
    """
    A source document for synthesis.

    Attributes:
        id: Document identifier
        content: Document content
        score: Relevance score
        metadata: Additional metadata
        claims: Extracted claims
    """

    id: str
    content: str
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    claims: List[str] = field(default_factory=list)

    @property
    def normalized_score(self) -> float:
        """Get score normalized to 0-1."""
        return min(max(self.score, 0.0), 1.0)


@dataclass
class SynthesizedAnswer:
    """
    A synthesized answer from multiple sources.

    Attributes:
        answer: The generated answer text
        sources: Source documents used
        confidence: Overall confidence score
        strategy: Synthesis strategy used
        key_points: Extracted key points
        conflicts: Identified conflicts between sources
        metadata: Additional metadata
    """

    answer: str
    sources: List[SourceDocument]
    confidence: float = 1.0
    strategy: SynthesisStrategy = SynthesisStrategy.FUSION
    key_points: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "strategy": self.strategy.value,
            "source_count": len(self.sources),
            "key_points": self.key_points,
            "has_conflicts": len(self.conflicts) > 0,
        }


@dataclass
class ConflictResolution:
    """
    Result of conflict resolution.

    Attributes:
        resolved: Whether conflict was resolved
        resolution: The resolved value
        conflicting_claims: Original conflicting claims
        confidence: Resolution confidence
    """

    resolved: bool
    resolution: str
    conflicting_claims: List[str]
    confidence: float = 0.5
    method: str = "voting"


# =============================================================================
# Claim Extraction
# =============================================================================


class ClaimExtractor:
    """Extracts key claims/facts from text."""

    def __init__(self, llm: Optional[LLMProtocol] = None):
        """Initialize extractor."""
        self.llm = llm

    async def extract(self, text: str) -> List[str]:
        """
        Extract claims from text.

        Args:
            text: Source text

        Returns:
            List of extracted claims
        """
        if self.llm is None:
            return self._heuristic_extract(text)

        prompt = f"""Extract the key factual claims from this text.
List each claim on a separate line, starting with a dash (-).

Text:
{text[:2000]}

Claims:"""

        try:
            response = await self.llm.generate(prompt)
            return self._parse_claims(response)
        except Exception as e:
            logger.warning(f"LLM claim extraction failed: {e}")
            return self._heuristic_extract(text)

    def _parse_claims(self, response: str) -> List[str]:
        """Parse claims from LLM response."""
        claims = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                claim = line[1:].strip()
                if claim:
                    claims.append(claim)
        return claims

    def _heuristic_extract(self, text: str) -> List[str]:
        """Extract claims using heuristics."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            # Filter for likely factual statements
            if len(sentence) > 20 and len(sentence) < 200:
                claims.append(sentence)

        return claims[:10]  # Limit to 10 claims


# =============================================================================
# Conflict Detection and Resolution
# =============================================================================


class ConflictDetector:
    """Detects conflicts between source claims."""

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        embedder: Optional[EmbedderProtocol] = None,
    ):
        """Initialize detector."""
        self.llm = llm
        self.embedder = embedder

    async def detect_conflicts(
        self,
        claims_by_source: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between sources.

        Args:
            claims_by_source: Claims grouped by source ID

        Returns:
            List of detected conflicts
        """
        all_claims = []
        for source_id, claims in claims_by_source.items():
            for claim in claims:
                all_claims.append({"source": source_id, "claim": claim})

        conflicts = []

        # Pairwise comparison
        for i, c1 in enumerate(all_claims):
            for c2 in all_claims[i + 1 :]:
                if c1["source"] == c2["source"]:
                    continue

                is_conflict = await self._check_conflict(c1["claim"], c2["claim"])

                if is_conflict:
                    conflicts.append(
                        {
                            "source1": c1["source"],
                            "claim1": c1["claim"],
                            "source2": c2["source"],
                            "claim2": c2["claim"],
                        }
                    )

        return conflicts

    async def _check_conflict(
        self,
        claim1: str,
        claim2: str,
    ) -> bool:
        """Check if two claims conflict."""
        if self.llm is None:
            return self._heuristic_conflict_check(claim1, claim2)

        prompt = f"""Do these two claims contradict each other?

Claim 1: {claim1}
Claim 2: {claim2}

Answer only YES or NO:"""

        try:
            response = await self.llm.generate(prompt)
            return "yes" in response.lower()
        except Exception:
            return self._heuristic_conflict_check(claim1, claim2)

    def _heuristic_conflict_check(
        self,
        claim1: str,
        claim2: str,
    ) -> bool:
        """Check conflict using heuristics."""
        # Simple negation detection
        negations = ["not", "no", "never", "neither", "none"]

        c1_has_neg = any(n in claim1.lower() for n in negations)
        c2_has_neg = any(n in claim2.lower() for n in negations)

        # If claims are similar but one has negation
        if c1_has_neg != c2_has_neg:
            # Check word overlap
            words1 = set(claim1.lower().split())
            words2 = set(claim2.lower().split())
            overlap = len(words1 & words2) / max(len(words1 | words2), 1)

            if overlap > 0.5:
                return True

        return False


class ConflictResolver:
    """Resolves conflicts between sources."""

    def __init__(self, llm: Optional[LLMProtocol] = None):
        """Initialize resolver."""
        self.llm = llm

    async def resolve(
        self,
        conflict: Dict[str, Any],
        source_scores: Dict[str, float],
    ) -> ConflictResolution:
        """
        Resolve a conflict between sources.

        Args:
            conflict: Conflict details
            source_scores: Relevance scores by source

        Returns:
            ConflictResolution result
        """
        claim1 = conflict["claim1"]
        claim2 = conflict["claim2"]
        source1 = conflict["source1"]
        source2 = conflict["source2"]

        # Weighted voting based on source scores
        score1 = source_scores.get(source1, 0.5)
        score2 = source_scores.get(source2, 0.5)

        if score1 > score2 * 1.5:  # Significant difference
            return ConflictResolution(
                resolved=True,
                resolution=claim1,
                conflicting_claims=[claim1, claim2],
                confidence=score1 / (score1 + score2),
                method="score_voting",
            )
        elif score2 > score1 * 1.5:
            return ConflictResolution(
                resolved=True,
                resolution=claim2,
                conflicting_claims=[claim1, claim2],
                confidence=score2 / (score1 + score2),
                method="score_voting",
            )

        # Use LLM for ambiguous cases
        if self.llm:
            return await self._llm_resolve(claim1, claim2)

        # Default to noting uncertainty
        return ConflictResolution(
            resolved=False,
            resolution=f"Sources disagree: '{claim1}' vs '{claim2}'",
            conflicting_claims=[claim1, claim2],
            confidence=0.3,
            method="unresolved",
        )

    async def _llm_resolve(
        self,
        claim1: str,
        claim2: str,
    ) -> ConflictResolution:
        """Use LLM to resolve conflict."""
        prompt = f"""Two sources provide conflicting information:

Source 1: {claim1}
Source 2: {claim2}

Based on general knowledge, which claim is more likely to be accurate?
Explain briefly and state the more reliable claim."""

        try:
            response = await self.llm.generate(prompt)
            return ConflictResolution(
                resolved=True,
                resolution=response[:500],
                conflicting_claims=[claim1, claim2],
                confidence=0.6,
                method="llm_reasoning",
            )
        except Exception as e:
            logger.warning(f"LLM conflict resolution failed: {e}")
            return ConflictResolution(
                resolved=False,
                resolution=f"Conflicting: {claim1} | {claim2}",
                conflicting_claims=[claim1, claim2],
                confidence=0.3,
                method="failed",
            )


# =============================================================================
# Synthesis Strategies
# =============================================================================


class SynthesisStrategyBase(ABC):
    """Base class for synthesis strategies."""

    @abstractmethod
    async def synthesize(
        self,
        query: str,
        documents: List[SourceDocument],
        **kwargs: Any,
    ) -> str:
        """Synthesize answer from documents."""
        pass


class ConcatStrategy(SynthesisStrategyBase):
    """Simple concatenation strategy."""

    async def synthesize(
        self,
        query: str,
        documents: List[SourceDocument],
        **kwargs: Any,
    ) -> str:
        """Concatenate document contents."""
        parts = []
        for i, doc in enumerate(documents, 1):
            parts.append(f"[Source {i}] {doc.content[:500]}")
        return "\n\n".join(parts)


class SummarizeStrategy(SynthesisStrategyBase):
    """Summarization-based synthesis."""

    def __init__(self, llm: LLMProtocol):
        """Initialize strategy."""
        self.llm = llm

    async def synthesize(
        self,
        query: str,
        documents: List[SourceDocument],
        **kwargs: Any,
    ) -> str:
        """Summarize documents to answer query."""
        context = "\n\n".join(
            [f"Source {i+1}: {doc.content[:800]}" for i, doc in enumerate(documents[:5])]
        )

        prompt = f"""Based on the following sources, provide a comprehensive answer to the question.

Question: {query}

Sources:
{context}

Answer (be thorough but concise):"""

        return await self.llm.generate(prompt)


class AggregateStrategy(SynthesisStrategyBase):
    """Key point aggregation strategy."""

    def __init__(self, llm: LLMProtocol):
        """Initialize strategy."""
        self.llm = llm
        self.claim_extractor = ClaimExtractor(llm)

    async def synthesize(
        self,
        query: str,
        documents: List[SourceDocument],
        **kwargs: Any,
    ) -> str:
        """Aggregate key points from documents."""
        # Extract claims from each document
        all_claims = []
        for doc in documents[:5]:
            claims = await self.claim_extractor.extract(doc.content)
            doc.claims = claims
            all_claims.extend(claims)

        # Deduplicate and format
        unique_claims = list(set(all_claims))[:15]

        prompt = f"""Based on these key points from multiple sources, answer the question.

Question: {query}

Key Points:
{chr(10).join(f"- {c}" for c in unique_claims)}

Synthesize a coherent answer:"""

        return await self.llm.generate(prompt)


class FusionStrategy(SynthesisStrategyBase):
    """Fusion-in-Decoder style synthesis."""

    def __init__(self, llm: LLMProtocol):
        """Initialize strategy."""
        self.llm = llm

    async def synthesize(
        self,
        query: str,
        documents: List[SourceDocument],
        **kwargs: Any,
    ) -> str:
        """FiD-style synthesis with parallel context."""
        # Format each document as a separate context
        contexts = []
        for i, doc in enumerate(documents[:5]):
            contexts.append(f"[{i+1}] {doc.content[:600]}")

        prompt = f"""Answer the question using ALL the provided contexts.
Each context may contain relevant information. Synthesize them into a coherent answer.

Question: {query}

Contexts:
{chr(10).join(contexts)}

Comprehensive Answer:"""

        return await self.llm.generate(prompt)


class WeightedStrategy(SynthesisStrategyBase):
    """Confidence-weighted synthesis."""

    def __init__(self, llm: LLMProtocol):
        """Initialize strategy."""
        self.llm = llm

    async def synthesize(
        self,
        query: str,
        documents: List[SourceDocument],
        **kwargs: Any,
    ) -> str:
        """Weighted synthesis based on relevance scores."""
        # Sort by score
        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)

        # Weighted context (more content from higher-scored docs)
        contexts = []
        for i, doc in enumerate(sorted_docs[:5]):
            max_len = int(800 * doc.normalized_score)
            contexts.append(f"[Relevance: {doc.score:.2f}] {doc.content[:max_len]}")

        prompt = f"""Answer the question, giving more weight to higher-relevance sources.

Question: {query}

Sources (ordered by relevance):
{chr(10).join(contexts)}

Answer (prioritize information from more relevant sources):"""

        return await self.llm.generate(prompt)


class ConsensusStrategy(SynthesisStrategyBase):
    """Consensus-based synthesis."""

    def __init__(self, llm: LLMProtocol):
        """Initialize strategy."""
        self.llm = llm
        self.claim_extractor = ClaimExtractor(llm)
        self.conflict_detector = ConflictDetector(llm)

    async def synthesize(
        self,
        query: str,
        documents: List[SourceDocument],
        **kwargs: Any,
    ) -> str:
        """Find consensus across sources."""
        # Extract claims per source
        claims_by_source: Dict[str, List[str]] = {}
        for doc in documents[:5]:
            claims = await self.claim_extractor.extract(doc.content)
            claims_by_source[doc.id] = claims

        # Find common claims (consensus)
        all_claims = []
        for claims in claims_by_source.values():
            all_claims.extend(claims)

        # Simple frequency-based consensus
        claim_counts: Dict[str, int] = {}
        for claim in all_claims:
            claim_lower = claim.lower()
            claim_counts[claim_lower] = claim_counts.get(claim_lower, 0) + 1

        # Get consensus claims (appear in multiple sources)
        consensus_claims = [
            claim for claim, count in claim_counts.items() if count >= 2 or len(documents) == 1
        ][:10]

        prompt = f"""Answer based on information that multiple sources agree on.

Question: {query}

Consensus Information:
{chr(10).join(f"- {c}" for c in consensus_claims) if consensus_claims else "No clear consensus found."}

Provide an answer focusing on well-supported information:"""

        return await self.llm.generate(prompt)


# =============================================================================
# Main Synthesizer
# =============================================================================


class ResponseSynthesizer:
    """
    Main response synthesizer for multi-source fusion.

    Combines information from multiple sources into coherent answers.

    Example:
        >>> synthesizer = ResponseSynthesizer(llm)
        >>> result = await synthesizer.synthesize(
        ...     query="What is machine learning?",
        ...     documents=[doc1, doc2, doc3],
        ...     strategy=SynthesisStrategy.FUSION
        ... )
    """

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        embedder: Optional[EmbedderProtocol] = None,
        default_strategy: SynthesisStrategy = SynthesisStrategy.FUSION,
    ):
        """
        Initialize synthesizer.

        Args:
            llm: Language model for synthesis
            embedder: Embedding model (optional)
            default_strategy: Default synthesis strategy
        """
        self.llm = llm
        self.embedder = embedder
        self.default_strategy = default_strategy

        self.claim_extractor = ClaimExtractor(llm)
        self.conflict_detector = ConflictDetector(llm, embedder)
        self.conflict_resolver = ConflictResolver(llm)

        # Initialize strategies
        self._strategies: Dict[SynthesisStrategy, SynthesisStrategyBase] = {
            SynthesisStrategy.CONCAT: ConcatStrategy(),
        }

        if llm:
            self._strategies[SynthesisStrategy.SUMMARIZE] = SummarizeStrategy(llm)
            self._strategies[SynthesisStrategy.AGGREGATE] = AggregateStrategy(llm)
            self._strategies[SynthesisStrategy.FUSION] = FusionStrategy(llm)
            self._strategies[SynthesisStrategy.WEIGHTED] = WeightedStrategy(llm)
            self._strategies[SynthesisStrategy.CONSENSUS] = ConsensusStrategy(llm)

    async def synthesize(
        self,
        query: str,
        documents: List[SourceDocument],
        *,
        strategy: Optional[SynthesisStrategy] = None,
        detect_conflicts: bool = True,
        resolve_conflicts: bool = True,
    ) -> SynthesizedAnswer:
        """
        Synthesize answer from multiple sources.

        Args:
            query: User query
            documents: Source documents
            strategy: Synthesis strategy
            detect_conflicts: Whether to detect conflicts
            resolve_conflicts: Whether to resolve conflicts

        Returns:
            SynthesizedAnswer with synthesized response
        """
        strategy = strategy or self.default_strategy

        if not documents:
            return SynthesizedAnswer(
                answer="No sources available to answer the question.",
                sources=[],
                confidence=0.0,
                strategy=strategy,
            )

        # Get strategy implementation
        strategy_impl = self._strategies.get(strategy)
        if strategy_impl is None:
            strategy_impl = self._strategies.get(SynthesisStrategy.CONCAT)

        # Extract claims for conflict detection
        conflicts = []
        if detect_conflicts and self.llm:
            claims_by_source = {}
            for doc in documents[:5]:
                claims = await self.claim_extractor.extract(doc.content)
                claims_by_source[doc.id] = claims
                doc.claims = claims

            conflicts = await self.conflict_detector.detect_conflicts(claims_by_source)

            # Resolve conflicts if requested
            if resolve_conflicts and conflicts:
                source_scores = {d.id: d.score for d in documents}
                for conflict in conflicts:
                    resolution = await self.conflict_resolver.resolve(conflict, source_scores)
                    conflict["resolution"] = resolution

        # Generate answer
        answer = await strategy_impl.synthesize(query, documents)

        # Extract key points
        key_points = []
        for doc in documents[:3]:
            key_points.extend(doc.claims[:3])

        # Calculate confidence
        avg_score = sum(d.score for d in documents) / len(documents)
        conflict_penalty = 0.1 * len(conflicts) if conflicts else 0
        confidence = max(0.1, avg_score - conflict_penalty)

        return SynthesizedAnswer(
            answer=answer,
            sources=documents,
            confidence=confidence,
            strategy=strategy,
            key_points=list(set(key_points))[:10],
            conflicts=conflicts,
        )

    async def synthesize_with_citations(
        self,
        query: str,
        documents: List[SourceDocument],
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Synthesize answer with inline citations.

        Args:
            query: User query
            documents: Source documents

        Returns:
            Tuple of (answer with citations, citation list)
        """
        if not self.llm:
            # Simple fallback
            result = await self.synthesize(query, documents)
            return result.answer, []

        # Format with citation markers
        context = "\n\n".join(
            [f"[{i+1}] {doc.content[:600]}" for i, doc in enumerate(documents[:5])]
        )

        prompt = f"""Answer the question using the sources. Include citations like [1], [2] etc.

Question: {query}

Sources:
{context}

Answer with citations:"""

        answer = await self.llm.generate(prompt)

        # Build citation list
        citations = [
            {"id": str(i + 1), "source": doc.id, "content": doc.content[:200]}
            for i, doc in enumerate(documents[:5])
        ]

        return answer, citations


# =============================================================================
# Convenience Functions
# =============================================================================


def create_synthesizer(
    llm: Optional[LLMProtocol] = None,
    strategy: SynthesisStrategy = SynthesisStrategy.FUSION,
) -> ResponseSynthesizer:
    """
    Create a response synthesizer.

    Args:
        llm: Language model
        strategy: Default strategy

    Returns:
        ResponseSynthesizer instance
    """
    return ResponseSynthesizer(llm=llm, default_strategy=strategy)


async def synthesize_answer(
    query: str,
    documents: List[Dict[str, Any]],
    llm: Optional[LLMProtocol] = None,
) -> str:
    """
    Quick synthesis of answer from documents.

    Args:
        query: User query
        documents: Document dicts with 'content' key

    Returns:
        Synthesized answer string
    """
    source_docs = [
        SourceDocument(
            id=str(i),
            content=doc.get("content", str(doc)),
            score=doc.get("score", 1.0),
        )
        for i, doc in enumerate(documents)
    ]

    synthesizer = ResponseSynthesizer(llm=llm)
    result = await synthesizer.synthesize(query, source_docs)
    return result.answer
