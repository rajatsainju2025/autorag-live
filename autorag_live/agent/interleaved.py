"""
Interleaved Retrieval-Generation (IRG) Implementation.

Implements fine-grained interleaving of retrieval and generation
for improved factuality and coherence.

Key features:
1. Generate sentence by sentence
2. Evaluate each sentence for factual support
3. Retrieve evidence for unsupported claims
4. Revise or confirm sentences based on evidence
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# ============================================================================
# Protocols and Types
# ============================================================================


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM providers."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for retrieval backends."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list["RetrievedDocument"]:
        """Retrieve documents for query."""
        ...


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""

    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Sentence Analysis Types
# ============================================================================


class SupportLevel(Enum):
    """Level of evidence support for a claim."""

    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NO_SUPPORT = "no_support"
    CONTRADICTED = "contradicted"
    NOT_A_CLAIM = "not_a_claim"  # For non-factual sentences


class SentenceType(Enum):
    """Type of sentence generated."""

    FACTUAL_CLAIM = "factual_claim"
    OPINION = "opinion"
    REASONING = "reasoning"
    TRANSITION = "transition"
    CONCLUSION = "conclusion"


@dataclass
class AnalyzedSentence:
    """A sentence with factual analysis."""

    text: str
    sentence_type: SentenceType
    support_level: SupportLevel
    supporting_docs: list[RetrievedDocument]
    confidence: float
    revision: str | None = None
    needs_retrieval: bool = False


@dataclass
class GenerationState:
    """Current state of interleaved generation."""

    query: str
    context: str
    sentences: list[AnalyzedSentence]
    all_retrieved_docs: list[RetrievedDocument]
    iteration: int = 0


# ============================================================================
# Sentence Analyzers
# ============================================================================


class SentenceAnalyzer(ABC):
    """Abstract base for sentence analysis."""

    @abstractmethod
    async def analyze(
        self,
        sentence: str,
        query: str,
        context: str,
    ) -> tuple[SentenceType, bool]:
        """
        Analyze sentence type and determine if retrieval needed.

        Returns:
            Tuple of (sentence_type, needs_retrieval)
        """
        ...


class LLMSentenceAnalyzer(SentenceAnalyzer):
    """LLM-based sentence analyzer."""

    ANALYSIS_PROMPT = """Analyze this sentence in the context of answering a question.

Question: {query}
Sentence: {sentence}

1. What type of sentence is this?
- FACTUAL_CLAIM: Makes a verifiable factual statement
- OPINION: Expresses an opinion or subjective view
- REASONING: Logical reasoning or inference
- TRANSITION: Connecting text
- CONCLUSION: Summarizing conclusion

2. Does this sentence contain claims that should be verified with external sources?
Answer: YES or NO

Analysis:
Type: """

    def __init__(self, llm: LLMProtocol):
        """Initialize with LLM provider."""
        self.llm = llm

    async def analyze(
        self,
        sentence: str,
        query: str,
        context: str,
    ) -> tuple[SentenceType, bool]:
        """Analyze sentence using LLM."""
        prompt = self.ANALYSIS_PROMPT.format(
            query=query,
            sentence=sentence,
        )

        response = await self.llm.generate(prompt, max_tokens=100, temperature=0.0)

        # Parse response
        sentence_type = self._parse_type(response)
        needs_retrieval = (
            "YES" in response.upper()
            and "SHOULD BE VERIFIED" in response.upper()
            or sentence_type == SentenceType.FACTUAL_CLAIM
        )

        return sentence_type, needs_retrieval

    def _parse_type(self, response: str) -> SentenceType:
        """Parse sentence type from response."""
        response_upper = response.upper()

        if "FACTUAL_CLAIM" in response_upper:
            return SentenceType.FACTUAL_CLAIM
        elif "OPINION" in response_upper:
            return SentenceType.OPINION
        elif "REASONING" in response_upper:
            return SentenceType.REASONING
        elif "TRANSITION" in response_upper:
            return SentenceType.TRANSITION
        elif "CONCLUSION" in response_upper:
            return SentenceType.CONCLUSION
        else:
            return SentenceType.FACTUAL_CLAIM  # Default to factual


class HeuristicSentenceAnalyzer(SentenceAnalyzer):
    """Fast heuristic-based sentence analyzer."""

    FACTUAL_INDICATORS = [
        r"\d{4}",  # Years
        r"\d+%",  # Percentages
        r"\$\d+",  # Dollar amounts
        r"according to",
        r"studies show",
        r"research indicates",
        r"is defined as",
        r"was founded",
        r"was born",
        r"discovered",
        r"invented",
    ]

    OPINION_INDICATORS = [
        "i think",
        "i believe",
        "in my opinion",
        "seems",
        "appears",
        "might",
        "could",
        "should",
        "probably",
    ]

    async def analyze(
        self,
        sentence: str,
        query: str,
        context: str,
    ) -> tuple[SentenceType, bool]:
        """Analyze sentence using heuristics."""
        import re

        sentence_lower = sentence.lower()

        # Check for opinion indicators
        for indicator in self.OPINION_INDICATORS:
            if indicator in sentence_lower:
                return SentenceType.OPINION, False

        # Check for factual indicators
        for pattern in self.FACTUAL_INDICATORS:
            if re.search(pattern, sentence_lower):
                return SentenceType.FACTUAL_CLAIM, True

        # Check for transition words
        transition_words = ["however", "therefore", "furthermore", "additionally", "moreover"]
        if any(sentence_lower.startswith(w) for w in transition_words):
            return SentenceType.TRANSITION, False

        # Check for conclusion
        conclusion_words = ["in conclusion", "to summarize", "in summary", "overall"]
        if any(w in sentence_lower for w in conclusion_words):
            return SentenceType.CONCLUSION, False

        # Default to factual claim needing verification
        return SentenceType.FACTUAL_CLAIM, True


# ============================================================================
# Evidence Verifier
# ============================================================================


class EvidenceVerifier:
    """Verifies claims against retrieved evidence."""

    VERIFICATION_PROMPT = """Does the following evidence support the claim?

Claim: {claim}

Evidence:
{evidence}

Assessment:
- FULLY_SUPPORTED: Evidence directly confirms the claim
- PARTIALLY_SUPPORTED: Evidence partially confirms or is related
- NO_SUPPORT: Evidence doesn't address the claim
- CONTRADICTED: Evidence contradicts the claim

Your assessment (one of the above):"""

    def __init__(self, llm: LLMProtocol):
        """Initialize with LLM provider."""
        self.llm = llm

    async def verify(
        self,
        claim: str,
        documents: list[RetrievedDocument],
    ) -> tuple[SupportLevel, float]:
        """
        Verify claim against documents.

        Returns:
            Tuple of (support_level, confidence)
        """
        if not documents:
            return SupportLevel.NO_SUPPORT, 0.0

        evidence_text = "\n".join(
            f"[{i}] {doc.content[:300]}" for i, doc in enumerate(documents, 1)
        )

        prompt = self.VERIFICATION_PROMPT.format(
            claim=claim,
            evidence=evidence_text,
        )

        response = await self.llm.generate(prompt, max_tokens=50, temperature=0.0)

        support_level = self._parse_support(response)
        confidence = self._estimate_confidence(support_level, documents)

        return support_level, confidence

    def _parse_support(self, response: str) -> SupportLevel:
        """Parse support level from response."""
        response_upper = response.upper()

        if "FULLY_SUPPORTED" in response_upper:
            return SupportLevel.FULLY_SUPPORTED
        elif "PARTIALLY_SUPPORTED" in response_upper:
            return SupportLevel.PARTIALLY_SUPPORTED
        elif "CONTRADICTED" in response_upper:
            return SupportLevel.CONTRADICTED
        else:
            return SupportLevel.NO_SUPPORT

    def _estimate_confidence(
        self,
        support: SupportLevel,
        docs: list[RetrievedDocument],
    ) -> float:
        """Estimate confidence based on support and doc scores."""
        if not docs:
            return 0.0

        # Base confidence from support level
        base_confidence = {
            SupportLevel.FULLY_SUPPORTED: 0.9,
            SupportLevel.PARTIALLY_SUPPORTED: 0.6,
            SupportLevel.NO_SUPPORT: 0.3,
            SupportLevel.CONTRADICTED: 0.1,
            SupportLevel.NOT_A_CLAIM: 0.5,
        }

        base = base_confidence.get(support, 0.5)

        # Adjust by document scores
        avg_score = sum(d.score for d in docs) / len(docs)

        return min(1.0, base * (0.5 + avg_score * 0.5))


# ============================================================================
# Sentence Reviser
# ============================================================================


class SentenceReviser:
    """Revises sentences based on evidence."""

    REVISION_PROMPT = """Revise this sentence to be accurate based on the evidence.

Original sentence: {sentence}

Evidence:
{evidence}

Support assessment: {support_level}

If the sentence is supported, keep it largely unchanged.
If not supported, revise to be accurate or add appropriate hedging.
If contradicted, correct the factual errors.

Revised sentence:"""

    def __init__(self, llm: LLMProtocol):
        """Initialize with LLM provider."""
        self.llm = llm

    async def revise(
        self,
        sentence: str,
        documents: list[RetrievedDocument],
        support_level: SupportLevel,
    ) -> str:
        """Revise sentence based on evidence."""
        if support_level == SupportLevel.FULLY_SUPPORTED:
            return sentence  # No revision needed

        evidence_text = "\n".join(
            f"[{i}] {doc.content[:300]}" for i, doc in enumerate(documents, 1)
        )

        prompt = self.REVISION_PROMPT.format(
            sentence=sentence,
            evidence=evidence_text,
            support_level=support_level.value,
        )

        revised = await self.llm.generate(prompt, max_tokens=150, temperature=0.3)
        return revised.strip()


# ============================================================================
# Interleaved Generator
# ============================================================================


@dataclass
class IRGConfig:
    """Configuration for Interleaved Retrieval-Generation."""

    max_sentences: int = 10
    top_k_retrieval: int = 3
    verify_all_claims: bool = True
    revise_unsupported: bool = True
    min_confidence: float = 0.5
    use_heuristic_analyzer: bool = False


@dataclass
class IRGResult:
    """Result from interleaved generation."""

    answer: str
    sentences: list[AnalyzedSentence]
    retrieval_count: int
    revision_count: int
    overall_confidence: float
    all_docs: list[RetrievedDocument]


class InterleavedGenerator:
    """
    Interleaved Retrieval-Generation (IRG).

    Generates sentence by sentence with factual verification.
    """

    GENERATION_PROMPT = """Generate the next sentence to answer this question.

Question: {query}

Context:
{context}

Answer so far:
{answer_so_far}

Generate the next sentence (one sentence only):"""

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol,
        config: IRGConfig | None = None,
    ):
        """
        Initialize interleaved generator.

        Args:
            llm: LLM provider
            retriever: Retrieval backend
            config: Generation configuration
        """
        self.llm = llm
        self.retriever = retriever
        self.config = config or IRGConfig()

        # Initialize components
        if self.config.use_heuristic_analyzer:
            self.analyzer = HeuristicSentenceAnalyzer()
        else:
            self.analyzer = LLMSentenceAnalyzer(llm)

        self.verifier = EvidenceVerifier(llm)
        self.reviser = SentenceReviser(llm)

    async def generate(
        self,
        query: str,
        initial_context: str = "",
    ) -> IRGResult:
        """
        Generate answer with interleaved retrieval.

        Args:
            query: User query
            initial_context: Initial context (optional)

        Returns:
            IRGResult with verified answer
        """
        state = GenerationState(
            query=query,
            context=initial_context,
            sentences=[],
            all_retrieved_docs=[],
        )

        retrieval_count = 0
        revision_count = 0

        for _ in range(self.config.max_sentences):
            # Generate next sentence
            sentence_text = await self._generate_sentence(state)

            if not sentence_text:
                break

            # Analyze sentence
            sentence_type, needs_retrieval = await self.analyzer.analyze(
                sentence_text, query, state.context
            )

            # Initialize analyzed sentence
            analyzed = AnalyzedSentence(
                text=sentence_text,
                sentence_type=sentence_type,
                support_level=SupportLevel.NOT_A_CLAIM,
                supporting_docs=[],
                confidence=1.0,
                needs_retrieval=needs_retrieval,
            )

            # Retrieve and verify if needed
            if needs_retrieval and self.config.verify_all_claims:
                docs = await self.retriever.retrieve(
                    sentence_text, top_k=self.config.top_k_retrieval
                )
                retrieval_count += 1

                analyzed.supporting_docs = docs
                state.all_retrieved_docs.extend(docs)

                # Verify claim
                support_level, confidence = await self.verifier.verify(sentence_text, docs)
                analyzed.support_level = support_level
                analyzed.confidence = confidence

                # Revise if needed
                if self.config.revise_unsupported:
                    if support_level in [
                        SupportLevel.NO_SUPPORT,
                        SupportLevel.CONTRADICTED,
                    ]:
                        revised = await self.reviser.revise(sentence_text, docs, support_level)
                        analyzed.revision = revised
                        analyzed.text = revised
                        revision_count += 1

                # Update context with retrieved docs
                state.context = self._update_context(state.context, docs)

            state.sentences.append(analyzed)

            # Check for completion
            if self._is_complete(sentence_text):
                break

        # Calculate overall confidence
        confidences = [s.confidence for s in state.sentences]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return IRGResult(
            answer=" ".join(s.text for s in state.sentences),
            sentences=state.sentences,
            retrieval_count=retrieval_count,
            revision_count=revision_count,
            overall_confidence=overall_confidence,
            all_docs=state.all_retrieved_docs,
        )

    async def _generate_sentence(self, state: GenerationState) -> str:
        """Generate the next sentence."""
        answer_so_far = " ".join(s.text for s in state.sentences)

        prompt = self.GENERATION_PROMPT.format(
            query=state.query,
            context=state.context[:2000],
            answer_so_far=answer_so_far,
        )

        response = await self.llm.generate(prompt, max_tokens=100, temperature=0.3)

        # Extract first sentence
        sentence = self._extract_sentence(response)
        return sentence

    def _extract_sentence(self, text: str) -> str:
        """Extract first complete sentence."""
        text = text.strip()

        # Find sentence boundary
        for end_char in [".", "!", "?"]:
            idx = text.find(end_char)
            if idx != -1:
                return text[: idx + 1].strip()

        # Return full text if no boundary found
        return text[:200] if len(text) > 200 else text

    def _update_context(
        self,
        context: str,
        docs: list[RetrievedDocument],
    ) -> str:
        """Update context with new documents."""
        new_context = "\n".join(d.content[:300] for d in docs)
        return f"{context}\n\nRetrieved Information:\n{new_context}"

    def _is_complete(self, sentence: str) -> bool:
        """Check if generation should stop."""
        completion_phrases = [
            "in conclusion",
            "to summarize",
            "in summary",
            "overall",
            "finally",
        ]
        return any(phrase in sentence.lower() for phrase in completion_phrases)


# ============================================================================
# Parallel Interleaved Generation
# ============================================================================


class ParallelIRG:
    """
    Parallel Interleaved Generation.

    Generates multiple candidate sentences and selects best.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol,
        num_candidates: int = 3,
        config: IRGConfig | None = None,
    ):
        """Initialize parallel IRG."""
        self.llm = llm
        self.retriever = retriever
        self.num_candidates = num_candidates
        self.config = config or IRGConfig()
        self.verifier = EvidenceVerifier(llm)

    async def generate(
        self,
        query: str,
        initial_context: str = "",
    ) -> IRGResult:
        """Generate with parallel candidate selection."""
        sentences: list[AnalyzedSentence] = []
        all_docs: list[RetrievedDocument] = []
        retrieval_count = 0
        context = initial_context

        for _ in range(self.config.max_sentences):
            # Generate multiple candidates
            candidates = await self._generate_candidates(query, context, sentences)

            if not candidates:
                break

            # Retrieve evidence
            all_candidate_docs: list[RetrievedDocument] = []
            for candidate in candidates:
                docs = await self.retriever.retrieve(candidate, top_k=self.config.top_k_retrieval)
                all_candidate_docs.extend(docs)
            retrieval_count += 1

            # Deduplicate docs
            seen_content: set[str] = set()
            unique_docs = []
            for doc in all_candidate_docs:
                if doc.content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.content)

            # Score each candidate
            best_candidate = None
            best_score = -1.0

            for candidate in candidates:
                support, confidence = await self.verifier.verify(candidate, unique_docs)
                score = self._calculate_score(support, confidence)

                if score > best_score:
                    best_score = score
                    best_candidate = AnalyzedSentence(
                        text=candidate,
                        sentence_type=SentenceType.FACTUAL_CLAIM,
                        support_level=support,
                        supporting_docs=unique_docs,
                        confidence=confidence,
                    )

            if best_candidate:
                sentences.append(best_candidate)
                all_docs.extend(unique_docs)
                context = self._update_context(context, unique_docs)

            # Check completion
            if best_candidate and self._is_complete(best_candidate.text):
                break

        confidences = [s.confidence for s in sentences]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return IRGResult(
            answer=" ".join(s.text for s in sentences),
            sentences=sentences,
            retrieval_count=retrieval_count,
            revision_count=0,
            overall_confidence=overall_confidence,
            all_docs=all_docs,
        )

    async def _generate_candidates(
        self,
        query: str,
        context: str,
        sentences: list[AnalyzedSentence],
    ) -> list[str]:
        """Generate multiple candidate sentences."""
        answer_so_far = " ".join(s.text for s in sentences)

        prompt = f"""Generate {self.num_candidates} different next sentences to continue this answer.

Question: {query}
Context: {context[:1000]}
Answer so far: {answer_so_far}

Generate {self.num_candidates} different options (one per line):"""

        response = await self.llm.generate(prompt, max_tokens=300, temperature=0.7)

        # Parse candidates
        candidates = []
        for line in response.split("\n"):
            line = line.strip()
            if line and not line.startswith("Option"):
                # Remove numbering
                if line[0].isdigit():
                    line = line.split(".", 1)[-1].strip()
                    line = line.split(")", 1)[-1].strip()
                if line:
                    candidates.append(line)

        return candidates[: self.num_candidates]

    def _calculate_score(
        self,
        support: SupportLevel,
        confidence: float,
    ) -> float:
        """Calculate overall score for a candidate."""
        support_weights = {
            SupportLevel.FULLY_SUPPORTED: 1.0,
            SupportLevel.PARTIALLY_SUPPORTED: 0.7,
            SupportLevel.NO_SUPPORT: 0.3,
            SupportLevel.CONTRADICTED: 0.0,
            SupportLevel.NOT_A_CLAIM: 0.5,
        }

        support_score = support_weights.get(support, 0.5)
        return support_score * 0.6 + confidence * 0.4

    def _update_context(
        self,
        context: str,
        docs: list[RetrievedDocument],
    ) -> str:
        """Update context with documents."""
        new_info = "\n".join(d.content[:200] for d in docs[:3])
        return f"{context}\n\n{new_info}"

    def _is_complete(self, sentence: str) -> bool:
        """Check if generation is complete."""
        indicators = ["in conclusion", "finally", "to summarize"]
        return any(ind in sentence.lower() for ind in indicators)


# ============================================================================
# Factory Functions
# ============================================================================


def create_interleaved_generator(
    llm: LLMProtocol,
    retriever: RetrieverProtocol,
    verify_all: bool = True,
    revise_unsupported: bool = True,
) -> InterleavedGenerator:
    """
    Create an interleaved generator.

    Args:
        llm: LLM provider
        retriever: Retrieval backend
        verify_all: Verify all factual claims
        revise_unsupported: Revise unsupported claims

    Returns:
        Configured InterleavedGenerator
    """
    config = IRGConfig(
        verify_all_claims=verify_all,
        revise_unsupported=revise_unsupported,
    )
    return InterleavedGenerator(llm, retriever, config)


def create_parallel_irg(
    llm: LLMProtocol,
    retriever: RetrieverProtocol,
    num_candidates: int = 3,
) -> ParallelIRG:
    """
    Create a parallel interleaved generator.

    Args:
        llm: LLM provider
        retriever: Retrieval backend
        num_candidates: Number of candidate sentences

    Returns:
        Configured ParallelIRG
    """
    return ParallelIRG(llm, retriever, num_candidates)


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example demonstrating IRG."""
    # This would use actual implementations
    # llm = OpenAILLM(api_key="...")
    # retriever = VectorStoreRetriever(...)

    # Standard IRG
    # irg = create_interleaved_generator(llm, retriever)
    # result = await irg.generate("What is quantum computing?")

    # Parallel IRG
    # parallel = create_parallel_irg(llm, retriever, num_candidates=3)
    # result = await parallel.generate("Explain photosynthesis")

    print("Interleaved Retrieval-Generation Implementation Ready")
    print("=" * 50)
    print("Features:")
    print("- Sentence-by-sentence generation")
    print("- LLM and heuristic sentence analysis")
    print("- Evidence verification per claim")
    print("- Automatic revision of unsupported claims")
    print("- Parallel candidate generation and selection")
    print("- Confidence tracking throughout")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
