"""
FLARE (Forward-Looking Active REtrieval) Implementation.

Based on: "Active Retrieval Augmented Generation" (Jiang et al., 2023)

FLARE improves RAG by:
1. Generating text until a low-confidence token is detected
2. Using the generated text to formulate a retrieval query
3. Retrieving relevant documents based on upcoming content
4. Continuing generation with the new context

This enables dynamic, on-demand retrieval during generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Protocol, runtime_checkable

# ============================================================================
# Protocols and Types
# ============================================================================


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM providers with logprobs support."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...

    async def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 100,
        **kwargs: Any,
    ) -> "GenerationWithLogprobs":
        """Generate text with token log probabilities."""
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


@dataclass
class TokenWithLogprob:
    """A token with its log probability."""

    token: str
    logprob: float
    top_logprobs: dict[str, float] = field(default_factory=dict)


@dataclass
class GenerationWithLogprobs:
    """Generation result with token-level log probabilities."""

    text: str
    tokens: list[TokenWithLogprob]
    finish_reason: str = ""


# ============================================================================
# FLARE Configuration
# ============================================================================


class FLAREMode(Enum):
    """FLARE triggering modes."""

    CONFIDENCE = "confidence"  # Trigger on low confidence tokens
    SENTENCE = "sentence"  # Trigger at sentence boundaries
    LOOKAHEAD = "lookahead"  # Trigger on uncertain lookahead
    HYBRID = "hybrid"  # Combine multiple triggers


@dataclass
class FLAREConfig:
    """Configuration for FLARE."""

    mode: FLAREMode = FLAREMode.CONFIDENCE
    confidence_threshold: float = 0.5  # Trigger retrieval below this
    max_generation_tokens: int = 50  # Max tokens before forced retrieval
    lookahead_tokens: int = 10  # Tokens to generate for lookahead query
    top_k_retrieval: int = 3
    max_iterations: int = 10
    use_regeneration: bool = True  # Regenerate after retrieval


# ============================================================================
# Confidence Monitoring
# ============================================================================


class ConfidenceMonitor:
    """
    Monitors token-level confidence during generation.

    Detects when the model is uncertain and should retrieve.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        window_size: int = 3,
    ):
        """
        Initialize confidence monitor.

        Args:
            threshold: Confidence threshold for triggering
            window_size: Number of tokens to average for smoothing
        """
        self.threshold = threshold
        self.window_size = window_size
        self._confidence_history: list[float] = []

    def should_retrieve(
        self,
        token: TokenWithLogprob,
    ) -> bool:
        """
        Check if retrieval should be triggered.

        Args:
            token: Token with log probability

        Returns:
            True if retrieval should be triggered
        """
        import math

        # Convert logprob to probability
        prob = math.exp(token.logprob)
        self._confidence_history.append(prob)

        # Check immediate threshold
        if prob < self.threshold:
            return True

        # Check windowed average
        if len(self._confidence_history) >= self.window_size:
            window = self._confidence_history[-self.window_size :]
            avg_prob = sum(window) / len(window)
            if avg_prob < self.threshold:
                return True

        return False

    def get_uncertainty_score(self, token: TokenWithLogprob) -> float:
        """Get uncertainty score for a token (0 = certain, 1 = uncertain)."""
        import math

        prob = math.exp(token.logprob)
        return 1.0 - prob

    def reset(self):
        """Reset confidence history."""
        self._confidence_history = []


class EntropyMonitor:
    """
    Monitors token-level entropy for uncertainty detection.

    High entropy indicates uncertainty across multiple possible tokens.
    """

    def __init__(
        self,
        entropy_threshold: float = 2.0,
    ):
        """Initialize entropy monitor."""
        self.entropy_threshold = entropy_threshold

    def calculate_entropy(self, token: TokenWithLogprob) -> float:
        """Calculate entropy from top logprobs."""
        import math

        if not token.top_logprobs:
            # Use single token logprob
            prob = math.exp(token.logprob)
            return -prob * math.log2(prob + 1e-10)

        entropy = 0.0
        for logprob in token.top_logprobs.values():
            prob = math.exp(logprob)
            entropy -= prob * math.log2(prob + 1e-10)

        return entropy

    def should_retrieve(self, token: TokenWithLogprob) -> bool:
        """Check if entropy exceeds threshold."""
        entropy = self.calculate_entropy(token)
        return entropy > self.entropy_threshold


# ============================================================================
# Query Formulation
# ============================================================================


class QueryFormulator(ABC):
    """Abstract base for formulating retrieval queries."""

    @abstractmethod
    async def formulate_query(
        self,
        generated_text: str,
        original_query: str,
    ) -> str:
        """Formulate a retrieval query from generated text."""
        ...


class LookaheadQueryFormulator(QueryFormulator):
    """
    Formulates queries using lookahead generation.

    Generates a few more tokens to understand what information is needed.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        lookahead_tokens: int = 10,
    ):
        """Initialize lookahead formulator."""
        self.llm = llm
        self.lookahead_tokens = lookahead_tokens

    async def formulate_query(
        self,
        generated_text: str,
        original_query: str,
    ) -> str:
        """Generate lookahead and extract query."""
        # Generate lookahead
        prompt = f"{original_query}\n\n{generated_text}"
        lookahead = await self.llm.generate(
            prompt,
            max_tokens=self.lookahead_tokens,
            temperature=1.0,  # Higher temperature for exploration
        )

        # Use lookahead as query
        # In practice, we might extract key terms or use LLM to reformulate
        return lookahead.strip()


class MaskedQueryFormulator(QueryFormulator):
    """
    Formulates queries by masking uncertain spans.

    Creates queries that target the uncertain information.
    """

    def __init__(self, llm: LLMProtocol):
        """Initialize masked formulator."""
        self.llm = llm

    async def formulate_query(
        self,
        generated_text: str,
        original_query: str,
    ) -> str:
        """Create query by identifying information gap."""
        prompt = f"""Based on the original question and partial answer, identify what specific information is needed next.

Original Question: {original_query}

Partial Answer: {generated_text}

What specific information should be retrieved to continue the answer? Write a search query:"""

        query = await self.llm.generate(prompt, max_tokens=50, temperature=0.3)
        return query.strip()


class DirectQueryFormulator(QueryFormulator):
    """Simple formulator that uses the last sentence as query."""

    async def formulate_query(
        self,
        generated_text: str,
        original_query: str,
    ) -> str:
        """Use last sentence or clause as query."""
        # Split into sentences
        sentences = generated_text.split(".")
        if sentences:
            last_sentence = sentences[-1].strip()
            if len(last_sentence) > 10:
                return last_sentence

        # Fall back to last N words
        words = generated_text.split()
        return " ".join(words[-10:]) if len(words) > 10 else generated_text


# ============================================================================
# FLARE Generator
# ============================================================================


@dataclass
class FLAREResult:
    """Result from FLARE generation."""

    answer: str
    retrieval_count: int
    retrieved_docs: list[RetrievedDocument]
    generation_steps: list[str]
    confidence_scores: list[float]


class FLAREGenerator:
    """
    FLARE (Forward-Looking Active REtrieval) Generator.

    Dynamically retrieves during generation based on confidence.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol,
        config: FLAREConfig | None = None,
        query_formulator: QueryFormulator | None = None,
    ):
        """
        Initialize FLARE generator.

        Args:
            llm: LLM provider with logprobs support
            retriever: Retrieval backend
            config: FLARE configuration
            query_formulator: Strategy for formulating queries
        """
        self.llm = llm
        self.retriever = retriever
        self.config = config or FLAREConfig()
        self.query_formulator = query_formulator or DirectQueryFormulator()
        self.confidence_monitor = ConfidenceMonitor(threshold=self.config.confidence_threshold)

    async def generate(
        self,
        query: str,
        context: str = "",
    ) -> FLAREResult:
        """
        Generate answer with active retrieval.

        Args:
            query: User query
            context: Initial context (optional)

        Returns:
            FLAREResult with answer and retrieval info
        """
        generated_text = ""
        all_retrieved_docs: list[RetrievedDocument] = []
        generation_steps: list[str] = []
        confidence_scores: list[float] = []
        retrieval_count = 0

        # Build initial prompt
        current_context = context

        for iteration in range(self.config.max_iterations):
            # Reset confidence monitor for this iteration
            self.confidence_monitor.reset()

            # Generate with logprobs
            prompt = self._build_prompt(query, current_context, generated_text)
            generation = await self.llm.generate_with_logprobs(
                prompt,
                max_tokens=self.config.max_generation_tokens,
            )

            # Process tokens and check for retrieval trigger
            new_text, trigger_idx, avg_confidence = self._process_generation(generation)

            confidence_scores.append(avg_confidence)

            if trigger_idx is not None:
                # Retrieval triggered - use text up to trigger
                generated_text += new_text[:trigger_idx]
                generation_steps.append(
                    f"Generated: {new_text[:trigger_idx][:50]}... (retrieval triggered)"
                )

                # Formulate query
                retrieval_query = await self.query_formulator.formulate_query(generated_text, query)

                # Retrieve
                docs = await self.retriever.retrieve(
                    retrieval_query, top_k=self.config.top_k_retrieval
                )
                all_retrieved_docs.extend(docs)
                retrieval_count += 1

                generation_steps.append(
                    f"Retrieved {len(docs)} docs for: {retrieval_query[:50]}..."
                )

                # Update context
                new_context = self._format_retrieved_context(docs)
                current_context = f"{current_context}\n\n{new_context}"

                # Optionally regenerate the uncertain part
                if self.config.use_regeneration:
                    continue

            else:
                # No retrieval triggered - append all text
                generated_text += new_text
                generation_steps.append(f"Generated: {new_text[:50]}...")

                # Check for completion
                if generation.finish_reason == "stop":
                    break

        return FLAREResult(
            answer=generated_text.strip(),
            retrieval_count=retrieval_count,
            retrieved_docs=all_retrieved_docs,
            generation_steps=generation_steps,
            confidence_scores=confidence_scores,
        )

    def _build_prompt(
        self,
        query: str,
        context: str,
        generated_so_far: str,
    ) -> str:
        """Build prompt for generation."""
        parts = []

        if context:
            parts.append(f"Context:\n{context}")

        parts.append(f"Question: {query}")

        if generated_so_far:
            parts.append(f"Answer so far: {generated_so_far}")
            parts.append("Continue the answer:")
        else:
            parts.append("Answer:")

        return "\n\n".join(parts)

    def _process_generation(
        self,
        generation: GenerationWithLogprobs,
    ) -> tuple[str, int | None, float]:
        """
        Process generated tokens and find retrieval trigger.

        Returns:
            Tuple of (text, trigger_index, average_confidence)
        """
        import math

        text = ""
        trigger_idx = None
        total_confidence = 0.0

        for i, token in enumerate(generation.tokens):
            text += token.token
            prob = math.exp(token.logprob)
            total_confidence += prob

            if self.confidence_monitor.should_retrieve(token):
                trigger_idx = len(text)
                break

        avg_confidence = total_confidence / len(generation.tokens) if generation.tokens else 0.0

        return text, trigger_idx, avg_confidence

    def _format_retrieved_context(
        self,
        docs: list[RetrievedDocument],
    ) -> str:
        """Format retrieved documents as context."""
        parts = []
        for i, doc in enumerate(docs, 1):
            parts.append(f"[{i}] {doc.content[:500]}")
        return "\n\n".join(parts)


# ============================================================================
# Sentence-Level FLARE
# ============================================================================


class SentenceFLARE:
    """
    Sentence-level FLARE variant.

    Generates sentence by sentence and retrieves at boundaries.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol,
        config: FLAREConfig | None = None,
    ):
        """Initialize sentence-level FLARE."""
        self.llm = llm
        self.retriever = retriever
        self.config = config or FLAREConfig()

    async def generate(
        self,
        query: str,
        context: str = "",
    ) -> FLAREResult:
        """Generate with sentence-level retrieval."""
        sentences: list[str] = []
        all_docs: list[RetrievedDocument] = []
        generation_steps: list[str] = []
        retrieval_count = 0
        current_context = context

        for _ in range(self.config.max_iterations):
            # Generate next sentence
            prompt = self._build_sentence_prompt(query, current_context, sentences)

            response = await self.llm.generate(
                prompt,
                max_tokens=100,
                temperature=0.3,
            )

            sentence = self._extract_sentence(response)
            if not sentence:
                break

            sentences.append(sentence)
            generation_steps.append(f"Generated sentence: {sentence[:50]}...")

            # Check if sentence needs verification/retrieval
            needs_retrieval = await self._needs_retrieval(sentence, query)

            if needs_retrieval:
                docs = await self.retriever.retrieve(sentence, top_k=self.config.top_k_retrieval)
                all_docs.extend(docs)
                retrieval_count += 1

                current_context = f"{current_context}\n\n" f"{self._format_context(docs)}"
                generation_steps.append(f"Retrieved {len(docs)} docs")

            # Check for completion
            if self._is_complete(sentence):
                break

        return FLAREResult(
            answer=" ".join(sentences),
            retrieval_count=retrieval_count,
            retrieved_docs=all_docs,
            generation_steps=generation_steps,
            confidence_scores=[],
        )

    def _build_sentence_prompt(
        self,
        query: str,
        context: str,
        sentences: list[str],
    ) -> str:
        """Build prompt for next sentence."""
        parts = []
        if context:
            parts.append(f"Context:\n{context}")

        parts.append(f"Question: {query}")

        if sentences:
            parts.append(f"Answer so far: {' '.join(sentences)}")
            parts.append("Generate the next sentence:")
        else:
            parts.append("Generate the first sentence of the answer:")

        return "\n\n".join(parts)

    def _extract_sentence(self, response: str) -> str:
        """Extract first complete sentence from response."""
        response = response.strip()

        # Find sentence boundary
        for end_char in [".", "!", "?"]:
            idx = response.find(end_char)
            if idx != -1:
                return response[: idx + 1].strip()

        return response

    async def _needs_retrieval(
        self,
        sentence: str,
        query: str,
    ) -> bool:
        """Check if sentence needs retrieval for verification."""
        # Simple heuristics - could use LLM for better detection
        uncertainty_words = [
            "perhaps",
            "maybe",
            "might",
            "could",
            "possibly",
            "likely",
            "approximately",
            "about",
            "around",
        ]

        sentence_lower = sentence.lower()
        for word in uncertainty_words:
            if word in sentence_lower:
                return True

        # Check for factual claims (numbers, names, dates)
        import re

        if re.search(r"\d{4}|\d+%|\$\d+", sentence):
            return True

        return False

    def _is_complete(self, sentence: str) -> bool:
        """Check if generation should stop."""
        completion_indicators = [
            "in conclusion",
            "to summarize",
            "in summary",
            "therefore",
        ]

        sentence_lower = sentence.lower()
        return any(ind in sentence_lower for ind in completion_indicators)

    def _format_context(self, docs: list[RetrievedDocument]) -> str:
        """Format documents as context."""
        return "\n\n".join(f"[{i}] {d.content[:300]}" for i, d in enumerate(docs, 1))


# ============================================================================
# Streaming FLARE
# ============================================================================


class StreamingFLARE:
    """
    Streaming FLARE that yields tokens as they're generated.

    Enables real-time display with retrieval indicators.
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: RetrieverProtocol,
        config: FLAREConfig | None = None,
    ):
        """Initialize streaming FLARE."""
        self.llm = llm
        self.retriever = retriever
        self.config = config or FLAREConfig()
        self.confidence_monitor = ConfidenceMonitor(threshold=self.config.confidence_threshold)

    async def stream(
        self,
        query: str,
        context: str = "",
    ) -> AsyncIterator[str | dict[str, Any]]:
        """
        Stream generation with retrieval events.

        Yields:
            Either text tokens or retrieval event dicts
        """
        generated_text = ""
        current_context = context

        for _ in range(self.config.max_iterations):
            self.confidence_monitor.reset()

            prompt = self._build_prompt(query, current_context, generated_text)

            # This would use streaming generation in practice
            generation = await self.llm.generate_with_logprobs(
                prompt,
                max_tokens=self.config.max_generation_tokens,
            )

            for token in generation.tokens:
                if self.confidence_monitor.should_retrieve(token):
                    # Signal retrieval event
                    yield {
                        "type": "retrieval_start",
                        "text_so_far": generated_text,
                    }

                    # Retrieve
                    docs = await self.retriever.retrieve(
                        generated_text[-100:],
                        top_k=self.config.top_k_retrieval,
                    )

                    yield {
                        "type": "retrieval_complete",
                        "docs": len(docs),
                    }

                    # Update context
                    current_context = self._update_context(current_context, docs)
                    break

                generated_text += token.token
                yield token.token

            if generation.finish_reason == "stop":
                break

    def _build_prompt(
        self,
        query: str,
        context: str,
        generated: str,
    ) -> str:
        """Build prompt for streaming generation."""
        parts = []
        if context:
            parts.append(f"Context: {context}")
        parts.append(f"Q: {query}")
        if generated:
            parts.append(f"A: {generated}")
        return "\n".join(parts)

    def _update_context(
        self,
        context: str,
        docs: list[RetrievedDocument],
    ) -> str:
        """Update context with retrieved docs."""
        new_context = "\n".join(d.content[:200] for d in docs)
        return f"{context}\n\nRetrieved:\n{new_context}"


# ============================================================================
# Factory Functions
# ============================================================================


def create_flare_generator(
    llm: LLMProtocol,
    retriever: RetrieverProtocol,
    confidence_threshold: float = 0.5,
    mode: FLAREMode = FLAREMode.CONFIDENCE,
) -> FLAREGenerator:
    """
    Create a FLARE generator.

    Args:
        llm: LLM with logprobs support
        retriever: Retrieval backend
        confidence_threshold: Trigger threshold
        mode: FLARE mode

    Returns:
        Configured FLAREGenerator
    """
    config = FLAREConfig(
        mode=mode,
        confidence_threshold=confidence_threshold,
    )
    return FLAREGenerator(llm, retriever, config)


def create_sentence_flare(
    llm: LLMProtocol,
    retriever: RetrieverProtocol,
) -> SentenceFLARE:
    """
    Create a sentence-level FLARE generator.

    Args:
        llm: LLM provider
        retriever: Retrieval backend

    Returns:
        Configured SentenceFLARE
    """
    return SentenceFLARE(llm, retriever)


def create_streaming_flare(
    llm: LLMProtocol,
    retriever: RetrieverProtocol,
    confidence_threshold: float = 0.5,
) -> StreamingFLARE:
    """
    Create a streaming FLARE generator.

    Args:
        llm: LLM with logprobs support
        retriever: Retrieval backend
        confidence_threshold: Trigger threshold

    Returns:
        Configured StreamingFLARE
    """
    config = FLAREConfig(confidence_threshold=confidence_threshold)
    return StreamingFLARE(llm, retriever, config)


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example demonstrating FLARE."""
    # This would use actual implementations
    # llm = OpenAILLM(api_key="...")
    # retriever = VectorStoreRetriever(...)

    # Standard FLARE
    # flare = create_flare_generator(llm, retriever)
    # result = await flare.generate("What is quantum computing?")

    # Sentence-level FLARE
    # sentence_flare = create_sentence_flare(llm, retriever)
    # result = await sentence_flare.generate("Explain photosynthesis")

    # Streaming FLARE
    # streaming = create_streaming_flare(llm, retriever)
    # async for token_or_event in streaming.stream("What is AI?"):
    #     if isinstance(token_or_event, str):
    #         print(token_or_event, end="")
    #     else:
    #         print(f"\n[Retrieval: {token_or_event}]")

    print("FLARE Implementation Ready")
    print("=" * 50)
    print("Features:")
    print("- Confidence-based retrieval triggering")
    print("- Token log probability monitoring")
    print("- Entropy-based uncertainty detection")
    print("- Lookahead query formulation")
    print("- Sentence-level retrieval variant")
    print("- Streaming generation with retrieval events")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
