"""
Streaming support for agentic RAG operations.

Provides token-by-token streaming, progress tracking, and cancellation support.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Callable, Generator, Optional


class StreamEventType(Enum):
    """Types of streaming events."""

    START = "start"
    TOKEN = "token"
    THINKING = "thinking"
    ACTION = "action"
    OBSERVATION = "observation"
    PROGRESS = "progress"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class StreamEvent:
    """Single streaming event."""

    event_type: StreamEventType
    content: str
    metadata: Optional[dict] = field(default=None)
    timestamp: Optional[float] = field(default=None)

    def __post_init__(self):
        """Set default timestamp."""
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        """Convert event to dictionary."""
        return {
            "type": self.event_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class StreamBuffer:
    """Thread-safe buffer for streaming events."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize stream buffer.

        Args:
            max_size: Maximum buffer size
        """
        self.max_size = max_size
        self.events: list = []
        self.lock = threading.Lock()
        self.event = threading.Event()

    def put(self, event: StreamEvent) -> None:
        """Add event to buffer."""
        with self.lock:
            self.events.append(event)
            if len(self.events) > self.max_size:
                self.events = self.events[-self.max_size :]
            self.event.set()

    def get(self, timeout: Optional[float] = None) -> Optional[StreamEvent]:
        """Get next event from buffer."""
        if self.event.wait(timeout=timeout):
            with self.lock:
                if self.events:
                    event = self.events.pop(0)
                    if not self.events:
                        self.event.clear()
                    return event
        return None

    def get_all(self) -> list:
        """Get all events and clear buffer."""
        with self.lock:
            events = self.events[:]
            self.events.clear()
            self.event.clear()
            return events

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self.lock:
            return len(self.events) == 0


class ProgressTracker:
    """Tracks progress of long-running operations."""

    def __init__(self, total_steps: int):
        """
        Initialize progress tracker.

        Args:
            total_steps: Total steps in operation
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def update(self, step: int, description: str = "") -> dict:
        """
        Update progress.

        Args:
            step: Current step
            description: Step description

        Returns:
            Progress metadata
        """
        with self.lock:
            self.current_step = step
            elapsed = time.time() - self.start_time

            if self.current_step > 0:
                rate = self.current_step / elapsed
                remaining = (self.total_steps - self.current_step) / rate
            else:
                rate = 0
                remaining = 0

            return {
                "current": self.current_step,
                "total": self.total_steps,
                "percentage": (self.current_step / self.total_steps * 100)
                if self.total_steps > 0
                else 0,
                "elapsed_seconds": elapsed,
                "estimated_remaining": remaining,
                "description": description,
            }

    def get_progress(self) -> dict:
        """Get current progress."""
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                "current": self.current_step,
                "total": self.total_steps,
                "percentage": (self.current_step / self.total_steps * 100)
                if self.total_steps > 0
                else 0,
                "elapsed_seconds": elapsed,
            }


class StreamingResponseHandler:
    """Handles streaming responses from agents."""

    def __init__(self, callback: Optional[Callable] = None):
        """
        Initialize streaming response handler.

        Args:
            callback: Optional callback for each event
        """
        self.buffer = StreamBuffer()
        self.callback = callback
        self.logger = logging.getLogger("StreamingResponseHandler")
        self._cancelled = False

    def stream_events(self) -> Generator[StreamEvent, None, None]:
        """
        Stream events as they arrive.

        Yields:
            Stream events
        """
        timeout = 1.0
        while not self._cancelled:
            event = self.buffer.get(timeout=timeout)
            if event:
                if self.callback:
                    try:
                        self.callback(event)
                    except Exception as e:
                        self.logger.error(f"Callback error: {str(e)}")

                yield event

            # Check if we should exit
            if self._cancelled and self.buffer.is_empty():
                break

    async def stream_events_async(self) -> AsyncGenerator[StreamEvent, None]:
        """
        Async stream events as they arrive.

        Yields:
            Stream events
        """
        while not self._cancelled:
            event = self.buffer.get(timeout=0.1)
            if event:
                if self.callback:
                    try:
                        self.callback(event)
                    except Exception as e:
                        self.logger.error(f"Callback error: {str(e)}")

                yield event
            else:
                await asyncio.sleep(0.01)

            # Check if we should exit
            if self._cancelled and self.buffer.is_empty():
                break

    def add_event(
        self, event_type: StreamEventType, content: str, metadata: Optional[dict] = None
    ) -> None:
        """
        Add event to stream.

        Args:
            event_type: Type of event
            content: Event content
            metadata: Optional metadata
        """
        event = StreamEvent(event_type=event_type, content=content, metadata=metadata or {})
        self.buffer.put(event)

    def start_streaming(self) -> None:
        """Mark streaming as started."""
        self.add_event(StreamEventType.START, "Streaming started")

    def add_token(self, token: str) -> None:
        """Add token to stream."""
        self.add_event(StreamEventType.TOKEN, token, {"is_token": True})

    def add_thinking(self, thought: str) -> None:
        """Add reasoning step to stream."""
        self.add_event(StreamEventType.THINKING, thought)

    def add_action(self, action_name: str, details: Optional[dict] = None) -> None:
        """Add action to stream."""
        self.add_event(StreamEventType.ACTION, action_name, {"action_details": details or {}})

    def add_observation(self, observation: str) -> None:
        """Add tool observation to stream."""
        self.add_event(StreamEventType.OBSERVATION, observation)

    def add_progress(self, progress_info: dict) -> None:
        """Add progress update to stream."""
        self.add_event(StreamEventType.PROGRESS, "Progress update", metadata=progress_info)

    def add_error(self, error_message: str) -> None:
        """Add error to stream."""
        self.add_event(StreamEventType.ERROR, error_message)

    def finish_streaming(self) -> None:
        """Mark streaming as complete."""
        self.add_event(StreamEventType.COMPLETE, "Streaming complete")
        self._cancelled = True

    def cancel(self) -> None:
        """Cancel streaming."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Check if streaming was cancelled."""
        return self._cancelled

    def get_all_events(self) -> list:
        """Get all buffered events."""
        return self.buffer.get_all()


class StreamingAgent:
    """Agent with streaming support."""

    def __init__(self, agent, streaming_handler: Optional[StreamingResponseHandler] = None):
        """
        Initialize streaming agent.

        Args:
            agent: Base agent instance
            streaming_handler: Optional streaming handler
        """
        self.agent = agent
        self.streaming_handler = streaming_handler or StreamingResponseHandler()
        self.logger = logging.getLogger("StreamingAgent")

    def process_with_streaming(self, query: str) -> Generator[StreamEvent, None, None]:
        """
        Process query with streaming output.

        Args:
            query: User query

        Yields:
            Stream events
        """
        self.streaming_handler.start_streaming()

        try:
            # Add thinking phase
            self.streaming_handler.add_thinking(f"Analyzing query: '{query}'")
            self.agent.memory.add_message("user", query)

            # Simulate token-by-token response
            response_tokens = [
                "Based",
                " on",
                " your",
                " query,",
                " here",
                " is",
                " the",
                " answer:",
            ]

            for i, token in enumerate(response_tokens):
                if self.streaming_handler.is_cancelled():
                    self.streaming_handler.add_error("Streaming cancelled")
                    break

                self.streaming_handler.add_token(token)

                # Update progress
                progress = {
                    "current": i + 1,
                    "total": len(response_tokens),
                    "percentage": ((i + 1) / len(response_tokens) * 100),
                }
                self.streaming_handler.add_progress(progress)

                # Yield events
                for event in self.streaming_handler.buffer.get_all():
                    yield event

        except Exception as e:
            self.streaming_handler.add_error(str(e))
            self.logger.error(f"Error during streaming: {str(e)}")

        finally:
            self.streaming_handler.finish_streaming()
            for event in self.streaming_handler.buffer.get_all():
                yield event

    def collect_all_events(self, query: str) -> list:
        """
        Collect all events from processing.

        Args:
            query: User query

        Returns:
            List of all events
        """
        events = []
        for event in self.process_with_streaming(query):
            events.append(event)
        return events


def create_streaming_response_text(events: list) -> str:
    """
    Create response text from streaming events.

    Args:
        events: List of stream events

    Returns:
        Combined response text
    """
    tokens = []
    for event in events:
        if event.event_type == StreamEventType.TOKEN:
            tokens.append(event.content)

    return "".join(tokens)


def print_streaming_events(events: list) -> None:
    """
    Print streaming events in human-readable format.

    Args:
        events: List of stream events
    """
    for event in events:
        if event.event_type == StreamEventType.START:
            print("▶ Starting...", flush=True)
        elif event.event_type == StreamEventType.TOKEN:
            print(event.content, end="", flush=True)
        elif event.event_type == StreamEventType.THINKING:
            print(f"\n💭 {event.content}\n", flush=True)
        elif event.event_type == StreamEventType.ACTION:
            print(f"\n🔧 Executing: {event.content}\n", flush=True)
        elif event.event_type == StreamEventType.OBSERVATION:
            print(f"\n📊 Observation: {event.content}\n", flush=True)
        elif event.event_type == StreamEventType.PROGRESS:
            progress = event.metadata
            percentage = progress.get("percentage", 0)
            print(f"\n⏳ Progress: {percentage:.0f}%\n", flush=True)
        elif event.event_type == StreamEventType.ERROR:
            print(f"\n❌ Error: {event.content}\n", flush=True)
        elif event.event_type == StreamEventType.COMPLETE:
            print("\n✅ Complete\n", flush=True)


# =============================================================================
# Semantic-Aware Streaming - State-of-the-Art Optimization
# =============================================================================


@dataclass
class SemanticChunk:
    """A semantically complete chunk of streamed content."""

    content: str
    chunk_type: str  # "sentence", "paragraph", "token", "citation"
    is_complete: bool
    tokens_count: int
    citations: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class SemanticStreamBuffer:
    """
    Buffer that delivers complete semantic units for improved UX.

    Instead of word-by-word streaming, this buffer accumulates tokens
    and delivers complete sentences. This improves:
    1. Perceived quality - users see complete thoughts
    2. Citation tracking - can identify sources per sentence
    3. Downstream processing - complete units for TTS/translation

    Based on streaming patterns from Perplexity.ai and ChatGPT.

    Example:
        >>> buffer = SemanticStreamBuffer()
        >>> for token in llm_stream:
        ...     sentence = await buffer.add_token(token)
        ...     if sentence:
        ...         yield sentence  # Complete sentence ready
    """

    def __init__(
        self,
        min_sentence_tokens: int = 5,
        max_buffer_tokens: int = 50,
        sentence_enders: str = ".!?",
        enable_citation_tracking: bool = True,
    ):
        """
        Initialize semantic stream buffer.

        Args:
            min_sentence_tokens: Minimum tokens before checking for sentence end
            max_buffer_tokens: Force flush after this many tokens
            sentence_enders: Characters that end sentences
            enable_citation_tracking: Track citation markers in content
        """
        self.min_sentence_tokens = min_sentence_tokens
        self.max_buffer_tokens = max_buffer_tokens
        self.sentence_enders = sentence_enders
        self.enable_citation_tracking = enable_citation_tracking

        self._buffer = ""
        self._token_count = 0
        self._total_sentences = 0
        self._pending_citations: list = []

    async def add_token(self, token: str) -> Optional[SemanticChunk]:
        """
        Add token to buffer, return complete sentence if available.

        Args:
            token: Token string from LLM stream

        Returns:
            SemanticChunk if a complete sentence is ready, else None
        """
        self._buffer += token
        self._token_count += 1

        # Track citations like [1], [2], etc.
        if self.enable_citation_tracking:
            self._extract_citations(token)

        # Check if we should flush
        if self._should_flush():
            return self._extract_sentence()

        return None

    def _should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        # Don't flush if too few tokens
        if self._token_count < self.min_sentence_tokens:
            return False

        # Flush if we have a complete sentence
        for ender in self.sentence_enders:
            # Look for sentence ender followed by space or end
            for i, char in enumerate(self._buffer):
                if char == ender:
                    # Check if this looks like end of sentence (not abbreviation)
                    if self._is_sentence_end(i):
                        return True

        # Force flush if buffer too large
        return self._token_count >= self.max_buffer_tokens

    def _is_sentence_end(self, pos: int) -> bool:
        """Check if position is a true sentence end (not abbreviation)."""
        if pos >= len(self._buffer) - 1:
            return True

        next_char = self._buffer[pos + 1] if pos + 1 < len(self._buffer) else ""

        # If followed by space and uppercase, likely sentence end
        if next_char == " ":
            after_space = self._buffer[pos + 2] if pos + 2 < len(self._buffer) else ""
            if after_space.isupper() or after_space == "":
                return True

        # Common abbreviations that don't end sentences
        before = self._buffer[max(0, pos - 3) : pos].lower()
        abbrevs = ["mr.", "ms.", "dr.", "vs.", "e.g", "i.e", "etc"]
        for abbrev in abbrevs:
            if before.endswith(abbrev[:-1]):
                return False

        return next_char in (" ", "\n", "")

    def _extract_sentence(self) -> Optional[SemanticChunk]:
        """Extract complete sentence from buffer."""
        if not self._buffer:
            return None

        # Find sentence boundary
        sentence_end = -1
        for i, char in enumerate(self._buffer):
            if char in self.sentence_enders and self._is_sentence_end(i):
                sentence_end = i + 1
                break

        if sentence_end > 0:
            sentence = self._buffer[:sentence_end]
            self._buffer = self._buffer[sentence_end:].lstrip()
        else:
            # Force flush entire buffer
            sentence = self._buffer
            self._buffer = ""

        self._total_sentences += 1
        chunk_tokens = len(sentence) // 4  # Estimate

        # Get citations for this sentence
        citations = self._pending_citations.copy()
        self._pending_citations.clear()

        self._token_count = len(self._buffer) // 4

        return SemanticChunk(
            content=sentence.strip(),
            chunk_type="sentence",
            is_complete=True,
            tokens_count=chunk_tokens,
            citations=citations,
        )

    def _extract_citations(self, token: str) -> None:
        """Extract citation markers from token."""
        import re

        # Match patterns like [1], [2,3], etc.
        citation_pattern = r"\[(\d+(?:,\s*\d+)*)\]"
        matches = re.findall(citation_pattern, token)

        for match in matches:
            # Split "1,2,3" into individual citations
            for cit in match.split(","):
                cit = cit.strip()
                if cit and cit not in self._pending_citations:
                    self._pending_citations.append(cit)

    def flush(self) -> Optional[SemanticChunk]:
        """Flush remaining buffer content."""
        if not self._buffer.strip():
            return None

        content = self._buffer.strip()
        self._buffer = ""
        self._token_count = 0

        citations = self._pending_citations.copy()
        self._pending_citations.clear()

        return SemanticChunk(
            content=content,
            chunk_type="sentence",
            is_complete=True,
            tokens_count=len(content) // 4,
            citations=citations,
        )

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            "total_sentences": self._total_sentences,
            "buffer_length": len(self._buffer),
            "pending_tokens": self._token_count,
            "pending_citations": len(self._pending_citations),
        }


@dataclass
class CitationInfo:
    """Information about a citation in streamed content."""

    citation_id: str
    source_doc: str
    relevance_score: float
    sentence_index: int


class StreamingRAGGenerator:
    """
    RAG generator with semantic streaming and inline citation tracking.

    Provides improved streaming UX by:
    1. Delivering complete sentences instead of raw tokens
    2. Tracking citations per sentence
    3. Supporting real-time citation verification
    4. Enabling downstream processing (TTS, translation)

    Example:
        >>> generator = StreamingRAGGenerator(llm, documents)
        >>> async for chunk in generator.stream_with_citations(query):
        ...     if chunk["type"] == "sentence":
        ...         print(chunk["content"])
        ...         for cit in chunk["citations"]:
        ...             print(f"  Source: {cit}")
    """

    def __init__(
        self,
        llm,
        documents: list,
        min_sentence_tokens: int = 5,
        max_buffer_tokens: int = 50,
    ):
        """
        Initialize streaming RAG generator.

        Args:
            llm: Language model with streaming support
            documents: Source documents for citation tracking
            min_sentence_tokens: Minimum tokens per sentence
            max_buffer_tokens: Maximum buffer size
        """
        self.llm = llm
        self.documents = documents
        self.min_sentence_tokens = min_sentence_tokens
        self.max_buffer_tokens = max_buffer_tokens

        self._citation_map: dict = {}
        self._build_citation_map()

    def _build_citation_map(self) -> None:
        """Build map of citation IDs to documents."""
        for i, doc in enumerate(self.documents):
            doc_id = doc.get("id", str(i + 1))
            self._citation_map[str(i + 1)] = {
                "id": doc_id,
                "content": doc.get("content", "")[:200],
                "source": doc.get("source", "Unknown"),
            }

    async def stream_with_citations(
        self,
        query: str,
        prompt_template: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream response with inline citation tracking.

        Yields semantic chunks (sentences) with associated citations.

        Args:
            query: User query
            prompt_template: Optional custom prompt template

        Yields:
            Dicts with type, content, and citations
        """
        # Build prompt with documents
        context = self._build_context()
        prompt = prompt_template or self._default_prompt(query, context)

        # Initialize semantic buffer
        buffer = SemanticStreamBuffer(
            min_sentence_tokens=self.min_sentence_tokens,
            max_buffer_tokens=self.max_buffer_tokens,
            enable_citation_tracking=True,
        )

        # Track all citations
        all_citations: list = []
        sentence_count = 0

        yield {"type": "start", "content": "", "citations": []}

        # Stream from LLM
        if hasattr(self.llm, "stream"):
            async for token in self.llm.stream(prompt):
                # Also yield raw token for real-time display
                yield {"type": "token", "content": token, "citations": []}

                # Try to get complete sentence
                sentence = await buffer.add_token(token)
                if sentence:
                    sentence_count += 1

                    # Resolve citation references
                    resolved_citations = self._resolve_citations(sentence.citations)
                    all_citations.extend(resolved_citations)

                    yield {
                        "type": "sentence",
                        "content": sentence.content,
                        "citations": resolved_citations,
                        "sentence_index": sentence_count,
                    }

        # Flush remaining content
        remaining = buffer.flush()
        if remaining:
            sentence_count += 1
            resolved_citations = self._resolve_citations(remaining.citations)
            all_citations.extend(resolved_citations)

            yield {
                "type": "sentence",
                "content": remaining.content,
                "citations": resolved_citations,
                "sentence_index": sentence_count,
            }

        yield {
            "type": "done",
            "content": "",
            "citations": [],
            "total_sentences": sentence_count,
            "total_citations": len(set(c["id"] for c in all_citations)),
        }

    def _build_context(self) -> str:
        """Build context from documents."""
        context_parts = []
        for i, doc in enumerate(self.documents[:5]):  # Limit to top 5
            content = doc.get("content", "")[:500]
            context_parts.append(f"[{i + 1}] {content}")
        return "\n\n".join(context_parts)

    def _default_prompt(self, query: str, context: str) -> str:
        """Build default RAG prompt."""
        return f"""Answer the question based on the context.
Include citation numbers [1], [2], etc. when referencing information.

Context:
{context}

Question: {query}

Answer:"""

    def _resolve_citations(self, citation_ids: list) -> list:
        """Resolve citation IDs to full citation info."""
        resolved = []
        for cit_id in citation_ids:
            if cit_id in self._citation_map:
                resolved.append(self._citation_map[cit_id])
        return resolved


class AdaptiveStreamThrottler:
    """
    Adaptive throttling for streaming to optimize perceived latency.

    Delivers tokens faster initially, then slows down for smoother UX.
    """

    def __init__(
        self,
        initial_delay_ms: float = 10,
        target_delay_ms: float = 30,
        ramp_tokens: int = 20,
    ):
        """
        Initialize adaptive throttler.

        Args:
            initial_delay_ms: Initial delay between tokens (fast start)
            target_delay_ms: Target delay after ramp-up
            ramp_tokens: Number of tokens to ramp up over
        """
        self.initial_delay = initial_delay_ms / 1000
        self.target_delay = target_delay_ms / 1000
        self.ramp_tokens = ramp_tokens
        self._token_count = 0

    async def throttle(self) -> None:
        """Apply adaptive delay based on token count."""
        if self._token_count < self.ramp_tokens:
            # Linear ramp from initial to target
            progress = self._token_count / self.ramp_tokens
            delay = self.initial_delay + (self.target_delay - self.initial_delay) * progress
        else:
            delay = self.target_delay

        self._token_count += 1
        await asyncio.sleep(delay)

    def reset(self) -> None:
        """Reset throttler for new stream."""
        self._token_count = 0
