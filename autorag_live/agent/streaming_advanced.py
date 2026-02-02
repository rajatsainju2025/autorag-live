"""
Advanced streaming support for agentic RAG with Server-Sent Events (SSE).

Provides real-time token generation, progress tracking, and
client-server streaming protocols for production use.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional


class StreamEventType(Enum):
    """Types of streaming events."""

    TOKEN = "token"  # Individual token
    CHUNK = "chunk"  # Text chunk
    REASONING = "reasoning"  # Reasoning step
    RETRIEVAL = "retrieval"  # Retrieval progress
    METADATA = "metadata"  # Metadata update
    ERROR = "error"  # Error event
    DONE = "done"  # Completion event
    PROGRESS = "progress"  # Progress update


@dataclass
class StreamMessage:
    """Structured streaming message."""

    event_type: StreamEventType
    data: Any
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        payload = {
            "type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        return f"data: {json.dumps(payload)}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class StreamBuffer:
    """Buffer for streaming output with backpressure control."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._closed = False

    async def put(self, message: StreamMessage) -> None:
        """Put message in buffer."""
        if self._closed:
            return
        await self.queue.put(message)

    async def get(self) -> Optional[StreamMessage]:
        """Get message from buffer."""
        if self._closed and self.queue.empty():
            return None
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    def close(self) -> None:
        """Close buffer."""
        self._closed = True

    @property
    def size(self) -> int:
        """Current buffer size."""
        return self.queue.qsize()

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.queue.full()


class TokenStreamProcessor:
    """Process and stream tokens with formatting."""

    def __init__(
        self,
        enable_word_boundary: bool = True,
        min_chunk_size: int = 5,
    ):
        self.enable_word_boundary = enable_word_boundary
        self.min_chunk_size = min_chunk_size
        self._buffer = []

    async def process_token(self, token: str) -> AsyncIterator[str]:
        """
        Process and yield tokens with smart boundary detection.

        Args:
            token: Input token

        Yields:
            Processed tokens
        """
        self._buffer.append(token)

        # Yield when we hit word boundary
        if self.enable_word_boundary:
            combined = "".join(self._buffer)
            if combined.endswith((" ", ".", ",", "!", "?", "\n")):
                yield combined
                self._buffer.clear()
        else:
            # Yield when buffer reaches min size
            if len(self._buffer) >= self.min_chunk_size:
                yield "".join(self._buffer)
                self._buffer.clear()

    async def flush(self) -> Optional[str]:
        """Flush remaining buffer."""
        if self._buffer:
            result = "".join(self._buffer)
            self._buffer.clear()
            return result
        return None


class StreamingRAGOrchestrator:
    """
    Orchestrates streaming RAG with progress tracking.

    Manages concurrent streaming of:
    - Token generation
    - Retrieval progress
    - Reasoning steps
    - Metadata updates
    """

    def __init__(
        self,
        enable_progress: bool = True,
        enable_reasoning_stream: bool = True,
    ):
        self.enable_progress = enable_progress
        self.enable_reasoning_stream = enable_reasoning_stream
        self.buffer = StreamBuffer()
        self.token_processor = TokenStreamProcessor()

    async def stream_rag_response(
        self,
        query: str,
        retriever_fn: Optional[Any] = None,
        generator_fn: Optional[Any] = None,
    ) -> AsyncIterator[StreamMessage]:
        """
        Stream complete RAG response with progress.

        Args:
            query: User query
            retriever_fn: Retrieval function
            generator_fn: Generation function

        Yields:
            Streaming messages
        """
        # Start event
        yield StreamMessage(
            event_type=StreamEventType.METADATA,
            data={"status": "started", "query": query},
        )

        # Retrieval phase
        if self.enable_progress:
            yield StreamMessage(
                event_type=StreamEventType.PROGRESS,
                data={"phase": "retrieval", "progress": 0.0},
            )

        documents = await self._retrieve_documents(query, retriever_fn)

        yield StreamMessage(
            event_type=StreamEventType.RETRIEVAL,
            data={"num_documents": len(documents), "documents": documents[:3]},
        )

        # Reasoning phase
        if self.enable_reasoning_stream:
            yield StreamMessage(
                event_type=StreamEventType.REASONING,
                data={"step": "Analyzing retrieved documents..."},
            )

        if self.enable_progress:
            yield StreamMessage(
                event_type=StreamEventType.PROGRESS,
                data={"phase": "generation", "progress": 0.5},
            )

        # Generation phase - stream tokens
        async for token in self._generate_tokens(query, documents, generator_fn):
            yield StreamMessage(event_type=StreamEventType.TOKEN, data=token)

        if self.enable_progress:
            yield StreamMessage(
                event_type=StreamEventType.PROGRESS,
                data={"phase": "complete", "progress": 1.0},
            )

        # Done event
        yield StreamMessage(
            event_type=StreamEventType.DONE,
            data={"status": "completed"},
        )

    async def _retrieve_documents(
        self, query: str, retriever_fn: Optional[Any]
    ) -> List[Dict[str, Any]]:
        """Retrieve documents asynchronously."""
        await asyncio.sleep(0.05)  # Simulate retrieval

        # Mock documents
        return [
            {"text": "Document 1 about " + query, "score": 0.9},
            {"text": "Document 2 related to " + query, "score": 0.8},
            {"text": "Document 3 mentioning " + query, "score": 0.7},
        ]

    async def _generate_tokens(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        generator_fn: Optional[Any],
    ) -> AsyncIterator[str]:
        """Generate answer tokens asynchronously."""
        # Simulate LLM token generation
        answer = f"Based on the retrieved documents about {query}, here is a comprehensive answer that synthesizes the key information."

        words = answer.split()
        for word in words:
            await asyncio.sleep(0.02)  # Simulate token latency
            yield word + " "


class SSEFormatter:
    """Format messages for Server-Sent Events protocol."""

    @staticmethod
    def format_message(message: StreamMessage) -> str:
        """Format message as SSE."""
        return message.to_sse()

    @staticmethod
    def format_batch(messages: List[StreamMessage]) -> str:
        """Format batch of messages as SSE."""
        return "".join([msg.to_sse() for msg in messages])

    @staticmethod
    def format_error(error: Exception) -> str:
        """Format error as SSE."""
        error_msg = StreamMessage(
            event_type=StreamEventType.ERROR,
            data={"error": str(error), "type": type(error).__name__},
        )
        return error_msg.to_sse()

    @staticmethod
    def format_done() -> str:
        """Format completion as SSE."""
        done_msg = StreamMessage(event_type=StreamEventType.DONE, data={"status": "complete"})
        return done_msg.to_sse()


class StreamingSession:
    """Manage streaming session with client."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.buffer = StreamBuffer()
        self.start_time = time.time()
        self._active = True
        self._token_count = 0

    async def send(self, message: StreamMessage) -> None:
        """Send message to client."""
        if not self._active:
            return

        await self.buffer.put(message)

        if message.event_type == StreamEventType.TOKEN:
            self._token_count += 1

    async def receive(self) -> Optional[StreamMessage]:
        """Receive message from buffer."""
        return await self.buffer.get()

    def close(self) -> None:
        """Close session."""
        self._active = False
        self.buffer.close()

    @property
    def duration(self) -> float:
        """Session duration in seconds."""
        return time.time() - self.start_time

    @property
    def token_count(self) -> int:
        """Total tokens sent."""
        return self._token_count

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.duration > 0:
            return self._token_count / self.duration
        return 0.0


class StreamingMetrics:
    """Track streaming performance metrics."""

    def __init__(self):
        self.start_time = time.time()
        self.first_token_time: Optional[float] = None
        self.token_count = 0
        self.chunk_count = 0
        self.total_bytes = 0

    def record_first_token(self) -> None:
        """Record first token timestamp."""
        if self.first_token_time is None:
            self.first_token_time = time.time()

    def record_token(self, token: str) -> None:
        """Record token generation."""
        self.token_count += 1
        self.total_bytes += len(token.encode("utf-8"))

    def record_chunk(self, chunk: str) -> None:
        """Record chunk generation."""
        self.chunk_count += 1
        self.total_bytes += len(chunk.encode("utf-8"))

    @property
    def time_to_first_token(self) -> Optional[float]:
        """Time to first token in seconds."""
        if self.first_token_time:
            return self.first_token_time - self.start_time
        return None

    @property
    def total_duration(self) -> float:
        """Total duration in seconds."""
        return time.time() - self.start_time

    @property
    def tokens_per_second(self) -> float:
        """Tokens per second throughput."""
        if self.total_duration > 0:
            return self.token_count / self.total_duration
        return 0.0

    @property
    def bytes_per_second(self) -> float:
        """Bytes per second throughput."""
        if self.total_duration > 0:
            return self.total_bytes / self.total_duration
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "token_count": self.token_count,
            "chunk_count": self.chunk_count,
            "total_bytes": self.total_bytes,
            "total_duration": self.total_duration,
            "time_to_first_token": self.time_to_first_token,
            "tokens_per_second": self.tokens_per_second,
            "bytes_per_second": self.bytes_per_second,
        }


# High-level API
async def stream_rag_query(
    query: str,
    retriever_fn: Optional[Any] = None,
    generator_fn: Optional[Any] = None,
) -> AsyncIterator[StreamMessage]:
    """
    Stream RAG query response.

    Args:
        query: User query
        retriever_fn: Optional retrieval function
        generator_fn: Optional generation function

    Yields:
        Streaming messages
    """
    orchestrator = StreamingRAGOrchestrator()
    async for message in orchestrator.stream_rag_response(query, retriever_fn, generator_fn):
        yield message


async def stream_to_sse(
    query: str,
    retriever_fn: Optional[Any] = None,
    generator_fn: Optional[Any] = None,
) -> AsyncIterator[str]:
    """
    Stream RAG query as Server-Sent Events.

    Args:
        query: User query
        retriever_fn: Optional retrieval function
        generator_fn: Optional generation function

    Yields:
        SSE-formatted strings
    """
    formatter = SSEFormatter()
    async for message in stream_rag_query(query, retriever_fn, generator_fn):
        yield formatter.format_message(message)


class StreamingClient:
    """Client for consuming streaming RAG responses."""

    def __init__(self):
        self.metrics = StreamingMetrics()
        self._buffer = []

    async def consume_stream(self, stream: AsyncIterator[StreamMessage]) -> Dict[str, Any]:
        """
        Consume streaming response and collect results.

        Args:
            stream: Streaming message iterator

        Returns:
            Complete response with metrics
        """
        tokens = []
        reasoning_steps = []
        retrieval_info = None

        async for message in stream:
            if message.event_type == StreamEventType.TOKEN:
                if not self.metrics.first_token_time:
                    self.metrics.record_first_token()
                self.metrics.record_token(message.data)
                tokens.append(message.data)

            elif message.event_type == StreamEventType.REASONING:
                reasoning_steps.append(message.data)

            elif message.event_type == StreamEventType.RETRIEVAL:
                retrieval_info = message.data

            elif message.event_type == StreamEventType.DONE:
                break

        return {
            "answer": "".join(tokens),
            "reasoning": reasoning_steps,
            "retrieval": retrieval_info,
            "metrics": self.metrics.to_dict(),
        }


# Example usage
async def example_streaming_rag():
    """Example of streaming RAG usage."""
    query = "What is machine learning?"

    # Stream with structured messages
    print("Streaming with structured messages:")
    async for message in stream_rag_query(query):
        print(f"[{message.event_type.value}] {message.data}")

    print("\n" + "=" * 50 + "\n")

    # Stream as SSE
    print("Streaming as Server-Sent Events:")
    async for sse_event in stream_to_sse(query):
        print(sse_event, end="")

    print("\n" + "=" * 50 + "\n")

    # Use client to consume stream
    print("Using StreamingClient:")
    client = StreamingClient()
    result = await client.consume_stream(stream_rag_query(query))
    print(f"Answer: {result['answer']}")
    print(f"Metrics: {result['metrics']}")


if __name__ == "__main__":
    asyncio.run(example_streaming_rag())
