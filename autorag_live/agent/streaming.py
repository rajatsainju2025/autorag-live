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
