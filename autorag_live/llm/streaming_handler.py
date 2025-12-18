"""
Streaming handler module for AutoRAG-Live.

Provides comprehensive streaming support for LLM responses
with buffering, parsing, and event handling.

Features:
- Token-by-token streaming
- Chunk buffering
- JSON streaming and parsing
- Server-sent events (SSE) support
- Stream aggregation
- Timeout handling
- Error recovery

Example usage:
    >>> handler = StreamingHandler()
    >>> async for chunk in handler.stream(llm_response):
    ...     print(chunk.text, end="", flush=True)
    >>> 
    >>> # With callbacks
    >>> handler.on_token(lambda t: print(t, end=""))
    >>> await handler.process_stream(response)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class StreamEventType(Enum):
    """Types of streaming events."""
    
    START = auto()
    TOKEN = auto()
    CHUNK = auto()
    TOOL_CALL = auto()
    CITATION = auto()
    ERROR = auto()
    END = auto()
    HEARTBEAT = auto()


@dataclass
class StreamEvent:
    """A streaming event."""
    
    event_type: StreamEventType
    data: Any = None
    
    # Timing
    timestamp: float = field(default_factory=time.time)
    
    # Position
    token_index: int = 0
    chunk_index: int = 0
    
    # Metadata
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A chunk of streamed content."""
    
    text: str
    index: int
    
    # Timing
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    
    # Token info
    token_count: int = 0
    
    # Finish info
    is_final: bool = False
    finish_reason: Optional[str] = None
    
    # Tool calls
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamResult:
    """Final result of streaming."""
    
    text: str
    chunks: List[StreamChunk]
    
    # Statistics
    total_tokens: int = 0
    chunk_count: int = 0
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    first_token_time: float = 0.0
    
    # Aggregated
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error info
    error: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_time_ms(self) -> float:
        """Total streaming time."""
        return (self.end_time - self.start_time) * 1000
    
    @property
    def time_to_first_token_ms(self) -> float:
        """Time to first token."""
        if self.first_token_time > 0:
            return (self.first_token_time - self.start_time) * 1000
        return 0.0
    
    @property
    def tokens_per_second(self) -> float:
        """Token generation rate."""
        duration = self.end_time - self.start_time
        if duration > 0:
            return self.total_tokens / duration
        return 0.0


class StreamBuffer:
    """Buffer for stream chunks."""
    
    def __init__(
        self,
        flush_size: int = 10,
        flush_timeout: float = 0.1,
    ):
        """
        Initialize buffer.
        
        Args:
            flush_size: Flush when buffer reaches this size
            flush_timeout: Flush after this many seconds
        """
        self.flush_size = flush_size
        self.flush_timeout = flush_timeout
        
        self._buffer: List[str] = []
        self._last_flush: float = time.time()
    
    def add(self, text: str) -> Optional[str]:
        """
        Add text to buffer.
        
        Returns flushed content if flush triggered.
        """
        self._buffer.append(text)
        
        # Check flush conditions
        if len(self._buffer) >= self.flush_size:
            return self.flush()
        
        if time.time() - self._last_flush > self.flush_timeout:
            return self.flush()
        
        return None
    
    def flush(self) -> str:
        """Flush buffer and return content."""
        content = ''.join(self._buffer)
        self._buffer = []
        self._last_flush = time.time()
        return content
    
    @property
    def content(self) -> str:
        """Get buffered content without flushing."""
        return ''.join(self._buffer)
    
    @property
    def size(self) -> int:
        """Get buffer size."""
        return len(self._buffer)


class JSONStreamParser:
    """Parser for streaming JSON."""
    
    def __init__(self):
        """Initialize parser."""
        self._buffer = ""
        self._depth = 0
        self._in_string = False
        self._escape_next = False
        self._objects: List[Any] = []
    
    def feed(self, chunk: str) -> List[Any]:
        """
        Feed chunk and return any complete objects.
        
        Args:
            chunk: Text chunk
            
        Returns:
            List of parsed JSON objects
        """
        result = []
        
        for char in chunk:
            if self._escape_next:
                self._escape_next = False
                self._buffer += char
                continue
            
            if char == '\\' and self._in_string:
                self._escape_next = True
                self._buffer += char
                continue
            
            if char == '"':
                self._in_string = not self._in_string
                self._buffer += char
                continue
            
            if not self._in_string:
                if char == '{':
                    if self._depth == 0:
                        self._buffer = ""
                    self._depth += 1
                    self._buffer += char
                elif char == '}':
                    self._buffer += char
                    self._depth -= 1
                    
                    if self._depth == 0:
                        try:
                            obj = json.loads(self._buffer)
                            result.append(obj)
                        except json.JSONDecodeError:
                            pass
                        self._buffer = ""
                elif self._depth > 0:
                    self._buffer += char
            else:
                self._buffer += char
        
        return result
    
    def reset(self) -> None:
        """Reset parser state."""
        self._buffer = ""
        self._depth = 0
        self._in_string = False
        self._escape_next = False


class SSEParser:
    """Parser for Server-Sent Events."""
    
    def __init__(self):
        """Initialize parser."""
        self._buffer = ""
    
    def feed(self, chunk: str) -> List[Dict[str, str]]:
        """
        Feed chunk and return parsed events.
        
        Args:
            chunk: Text chunk
            
        Returns:
            List of event dicts with 'event', 'data', 'id' keys
        """
        self._buffer += chunk
        events = []
        
        # Split on double newlines (event boundary)
        while '\n\n' in self._buffer:
            event_str, self._buffer = self._buffer.split('\n\n', 1)
            event = self._parse_event(event_str)
            if event:
                events.append(event)
        
        return events
    
    def _parse_event(self, event_str: str) -> Optional[Dict[str, str]]:
        """Parse single event."""
        event = {}
        
        for line in event_str.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'data':
                    if 'data' in event:
                        event['data'] += '\n' + value
                    else:
                        event['data'] = value
                else:
                    event[key] = value
        
        return event if event else None
    
    def reset(self) -> None:
        """Reset parser."""
        self._buffer = ""


class StreamingHandler:
    """
    Main streaming handler interface.
    
    Example:
        >>> handler = StreamingHandler()
        >>> 
        >>> # Register callbacks
        >>> handler.on_start(lambda: print("Starting..."))
        >>> handler.on_token(lambda t: print(t, end=""))
        >>> handler.on_end(lambda r: print(f"\\nDone: {r.total_tokens} tokens"))
        >>> 
        >>> # Process stream
        >>> async for chunk in some_llm_stream():
        ...     await handler.process_chunk(chunk)
        >>> 
        >>> result = handler.get_result()
    """
    
    def __init__(
        self,
        buffer_size: int = 10,
        timeout: float = 60.0,
        heartbeat_interval: float = 15.0,
    ):
        """
        Initialize handler.
        
        Args:
            buffer_size: Chunk buffer size
            timeout: Stream timeout
            heartbeat_interval: Heartbeat interval
        """
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.heartbeat_interval = heartbeat_interval
        
        # State
        self._chunks: List[StreamChunk] = []
        self._text = ""
        self._token_count = 0
        self._chunk_index = 0
        
        # Timing
        self._start_time: float = 0.0
        self._first_token_time: float = 0.0
        self._last_activity: float = 0.0
        
        # Callbacks
        self._on_start: List[Callable[[], None]] = []
        self._on_token: List[Callable[[str], None]] = []
        self._on_chunk: List[Callable[[StreamChunk], None]] = []
        self._on_error: List[Callable[[Exception], None]] = []
        self._on_end: List[Callable[[StreamResult], None]] = []
        
        # Buffer
        self._buffer = StreamBuffer(flush_size=buffer_size)
        
        # Tools
        self._tool_calls: List[Dict[str, Any]] = []
    
    def on_start(self, callback: Callable[[], None]) -> StreamingHandler:
        """Register start callback."""
        self._on_start.append(callback)
        return self
    
    def on_token(self, callback: Callable[[str], None]) -> StreamingHandler:
        """Register token callback."""
        self._on_token.append(callback)
        return self
    
    def on_chunk(self, callback: Callable[[StreamChunk], None]) -> StreamingHandler:
        """Register chunk callback."""
        self._on_chunk.append(callback)
        return self
    
    def on_error(self, callback: Callable[[Exception], None]) -> StreamingHandler:
        """Register error callback."""
        self._on_error.append(callback)
        return self
    
    def on_end(self, callback: Callable[[StreamResult], None]) -> StreamingHandler:
        """Register end callback."""
        self._on_end.append(callback)
        return self
    
    def start(self) -> None:
        """Signal stream start."""
        self._start_time = time.time()
        self._last_activity = time.time()
        
        for callback in self._on_start:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Start callback error: {e}")
    
    def process_token(self, token: str) -> None:
        """
        Process single token.
        
        Args:
            token: Token text
        """
        self._last_activity = time.time()
        
        if self._first_token_time == 0:
            self._first_token_time = time.time()
        
        self._text += token
        self._token_count += 1
        
        # Call token callbacks
        for callback in self._on_token:
            try:
                callback(token)
            except Exception as e:
                logger.warning(f"Token callback error: {e}")
        
        # Buffer for chunks
        flushed = self._buffer.add(token)
        if flushed:
            self._emit_chunk(flushed)
    
    def _emit_chunk(self, text: str) -> None:
        """Emit a chunk."""
        chunk = StreamChunk(
            text=text,
            index=self._chunk_index,
            latency_ms=(time.time() - self._start_time) * 1000,
            token_count=len(text.split()),
        )
        
        self._chunks.append(chunk)
        self._chunk_index += 1
        
        for callback in self._on_chunk:
            try:
                callback(chunk)
            except Exception as e:
                logger.warning(f"Chunk callback error: {e}")
    
    def process_tool_call(self, tool_call: Dict[str, Any]) -> None:
        """Process tool call from stream."""
        self._tool_calls.append(tool_call)
    
    def error(self, error: Exception) -> None:
        """Handle error."""
        for callback in self._on_error:
            try:
                callback(error)
            except Exception as e:
                logger.warning(f"Error callback error: {e}")
    
    def end(self, finish_reason: Optional[str] = None) -> StreamResult:
        """
        Signal stream end and get result.
        
        Args:
            finish_reason: Reason for stream end
            
        Returns:
            StreamResult
        """
        # Flush remaining buffer
        remaining = self._buffer.flush()
        if remaining:
            self._emit_chunk(remaining)
        
        # Mark final chunk
        if self._chunks:
            self._chunks[-1].is_final = True
            self._chunks[-1].finish_reason = finish_reason
        
        result = StreamResult(
            text=self._text,
            chunks=self._chunks,
            total_tokens=self._token_count,
            chunk_count=len(self._chunks),
            start_time=self._start_time,
            end_time=time.time(),
            first_token_time=self._first_token_time,
            tool_calls=self._tool_calls,
        )
        
        for callback in self._on_end:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"End callback error: {e}")
        
        return result
    
    def get_result(self) -> StreamResult:
        """Get current result without ending."""
        return StreamResult(
            text=self._text,
            chunks=self._chunks.copy(),
            total_tokens=self._token_count,
            chunk_count=len(self._chunks),
            start_time=self._start_time,
            end_time=time.time(),
            first_token_time=self._first_token_time,
            tool_calls=self._tool_calls.copy(),
        )
    
    def reset(self) -> None:
        """Reset handler state."""
        self._chunks = []
        self._text = ""
        self._token_count = 0
        self._chunk_index = 0
        self._start_time = 0.0
        self._first_token_time = 0.0
        self._last_activity = 0.0
        self._tool_calls = []
        self._buffer = StreamBuffer(flush_size=self.buffer_size)
    
    async def stream(
        self,
        response: AsyncIterator[str],
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream from async iterator.
        
        Args:
            response: Async iterator of tokens
            
        Yields:
            StreamChunk objects
        """
        self.start()
        
        try:
            async for token in response:
                self.process_token(token)
                
                # Check for timeout
                if time.time() - self._last_activity > self.timeout:
                    raise TimeoutError("Stream timeout")
                
                # Yield chunks as they're ready
                while self._chunks and self._chunk_index > len(self._chunks):
                    yield self._chunks[-1]
            
            # Final result
            result = self.end()
            if result.chunks:
                yield result.chunks[-1]
                
        except Exception as e:
            self.error(e)
            raise


class AsyncStreamProcessor:
    """Async stream processor with concurrency."""
    
    def __init__(
        self,
        handler: StreamingHandler,
        max_concurrent: int = 3,
    ):
        """
        Initialize processor.
        
        Args:
            handler: StreamingHandler instance
            max_concurrent: Max concurrent streams
        """
        self.handler = handler
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process(
        self,
        stream: AsyncIterator[str],
    ) -> StreamResult:
        """Process a stream."""
        async with self._semaphore:
            self.handler.reset()
            self.handler.start()
            
            try:
                async for token in stream:
                    self.handler.process_token(token)
                
                return self.handler.end()
                
            except Exception as e:
                self.handler.error(e)
                return self.handler.end("error")
    
    async def process_many(
        self,
        streams: List[AsyncIterator[str]],
    ) -> List[StreamResult]:
        """Process multiple streams concurrently."""
        tasks = [self.process(stream) for stream in streams]
        return await asyncio.gather(*tasks)


class StreamAggregator:
    """Aggregate multiple streams."""
    
    def __init__(self):
        """Initialize aggregator."""
        self._results: List[StreamResult] = []
    
    def add(self, result: StreamResult) -> None:
        """Add stream result."""
        self._results.append(result)
    
    def aggregate(self) -> Dict[str, Any]:
        """Aggregate statistics."""
        if not self._results:
            return {}
        
        total_tokens = sum(r.total_tokens for r in self._results)
        total_chunks = sum(r.chunk_count for r in self._results)
        total_time = sum(r.total_time_ms for r in self._results)
        
        ttft_values = [
            r.time_to_first_token_ms
            for r in self._results
            if r.time_to_first_token_ms > 0
        ]
        
        return {
            'stream_count': len(self._results),
            'total_tokens': total_tokens,
            'total_chunks': total_chunks,
            'total_time_ms': total_time,
            'avg_tokens_per_stream': total_tokens / len(self._results),
            'avg_time_per_stream_ms': total_time / len(self._results),
            'avg_time_to_first_token_ms': (
                sum(ttft_values) / len(ttft_values) if ttft_values else 0
            ),
            'min_time_to_first_token_ms': min(ttft_values) if ttft_values else 0,
            'max_time_to_first_token_ms': max(ttft_values) if ttft_values else 0,
        }
    
    def clear(self) -> None:
        """Clear results."""
        self._results = []


# Convenience functions

async def stream_tokens(
    response: AsyncIterator[str],
    on_token: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Stream tokens and return full text.
    
    Args:
        response: Token stream
        on_token: Optional token callback
        
    Returns:
        Complete text
    """
    handler = StreamingHandler()
    
    if on_token:
        handler.on_token(on_token)
    
    handler.start()
    
    async for token in response:
        handler.process_token(token)
    
    result = handler.end()
    return result.text


def create_sse_stream(
    events: List[Dict[str, Any]],
) -> Iterator[str]:
    """
    Create SSE formatted stream.
    
    Args:
        events: Events to stream
        
    Yields:
        SSE formatted strings
    """
    for event in events:
        lines = []
        
        if 'event' in event:
            lines.append(f"event: {event['event']}")
        
        if 'data' in event:
            data = event['data']
            if isinstance(data, (dict, list)):
                data = json.dumps(data)
            lines.append(f"data: {data}")
        
        if 'id' in event:
            lines.append(f"id: {event['id']}")
        
        yield '\n'.join(lines) + '\n\n'


def parse_sse_stream(
    stream: Iterator[str],
) -> Iterator[Dict[str, str]]:
    """
    Parse SSE stream.
    
    Args:
        stream: SSE formatted stream
        
    Yields:
        Parsed events
    """
    parser = SSEParser()
    
    for chunk in stream:
        events = parser.feed(chunk)
        for event in events:
            yield event
