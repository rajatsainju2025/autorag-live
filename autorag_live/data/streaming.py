"""
Streaming data ingestion for real-time document updates.

This module provides capabilities for real-time document ingestion,
incremental index updates, and stream processing for RAG systems.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set

from autorag_live.utils import get_logger

logger = get_logger(__name__)


class DocumentEvent(Enum):
    """Types of document events."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    BULK_ADD = "bulk_add"


@dataclass
class Document:
    """
    Document representation for streaming.

    Attributes:
        id: Unique document identifier
        content: Document text content
        metadata: Additional metadata
        timestamp: Creation/update timestamp
        hash: Content hash for deduplication
    """

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    hash: Optional[str] = None

    def __post_init__(self):
        """Compute content hash if not provided."""
        if self.hash is None:
            self.hash = hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class StreamEvent:
    """
    Event in the document stream.

    Attributes:
        event_type: Type of event
        document: Document involved in the event
        timestamp: Event timestamp
        metadata: Additional event metadata
    """

    event_type: DocumentEvent
    document: Document
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamBuffer:
    """
    Buffer for streaming documents with batching.

    Handles buffering, batching, and flushing of document streams.
    """

    def __init__(
        self,
        max_size: int = 100,
        max_wait_seconds: float = 5.0,
        deduplicate: bool = True,
    ):
        """
        Initialize stream buffer.

        Args:
            max_size: Maximum buffer size before auto-flush
            max_wait_seconds: Maximum wait time before auto-flush
            deduplicate: Whether to deduplicate by content hash
        """
        self.max_size = max_size
        self.max_wait_seconds = max_wait_seconds
        self.deduplicate = deduplicate

        self._buffer: Deque[StreamEvent] = deque()
        self._seen_hashes: Set[str] = set()
        self._last_flush_time = time.time()
        self._lock = asyncio.Lock()

    async def add(self, event: StreamEvent) -> bool:
        """
        Add event to buffer.

        Args:
            event: Stream event to add

        Returns:
            True if added, False if deduplicated
        """
        async with self._lock:
            # Deduplication check
            if self.deduplicate and event.document.hash:
                if event.document.hash in self._seen_hashes:
                    logger.debug(f"Deduplicated document: {event.document.id}")
                    return False
                self._seen_hashes.add(event.document.hash)

            self._buffer.append(event)
            return True

    async def should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        async with self._lock:
            if len(self._buffer) >= self.max_size:
                return True

            time_since_flush = time.time() - self._last_flush_time
            if time_since_flush >= self.max_wait_seconds and len(self._buffer) > 0:
                return True

            return False

    async def flush(self) -> List[StreamEvent]:
        """
        Flush buffer and return events.

        Returns:
            List of buffered events
        """
        async with self._lock:
            events = list(self._buffer)
            self._buffer.clear()
            self._last_flush_time = time.time()

            logger.debug(f"Flushed {len(events)} events from buffer")
            return events

    async def clear(self) -> None:
        """Clear buffer and deduplication cache."""
        async with self._lock:
            self._buffer.clear()
            self._seen_hashes.clear()
            self._last_flush_time = time.time()


class DocumentStream:
    """
    Asynchronous document stream processor.

    Handles real-time document ingestion with batching, deduplication,
    and callback processing.
    """

    def __init__(
        self,
        buffer_size: int = 100,
        buffer_wait_seconds: float = 5.0,
        deduplicate: bool = True,
        callback: Optional[Callable[[List[StreamEvent]], None]] = None,
    ):
        """
        Initialize document stream.

        Args:
            buffer_size: Maximum buffer size
            buffer_wait_seconds: Maximum buffer wait time
            deduplicate: Whether to deduplicate documents
            callback: Callback function for processing batches
        """
        self.buffer = StreamBuffer(buffer_size, buffer_wait_seconds, deduplicate)
        self.callback = callback
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._stats = {
            "events_processed": 0,
            "documents_added": 0,
            "documents_updated": 0,
            "documents_deleted": 0,
            "batches_processed": 0,
            "duplicates_skipped": 0,
        }

    async def start(self) -> None:
        """Start the stream processor."""
        if self._running:
            logger.warning("Stream already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("Document stream started")

    async def stop(self) -> None:
        """Stop the stream processor."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Flush remaining events
        remaining = await self.buffer.flush()
        if remaining and self.callback:
            await self._process_batch(remaining)

        logger.info("Document stream stopped")

    async def add_document(
        self,
        document: Document,
        event_type: DocumentEvent = DocumentEvent.ADD,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add document to stream.

        Args:
            document: Document to add
            event_type: Type of event
            metadata: Additional event metadata

        Returns:
            True if added successfully
        """
        event = StreamEvent(
            event_type=event_type,
            document=document,
            metadata=metadata or {},
        )

        added = await self.buffer.add(event)
        if not added:
            self._stats["duplicates_skipped"] += 1

        return added

    async def add_documents_bulk(
        self,
        documents: List[Document],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add multiple documents in bulk.

        Args:
            documents: List of documents to add
            metadata: Additional metadata for all documents

        Returns:
            Number of documents added
        """
        added_count = 0
        for doc in documents:
            added = await self.add_document(doc, DocumentEvent.BULK_ADD, metadata)
            if added:
                added_count += 1

        return added_count

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                # Check if buffer should be flushed
                if await self.buffer.should_flush():
                    events = await self.buffer.flush()
                    if events:
                        await self._process_batch(events)

                # Small delay to avoid busy waiting
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stream processing loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Back off on error

    async def _process_batch(self, events: List[StreamEvent]) -> None:
        """
        Process a batch of events.

        Args:
            events: List of stream events to process
        """
        if not events:
            return

        # Update statistics
        self._stats["batches_processed"] += 1
        self._stats["events_processed"] += len(events)

        for event in events:
            if event.event_type == DocumentEvent.ADD or event.event_type == DocumentEvent.BULK_ADD:
                self._stats["documents_added"] += 1
            elif event.event_type == DocumentEvent.UPDATE:
                self._stats["documents_updated"] += 1
            elif event.event_type == DocumentEvent.DELETE:
                self._stats["documents_deleted"] += 1

        # Call callback if provided
        if self.callback:
            try:
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(events)
                else:
                    self.callback(events)

                logger.debug(f"Processed batch of {len(events)} events")
            except Exception as e:
                logger.error(f"Error in batch callback: {e}", exc_info=True)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get stream statistics.

        Returns:
            Dictionary with stream statistics
        """
        return {
            **self._stats,
            "buffer_size": len(self.buffer._buffer),
            "is_running": self._running,
        }


class FileWatcher:
    """
    Watch a directory for new/modified files for streaming ingestion.

    Monitors filesystem changes and streams documents as they're added/modified.
    """

    def __init__(
        self,
        watch_path: Path,
        stream: DocumentStream,
        file_pattern: str = "*.txt",
        poll_interval: float = 1.0,
    ):
        """
        Initialize file watcher.

        Args:
            watch_path: Directory to watch
            stream: Document stream to feed
            file_pattern: Glob pattern for files to watch
            poll_interval: Polling interval in seconds
        """
        self.watch_path = Path(watch_path)
        self.stream = stream
        self.file_pattern = file_pattern
        self.poll_interval = poll_interval

        self._seen_files: Dict[str, float] = {}  # filepath -> mtime
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start watching directory."""
        if self._running:
            logger.warning("File watcher already running")
            return

        if not self.watch_path.exists():
            raise ValueError(f"Watch path does not exist: {self.watch_path}")

        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info(f"File watcher started: {self.watch_path}")

    async def stop(self) -> None:
        """Stop watching directory."""
        if not self._running:
            return

        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("File watcher stopped")

    async def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._running:
            try:
                await self._scan_directory()
                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in file watcher loop: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

    async def _scan_directory(self) -> None:
        """Scan directory for new/modified files."""
        for file_path in self.watch_path.glob(self.file_pattern):
            if not file_path.is_file():
                continue

            file_str = str(file_path)
            mtime = file_path.stat().st_mtime

            # Check if new or modified
            if file_str not in self._seen_files:
                # New file
                await self._process_file(file_path, is_new=True)
                self._seen_files[file_str] = mtime

            elif self._seen_files[file_str] < mtime:
                # Modified file
                await self._process_file(file_path, is_new=False)
                self._seen_files[file_str] = mtime

    async def _process_file(self, file_path: Path, is_new: bool) -> None:
        """
        Process a file and add to stream.

        Args:
            file_path: Path to file
            is_new: Whether file is new or modified
        """
        try:
            content = file_path.read_text(encoding="utf-8")

            document = Document(
                id=str(file_path),
                content=content,
                metadata={
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size": len(content),
                },
            )

            event_type = DocumentEvent.ADD if is_new else DocumentEvent.UPDATE
            await self.stream.add_document(document, event_type)

            logger.debug(f"{'Added' if is_new else 'Updated'} file: {file_path.name}")

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")


# Convenience function for simple streaming setup
async def create_stream_from_directory(
    directory: Path,
    callback: Callable[[List[StreamEvent]], None],
    file_pattern: str = "*.txt",
    buffer_size: int = 100,
    poll_interval: float = 1.0,
) -> tuple[DocumentStream, FileWatcher]:
    """
    Create a document stream from a directory.

    Args:
        directory: Directory to watch
        callback: Callback for processing batches
        file_pattern: File pattern to watch
        buffer_size: Stream buffer size
        poll_interval: Directory poll interval

    Returns:
        Tuple of (DocumentStream, FileWatcher)
    """
    stream = DocumentStream(buffer_size=buffer_size, callback=callback)
    watcher = FileWatcher(directory, stream, file_pattern, poll_interval)

    await stream.start()
    await watcher.start()

    return stream, watcher
