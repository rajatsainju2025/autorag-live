"""
Tests for streaming data ingestion.
"""
import asyncio

import pytest

from autorag_live.data.streaming import (
    Document,
    DocumentEvent,
    DocumentStream,
    FileWatcher,
    StreamBuffer,
    StreamEvent,
    create_stream_from_directory,
)


@pytest.fixture
def sample_document():
    """Create a sample document."""
    return Document(id="doc1", content="Test content", metadata={"source": "test"})


@pytest.fixture
def sample_event(sample_document):
    """Create a sample stream event."""
    return StreamEvent(event_type=DocumentEvent.ADD, document=sample_document)


class TestDocument:
    """Test suite for Document class."""

    def test_document_creation(self):
        """Test document creation."""
        doc = Document(id="test", content="content", metadata={"key": "value"})

        assert doc.id == "test"
        assert doc.content == "content"
        assert doc.metadata == {"key": "value"}
        assert doc.hash is not None

    def test_document_hash_generation(self):
        """Test automatic hash generation."""
        doc1 = Document(id="doc1", content="same content")
        doc2 = Document(id="doc2", content="same content")

        assert doc1.hash == doc2.hash

    def test_document_different_content_hash(self):
        """Test different content produces different hashes."""
        doc1 = Document(id="doc1", content="content A")
        doc2 = Document(id="doc2", content="content B")

        assert doc1.hash != doc2.hash


class TestStreamBuffer:
    """Test suite for StreamBuffer."""

    @pytest.mark.asyncio
    async def test_buffer_add(self, sample_event):
        """Test adding events to buffer."""
        buffer = StreamBuffer(max_size=10)

        added = await buffer.add(sample_event)
        assert added is True
        assert len(buffer._buffer) == 1

    @pytest.mark.asyncio
    async def test_buffer_deduplication(self):
        """Test deduplication by content hash."""
        buffer = StreamBuffer(max_size=10, deduplicate=True)

        doc1 = Document(id="doc1", content="same content")
        doc2 = Document(id="doc2", content="same content")  # Same content, different ID

        event1 = StreamEvent(event_type=DocumentEvent.ADD, document=doc1)
        event2 = StreamEvent(event_type=DocumentEvent.ADD, document=doc2)

        added1 = await buffer.add(event1)
        added2 = await buffer.add(event2)

        assert added1 is True
        assert added2 is False  # Deduplicated
        assert len(buffer._buffer) == 1

    @pytest.mark.asyncio
    async def test_buffer_size_flush(self):
        """Test flush triggered by size."""
        buffer = StreamBuffer(max_size=3, max_wait_seconds=10.0)

        # Add events up to max_size
        for i in range(3):
            doc = Document(id=f"doc{i}", content=f"content{i}")
            event = StreamEvent(event_type=DocumentEvent.ADD, document=doc)
            await buffer.add(event)

        should_flush = await buffer.should_flush()
        assert should_flush is True

    @pytest.mark.asyncio
    async def test_buffer_time_flush(self):
        """Test flush triggered by time."""
        buffer = StreamBuffer(max_size=100, max_wait_seconds=0.1)

        doc = Document(id="doc1", content="content")
        event = StreamEvent(event_type=DocumentEvent.ADD, document=doc)
        await buffer.add(event)

        # Initially should not flush
        assert await buffer.should_flush() is False

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Now should flush
        assert await buffer.should_flush() is True

    @pytest.mark.asyncio
    async def test_buffer_flush(self):
        """Test flushing buffer."""
        buffer = StreamBuffer(max_size=10)

        # Add events
        for i in range(3):
            doc = Document(id=f"doc{i}", content=f"content{i}")
            event = StreamEvent(event_type=DocumentEvent.ADD, document=doc)
            await buffer.add(event)

        # Flush
        events = await buffer.flush()

        assert len(events) == 3
        assert len(buffer._buffer) == 0

    @pytest.mark.asyncio
    async def test_buffer_clear(self):
        """Test clearing buffer."""
        buffer = StreamBuffer(max_size=10, deduplicate=True)

        doc = Document(id="doc1", content="content")
        event = StreamEvent(event_type=DocumentEvent.ADD, document=doc)
        await buffer.add(event)

        await buffer.clear()

        assert len(buffer._buffer) == 0
        assert len(buffer._seen_hashes) == 0


class TestDocumentStream:
    """Test suite for DocumentStream."""

    @pytest.mark.asyncio
    async def test_stream_start_stop(self):
        """Test starting and stopping stream."""
        stream = DocumentStream(buffer_size=10)

        await stream.start()
        assert stream._running is True

        await stream.stop()
        assert stream._running is False

    @pytest.mark.asyncio
    async def test_add_document(self):
        """Test adding document to stream."""
        stream = DocumentStream(buffer_size=10)
        await stream.start()

        doc = Document(id="doc1", content="content")
        added = await stream.add_document(doc)

        assert added is True
        assert stream.get_stats()["events_processed"] == 0  # Not processed yet

        await stream.stop()

    @pytest.mark.asyncio
    async def test_add_documents_bulk(self):
        """Test bulk document addition."""
        stream = DocumentStream(buffer_size=10)
        await stream.start()

        docs = [Document(id=f"doc{i}", content=f"content{i}") for i in range(5)]
        added_count = await stream.add_documents_bulk(docs)

        assert added_count == 5

        await stream.stop()

    @pytest.mark.asyncio
    async def test_stream_callback(self):
        """Test stream callback processing."""
        processed_events = []

        def callback(events):
            processed_events.extend(events)

        stream = DocumentStream(buffer_size=2, callback=callback)
        await stream.start()

        # Add documents to trigger flush
        for i in range(3):
            doc = Document(id=f"doc{i}", content=f"content{i}")
            await stream.add_document(doc)

        # Wait for processing
        await asyncio.sleep(0.3)
        await stream.stop()

        # Check callback was called
        assert len(processed_events) >= 2

    @pytest.mark.asyncio
    async def test_stream_statistics(self):
        """Test stream statistics tracking."""
        stream = DocumentStream(buffer_size=10)
        await stream.start()

        # Add various events
        doc1 = Document(id="doc1", content="content1")
        await stream.add_document(doc1, DocumentEvent.ADD)

        doc2 = Document(id="doc2", content="content2")
        await stream.add_document(doc2, DocumentEvent.UPDATE)

        await asyncio.sleep(0.2)
        await stream.stop()

        stats = stream.get_stats()
        assert "events_processed" in stats
        assert "documents_added" in stats
        assert "is_running" in stats


class TestFileWatcher:
    """Test suite for FileWatcher."""

    @pytest.mark.asyncio
    async def test_file_watcher_initialization(self, tmp_path):
        """Test file watcher initialization."""
        stream = DocumentStream(buffer_size=10)

        watcher = FileWatcher(
            watch_path=tmp_path, stream=stream, file_pattern="*.txt", poll_interval=0.1
        )

        assert watcher.watch_path == tmp_path
        assert watcher.stream == stream

    @pytest.mark.asyncio
    async def test_file_watcher_detects_new_file(self, tmp_path):
        """Test detecting new files."""
        processed_docs = []

        def callback(events):
            processed_docs.extend([e.document for e in events])

        stream = DocumentStream(buffer_size=10, callback=callback)
        await stream.start()

        watcher = FileWatcher(
            watch_path=tmp_path, stream=stream, file_pattern="*.txt", poll_interval=0.1
        )
        await watcher.start()

        # Create a new file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        # Wait for detection
        await asyncio.sleep(0.3)

        await watcher.stop()
        await stream.stop()

        # Check file was processed
        assert len(processed_docs) > 0
        assert any("Test content" in doc.content for doc in processed_docs)

    @pytest.mark.asyncio
    async def test_file_watcher_detects_modified_file(self, tmp_path):
        """Test detecting modified files."""
        processed_docs = []

        def callback(events):
            processed_docs.extend([e.document for e in events])

        stream = DocumentStream(buffer_size=10, callback=callback)
        await stream.start()

        # Create initial file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Initial content")

        watcher = FileWatcher(
            watch_path=tmp_path, stream=stream, file_pattern="*.txt", poll_interval=0.1
        )
        await watcher.start()

        await asyncio.sleep(0.2)

        # Modify the file
        test_file.write_text("Modified content")

        # Wait for detection
        await asyncio.sleep(0.3)

        await watcher.stop()
        await stream.stop()

        # Check both versions were processed
        assert len(processed_docs) >= 2


class TestStreamIntegration:
    """Integration tests for streaming components."""

    @pytest.mark.asyncio
    async def test_create_stream_from_directory(self, tmp_path):
        """Test convenience function for stream creation."""
        processed_events = []

        def callback(events):
            processed_events.extend(events)

        stream, watcher = await create_stream_from_directory(
            directory=tmp_path,
            callback=callback,
            file_pattern="*.txt",
            buffer_size=10,
            poll_interval=0.1,
        )

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        # Wait for processing
        await asyncio.sleep(0.3)

        await watcher.stop()
        await stream.stop()

        # Verify processing
        assert len(processed_events) > 0

    @pytest.mark.asyncio
    async def test_end_to_end_streaming(self, tmp_path):
        """Test complete streaming workflow."""
        results = []

        def process_batch(events):
            for event in events:
                results.append({"type": event.event_type.value, "content": event.document.content})

        stream = DocumentStream(buffer_size=5, buffer_wait_seconds=0.5, callback=process_batch)
        await stream.start()

        # Add documents
        for i in range(10):
            doc = Document(id=f"doc{i}", content=f"Content {i}")
            await stream.add_document(doc)

        # Wait for processing
        await asyncio.sleep(1.0)
        await stream.stop()

        # Verify results
        assert len(results) == 10
        assert all(r["type"] == "add" for r in results)
