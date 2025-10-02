"""Tests for time-series note support."""

import tempfile
from datetime import datetime, timedelta

import numpy as np
import pytest

from autorag_live.data.time_series import (
    FFTEmbedder,
    TimeSeriesNote,
    TimeSeriesRetriever,
    create_time_series_retriever,
)


class TestTimeSeriesNote:
    """Test TimeSeriesNote functionality."""

    def test_note_creation(self):
        """Test creating a time-series note."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        metadata = {"category": "work", "priority": "high"}

        note = TimeSeriesNote(content="Meeting notes", timestamp=timestamp, metadata=metadata)

        assert note.content == "Meeting notes"
        assert note.timestamp == timestamp
        assert note.metadata == metadata
        assert note.embedding is None

    def test_note_with_embedding(self):
        """Test note with pre-computed embedding."""
        embedding = np.array([0.1, 0.2, 0.3])
        note = TimeSeriesNote(
            content="Test", timestamp=datetime.now(), metadata={}, embedding=embedding
        )

        assert note.embedding is not None
        assert np.array_equal(note.embedding, embedding)

    def test_note_invalid_timestamp(self):
        """Test note creation with invalid timestamp."""
        with pytest.raises(ValueError, match="timestamp must be a datetime object"):
            TimeSeriesNote(
                content="Test",
                timestamp="invalid",  # type: ignore # Invalid type for testing
                metadata={},
            )

    def test_note_to_dict(self):
        """Test converting note to dictionary."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        note = TimeSeriesNote(
            content="Test content", timestamp=timestamp, metadata={"key": "value"}
        )

        note_dict = note.to_dict()

        assert note_dict["content"] == "Test content"
        assert note_dict["timestamp"] == timestamp.isoformat()
        assert note_dict["metadata"] == {"key": "value"}
        assert note_dict["embedding"] is None

    def test_note_from_dict(self):
        """Test creating note from dictionary."""
        timestamp_str = "2023-01-01T12:00:00"
        embedding = [0.1, 0.2, 0.3]

        note_dict = {
            "content": "Test content",
            "timestamp": timestamp_str,
            "metadata": {"key": "value"},
            "embedding": embedding,
        }

        note = TimeSeriesNote.from_dict(note_dict)

        assert note.content == "Test content"
        assert note.timestamp == datetime.fromisoformat(timestamp_str)
        assert note.metadata == {"key": "value"}
        assert note.embedding is not None
        assert np.array_equal(note.embedding, np.array(embedding))


class TestFFTEmbedder:
    """Test FFT embedder functionality."""

    def test_embedder_initialization(self):
        """Test FFT embedder initialization."""
        embedder = FFTEmbedder(window_size=50, overlap=0.25, n_fft=64, embedding_dim=256)

        assert embedder.window_size == 50
        assert embedder.overlap == 0.25
        assert embedder.n_fft == 64
        assert embedder.embedding_dim == 256
        assert len(embedder.window) == 50

    def test_text_to_signal(self):
        """Test converting text to numerical signal."""
        embedder = FFTEmbedder()
        signal = embedder._text_to_signal("hello")

        assert isinstance(signal, np.ndarray)
        assert len(signal) == 5  # Length of "hello"
        assert signal.dtype == np.float32

    def test_embed_text(self):
        """Test generating embedding for text."""
        embedder = FFTEmbedder(embedding_dim=128)
        embedding = embedder.embed_text("This is a test document for embedding.")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (128,)
        assert embedding.dtype == np.float32

        # Check normalization
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6

    def test_embed_batch(self):
        """Test batch embedding generation."""
        embedder = FFTEmbedder(embedding_dim=64)
        texts = ["First document", "Second document", "Third document"]

        embeddings = embedder.embed_batch(texts)

        assert embeddings.shape == (3, 64)
        assert embeddings.dtype == np.float32

        # Check normalization for each embedding
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 1e-6


class TestTimeSeriesRetriever:
    """Test TimeSeriesRetriever functionality."""

    def test_retriever_initialization(self):
        """Test retriever initialization."""
        embedder = FFTEmbedder(embedding_dim=128)
        retriever = TimeSeriesRetriever(embedder=embedder, temporal_weight=0.4, content_weight=0.6)

        assert retriever.embedder == embedder
        assert retriever.temporal_weight == 0.4
        assert retriever.content_weight == 0.6
        assert retriever.notes == []
        assert retriever.embeddings is None

    def test_add_note(self):
        """Test adding a note to retriever."""
        retriever = TimeSeriesRetriever()
        timestamp = datetime(2023, 1, 1, 12, 0, 0)

        note = retriever.add_note(
            content="Meeting notes", timestamp=timestamp, metadata={"category": "work"}
        )

        assert len(retriever.notes) == 1
        assert retriever.notes[0] == note
        assert note.content == "Meeting notes"
        assert note.timestamp == timestamp
        assert note.metadata == {"category": "work"}
        assert note.embedding is not None
        assert retriever.embeddings is not None
        assert retriever.embeddings.shape == (1, 384)  # Default embedding dim

    def test_temporal_similarity(self):
        """Test temporal similarity calculation."""
        retriever = TimeSeriesRetriever()

        query_time = datetime(2023, 1, 1, 12, 0, 0)
        note_time = datetime(2023, 1, 2, 12, 0, 0)  # 1 day later

        similarity = retriever._temporal_similarity(query_time, note_time)

        # Should be high similarity for close timestamps
        assert 0.9 < similarity <= 1.0

        # Test with distant timestamp
        distant_time = datetime(2023, 2, 1, 12, 0, 0)  # 31 days later
        distant_similarity = retriever._temporal_similarity(query_time, distant_time)

        assert distant_similarity < similarity

    def test_content_similarity(self):
        """Test content similarity calculation."""
        retriever = TimeSeriesRetriever()

        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])  # Orthogonal vectors

        similarity = retriever._content_similarity(emb1, emb2)
        assert similarity == 0.0

        # Test identical vectors
        similarity_identical = retriever._content_similarity(emb1, emb1)
        assert similarity_identical == 1.0

    def test_search(self):
        """Test searching for notes."""
        retriever = TimeSeriesRetriever()

        # Add some test notes
        base_time = datetime(2023, 1, 1, 12, 0, 0)
        retriever.add_note("Python programming", base_time, {"topic": "coding"})
        retriever.add_note("Machine learning", base_time + timedelta(days=1), {"topic": "ML"})
        retriever.add_note("Data science", base_time + timedelta(days=2), {"topic": "data"})

        # Search
        results = retriever.search("programming", base_time, top_k=2)

        assert len(results) == 2
        assert all("score" in result for result in results)
        assert all("content" in result for result in results)
        assert all("timestamp" in result for result in results)

        # Results should be sorted by combined score
        scores = [r["combined_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_time_window(self):
        """Test search with time window filter."""
        retriever = TimeSeriesRetriever()

        base_time = datetime(2023, 1, 1, 12, 0, 0)
        retriever.add_note("Recent note", base_time, {})
        retriever.add_note("Old note", base_time - timedelta(days=10), {})

        # Search with 5-day window
        results = retriever.search("note", base_time, time_window_days=5)

        # Should only return the recent note
        assert len(results) == 1
        assert "Recent note" in results[0]["content"]

    def test_get_temporal_distribution(self):
        """Test getting temporal distribution."""
        retriever = TimeSeriesRetriever()

        base_time = datetime(2023, 1, 1, 12, 0, 0)
        for i in range(10):
            retriever.add_note(f"Note {i}", base_time + timedelta(days=i), {})

        distribution = retriever.get_temporal_distribution(bins=5)

        assert "distribution" in distribution
        assert len(distribution["distribution"]) == 5
        assert distribution["num_notes"] == 10

        total_count = sum(bin_data["count"] for bin_data in distribution["distribution"])
        assert total_count == 10

    def test_save_and_load(self):
        """Test saving and loading retriever."""
        retriever = TimeSeriesRetriever()

        # Add some notes
        base_time = datetime(2023, 1, 1, 12, 0, 0)
        retriever.add_note("Test note 1", base_time, {"id": 1})
        retriever.add_note("Test note 2", base_time + timedelta(days=1), {"id": 2})

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            temp_path = f.name

        try:
            # Save
            retriever.save(temp_path)

            # Load
            loaded_retriever = TimeSeriesRetriever.load(temp_path)

            assert len(loaded_retriever.notes) == 2
            assert loaded_retriever.temporal_weight == retriever.temporal_weight
            assert loaded_retriever.content_weight == retriever.content_weight

            # Check notes
            assert loaded_retriever.notes[0].content == "Test note 1"
            assert loaded_retriever.notes[1].content == "Test note 2"

        finally:
            import os

            os.unlink(temp_path)

    def test_empty_search(self):
        """Test search with no notes."""
        retriever = TimeSeriesRetriever()
        results = retriever.search("query", datetime.now())

        assert results == []


class TestCreateTimeSeriesRetriever:
    """Test time-series retriever factory function."""

    def test_create_retriever(self):
        """Test creating retriever with factory function."""
        retriever = create_time_series_retriever(temporal_weight=0.5, content_weight=0.5)

        assert isinstance(retriever, TimeSeriesRetriever)
        assert retriever.temporal_weight == 0.5
        assert retriever.content_weight == 0.5
