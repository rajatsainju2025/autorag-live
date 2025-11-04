"""Time-series note support with FFT-based embeddings."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from autorag_live.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TimeSeriesNote:
    """Represents a time-series note with temporal information."""

    content: str
    timestamp: datetime
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate timestamp and initialize metadata."""
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")

        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert note to dictionary representation."""
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeSeriesNote":
        """Create note from dictionary representation."""
        return cls(
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            embedding=np.array(data["embedding"]) if data.get("embedding") else None,
        )


class FFTEmbedder:
    """FFT-based embedder for time-series data."""

    def __init__(
        self,
        window_size: int = 100,
        overlap: float = 0.5,
        n_fft: Optional[int] = None,
        embedding_dim: int = 384,
    ):
        """Initialize FFT embedder.

        Args:
            window_size: Size of sliding window for FFT
            overlap: Overlap fraction between windows (0-1)
            n_fft: Number of FFT points (default: window_size)
            embedding_dim: Dimension of output embeddings
        """
        self.window_size = window_size
        self.overlap = overlap
        self.n_fft = n_fft or window_size
        self.embedding_dim = embedding_dim

        # Pre-compute window function
        self.window = np.hanning(window_size)

    def _text_to_signal(self, text: str) -> np.ndarray:
        """Convert text to numerical signal for FFT processing."""
        # Simple character-based encoding
        # Could be enhanced with word embeddings or other features
        chars = [ord(c) for c in text]
        signal = np.array(chars, dtype=np.float32)

        # Normalize to [-1, 1]
        if len(signal) > 0:
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        return signal

    def _compute_fft_features(self, signal: np.ndarray) -> np.ndarray:
        """Compute FFT-based features from signal."""
        # Apply window
        if len(signal) >= self.window_size:
            windowed_signal = signal[: self.window_size] * self.window
        else:
            # Pad shorter signals
            windowed_signal = np.pad(signal, (0, self.window_size - len(signal)))
            windowed_signal *= self.window

        # Compute FFT
        fft = np.fft.fft(windowed_signal, n=self.n_fft)
        fft_magnitude = np.abs(fft)

        # Extract features from different frequency bands
        # Low frequencies (0-10% of spectrum)
        low_freq = fft_magnitude[: self.n_fft // 10]
        # Mid frequencies (10-50% of spectrum)
        mid_freq = fft_magnitude[self.n_fft // 10 : self.n_fft // 2]
        # High frequencies (50-100% of spectrum)
        high_freq = fft_magnitude[self.n_fft // 2 :]

        # Statistical features
        features = []
        for freq_band in [low_freq, mid_freq, high_freq]:
            if len(freq_band) > 0:
                features.extend(
                    [
                        np.mean(freq_band),
                        np.std(freq_band),
                        np.max(freq_band),
                        np.min(freq_band),
                        np.sum(freq_band**2),  # Energy
                    ]
                )

        return np.array(features)

    def embed_text(self, text: str) -> np.ndarray:
        """Generate FFT-based embedding for text."""
        signal = self._text_to_signal(text)
        features = self._compute_fft_features(signal)

        # Project to desired embedding dimension
        if len(features) >= self.embedding_dim:
            embedding = features[: self.embedding_dim]
        else:
            # Pad with zeros if needed
            embedding = np.pad(features, (0, self.embedding_dim - len(features)))

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for batch of texts."""
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)

        return np.array(embeddings)


class TimeSeriesRetriever:
    """Retriever for time-series notes using temporal and content features."""

    def __init__(
        self,
        embedder: Optional[FFTEmbedder] = None,
        temporal_weight: float = 0.3,
        content_weight: float = 0.7,
    ):
        """Initialize time-series retriever.

        Args:
            embedder: FFT embedder for content (default: create new)
            temporal_weight: Weight for temporal similarity (0-1)
            content_weight: Weight for content similarity (0-1)
        """
        self.embedder = embedder or FFTEmbedder()
        self.temporal_weight = temporal_weight
        self.content_weight = content_weight

        self.notes: List[TimeSeriesNote] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_note(
        self, content: str, timestamp: datetime, metadata: Optional[Dict[str, Any]] = None
    ) -> TimeSeriesNote:
        """Add a note to the retriever."""
        note = TimeSeriesNote(content=content, timestamp=timestamp, metadata=metadata or {})

        # Generate embedding
        note.embedding = self.embedder.embed_text(content)

        self.notes.append(note)

        # Update embeddings matrix
        if self.embeddings is None:
            self.embeddings = note.embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, note.embedding])

        logger.info(f"Added note with timestamp {timestamp}")
        return note

    def add_notes(self, notes: List[TimeSeriesNote]):
        """Add multiple notes to the retriever."""
        for note in notes:
            if note.embedding is None:
                note.embedding = self.embedder.embed_text(note.content)
            self.notes.append(note)

        # Update embeddings matrix
        if self.notes:
            embeddings_list = [note.embedding for note in self.notes if note.embedding is not None]
            if embeddings_list:
                self.embeddings = np.array(embeddings_list)

        logger.info(f"Added {len(notes)} notes")

    def _temporal_similarity(self, query_time: datetime, note_time: datetime) -> float:
        """Calculate temporal similarity between query and note timestamps."""
        # Convert to timestamps for numerical comparison
        query_ts = query_time.timestamp()
        note_ts = note_time.timestamp()

        # Calculate time difference in days
        time_diff = abs(query_ts - note_ts) / (24 * 3600)

        # Exponential decay similarity (closer = more similar)
        # Half-life of 30 days
        similarity = np.exp(-time_diff / 30.0)

        return float(similarity)

    def _content_similarity(self, query_embedding: np.ndarray, note_embedding: np.ndarray) -> float:
        """Calculate content similarity using cosine similarity."""
        dot_product = np.dot(query_embedding, note_embedding)
        norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(note_embedding)

        if norm_product == 0:
            return 0.0

        return float(dot_product / norm_product)

    def search(
        self,
        query: str,
        query_time: datetime,
        top_k: int = 10,
        time_window_days: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for relevant notes using temporal and content features.

        Args:
            query: Search query text
            query_time: Query timestamp
            top_k: Number of results to return
            time_window_days: Optional time window filter (days)

        Returns:
            List of search results with scores and metadata
        """
        if not self.notes or self.embeddings is None:
            return []

        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        results = []

        for i, note in enumerate(self.notes):
            if note.embedding is None:
                continue

            # Filter by time window if specified
            if time_window_days is not None:
                time_diff = abs((query_time - note.timestamp).total_seconds()) / (24 * 3600)
                if time_diff > time_window_days:
                    continue

            # Calculate similarities
            temporal_sim = self._temporal_similarity(query_time, note.timestamp)
            content_sim = self._content_similarity(query_embedding, note.embedding)

            # Combine similarities
            combined_score = self.temporal_weight * temporal_sim + self.content_weight * content_sim

            result = {
                "content": note.content,
                "timestamp": note.timestamp,
                "temporal_score": temporal_sim,
                "content_score": content_sim,
                "combined_score": combined_score,
                "score": combined_score,  # Alias for compatibility
                "metadata": note.metadata,
                "time_diff_days": abs((query_time - note.timestamp).total_seconds()) / (24 * 3600),
            }
            results.append(result)

        # Sort by combined score
        results.sort(key=lambda x: x["combined_score"], reverse=True)

        return results[:top_k]

    def get_temporal_distribution(self, bins: int = 30) -> Dict[str, Any]:
        """Get temporal distribution of notes."""
        if not self.notes:
            return {}

        timestamps = [note.timestamp for note in self.notes]
        timestamps.sort()

        if len(timestamps) < 2:
            return {"timestamps": timestamps, "distribution": []}

        # Calculate time spans
        start_time = min(timestamps)
        end_time = max(timestamps)
        total_span = (end_time - start_time).total_seconds() / (24 * 3600)  # days

        # Create bins
        bin_edges = [start_time + timedelta(days=i * total_span / bins) for i in range(bins + 1)]

        # Count notes per bin
        distribution = []
        for i in range(bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            # Include the last bin's end edge
            if i == bins - 1:
                count = sum(1 for ts in timestamps if bin_start <= ts <= bin_end)
            else:
                count = sum(1 for ts in timestamps if bin_start <= ts < bin_end)
            distribution.append({"start": bin_start, "end": bin_end, "count": count})

        return {
            "start_time": start_time,
            "end_time": end_time,
            "total_span_days": total_span,
            "num_notes": len(self.notes),
            "distribution": distribution,
        }

    def save(self, path: str):
        """Save retriever state to file."""
        data = {
            "notes": [note.to_dict() for note in self.notes],
            "temporal_weight": self.temporal_weight,
            "content_weight": self.content_weight,
            "embedder_config": {
                "window_size": self.embedder.window_size,
                "overlap": self.embedder.overlap,
                "n_fft": self.embedder.n_fft,
                "embedding_dim": self.embedder.embedding_dim,
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved time-series retriever to {path}")

    @classmethod
    def load(cls, path: str) -> "TimeSeriesRetriever":
        """Load retriever from file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Recreate embedder
        embedder_config = data.get("embedder_config", {})
        embedder = FFTEmbedder(**embedder_config)

        # Create retriever
        retriever = cls(
            embedder=embedder,
            temporal_weight=data.get("temporal_weight", 0.3),
            content_weight=data.get("content_weight", 0.7),
        )

        # Load notes
        notes_data = data.get("notes", [])
        notes = []
        for note_data in notes_data:
            note = TimeSeriesNote.from_dict(note_data)
            notes.append(note)

        retriever.add_notes(notes)

        return retriever


def create_time_series_retriever(**kwargs) -> TimeSeriesRetriever:
    """Factory function to create time-series retriever."""
    return TimeSeriesRetriever(**kwargs)
