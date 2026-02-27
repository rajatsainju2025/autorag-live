"""
Checkpoint & persistence layer for long-running agentic RAG workflows.

Enables save/resume for agent state so that multi-step reasoning,
long-running plan-and-execute agents, and multi-turn conversations
can survive process restarts, network failures, or human-in-the-loop
pauses.

Design:
    - ``CheckpointStore`` protocol — pluggable backends (file, SQLite, …)
    - ``FileCheckpointStore`` — default JSON-on-disk backend
    - ``Checkpoint`` dataclass — a timestamped snapshot of ``RAGContext``
      plus the graph node the execution was at when paused

Example:
    >>> store = FileCheckpointStore("/tmp/checkpoints")
    >>> cp = Checkpoint.from_context(ctx, node="grade_docs")
    >>> store.save(cp)
    >>>
    >>> # Later, resume:
    >>> restored = store.load(cp.checkpoint_id)
    >>> ctx = restored.to_context()
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from autorag_live.core.context import RAGContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Checkpoint:
    """
    Immutable snapshot of an in-flight agentic RAG execution.

    Attributes:
        checkpoint_id:  Unique identifier (UUIDv4).
        context_id:     The RAGContext.context_id this checkpoint belongs to.
        node:           The graph node name where execution was paused.
        context_data:   Serialised RAGContext (dict form).
        step_index:     How many graph steps had completed.
        created_at:     ISO-8601 timestamp.
        metadata:       Arbitrary extra data (e.g. user_id, session_id).
    """

    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str = ""
    node: str = ""
    context_data: Dict[str, Any] = field(default_factory=dict)
    step_index: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ----- Factories ---------------------------------------------------------

    @classmethod
    def from_context(
        cls,
        context: RAGContext,
        node: str = "",
        step_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """Create a checkpoint from a live RAGContext."""
        return cls(
            context_id=context.context_id,
            node=node,
            context_data=context.to_dict(),
            step_index=step_index,
            metadata=metadata or {},
        )

    def to_context(self) -> RAGContext:
        """Restore a RAGContext from this checkpoint."""
        return RAGContext.from_dict(self.context_data)

    # ----- Serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "context_id": self.context_id,
            "node": self.node,
            "context_data": self.context_data,
            "step_index": self.step_index,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Checkpoint:
        return cls(
            checkpoint_id=data.get("checkpoint_id", str(uuid.uuid4())),
            context_id=data.get("context_id", ""),
            node=data.get("node", ""),
            context_data=data.get("context_data", {}),
            step_index=data.get("step_index", 0),
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# CheckpointStore protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for checkpoint persistence backends."""

    def save(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint."""
        ...

    def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint by ID.  Returns None if not found."""
        ...

    def list_for_context(self, context_id: str) -> List[Checkpoint]:
        """List all checkpoints for a given context_id, newest first."""
        ...

    def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.  Returns True if it existed."""
        ...

    def latest_for_context(self, context_id: str) -> Optional[Checkpoint]:
        """Return the most recent checkpoint for a context_id."""
        ...


# ---------------------------------------------------------------------------
# FileCheckpointStore — default JSON-on-disk backend
# ---------------------------------------------------------------------------


class FileCheckpointStore:
    """
    Simple file-based checkpoint store.

    Each checkpoint is saved as a JSON file named ``<checkpoint_id>.json``
    inside the given directory.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, checkpoint_id: str) -> Path:
        return self.directory / f"{checkpoint_id}.json"

    def save(self, checkpoint: Checkpoint) -> None:
        path = self._path(checkpoint.checkpoint_id)
        path.write_text(json.dumps(checkpoint.to_dict(), indent=2, default=str))
        logger.debug("Saved checkpoint %s to %s", checkpoint.checkpoint_id[:8], path)

    def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        path = self._path(checkpoint_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return Checkpoint.from_dict(data)

    def list_for_context(self, context_id: str) -> List[Checkpoint]:
        results: List[Checkpoint] = []
        for fp in self.directory.glob("*.json"):
            try:
                data = json.loads(fp.read_text())
                if data.get("context_id") == context_id:
                    results.append(Checkpoint.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping malformed checkpoint file: %s", fp)
        results.sort(key=lambda c: c.created_at, reverse=True)
        return results

    def delete(self, checkpoint_id: str) -> bool:
        path = self._path(checkpoint_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def latest_for_context(self, context_id: str) -> Optional[Checkpoint]:
        checkpoints = self.list_for_context(context_id)
        return checkpoints[0] if checkpoints else None

    def __repr__(self) -> str:
        return f"FileCheckpointStore(dir={self.directory})"


# ---------------------------------------------------------------------------
# InMemoryCheckpointStore — useful for testing
# ---------------------------------------------------------------------------


class InMemoryCheckpointStore:
    """In-memory checkpoint store for testing and short-lived pipelines."""

    def __init__(self) -> None:
        self._store: Dict[str, Checkpoint] = {}

    def save(self, checkpoint: Checkpoint) -> None:
        self._store[checkpoint.checkpoint_id] = checkpoint

    def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        return self._store.get(checkpoint_id)

    def list_for_context(self, context_id: str) -> List[Checkpoint]:
        results = [cp for cp in self._store.values() if cp.context_id == context_id]
        results.sort(key=lambda c: c.created_at, reverse=True)
        return results

    def delete(self, checkpoint_id: str) -> bool:
        return self._store.pop(checkpoint_id, None) is not None

    def latest_for_context(self, context_id: str) -> Optional[Checkpoint]:
        cps = self.list_for_context(context_id)
        return cps[0] if cps else None

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
