"""Tests for the checkpoint/persistence layer."""

import tempfile
from pathlib import Path

from autorag_live.core.checkpoint import Checkpoint, FileCheckpointStore, InMemoryCheckpointStore
from autorag_live.core.context import RAGContext, RetrievedDocument

# ---------------------------------------------------------------------------
# Checkpoint dataclass tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_from_context_round_trip(self):
        ctx = RAGContext.create(query="What is RLHF?")
        ctx = ctx.add_documents([RetrievedDocument(doc_id="d1", content="doc", score=0.9)])
        ctx = ctx.with_answer("An answer", confidence=0.8)

        cp = Checkpoint.from_context(ctx, node="generate", step_index=3)
        assert cp.context_id == ctx.context_id
        assert cp.node == "generate"
        assert cp.step_index == 3

        restored = cp.to_context()
        assert restored.query == "What is RLHF?"
        assert restored.document_count == 1
        assert restored.answer == "An answer"

    def test_serialise_deserialise(self):
        ctx = RAGContext.create(query="test")
        cp = Checkpoint.from_context(ctx, node="retrieve", metadata={"user": "alice"})

        data = cp.to_dict()
        restored = Checkpoint.from_dict(data)

        assert restored.checkpoint_id == cp.checkpoint_id
        assert restored.node == "retrieve"
        assert restored.metadata["user"] == "alice"


# ---------------------------------------------------------------------------
# InMemoryCheckpointStore tests
# ---------------------------------------------------------------------------


class TestInMemoryCheckpointStore:
    def test_save_and_load(self):
        store = InMemoryCheckpointStore()
        ctx = RAGContext.create(query="test")
        cp = Checkpoint.from_context(ctx, node="a")

        store.save(cp)
        loaded = store.load(cp.checkpoint_id)
        assert loaded is not None
        assert loaded.checkpoint_id == cp.checkpoint_id

    def test_load_missing_returns_none(self):
        store = InMemoryCheckpointStore()
        assert store.load("nonexistent") is None

    def test_list_for_context(self):
        store = InMemoryCheckpointStore()
        ctx = RAGContext.create(query="test")
        cp1 = Checkpoint.from_context(ctx, node="a", step_index=1)
        cp2 = Checkpoint.from_context(ctx, node="b", step_index=2)
        other_ctx = RAGContext.create(query="other")
        cp3 = Checkpoint.from_context(other_ctx, node="c")

        store.save(cp1)
        store.save(cp2)
        store.save(cp3)

        results = store.list_for_context(ctx.context_id)
        assert len(results) == 2

    def test_delete(self):
        store = InMemoryCheckpointStore()
        ctx = RAGContext.create(query="test")
        cp = Checkpoint.from_context(ctx, node="a")
        store.save(cp)

        assert store.delete(cp.checkpoint_id) is True
        assert store.delete(cp.checkpoint_id) is False
        assert store.load(cp.checkpoint_id) is None

    def test_latest_for_context(self):
        store = InMemoryCheckpointStore()
        ctx = RAGContext.create(query="test")
        cp1 = Checkpoint.from_context(ctx, node="a", step_index=1)
        cp2 = Checkpoint.from_context(ctx, node="b", step_index=2)
        store.save(cp1)
        store.save(cp2)

        latest = store.latest_for_context(ctx.context_id)
        assert latest is not None

    def test_clear(self):
        store = InMemoryCheckpointStore()
        store.save(Checkpoint.from_context(RAGContext.create(query="t")))
        assert len(store) == 1
        store.clear()
        assert len(store) == 0


# ---------------------------------------------------------------------------
# FileCheckpointStore tests
# ---------------------------------------------------------------------------


class TestFileCheckpointStore:
    def test_save_and_load(self, tmp_path: Path):
        store = FileCheckpointStore(tmp_path / "checkpoints")
        ctx = RAGContext.create(query="test")
        cp = Checkpoint.from_context(ctx, node="retrieve")

        store.save(cp)
        loaded = store.load(cp.checkpoint_id)
        assert loaded is not None
        assert loaded.context_id == ctx.context_id
        assert loaded.node == "retrieve"

    def test_load_missing_returns_none(self, tmp_path: Path):
        store = FileCheckpointStore(tmp_path / "checkpoints")
        assert store.load("nonexistent") is None

    def test_delete(self, tmp_path: Path):
        store = FileCheckpointStore(tmp_path / "checkpoints")
        ctx = RAGContext.create(query="test")
        cp = Checkpoint.from_context(ctx, node="a")
        store.save(cp)

        assert store.delete(cp.checkpoint_id) is True
        assert store.delete(cp.checkpoint_id) is False

    def test_list_for_context(self, tmp_path: Path):
        store = FileCheckpointStore(tmp_path / "checkpoints")
        ctx = RAGContext.create(query="test")
        cp1 = Checkpoint.from_context(ctx, node="a")
        cp2 = Checkpoint.from_context(ctx, node="b")
        store.save(cp1)
        store.save(cp2)

        results = store.list_for_context(ctx.context_id)
        assert len(results) == 2

    def test_creates_directory(self, tmp_path: Path):
        target = tmp_path / "nested" / "dir" / "checkpoints"
        FileCheckpointStore(target)
        assert target.exists()

    def test_protocol_compliance(self):
        """FileCheckpointStore should satisfy the CheckpointStore protocol."""
        from autorag_live.core.checkpoint import CheckpointStore

        assert isinstance(FileCheckpointStore(tempfile.mkdtemp()), CheckpointStore)

    def test_inmemory_protocol_compliance(self):
        from autorag_live.core.checkpoint import CheckpointStore

        assert isinstance(InMemoryCheckpointStore(), CheckpointStore)
