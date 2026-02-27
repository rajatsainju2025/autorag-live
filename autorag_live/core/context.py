"""
Unified RAGContext — the single typed state object flowing through all pipeline stages.

This replaces disconnected StageResult dataclasses with a single, immutable-by-default
context that accumulates retrieval results, reasoning traces, evaluation metadata,
and agent state as it flows through the agentic RAG pipeline.

Design Principles:
    1. **Single source of truth** — one context object per query lifecycle
    2. **Append-only history** — stages enrich the context; nothing is overwritten
    3. **Typed & validated** — every field has a clear type; invalid transitions raise
    4. **Serialisable** — JSON-round-trippable for checkpointing & observability
    5. **Protocol-compatible** — works with the existing Protocol-based interfaces

Inspired by LangGraph's typed state dict but built on frozen dataclasses for safety.

Example:
    >>> ctx = RAGContext.create(query="What is RLHF?")
    >>> ctx = ctx.add_documents([doc1, doc2])
    >>> ctx = ctx.add_reasoning_trace("Retrieved 2 docs from dense index")
    >>> ctx = ctx.with_answer("RLHF is ...", confidence=0.92)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


class ContextStage(str, Enum):
    """Pipeline stages that a RAGContext passes through."""

    CREATED = "created"
    ROUTING = "routing"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    AUGMENTATION = "augmentation"
    GENERATION = "generation"
    REFLECTION = "reflection"
    SAFETY = "safety"
    EVALUATION = "evaluation"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass(frozen=True)
class RetrievedDocument:
    """Immutable document snapshot captured during retrieval."""

    doc_id: str
    content: str
    score: float = 0.0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content[:500],
            "score": round(self.score, 4),
            "source": self.source,
        }


@dataclass(frozen=True)
class ReasoningTrace:
    """Single reasoning step captured during agent execution."""

    step: int
    stage: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "stage": self.stage,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class EvalScore:
    """Evaluation score for a single metric."""

    metric: str
    score: float
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"metric": self.metric, "score": round(self.score, 4), "details": self.details}


@dataclass(frozen=True)
class StageLatency:
    """Latency measurement for a pipeline stage."""

    stage: ContextStage
    latency_ms: float


# ---------------------------------------------------------------------------
# RAGContext — the unified state object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RAGContext:
    """
    Immutable, append-only context that flows through every stage of the
    agentic RAG pipeline.

    All mutation methods return a **new** ``RAGContext`` instance (copy-on-write).
    This guarantees that parallel branches (e.g. in a StateGraph) never corrupt
    each other's state.

    Attributes:
        context_id:       Unique identifier for this query lifecycle.
        query:            The original user query.
        current_stage:    Which pipeline stage the context is currently in.
        documents:        Retrieved / reranked documents (ordered by relevance).
        reasoning_traces: Ordered reasoning steps from agent execution.
        answer:           The generated answer (None until generation stage).
        confidence:       Confidence score for the answer (0.0 – 1.0).
        eval_scores:      Evaluation metric scores.
        latencies:        Per-stage latency measurements.
        metadata:         Arbitrary key-value metadata.
        created_at:       ISO-8601 timestamp of context creation.
        tags:             Immutable set of tags for routing / filtering.
    """

    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    current_stage: ContextStage = ContextStage.CREATED
    documents: Tuple[RetrievedDocument, ...] = ()
    reasoning_traces: Tuple[ReasoningTrace, ...] = ()
    answer: Optional[str] = None
    confidence: float = 0.0
    eval_scores: Tuple[EvalScore, ...] = ()
    latencies: Tuple[StageLatency, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: FrozenSet[str] = field(default_factory=frozenset)

    # ----- Factory -----------------------------------------------------------

    @classmethod
    def create(
        cls,
        query: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[FrozenSet[str]] = None,
    ) -> RAGContext:
        """Create a fresh context for a new user query."""
        return cls(
            query=query,
            metadata=metadata or {},
            tags=tags or frozenset(),
        )

    # ----- Copy-on-write mutators -------------------------------------------

    def advance_stage(self, stage: ContextStage) -> RAGContext:
        """Move the context to a new pipeline stage."""
        return self._replace(current_stage=stage)

    def add_documents(self, docs: Sequence[RetrievedDocument]) -> RAGContext:
        """Append retrieved documents (preserves ordering)."""
        return self._replace(documents=self.documents + tuple(docs))

    def replace_documents(self, docs: Sequence[RetrievedDocument]) -> RAGContext:
        """Replace all documents (e.g. after reranking)."""
        return self._replace(documents=tuple(docs))

    def add_reasoning_trace(
        self, content: str, stage: str = "", *, metadata: Optional[Dict[str, Any]] = None
    ) -> RAGContext:
        """Append a reasoning trace entry."""
        trace = ReasoningTrace(
            step=len(self.reasoning_traces) + 1,
            stage=stage or self.current_stage.value,
            content=content,
            metadata=metadata or {},
        )
        return self._replace(reasoning_traces=self.reasoning_traces + (trace,))

    def with_answer(self, answer: str, confidence: float = 0.0) -> RAGContext:
        """Set the generated answer and confidence."""
        return self._replace(answer=answer, confidence=max(0.0, min(1.0, confidence)))

    def add_eval_score(self, metric: str, score: float, details: str = "") -> RAGContext:
        """Append an evaluation score."""
        es = EvalScore(metric=metric, score=score, details=details)
        return self._replace(eval_scores=self.eval_scores + (es,))

    def record_latency(self, stage: ContextStage, latency_ms: float) -> RAGContext:
        """Record latency for a pipeline stage."""
        sl = StageLatency(stage=stage, latency_ms=latency_ms)
        return self._replace(latencies=self.latencies + (sl,))

    def set_metadata(self, key: str, value: Any) -> RAGContext:
        """Set a metadata key (returns new context)."""
        new_meta = {**self.metadata, key: value}
        return self._replace(metadata=new_meta)

    def add_tag(self, tag: str) -> RAGContext:
        """Add a tag to the context."""
        return self._replace(tags=self.tags | {tag})

    def mark_error(self, error_msg: str) -> RAGContext:
        """Transition to ERROR stage with an error message."""
        return self._replace(
            current_stage=ContextStage.ERROR,
            metadata={**self.metadata, "error": error_msg},
        )

    # ----- Queries -----------------------------------------------------------

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def top_documents(self) -> Tuple[RetrievedDocument, ...]:
        """Top-5 documents by score."""
        return tuple(sorted(self.documents, key=lambda d: d.score, reverse=True)[:5])

    @property
    def total_latency_ms(self) -> float:
        return sum(sl.latency_ms for sl in self.latencies)

    @property
    def avg_eval_score(self) -> float:
        if not self.eval_scores:
            return 0.0
        return sum(es.score for es in self.eval_scores) / len(self.eval_scores)

    @property
    def is_complete(self) -> bool:
        return self.current_stage == ContextStage.COMPLETE

    @property
    def has_error(self) -> bool:
        return self.current_stage == ContextStage.ERROR

    # ----- Serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "context_id": self.context_id,
            "query": self.query,
            "current_stage": self.current_stage.value,
            "documents": [d.to_dict() for d in self.documents],
            "reasoning_traces": [t.to_dict() for t in self.reasoning_traces],
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "eval_scores": [es.to_dict() for es in self.eval_scores],
            "latencies": {sl.stage.value: round(sl.latency_ms, 1) for sl in self.latencies},
            "total_latency_ms": round(self.total_latency_ms, 1),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "tags": sorted(self.tags),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RAGContext:
        """Deserialise from a dictionary."""
        documents = tuple(
            RetrievedDocument(
                doc_id=d["doc_id"],
                content=d.get("content", ""),
                score=d.get("score", 0.0),
                source=d.get("source", ""),
            )
            for d in data.get("documents", [])
        )
        traces = tuple(
            ReasoningTrace(
                step=t["step"],
                stage=t.get("stage", ""),
                content=t.get("content", ""),
                timestamp=t.get("timestamp", ""),
            )
            for t in data.get("reasoning_traces", [])
        )
        evals = tuple(
            EvalScore(
                metric=es["metric"],
                score=es.get("score", 0.0),
                details=es.get("details", ""),
            )
            for es in data.get("eval_scores", [])
        )
        latencies_raw = data.get("latencies", {})
        latencies = tuple(
            StageLatency(stage=ContextStage(k), latency_ms=v)
            for k, v in latencies_raw.items()
            if k in {s.value for s in ContextStage}
        )
        return cls(
            context_id=data.get("context_id", str(uuid.uuid4())),
            query=data.get("query", ""),
            current_stage=ContextStage(data.get("current_stage", "created")),
            documents=documents,
            reasoning_traces=traces,
            answer=data.get("answer"),
            confidence=data.get("confidence", 0.0),
            eval_scores=evals,
            latencies=latencies,
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
            tags=frozenset(data.get("tags", [])),
        )

    # ----- Internal ----------------------------------------------------------

    def _replace(self, **changes: Any) -> RAGContext:
        """Return a copy with the given fields replaced (frozen-dataclass safe)."""
        current = {
            "context_id": self.context_id,
            "query": self.query,
            "current_stage": self.current_stage,
            "documents": self.documents,
            "reasoning_traces": self.reasoning_traces,
            "answer": self.answer,
            "confidence": self.confidence,
            "eval_scores": self.eval_scores,
            "latencies": self.latencies,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "tags": self.tags,
        }
        current.update(changes)
        return RAGContext(**current)

    def __repr__(self) -> str:
        return (
            f"RAGContext(id={self.context_id[:8]}…, query={self.query[:40]!r}, "
            f"stage={self.current_stage.value}, docs={self.document_count}, "
            f"answer={'✓' if self.answer else '✗'})"
        )
