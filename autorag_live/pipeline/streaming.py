"""
End-to-end streaming pipeline for agentic RAG.

The project had streaming in isolated modules (streaming_handler,
streaming_advanced, streaming_orchestrator) but no unified way to
stream from query→retrieve→generate→response as async events.

This module provides:
    1. ``StreamEvent`` — typed event emitted at each pipeline stage
    2. ``StreamingPipeline`` — wraps a ``CompiledGraph`` and yields events
    3. Consumers can render partial answers, show retrieval progress, etc.

Architecture:
    - Each graph node emits ``StreamEvent`` objects via an async queue
    - ``StreamingPipeline.stream()`` is an ``AsyncIterator[StreamEvent]``
    - Events carry the stage name, event type, and payload
    - Compatible with SSE, WebSocket, or Streamlit rendering

Example:
    >>> pipeline = StreamingPipeline(compiled_graph)
    >>> async for event in pipeline.stream(RAGContext.create(query="What is RLHF?")):
    ...     if event.event_type == EventType.STAGE_START:
    ...         print(f"Starting {event.stage}...")
    ...     elif event.event_type == EventType.TOKEN:
    ...         print(event.data["token"], end="")
    ...     elif event.event_type == EventType.COMPLETE:
    ...         print(f"\\nDone! Confidence: {event.data['confidence']}")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

from autorag_live.core.context import RAGContext
from autorag_live.core.state_graph import CompiledGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Types of streaming events."""

    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    DOCUMENTS_RETRIEVED = "documents_retrieved"
    TOKEN = "token"  # Incremental generation token
    REASONING = "reasoning"  # Agent reasoning step
    ERROR = "error"
    COMPLETE = "complete"
    METADATA = "metadata"


@dataclass(frozen=True)
class StreamEvent:
    """
    A single streaming event from the pipeline.

    Attributes:
        event_type:  What kind of event this is.
        stage:       Which pipeline node emitted it.
        data:        Event-specific payload.
        timestamp:   When the event was created.
        context_id:  RAGContext ID for correlation.
    """

    event_type: EventType
    stage: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event_type.value,
            "stage": self.stage,
            "data": self.data,
            "timestamp": self.timestamp,
            "context_id": self.context_id,
        }

    def to_sse(self) -> str:
        """Format as Server-Sent Event string."""
        import json

        return f"event: {self.event_type.value}\ndata: {json.dumps(self.data)}\n\n"


# ---------------------------------------------------------------------------
# StreamingPipeline
# ---------------------------------------------------------------------------


class StreamingPipeline:
    """
    Wraps a ``CompiledGraph`` and yields ``StreamEvent`` objects as the
    pipeline executes, enabling real-time UI rendering.
    """

    def __init__(
        self,
        graph: CompiledGraph,
        *,
        buffer_size: int = 100,
    ):
        self._graph = graph
        self._buffer_size = buffer_size

    async def stream(
        self,
        context: RAGContext,
        *,
        max_steps: int = 50,
    ) -> AsyncIterator[StreamEvent]:
        """
        Execute the graph and yield streaming events.

        This is the primary API for consumers — iterate with ``async for``.
        """
        queue: asyncio.Queue[Optional[StreamEvent]] = asyncio.Queue(maxsize=self._buffer_size)

        async def _run() -> None:
            """Run graph in background, pushing events to queue."""
            try:
                await queue.put(
                    StreamEvent(
                        event_type=EventType.PIPELINE_START,
                        data={"query": context.query, "nodes": self._graph.node_names},
                        context_id=context.context_id,
                    )
                )

                def on_start(node: str, ctx: RAGContext) -> None:
                    queue.put_nowait(
                        StreamEvent(
                            event_type=EventType.STAGE_START,
                            stage=node,
                            data={"doc_count": ctx.document_count},
                            context_id=ctx.context_id,
                        ),
                    )

                def on_end(node: str, ctx: RAGContext, latency_ms: float) -> None:
                    event_data: Dict[str, Any] = {
                        "latency_ms": round(latency_ms, 1),
                        "doc_count": ctx.document_count,
                    }
                    if ctx.answer:
                        event_data["answer_preview"] = ctx.answer[:100]
                    if ctx.documents:
                        event_data["top_score"] = max(d.score for d in ctx.documents)

                    queue.put_nowait(
                        StreamEvent(
                            event_type=EventType.STAGE_END,
                            stage=node,
                            data=event_data,
                            context_id=ctx.context_id,
                        ),
                    )

                result = await self._graph.invoke(
                    context,
                    max_steps=max_steps,
                    on_node_start=on_start,
                    on_node_end=on_end,
                )

                # Emit completion event
                final_ctx = result.context
                await queue.put(
                    StreamEvent(
                        event_type=EventType.COMPLETE,
                        data={
                            "answer": final_ctx.answer,
                            "confidence": final_ctx.confidence,
                            "total_latency_ms": round(result.total_latency_ms, 1),
                            "node_trace": result.node_trace,
                            "has_error": final_ctx.has_error,
                        },
                        context_id=final_ctx.context_id,
                    )
                )

            except Exception as exc:
                await queue.put(
                    StreamEvent(
                        event_type=EventType.ERROR,
                        data={"error": str(exc)},
                        context_id=context.context_id,
                    )
                )
            finally:
                await queue.put(None)  # sentinel

        # Start graph execution in background
        task = asyncio.create_task(_run())

        # Yield events as they arrive
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

        # Ensure task is cleaned up
        await task


# ---------------------------------------------------------------------------
# Helper: collect all events (useful for testing)
# ---------------------------------------------------------------------------


async def collect_stream_events(
    pipeline: StreamingPipeline,
    context: RAGContext,
    **kwargs: Any,
) -> List[StreamEvent]:
    """Collect all streaming events into a list (for testing)."""
    events: List[StreamEvent] = []
    async for event in pipeline.stream(context, **kwargs):
        events.append(event)
    return events
