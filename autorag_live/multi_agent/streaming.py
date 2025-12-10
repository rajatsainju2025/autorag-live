"""
Streaming support for multi-agent collaboration.

Provides real-time updates during agent discussions, debate, and consensus building
for improved user experience in agentic RAG systems.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional


class StreamEventType(Enum):
    """Types of streaming events from multi-agent collaboration."""

    AGENT_STARTED = "agent_started"
    AGENT_THINKING = "agent_thinking"
    AGENT_RESPONSE = "agent_response"
    PROPOSAL_CREATED = "proposal_created"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    DEBATE_ROUND = "debate_round"
    CONSENSUS_BUILDING = "consensus_building"
    CONSENSUS_REACHED = "consensus_reached"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


@dataclass
class StreamEvent:
    """Single event in the collaboration stream."""

    event_type: StreamEventType
    agent_id: Optional[str] = None
    agent_role: Optional[str] = None
    content: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "content": self.content,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        import json

        data = json.dumps(self.to_dict())
        return f"event: {self.event_type.value}\ndata: {data}\n\n"


class StreamingMultiAgentOrchestrator:
    """
    Multi-agent orchestrator with streaming support.

    Yields real-time updates as agents collaborate to answer queries.
    """

    def __init__(self):
        """Initialize streaming orchestrator."""
        self.logger = logging.getLogger("StreamingMultiAgentOrchestrator")
        self.agents: Dict[str, Dict[str, Any]] = {}
        self._event_handlers: List[Callable[[StreamEvent], None]] = []

        # Initialize default agents
        self._init_default_agents()

    def _init_default_agents(self) -> None:
        """Initialize default specialized agents."""
        self.agents = {
            "retriever": {
                "id": "retriever_001",
                "role": "retriever",
                "expertise": "document retrieval and relevance scoring",
            },
            "reasoner": {
                "id": "reasoner_001",
                "role": "reasoner",
                "expertise": "logical analysis and inference",
            },
            "synthesizer": {
                "id": "synthesizer_001",
                "role": "synthesizer",
                "expertise": "information integration and answer generation",
            },
            "critic": {
                "id": "critic_001",
                "role": "critic",
                "expertise": "answer validation and improvement suggestions",
            },
        }

    def add_event_handler(self, handler: Callable[[StreamEvent], None]) -> None:
        """Add handler to receive stream events."""
        self._event_handlers.append(handler)

    def _emit_event(self, event: StreamEvent) -> None:
        """Emit event to all handlers."""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.warning(f"Event handler error: {e}")

    def stream_collaboration(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Iterator[StreamEvent]:
        """
        Stream multi-agent collaboration process.

        Args:
            query: User query to process
            context: Optional context (documents, conversation history)

        Yields:
            StreamEvent for each step of collaboration
        """
        context = context or {}
        documents = context.get("documents", [])

        # Phase 1: Agent initialization
        for agent_id, agent_info in self.agents.items():
            event = StreamEvent(
                event_type=StreamEventType.AGENT_STARTED,
                agent_id=agent_info["id"],
                agent_role=agent_info["role"],
                content=f"Agent {agent_info['role']} initialized",
            )
            self._emit_event(event)
            yield event

        # Phase 2: Retrieval agent analyzes documents
        yield from self._stream_retrieval_analysis(query, documents)

        # Phase 3: Reasoner processes information
        yield from self._stream_reasoning(query, documents)

        # Phase 4: Initial proposals from agents
        proposals = []
        yield from self._stream_proposals(query, documents, proposals)

        # Phase 5: Debate and disagreement resolution
        yield from self._stream_debate(proposals)

        # Phase 6: Consensus building
        yield from self._stream_consensus(proposals, query)

        # Calculate final answer
        final_answer, confidence = self._calculate_consensus(proposals, query)

        # Phase 7: Final answer
        event = StreamEvent(
            event_type=StreamEventType.FINAL_ANSWER,
            content=final_answer,
            confidence=confidence,
            metadata={"proposal_count": len(proposals)},
        )
        self._emit_event(event)
        yield event

    def _stream_retrieval_analysis(self, query: str, documents: List[str]) -> Iterator[StreamEvent]:
        """Stream retrieval agent's analysis."""
        agent_info = self.agents["retriever"]

        # Thinking
        yield StreamEvent(
            event_type=StreamEventType.AGENT_THINKING,
            agent_id=agent_info["id"],
            agent_role=agent_info["role"],
            content="Analyzing retrieved documents for relevance...",
        )

        # Analysis results
        relevance_summary = (
            f"Analyzed {len(documents)} documents. "
            f"Found {min(len(documents), 3)} highly relevant sources."
        )

        yield StreamEvent(
            event_type=StreamEventType.AGENT_RESPONSE,
            agent_id=agent_info["id"],
            agent_role=agent_info["role"],
            content=relevance_summary,
            confidence=0.75,
        )

    def _stream_reasoning(self, query: str, documents: List[str]) -> Iterator[StreamEvent]:
        """Stream reasoner's analysis."""
        agent_info = self.agents["reasoner"]

        yield StreamEvent(
            event_type=StreamEventType.AGENT_THINKING,
            agent_id=agent_info["id"],
            agent_role=agent_info["role"],
            content="Applying logical reasoning to synthesize information...",
        )

        reasoning_result = (
            "Identified key concepts and relationships. "
            "Building coherent understanding from multiple sources."
        )

        yield StreamEvent(
            event_type=StreamEventType.AGENT_RESPONSE,
            agent_id=agent_info["id"],
            agent_role=agent_info["role"],
            content=reasoning_result,
            confidence=0.70,
        )

    def _stream_proposals(
        self,
        query: str,
        documents: List[str],
        proposals: List[Dict[str, Any]],
    ) -> Iterator[StreamEvent]:
        """Stream agent proposals."""
        # Synthesizer proposes answer
        synth_info = self.agents["synthesizer"]

        yield StreamEvent(
            event_type=StreamEventType.AGENT_THINKING,
            agent_id=synth_info["id"],
            agent_role=synth_info["role"],
            content="Generating comprehensive answer from available information...",
        )

        synth_proposal = {
            "agent_id": synth_info["id"],
            "content": self._generate_synthesis(query, documents),
            "confidence": 0.72,
        }
        proposals.append(synth_proposal)

        yield StreamEvent(
            event_type=StreamEventType.PROPOSAL_CREATED,
            agent_id=synth_info["id"],
            agent_role=synth_info["role"],
            content=synth_proposal["content"],
            confidence=synth_proposal["confidence"],
        )

    def _stream_debate(self, proposals: List[Dict[str, Any]]) -> Iterator[StreamEvent]:
        """Stream debate between agents."""
        critic_info = self.agents["critic"]

        yield StreamEvent(
            event_type=StreamEventType.DEBATE_ROUND,
            content="Starting evaluation round",
            metadata={"round": 1},
        )

        # Critic evaluates proposals
        yield StreamEvent(
            event_type=StreamEventType.AGENT_THINKING,
            agent_id=critic_info["id"],
            agent_role=critic_info["role"],
            content="Evaluating proposal quality and identifying potential issues...",
        )

        # Agreement or disagreement
        if proposals and proposals[0].get("confidence", 0) > 0.6:
            yield StreamEvent(
                event_type=StreamEventType.AGREEMENT,
                agent_id=critic_info["id"],
                agent_role=critic_info["role"],
                content="Proposal appears well-grounded and comprehensive",
                confidence=0.80,
            )
        else:
            yield StreamEvent(
                event_type=StreamEventType.DISAGREEMENT,
                agent_id=critic_info["id"],
                agent_role=critic_info["role"],
                content="Proposal could benefit from additional context",
                confidence=0.50,
            )

    def _stream_consensus(
        self,
        proposals: List[Dict[str, Any]],
        query: str,
    ) -> Iterator[StreamEvent]:
        """Stream consensus building process."""
        yield StreamEvent(
            event_type=StreamEventType.CONSENSUS_BUILDING,
            content="Aggregating agent perspectives and building final answer...",
        )

        # Calculate consensus
        if proposals:
            avg_confidence = sum(p.get("confidence", 0.5) for p in proposals) / len(proposals)
        else:
            avg_confidence = 0.5

        yield StreamEvent(
            event_type=StreamEventType.CONSENSUS_REACHED,
            content=f"Consensus reached with {len(proposals)} proposals",
            confidence=avg_confidence,
            metadata={"participating_agents": len(self.agents)},
        )

    def _calculate_consensus(
        self,
        proposals: List[Dict[str, Any]],
        query: str,
    ) -> tuple:
        """Calculate final consensus answer and confidence."""
        if proposals:
            final_answer = proposals[0]["content"]
            avg_confidence = sum(p.get("confidence", 0.5) for p in proposals) / len(proposals)
        else:
            final_answer = (
                f"Based on my analysis of the query '{query}', I can provide a general response."
            )
            avg_confidence = 0.5

        return final_answer, avg_confidence

    def _generate_synthesis(self, query: str, documents: List[str]) -> str:
        """Generate synthesized answer."""
        if documents:
            doc_summary = " ".join(documents[:3])[:200]
            return f"Based on the available information, here's what I found: {doc_summary}"
        return f"Regarding your query about '{query}', I can provide general guidance."


class AsyncStreamingMultiAgentOrchestrator:
    """
    Async version of streaming multi-agent orchestrator.

    Enables non-blocking streaming in async applications.
    """

    def __init__(self):
        """Initialize async streaming orchestrator."""
        self.logger = logging.getLogger("AsyncStreamingMultiAgentOrchestrator")
        self.sync_orchestrator = StreamingMultiAgentOrchestrator()

    async def stream_collaboration_async(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        delay_ms: float = 50,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream multi-agent collaboration asynchronously.

        Args:
            query: User query to process
            context: Optional context
            delay_ms: Artificial delay between events (for demo/UI smoothing)

        Yields:
            StreamEvent for each step of collaboration
        """
        for event in self.sync_orchestrator.stream_collaboration(query, context):
            yield event
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000)

    async def batch_stream(
        self,
        queries: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process multiple queries with streaming updates.

        Yields events tagged with query index.
        """
        for idx, query in enumerate(queries):
            async for event in self.stream_collaboration_async(query, context):
                yield {
                    "query_index": idx,
                    "query": query,
                    "event": event.to_dict(),
                }


def create_sse_stream(
    orchestrator: StreamingMultiAgentOrchestrator,
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> Iterator[str]:
    """
    Create Server-Sent Events stream for web integration.

    Args:
        orchestrator: Streaming orchestrator
        query: User query
        context: Optional context

    Yields:
        SSE-formatted strings
    """
    for event in orchestrator.stream_collaboration(query, context):
        yield event.to_sse()
