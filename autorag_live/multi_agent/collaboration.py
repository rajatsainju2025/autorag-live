"""
Multi-agent collaboration framework for distributed agentic reasoning.

Enables multiple specialized agents to coordinate, communicate, and resolve
disagreements through consensus mechanisms.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentRole(Enum):
    """Specialized agent roles."""

    RETRIEVER = "retriever"  # Document retrieval specialist
    REASONER = "reasoner"  # Logical reasoning specialist
    SYNTHESIZER = "synthesizer"  # Answer synthesis specialist
    EVALUATOR = "evaluator"  # Quality evaluation specialist
    CRITIC = "critic"  # Answer criticism and refinement


@dataclass
class AgentMessage:
    """Message from one agent to another."""

    sender_id: str
    sender_role: AgentRole
    content: str
    message_type: str  # "proposal", "agreement", "disagreement", "refinement"
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "sender_id": self.sender_id,
            "sender_role": self.sender_role.value,
            "content": self.content,
            "message_type": self.message_type,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class AgentProposal:
    """Proposal from an agent."""

    proposer_id: str
    proposer_role: AgentRole
    proposal: str
    reasoning: str
    confidence: float = 0.5
    supporting_evidence: List[str] = field(default_factory=list)
    agreements: List[str] = field(default_factory=list)
    disagreements: List[str] = field(default_factory=list)

    def get_consensus_score(self) -> float:
        """Calculate consensus score for this proposal."""
        total_votes = len(self.agreements) + len(self.disagreements)

        if total_votes == 0:
            return self.confidence

        agreement_ratio = len(self.agreements) / total_votes
        return (self.confidence * 0.5) + (agreement_ratio * 0.5)


class SpecializedAgent:
    """Base class for specialized agents in multi-agent system."""

    def __init__(self, agent_id: str, role: AgentRole, expertise_domain: str):
        """Initialize specialized agent."""
        self.agent_id = agent_id
        self.role = role
        self.expertise_domain = expertise_domain
        self.logger = logging.getLogger(f"Agent.{agent_id}")
        self.message_history: List[AgentMessage] = []
        self.proposals: List[AgentProposal] = []

    def process_query(self, query: str) -> str:
        """Process query according to agent's specialization."""
        if self.role == AgentRole.RETRIEVER:
            return self._retrieve(query)
        elif self.role == AgentRole.REASONER:
            return self._reason(query)
        elif self.role == AgentRole.SYNTHESIZER:
            return self._synthesize(query)
        elif self.role == AgentRole.EVALUATOR:
            return self._evaluate(query)
        elif self.role == AgentRole.CRITIC:
            return self._critique(query)
        else:
            return ""

    def propose(
        self,
        content: str,
        reasoning: str = "",
        confidence: float = 0.5,
        evidence: Optional[List[str]] = None,
    ) -> AgentProposal:
        """Create proposal for other agents to evaluate."""
        proposal = AgentProposal(
            proposer_id=self.agent_id,
            proposer_role=self.role,
            proposal=content,
            reasoning=reasoning,
            confidence=confidence,
            supporting_evidence=evidence or [],
        )

        self.proposals.append(proposal)
        return proposal

    def agree_with(self, proposal: AgentProposal, reasoning: str = "") -> None:
        """Express agreement with another agent's proposal."""
        proposal.agreements.append(self.agent_id)

        message = AgentMessage(
            sender_id=self.agent_id,
            sender_role=self.role,
            content=f"Agree with {proposal.proposer_id}",
            message_type="agreement",
            confidence=0.8,
            metadata={"reasoning": reasoning},
        )

        self.message_history.append(message)

    def disagree_with(self, proposal: AgentProposal, reason: str = "") -> None:
        """Express disagreement with another agent's proposal."""
        proposal.disagreements.append(self.agent_id)

        message = AgentMessage(
            sender_id=self.agent_id,
            sender_role=self.role,
            content=f"Disagree with {proposal.proposer_id}: {reason}",
            message_type="disagreement",
            confidence=0.5,
        )

        self.message_history.append(message)

    def _retrieve(self, query: str) -> str:
        """Specialized retrieval."""
        return f"Retrieved documents for: {query}"

    def _reason(self, query: str) -> str:
        """Specialized reasoning."""
        return f"Reasoning about: {query}"

    def _synthesize(self, query: str) -> str:
        """Specialized synthesis."""
        return f"Synthesized answer for: {query}"

    def _evaluate(self, query: str) -> str:
        """Specialized evaluation."""
        return f"Evaluated quality of: {query}"

    def _critique(self, query: str) -> str:
        """Specialized criticism."""
        return f"Critical analysis of: {query}"


class MultiAgentOrchestrator:
    """
    Orchestrates multiple specialized agents.

    Coordinates communication, manages proposals, and achieves consensus.
    """

    def __init__(self):
        """Initialize orchestrator."""
        self.logger = logging.getLogger("MultiAgentOrchestrator")
        self.agents: Dict[str, SpecializedAgent] = {}
        self.conversation_history: List[AgentMessage] = []
        self.proposals: List[AgentProposal] = []
        self.consensus_threshold = 0.6
        self._executor = ThreadPoolExecutor(max_workers=8)

    def register_agent(self, agent: SpecializedAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.agent_id}")

    def orchestrate_query(self, query: str) -> Dict[str, Any]:
        """
        Orchestrate multiple agents to answer a query.

        Args:
            query: Input query

        Returns:
            Final answer with consensus information
        """
        results = {}

        # Phase 1: Initial processing
        for agent_id, agent in self.agents.items():
            result = agent.process_query(query)
            results[agent_id] = result

            self.logger.info(f"Agent {agent_id} produced: {result[:50]}...")

        # Phase 2: Proposal and agreement
        proposals = []
        for agent_id, agent in self.agents.items():
            proposal = agent.propose(
                content=results[agent_id],
                reasoning=f"From {agent.role.value} perspective",
                confidence=0.7,
            )
            proposals.append(proposal)

        self.proposals.extend(proposals)

        # Phase 3: Consensus building
        consensus_result = self._build_consensus(proposals)

        return {
            "query": query,
            "initial_results": results,
            "proposals": [p.to_dict() for p in proposals],
            "consensus": consensus_result,
        }

    async def orchestrate_query_async(self, query: str) -> Dict[str, Any]:
        """Orchestrate agents with async parallel execution.

        Phases 1 (processing) and 2 (proposals) fan out to all agents
        concurrently via asyncio.gather + ThreadPoolExecutor, giving ~N×
        speedup where N = number of agents.

        Args:
            query: Input query

        Returns:
            Final answer with consensus information
        """
        loop = asyncio.get_event_loop()

        # Phase 1: Process all agents in parallel
        async def _run_agent(agent_id: str, agent: SpecializedAgent) -> tuple:
            result = await loop.run_in_executor(self._executor, agent.process_query, query)
            self.logger.info("Agent %s produced: %s...", agent_id, result[:50])
            return agent_id, result

        agent_tasks = [_run_agent(aid, ag) for aid, ag in self.agents.items()]
        results_list = await asyncio.gather(*agent_tasks)
        results = dict(results_list)

        # Phase 2: Proposals (cheap, but parallelise for consistency)
        proposals: List[AgentProposal] = []
        for agent_id, agent in self.agents.items():
            proposal = agent.propose(
                content=results[agent_id],
                reasoning=f"From {agent.role.value} perspective",
                confidence=0.7,
            )
            proposals.append(proposal)

        self.proposals.extend(proposals)

        # Phase 3: Consensus (lightweight, stays sync)
        consensus_result = self._build_consensus(proposals)

        return {
            "query": query,
            "initial_results": results,
            "proposals": [p.to_dict() for p in proposals],
            "consensus": consensus_result,
        }

    def _build_consensus(self, proposals: List[AgentProposal]) -> Dict[str, Any]:
        """Build consensus from proposals."""
        if not proposals:
            return {"status": "no_proposals"}

        # Calculate consensus scores
        scores = [(p, p.get_consensus_score()) for p in proposals]
        scores.sort(key=lambda x: x[1], reverse=True)

        best_proposal = scores[0][0]
        best_score = scores[0][1]

        # Check if consensus threshold is met
        if best_score >= self.consensus_threshold:
            status = "consensus"
        elif best_score >= 0.5:
            status = "majority"
        else:
            status = "disagreement"

        return {
            "status": status,
            "chosen_proposal": best_proposal.proposal,
            "proposer": best_proposal.proposer_id,
            "consensus_score": best_score,
            "all_scores": [(p.proposer_id, s) for p, s in scores],
        }

    def resolve_disagreement(self, proposals: List[AgentProposal]) -> str:
        """
        Resolve disagreement between agents.

        Implements debate-style resolution.
        """
        if not proposals:
            return ""

        # Request critiques
        critiques = {}
        for proposal in proposals:
            for agent_id, agent in self.agents.items():
                if agent.role == AgentRole.CRITIC:
                    critique = agent.process_query(f"Critique: {proposal.proposal}")
                    if agent_id not in critiques:
                        critiques[agent_id] = []
                    critiques[agent_id].append(critique)

        # Synthesize critiques into refined proposal
        refinement = "Refined proposal considering: "
        refinement += "; ".join([c[0] for c in list(critiques.values())[:3]])

        return refinement

    def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary of all agents and their proposals."""
        return {
            "num_agents": len(self.agents),
            "agents": list(self.agents.keys()),
            "num_proposals": len(self.proposals),
            "avg_consensus_score": (
                sum(p.get_consensus_score() for p in self.proposals) / len(self.proposals)
                if self.proposals
                else 0.0
            ),
        }

    def clear_history(self) -> None:
        """Clear conversation and proposal history."""
        self.conversation_history.clear()
        self.proposals.clear()

        for agent in self.agents.values():
            agent.message_history.clear()
            agent.proposals.clear()


def create_default_agents() -> List[SpecializedAgent]:
    """Create default set of specialized agents."""
    agents = [
        SpecializedAgent("retriever_1", AgentRole.RETRIEVER, "document_search"),
        SpecializedAgent("reasoner_1", AgentRole.REASONER, "logical_analysis"),
        SpecializedAgent("synthesizer_1", AgentRole.SYNTHESIZER, "answer_synthesis"),
        SpecializedAgent("evaluator_1", AgentRole.EVALUATOR, "quality_check"),
        SpecializedAgent("critic_1", AgentRole.CRITIC, "critical_analysis"),
    ]

    return agents
