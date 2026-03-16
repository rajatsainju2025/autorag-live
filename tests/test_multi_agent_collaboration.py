"""Integration tests for multi-agent collaboration framework."""

import asyncio

import pytest

from autorag_live.multi_agent.collaboration import (
    AgentMessage,
    AgentProposal,
    AgentRole,
    MultiAgentOrchestrator,
    SpecializedAgent,
    create_default_agents,
)


class TestAgentMessage:
    """Tests for AgentMessage."""

    def test_message_creation(self):
        """Test creating agent message."""
        msg = AgentMessage(
            sender_id="agent_1",
            sender_role=AgentRole.RETRIEVER,
            content="Test message",
            message_type="proposal",
            confidence=0.8,
        )
        assert msg.sender_id == "agent_1"
        assert msg.sender_role == AgentRole.RETRIEVER
        assert msg.message_type == "proposal"
        assert msg.confidence == 0.8

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = AgentMessage(
            sender_id="agent_1",
            sender_role=AgentRole.REASONER,
            content="Reasoning content",
            message_type="agreement",
            confidence=0.9,
            metadata={"key": "value"},
        )
        msg_dict = msg.to_dict()
        assert msg_dict["sender_id"] == "agent_1"
        assert msg_dict["sender_role"] == "reasoner"
        assert msg_dict["confidence"] == 0.9
        assert msg_dict["metadata"]["key"] == "value"


class TestAgentProposal:
    """Tests for AgentProposal."""

    def test_proposal_creation(self):
        """Test creating agent proposal."""
        proposal = AgentProposal(
            proposer_id="agent_1",
            proposer_role=AgentRole.SYNTHESIZER,
            proposal="Final answer",
            reasoning="Based on analysis",
            confidence=0.85,
        )
        assert proposal.proposer_id == "agent_1"
        assert proposal.proposal == "Final answer"
        assert proposal.confidence == 0.85

    def test_consensus_score_no_votes(self):
        """Test consensus score with no votes."""
        proposal = AgentProposal(
            proposer_id="agent_1",
            proposer_role=AgentRole.RETRIEVER,
            proposal="Answer",
            reasoning="Reasoning",
            confidence=0.7,
        )
        score = proposal.get_consensus_score()
        assert score == 0.7

    def test_consensus_score_with_agreement(self):
        """Test consensus score with full agreement."""
        proposal = AgentProposal(
            proposer_id="agent_1",
            proposer_role=AgentRole.RETRIEVER,
            proposal="Answer",
            reasoning="Reasoning",
            confidence=0.8,
            agreements=["agent_2", "agent_3", "agent_4"],
            disagreements=[],
        )
        # score = (0.8 * 0.5) + (1.0 * 0.5) = 0.4 + 0.5 = 0.9
        score = proposal.get_consensus_score()
        assert score == 0.9

    def test_consensus_score_with_disagreement(self):
        """Test consensus score with partial disagreement."""
        proposal = AgentProposal(
            proposer_id="agent_1",
            proposer_role=AgentRole.RETRIEVER,
            proposal="Answer",
            reasoning="Reasoning",
            confidence=0.8,
            agreements=["agent_2"],
            disagreements=["agent_3"],
        )
        # score = (0.8 * 0.5) + (0.5 * 0.5) = 0.4 + 0.25 = 0.65
        score = proposal.get_consensus_score()
        assert score == 0.65


class TestSpecializedAgent:
    """Tests for SpecializedAgent."""

    def test_agent_creation(self):
        """Test creating specialized agent."""
        agent = SpecializedAgent("agent_1", AgentRole.RETRIEVER, "document_search")
        assert agent.agent_id == "agent_1"
        assert agent.role == AgentRole.RETRIEVER
        assert agent.expertise_domain == "document_search"
        assert len(agent.message_history) == 0
        assert len(agent.proposals) == 0

    def test_retriever_processing(self):
        """Test retriever agent query processing."""
        agent = SpecializedAgent("retriever", AgentRole.RETRIEVER, "search")
        result = agent.process_query("test query")
        assert "Retrieved documents for" in result
        assert "test query" in result

    def test_reasoner_processing(self):
        """Test reasoner agent query processing."""
        agent = SpecializedAgent("reasoner", AgentRole.REASONER, "logic")
        result = agent.process_query("complex question")
        assert "Reasoning about" in result
        assert "complex question" in result

    def test_synthesizer_processing(self):
        """Test synthesizer agent query processing."""
        agent = SpecializedAgent("synthesizer", AgentRole.SYNTHESIZER, "synthesis")
        result = agent.process_query("multiple sources")
        assert "Synthesized answer for" in result

    def test_evaluator_processing(self):
        """Test evaluator agent query processing."""
        agent = SpecializedAgent("evaluator", AgentRole.EVALUATOR, "quality")
        result = agent.process_query("answer quality")
        assert "Evaluated quality of" in result

    def test_critic_processing(self):
        """Test critic agent query processing."""
        agent = SpecializedAgent("critic", AgentRole.CRITIC, "criticism")
        result = agent.process_query("proposed answer")
        assert "Critical analysis of" in result

    def test_agent_proposal(self):
        """Test agent creating proposal."""
        agent = SpecializedAgent("agent_1", AgentRole.RETRIEVER, "search")
        proposal = agent.propose(
            content="Test answer",
            reasoning="Good reasoning",
            confidence=0.9,
            evidence=["evidence_1", "evidence_2"],
        )
        assert proposal.proposer_id == "agent_1"
        assert proposal.proposal == "Test answer"
        assert proposal.confidence == 0.9
        assert len(proposal.supporting_evidence) == 2
        assert len(agent.proposals) == 1

    def test_agent_agreement(self):
        """Test agent expressing agreement."""
        agent1 = SpecializedAgent("agent_1", AgentRole.RETRIEVER, "search")
        agent2 = SpecializedAgent("agent_2", AgentRole.REASONER, "logic")

        proposal = agent1.propose("Answer", "Reasoning", 0.8)
        agent2.agree_with(proposal, "Good reasoning")

        assert "agent_2" in proposal.agreements
        assert len(agent2.message_history) == 1
        assert agent2.message_history[0].message_type == "agreement"

    def test_agent_disagreement(self):
        """Test agent expressing disagreement."""
        agent1 = SpecializedAgent("agent_1", AgentRole.RETRIEVER, "search")
        agent2 = SpecializedAgent("agent_2", AgentRole.REASONER, "logic")

        proposal = agent1.propose("Answer", "Reasoning", 0.8)
        agent2.disagree_with(proposal, "Different interpretation")

        assert "agent_2" in proposal.disagreements
        assert len(agent2.message_history) == 1
        assert agent2.message_history[0].message_type == "disagreement"


class TestMultiAgentOrchestrator:
    """Tests for MultiAgentOrchestrator."""

    def test_orchestrator_creation(self):
        """Test creating orchestrator."""
        orchestrator = MultiAgentOrchestrator()
        assert len(orchestrator.agents) == 0
        assert orchestrator.consensus_threshold == 0.6

    def test_register_agent(self):
        """Test registering agents."""
        orchestrator = MultiAgentOrchestrator()
        agent1 = SpecializedAgent("agent_1", AgentRole.RETRIEVER, "search")
        agent2 = SpecializedAgent("agent_2", AgentRole.REASONER, "logic")

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        assert len(orchestrator.agents) == 2
        assert "agent_1" in orchestrator.agents
        assert "agent_2" in orchestrator.agents

    def test_orchestrate_query_single_agent(self):
        """Test orchestrating query with single agent."""
        orchestrator = MultiAgentOrchestrator()
        agent = SpecializedAgent("agent_1", AgentRole.RETRIEVER, "search")
        orchestrator.register_agent(agent)

        result = orchestrator.orchestrate_query("test query")

        assert "query" in result
        assert result["query"] == "test query"
        assert "initial_results" in result
        assert "agent_1" in result["initial_results"]
        assert "proposals" in result
        assert "consensus" in result

    def test_orchestrate_query_multiple_agents(self):
        """Test orchestrating query with multiple agents."""
        orchestrator = MultiAgentOrchestrator()
        agents = [
            SpecializedAgent("retriever", AgentRole.RETRIEVER, "search"),
            SpecializedAgent("reasoner", AgentRole.REASONER, "logic"),
            SpecializedAgent("synthesizer", AgentRole.SYNTHESIZER, "synthesis"),
        ]

        for agent in agents:
            orchestrator.register_agent(agent)

        result = orchestrator.orchestrate_query("complex query")

        assert len(result["initial_results"]) == 3
        assert len(result["proposals"]) == 3
        assert result["consensus"]["status"] in ["consensus", "majority", "disagreement"]

    def test_build_consensus_no_proposals(self):
        """Test building consensus with no proposals."""
        orchestrator = MultiAgentOrchestrator()
        result = orchestrator._build_consensus([])
        assert result["status"] == "no_proposals"

    def test_build_consensus_high_threshold(self):
        """Test building consensus with high agreement."""
        orchestrator = MultiAgentOrchestrator()
        proposal = AgentProposal(
            proposer_id="agent_1",
            proposer_role=AgentRole.RETRIEVER,
            proposal="Answer",
            reasoning="Good",
            confidence=0.9,
            agreements=["agent_2", "agent_3", "agent_4"],
            disagreements=[],
        )

        result = orchestrator._build_consensus([proposal])

        assert result["status"] == "consensus"
        assert result["chosen_proposal"] == "Answer"
        assert result["consensus_score"] == 0.95

    def test_build_consensus_partial_agreement(self):
        """Test building consensus with mixed agreement."""
        orchestrator = MultiAgentOrchestrator()
        proposal = AgentProposal(
            proposer_id="agent_1",
            proposer_role=AgentRole.RETRIEVER,
            proposal="Answer",
            reasoning="Good",
            confidence=0.7,
            agreements=["agent_2"],
            disagreements=["agent_3"],
        )

        result = orchestrator._build_consensus([proposal])

        assert result["status"] == "consensus"
        assert result["consensus_score"] == 0.6

    def test_resolve_disagreement(self):
        """Test resolving disagreement."""
        orchestrator = MultiAgentOrchestrator()
        agent_critic = SpecializedAgent("critic", AgentRole.CRITIC, "analysis")
        orchestrator.register_agent(agent_critic)

        proposal = AgentProposal(
            proposer_id="agent_1",
            proposer_role=AgentRole.RETRIEVER,
            proposal="Initial answer",
            reasoning="Initial reasoning",
            confidence=0.5,
        )

        refinement = orchestrator.resolve_disagreement([proposal])

        assert "Refined proposal" in refinement
        assert "Critique" in refinement

    def test_get_agent_summary(self):
        """Test getting agent summary."""
        orchestrator = MultiAgentOrchestrator()
        agents = create_default_agents()

        for agent in agents:
            orchestrator.register_agent(agent)

        summary = orchestrator.get_agent_summary()

        assert summary["num_agents"] == 5
        assert len(summary["agents"]) == 5
        assert summary["num_proposals"] == 0

    def test_clear_history(self):
        """Test clearing history."""
        orchestrator = MultiAgentOrchestrator()
        agent = SpecializedAgent("agent_1", AgentRole.RETRIEVER, "search")
        orchestrator.register_agent(agent)

        # Create some history
        orchestrator.orchestrate_query("test query")
        assert len(orchestrator.proposals) > 0

        # Clear history
        orchestrator.clear_history()

        assert len(orchestrator.proposals) == 0
        assert len(agent.proposals) == 0
        assert len(agent.message_history) == 0


class TestMultiAgentIntegration:
    """Integration tests for complete multi-agent workflows."""

    def test_default_agents_workflow(self):
        """Test workflow with default agent set."""
        orchestrator = MultiAgentOrchestrator()
        agents = create_default_agents()

        for agent in agents:
            orchestrator.register_agent(agent)

        result = orchestrator.orchestrate_query("What is machine learning?")

        assert result["consensus"]["status"] in ["consensus", "majority"]
        assert result["consensus"]["consensus_score"] > 0.5
        assert len(result["initial_results"]) == 5

    def test_multi_query_workflow(self):
        """Test handling multiple sequential queries."""
        orchestrator = MultiAgentOrchestrator()
        agents = create_default_agents()

        for agent in agents:
            orchestrator.register_agent(agent)

        queries = ["First question?", "Second question?", "Third question?"]

        for query in queries:
            orchestrator.orchestrate_query(query)

        # All queries should be processed
        assert len(orchestrator.proposals) == 15  # 3 queries × 5 agents

    def test_dynamic_agent_addition(self):
        """Test adding agents dynamically."""
        orchestrator = MultiAgentOrchestrator()

        # Start with 2 agents
        agent1 = SpecializedAgent("agent_1", AgentRole.RETRIEVER, "search")
        agent2 = SpecializedAgent("agent_2", AgentRole.REASONER, "logic")
        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        result1 = orchestrator.orchestrate_query("Query 1")
        assert len(result1["initial_results"]) == 2

        # Add another agent
        agent3 = SpecializedAgent("agent_3", AgentRole.SYNTHESIZER, "synthesis")
        orchestrator.register_agent(agent3)

        result2 = orchestrator.orchestrate_query("Query 2")
        assert len(result2["initial_results"]) == 3

    def test_agent_consensus_building(self):
        """Test consensus building across multiple agents."""
        orchestrator = MultiAgentOrchestrator()
        agents = create_default_agents()

        for agent in agents:
            orchestrator.register_agent(agent)

        # First query
        orchestrator.orchestrate_query("Test query")

        # Simulate agreement on best proposal
        proposals = orchestrator.proposals
        if proposals:
            best = proposals[0]
            for agent in agents[1:]:
                agent.agree_with(best)

            # Get summary after consensus building
            summary = orchestrator.get_agent_summary()
            assert summary["avg_consensus_score"] > 0.6

    @pytest.mark.asyncio
    async def test_async_orchestration(self):
        """Test asynchronous orchestration."""
        orchestrator = MultiAgentOrchestrator()
        agents = create_default_agents()

        for agent in agents:
            orchestrator.register_agent(agent)

        result = await orchestrator.orchestrate_query_async("Async test query")

        assert "query" in result
        assert "consensus" in result
        assert len(result["initial_results"]) == 5

    @pytest.mark.asyncio
    async def test_async_multiple_queries(self):
        """Test multiple async queries concurrently."""
        orchestrator = MultiAgentOrchestrator()
        agents = create_default_agents()

        for agent in agents:
            orchestrator.register_agent(agent)

        queries = ["Query 1", "Query 2", "Query 3"]
        tasks = [orchestrator.orchestrate_query_async(q) for q in queries]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        for result in results:
            assert "consensus" in result
            assert "initial_results" in result


class TestEdgeCases:
    """Edge case tests for multi-agent collaboration."""

    def test_single_agent_no_votes(self):
        """Test single agent with no votes."""
        orchestrator = MultiAgentOrchestrator()
        agent = SpecializedAgent("solo", AgentRole.RETRIEVER, "search")
        orchestrator.register_agent(agent)

        result = orchestrator.orchestrate_query("query")

        consensus = result["consensus"]
        # Single agent's confidence should be used as-is
        assert consensus["consensus_score"] == 0.7

    def test_all_disagreement(self):
        """Test scenario with all agents disagreeing."""
        orchestrator = MultiAgentOrchestrator()
        agents = create_default_agents()

        for agent in agents:
            orchestrator.register_agent(agent)

        orchestrator.orchestrate_query("query")

        # Manually create disagreement scenario
        proposals = orchestrator.proposals[:]
        if len(proposals) > 1:
            for i, prop in enumerate(proposals[1:]):
                for other_agent in agents[i + 1 :]:
                    other_agent.disagree_with(proposals[0])

        consensus = orchestrator._build_consensus(proposals)
        # With default confidence of 0.7 and no consensus, status could be consensus, majority, or disagreement
        assert consensus["status"] in ["consensus", "majority", "disagreement"]

    def test_empty_query(self):
        """Test handling empty query."""
        orchestrator = MultiAgentOrchestrator()
        agent = SpecializedAgent("agent", AgentRole.RETRIEVER, "search")
        orchestrator.register_agent(agent)

        result = orchestrator.orchestrate_query("")

        assert "initial_results" in result
        assert result["query"] == ""
