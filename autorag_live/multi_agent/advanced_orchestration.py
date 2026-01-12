"""
Enhanced Multi-Agent Orchestration for Agentic RAG.

Implements state-of-the-art multi-agent patterns:
- Agent debate and consensus mechanisms
- Hierarchical agent organization
- Specialization-based routing
- Collaborative refinement
- Critic-generator patterns

References:
- Multi-Agent Debate (Du et al., 2023)
- CAMEL: Communicative Agents (Li et al., 2023)
- AutoGen (Microsoft, 2023)

Example:
    >>> orchestrator = MultiAgentOrchestrator()
    >>> orchestrator.add_agent("researcher", ResearcherAgent())
    >>> orchestrator.add_agent("critic", CriticAgent())
    >>> result = await orchestrator.execute_with_debate(query)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for agent implementations."""

    @property
    def name(self) -> str:
        """Agent identifier."""
        ...

    @property
    def role(self) -> str:
        """Agent role description."""
        ...

    async def process(
        self,
        message: "AgentMessage",
        context: Optional[Dict[str, Any]] = None,
    ) -> "AgentResponse":
        """Process a message and return response."""
        ...


# =============================================================================
# Enums and Types
# =============================================================================


class AgentRole(Enum):
    """Standard agent roles."""

    RESEARCHER = auto()  # Information gathering
    CRITIC = auto()  # Quality assessment
    GENERATOR = auto()  # Content generation
    PLANNER = auto()  # Task planning
    EXECUTOR = auto()  # Task execution
    SUMMARIZER = auto()  # Summarization
    VERIFIER = auto()  # Fact verification
    ORCHESTRATOR = auto()  # Coordination


class MessageType(Enum):
    """Types of inter-agent messages."""

    QUERY = auto()  # Initial query
    RESPONSE = auto()  # Agent response
    CRITIQUE = auto()  # Critical feedback
    REFINEMENT = auto()  # Refined response
    VOTE = auto()  # Voting message
    CONSENSUS = auto()  # Consensus result
    DELEGATE = auto()  # Task delegation
    FEEDBACK = auto()  # General feedback


class OrchestrationStrategy(Enum):
    """Multi-agent orchestration strategies."""

    SEQUENTIAL = auto()  # One after another
    PARALLEL = auto()  # Simultaneous execution
    DEBATE = auto()  # Debate until consensus
    HIERARCHICAL = auto()  # Manager-worker pattern
    COLLABORATIVE = auto()  # Collaborative refinement
    CRITIC_GENERATOR = auto()  # Generate then critique


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AgentMessage:
    """Message passed between agents."""

    content: str
    message_type: MessageType
    sender: str = ""
    recipient: str = ""
    turn: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "type": self.message_type.name,
            "sender": self.sender,
            "recipient": self.recipient,
            "turn": self.turn,
            "metadata": self.metadata,
        }


@dataclass
class AgentResponse:
    """Response from an agent."""

    content: str
    confidence: float = 0.5
    reasoning: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateRound:
    """A single round of debate."""

    round_num: int
    responses: Dict[str, AgentResponse]
    critiques: Dict[str, str] = field(default_factory=dict)
    consensus_reached: bool = False
    winning_response: Optional[str] = None


@dataclass
class OrchestrationResult:
    """Result from multi-agent orchestration."""

    query: str
    final_answer: str
    strategy: OrchestrationStrategy
    agent_responses: Dict[str, AgentResponse] = field(default_factory=dict)
    debate_rounds: List[DebateRound] = field(default_factory=list)
    consensus_score: float = 0.0
    total_turns: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base Agent Class
# =============================================================================


class BaseAgent(ABC):
    """Base class for agents in multi-agent system."""

    def __init__(
        self,
        name: str,
        role: AgentRole,
        llm: Any = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize agent."""
        self._name = name
        self._role = role
        self.llm = llm
        self.system_prompt = system_prompt or self._default_prompt()

    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> str:
        return self._role.name.lower()

    @abstractmethod
    async def process(
        self,
        message: AgentMessage,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Process message and generate response."""
        ...

    def _default_prompt(self) -> str:
        """Get default system prompt for role."""
        prompts = {
            AgentRole.RESEARCHER: "You are a research agent. Gather and synthesize information.",
            AgentRole.CRITIC: "You are a critic agent. Evaluate responses for accuracy and completeness.",
            AgentRole.GENERATOR: "You are a generator agent. Create high-quality responses.",
            AgentRole.PLANNER: "You are a planning agent. Break down tasks into steps.",
            AgentRole.VERIFIER: "You are a verification agent. Check facts and claims.",
        }
        return prompts.get(self._role, "You are a helpful assistant.")


class LLMAgent(BaseAgent):
    """Agent powered by an LLM."""

    def __init__(
        self,
        name: str,
        role: AgentRole,
        llm: Any,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """Initialize LLM agent."""
        super().__init__(name, role, llm, system_prompt)
        self.temperature = temperature

    async def process(
        self,
        message: AgentMessage,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Process message using LLM."""
        prompt = self._build_prompt(message, context)

        if hasattr(self.llm, "generate"):
            response_text = await self.llm.generate(
                prompt,
                temperature=self.temperature,
            )
        else:
            response_text = str(self.llm)  # Fallback for testing

        return AgentResponse(
            content=response_text,
            confidence=0.8,
            metadata={"agent": self.name, "role": self.role},
        )

    def _build_prompt(
        self,
        message: AgentMessage,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Build prompt for LLM."""
        parts = [self.system_prompt, "", f"Message: {message.content}"]

        if context:
            if "history" in context:
                parts.insert(2, f"Previous discussion:\n{context['history']}")
            if "documents" in context:
                docs_text = "\n".join(context["documents"][:5])
                parts.insert(2, f"Relevant documents:\n{docs_text}")

        return "\n".join(parts)


# =============================================================================
# Specialized Agents
# =============================================================================


class CriticAgent(LLMAgent):
    """Agent specialized in critiquing responses."""

    CRITIQUE_PROMPT = """You are a critical evaluator. Analyze the response below and provide:
1. Accuracy assessment (0-10)
2. Completeness assessment (0-10)
3. Specific issues found
4. Suggestions for improvement

Response to evaluate:
{response}

Original question:
{question}"""

    def __init__(self, name: str, llm: Any):
        """Initialize critic agent."""
        super().__init__(name, AgentRole.CRITIC, llm)

    async def critique(
        self,
        response: str,
        question: str,
    ) -> Dict[str, Any]:
        """Generate critique of a response."""
        prompt = self.CRITIQUE_PROMPT.format(response=response, question=question)

        message = AgentMessage(
            content=prompt,
            message_type=MessageType.QUERY,
        )

        result = await self.process(message)

        # Parse critique
        return {
            "critique": result.content,
            "confidence": result.confidence,
        }


class GeneratorAgent(LLMAgent):
    """Agent specialized in generating responses."""

    def __init__(self, name: str, llm: Any):
        """Initialize generator agent."""
        super().__init__(name, AgentRole.GENERATOR, llm)

    async def generate(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        feedback: Optional[str] = None,
    ) -> AgentResponse:
        """Generate response, optionally incorporating feedback."""
        content = query
        if feedback:
            content = f"{query}\n\nPrevious feedback to address:\n{feedback}"

        message = AgentMessage(
            content=content,
            message_type=MessageType.QUERY,
        )

        return await self.process(message, context)


class VerifierAgent(LLMAgent):
    """Agent specialized in verifying facts."""

    VERIFY_PROMPT = """Verify the following claim. Is it accurate based on the provided context?

Claim: {claim}

Context:
{context}

Respond with:
1. VERIFIED or UNVERIFIED
2. Confidence level (0-100%)
3. Explanation"""

    def __init__(self, name: str, llm: Any):
        """Initialize verifier agent."""
        super().__init__(name, AgentRole.VERIFIER, llm)

    async def verify(
        self,
        claim: str,
        context: str,
    ) -> Dict[str, Any]:
        """Verify a claim against context."""
        prompt = self.VERIFY_PROMPT.format(claim=claim, context=context)

        message = AgentMessage(
            content=prompt,
            message_type=MessageType.QUERY,
        )

        result = await self.process(message)

        # Parse verification result
        is_verified = "VERIFIED" in result.content.upper()

        return {
            "verified": is_verified,
            "explanation": result.content,
            "confidence": result.confidence,
        }


# =============================================================================
# Consensus Mechanisms
# =============================================================================


class ConsensusMechanism(ABC):
    """Base class for consensus mechanisms."""

    @abstractmethod
    def compute_consensus(
        self,
        responses: Dict[str, AgentResponse],
    ) -> Tuple[str, float]:
        """
        Compute consensus from agent responses.

        Returns:
            Tuple of (winning_agent_name, consensus_score)
        """
        ...


class MajorityVoting(ConsensusMechanism):
    """Simple majority voting consensus."""

    def __init__(self, threshold: float = 0.5):
        """Initialize with vote threshold."""
        self.threshold = threshold

    def compute_consensus(
        self,
        responses: Dict[str, AgentResponse],
    ) -> Tuple[str, float]:
        """Compute consensus via confidence-weighted voting."""
        if not responses:
            return "", 0.0

        # Weight by confidence
        total_confidence = sum(r.confidence for r in responses.values())

        if total_confidence == 0:
            # Equal weight fallback
            return list(responses.keys())[0], 1.0 / len(responses)

        # Find highest weighted response
        weighted_scores = {
            name: resp.confidence / total_confidence for name, resp in responses.items()
        }

        winner = max(weighted_scores, key=weighted_scores.get)
        score = weighted_scores[winner]

        return winner, score


class DebateConsensus(ConsensusMechanism):
    """Consensus through iterative debate and refinement."""

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        max_rounds: int = 3,
    ):
        """Initialize debate consensus."""
        self.similarity_threshold = similarity_threshold
        self.max_rounds = max_rounds

    def compute_consensus(
        self,
        responses: Dict[str, AgentResponse],
    ) -> Tuple[str, float]:
        """Compute consensus based on response similarity."""
        if len(responses) <= 1:
            return (list(responses.keys())[0] if responses else "", 1.0)

        # Compute pairwise similarity
        names = list(responses.keys())
        similarity_matrix: Dict[Tuple[str, str], float] = {}

        for i, name1 in enumerate(names):
            for name2 in names[i + 1 :]:
                sim = self._compute_similarity(
                    responses[name1].content,
                    responses[name2].content,
                )
                similarity_matrix[(name1, name2)] = sim

        # Find highest average similarity
        avg_similarities = {}
        for name in names:
            sims = []
            for (n1, n2), sim in similarity_matrix.items():
                if name in (n1, n2):
                    sims.append(sim)
            avg_similarities[name] = sum(sims) / len(sims) if sims else 0

        winner = max(avg_similarities, key=avg_similarities.get)
        return winner, avg_similarities[winner]

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


# =============================================================================
# Multi-Agent Orchestrator
# =============================================================================


class MultiAgentOrchestrator:
    """
    Orchestrates multiple agents for complex RAG tasks.

    Supports various strategies including debate, hierarchical,
    and collaborative patterns.
    """

    def __init__(
        self,
        default_strategy: OrchestrationStrategy = OrchestrationStrategy.COLLABORATIVE,
        consensus_mechanism: Optional[ConsensusMechanism] = None,
        max_debate_rounds: int = 3,
    ):
        """Initialize orchestrator."""
        self.default_strategy = default_strategy
        self.consensus = consensus_mechanism or MajorityVoting()
        self.max_debate_rounds = max_debate_rounds

        self._agents: Dict[str, AgentProtocol] = {}
        self._role_mapping: Dict[AgentRole, List[str]] = {}

    def add_agent(
        self,
        name: str,
        agent: AgentProtocol,
        role: Optional[AgentRole] = None,
    ) -> None:
        """Add agent to orchestrator."""
        self._agents[name] = agent

        # Track by role
        agent_role = role or AgentRole.GENERATOR
        if agent_role not in self._role_mapping:
            self._role_mapping[agent_role] = []
        self._role_mapping[agent_role].append(name)

        logger.debug(f"Added agent: {name}")

    def remove_agent(self, name: str) -> None:
        """Remove agent from orchestrator."""
        if name in self._agents:
            del self._agents[name]

    def get_agent(self, name: str) -> Optional[AgentProtocol]:
        """Get agent by name."""
        return self._agents.get(name)

    def get_agents_by_role(self, role: AgentRole) -> List[AgentProtocol]:
        """Get all agents with a specific role."""
        names = self._role_mapping.get(role, [])
        return [self._agents[n] for n in names if n in self._agents]

    async def execute(
        self,
        query: str,
        strategy: Optional[OrchestrationStrategy] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """
        Execute multi-agent orchestration.

        Args:
            query: User query
            strategy: Orchestration strategy
            context: Additional context

        Returns:
            OrchestrationResult with final answer
        """
        selected_strategy = strategy or self.default_strategy

        if selected_strategy == OrchestrationStrategy.SEQUENTIAL:
            return await self._execute_sequential(query, context)
        elif selected_strategy == OrchestrationStrategy.PARALLEL:
            return await self._execute_parallel(query, context)
        elif selected_strategy == OrchestrationStrategy.DEBATE:
            return await self._execute_debate(query, context)
        elif selected_strategy == OrchestrationStrategy.COLLABORATIVE:
            return await self._execute_collaborative(query, context)
        elif selected_strategy == OrchestrationStrategy.CRITIC_GENERATOR:
            return await self._execute_critic_generator(query, context)
        else:
            return await self._execute_parallel(query, context)

    async def _execute_sequential(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> OrchestrationResult:
        """Execute agents sequentially, each building on previous."""
        responses: Dict[str, AgentResponse] = {}
        current_context = context or {}
        accumulated_response = ""

        for name, agent in self._agents.items():
            message = AgentMessage(
                content=query,
                message_type=MessageType.QUERY,
            )

            if accumulated_response:
                current_context["previous_response"] = accumulated_response

            response = await agent.process(message, current_context)
            responses[name] = response
            accumulated_response = response.content

        # Final answer is last response
        final_answer = accumulated_response

        return OrchestrationResult(
            query=query,
            final_answer=final_answer,
            strategy=OrchestrationStrategy.SEQUENTIAL,
            agent_responses=responses,
            total_turns=len(self._agents),
        )

    async def _execute_parallel(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> OrchestrationResult:
        """Execute all agents in parallel and aggregate."""
        message = AgentMessage(
            content=query,
            message_type=MessageType.QUERY,
        )

        # Execute in parallel
        tasks = [agent.process(message, context) for agent in self._agents.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect valid responses
        responses = {}
        for name, result in zip(self._agents.keys(), results):
            if isinstance(result, AgentResponse):
                responses[name] = result

        # Compute consensus
        winner, score = self.consensus.compute_consensus(responses)
        final_answer = responses[winner].content if winner else ""

        return OrchestrationResult(
            query=query,
            final_answer=final_answer,
            strategy=OrchestrationStrategy.PARALLEL,
            agent_responses=responses,
            consensus_score=score,
            total_turns=1,
        )

    async def _execute_debate(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> OrchestrationResult:
        """Execute debate between agents until consensus."""
        debate_rounds: List[DebateRound] = []
        responses: Dict[str, AgentResponse] = {}

        # Initial responses
        message = AgentMessage(
            content=query,
            message_type=MessageType.QUERY,
        )

        tasks = [agent.process(message, context) for agent in self._agents.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for name, result in zip(self._agents.keys(), results):
            if isinstance(result, AgentResponse):
                responses[name] = result

        # Debate rounds
        for round_num in range(self.max_debate_rounds):
            round_data = DebateRound(
                round_num=round_num + 1,
                responses=dict(responses),
            )

            # Each agent critiques others
            critiques = await self._generate_critiques(query, responses)
            round_data.critiques = critiques

            # Check for consensus
            winner, score = self.consensus.compute_consensus(responses)

            if score >= 0.8:  # High consensus threshold
                round_data.consensus_reached = True
                round_data.winning_response = winner
                debate_rounds.append(round_data)
                break

            debate_rounds.append(round_data)

            # Agents refine based on critiques
            responses = await self._refine_responses(query, responses, critiques, context)

        # Final consensus
        winner, score = self.consensus.compute_consensus(responses)
        final_answer = responses[winner].content if winner else ""

        return OrchestrationResult(
            query=query,
            final_answer=final_answer,
            strategy=OrchestrationStrategy.DEBATE,
            agent_responses=responses,
            debate_rounds=debate_rounds,
            consensus_score=score,
            total_turns=len(debate_rounds),
        )

    async def _execute_collaborative(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> OrchestrationResult:
        """Execute collaborative refinement pattern."""
        responses: Dict[str, AgentResponse] = {}

        # Get initial response from generator
        generators = self.get_agents_by_role(AgentRole.GENERATOR)
        if generators:
            message = AgentMessage(content=query, message_type=MessageType.QUERY)
            initial = await generators[0].process(message, context)
            responses["generator"] = initial
        else:
            # Fallback to first agent
            first_agent = list(self._agents.values())[0]
            message = AgentMessage(content=query, message_type=MessageType.QUERY)
            initial = await first_agent.process(message, context)
            responses["initial"] = initial

        current_response = initial.content

        # Critique phase
        critics = self.get_agents_by_role(AgentRole.CRITIC)
        if critics:
            critique_msg = AgentMessage(
                content=f"Critique this response to: {query}\n\nResponse: {current_response}",
                message_type=MessageType.CRITIQUE,
            )
            critique = await critics[0].process(critique_msg, context)
            responses["critic"] = critique

            # Refinement based on critique
            if generators:
                refine_msg = AgentMessage(
                    content=f"Improve response based on feedback:\nOriginal: {current_response}\nFeedback: {critique.content}",
                    message_type=MessageType.REFINEMENT,
                )
                refined = await generators[0].process(refine_msg, context)
                responses["refined"] = refined
                current_response = refined.content

        return OrchestrationResult(
            query=query,
            final_answer=current_response,
            strategy=OrchestrationStrategy.COLLABORATIVE,
            agent_responses=responses,
            total_turns=len(responses),
        )

    async def _execute_critic_generator(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> OrchestrationResult:
        """Execute generator-critic loop."""
        generators = self.get_agents_by_role(AgentRole.GENERATOR)
        critics = self.get_agents_by_role(AgentRole.CRITIC)

        if not generators:
            generators = [list(self._agents.values())[0]]

        responses: Dict[str, AgentResponse] = {}

        # Generate initial response
        gen_msg = AgentMessage(content=query, message_type=MessageType.QUERY)
        response = await generators[0].process(gen_msg, context)
        responses["generation_1"] = response

        current_response = response.content
        iterations = 1
        max_iterations = 3

        while critics and iterations < max_iterations:
            # Critique
            critic_msg = AgentMessage(
                content=f"Evaluate: {current_response}",
                message_type=MessageType.CRITIQUE,
            )
            critique = await critics[0].process(critic_msg, context)
            responses[f"critique_{iterations}"] = critique

            # Check if improvement needed
            if "good" in critique.content.lower() and critique.confidence > 0.8:
                break

            # Regenerate with feedback
            improve_msg = AgentMessage(
                content=f"Question: {query}\nPrevious: {current_response}\nFeedback: {critique.content}\nImprove:",
                message_type=MessageType.REFINEMENT,
            )
            response = await generators[0].process(improve_msg, context)
            responses[f"generation_{iterations + 1}"] = response
            current_response = response.content
            iterations += 1

        return OrchestrationResult(
            query=query,
            final_answer=current_response,
            strategy=OrchestrationStrategy.CRITIC_GENERATOR,
            agent_responses=responses,
            total_turns=iterations,
        )

    async def _generate_critiques(
        self,
        query: str,
        responses: Dict[str, AgentResponse],
    ) -> Dict[str, str]:
        """Generate critiques for each response."""
        critiques = {}

        for name, response in responses.items():
            # Each other agent critiques
            other_agents = [a for n, a in self._agents.items() if n != name]
            if other_agents:
                critic = other_agents[0]
                msg = AgentMessage(
                    content=f"Critique for '{query}':\n{response.content}",
                    message_type=MessageType.CRITIQUE,
                )
                result = await critic.process(msg)
                critiques[name] = result.content

        return critiques

    async def _refine_responses(
        self,
        query: str,
        responses: Dict[str, AgentResponse],
        critiques: Dict[str, str],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, AgentResponse]:
        """Refine responses based on critiques."""
        refined = {}

        for name, response in responses.items():
            if name not in self._agents:
                continue

            agent = self._agents[name]
            critique = critiques.get(name, "")

            msg = AgentMessage(
                content=f"Question: {query}\nYour response: {response.content}\nCritique: {critique}\nProvide improved response:",
                message_type=MessageType.REFINEMENT,
            )

            result = await agent.process(msg, context)
            refined[name] = result

        return refined

    @property
    def agent_count(self) -> int:
        """Number of registered agents."""
        return len(self._agents)

    @property
    def agent_names(self) -> List[str]:
        """List of agent names."""
        return list(self._agents.keys())
