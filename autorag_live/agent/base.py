"""
Core agent framework for agentic RAG.

Provides base Agent class with state management, memory buffers, and action planning
for multi-turn agentic RAG conversations.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class AgentState(Enum):
    """Agent execution states."""

    IDLE = "idle"  # Waiting for input
    THINKING = "thinking"  # Processing/reasoning
    ACTING = "acting"  # Executing tools
    OBSERVING = "observing"  # Processing observations
    COMPLETE = "complete"  # Task completed
    ERROR = "error"  # Error state


@dataclass
class Message:
    """Single message in conversation history."""

    role: str  # "user", "agent", "tool", "observation"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Action:
    """Represents an action/tool call to execute."""

    tool_name: str
    tool_input: Dict[str, Any]
    action_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "reasoning": self.reasoning,
        }


@dataclass
class Observation:
    """Result/observation from tool execution."""

    action_id: str
    tool_name: str
    result: Any
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary."""
        return {
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "result": self.result,
            "success": self.success,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


class AgentMemory:
    """Multi-level memory system for agent."""

    def __init__(self, max_messages: int = 100, max_observations: int = 50):
        """
        Initialize agent memory.

        Args:
            max_messages: Max conversation messages to keep
            max_observations: Max observations to keep
        """
        self.messages: List[Message] = []
        self.observations: List[Observation] = []
        self.max_messages = max_messages
        self.max_observations = max_observations
        self.metadata: Dict[str, Any] = {}

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add message to history."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)

        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_observation(self, observation: Observation) -> None:
        """Add tool observation to history."""
        self.observations.append(observation)

        # Keep only recent observations
        if len(self.observations) > self.max_observations:
            self.observations = self.observations[-self.max_observations :]

    def get_conversation_context(self, max_tokens: Optional[int] = None) -> str:
        """
        Get formatted conversation context.

        Args:
            max_tokens: Optional max token limit

        Returns:
            Formatted conversation string
        """
        lines = []
        for msg in self.messages:
            lines.append(f"{msg.role.upper()}: {msg.content}")

        context = "\n".join(lines)
        # Simple token estimation (1 token ≈ 4 chars)
        if max_tokens and len(context) > max_tokens * 4:
            context = context[-(max_tokens * 4) :]

        return context

    def get_last_observation(self) -> Optional[Observation]:
        """Get most recent observation."""
        return self.observations[-1] if self.observations else None

    def clear(self) -> None:
        """Clear all memory."""
        self.messages.clear()
        self.observations.clear()
        self.metadata.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export memory as dictionary."""
        return {
            "messages": [m.to_dict() for m in self.messages],
            "observations": [o.to_dict() for o in self.observations],
            "metadata": self.metadata,
        }


class Agent:
    """
    Base agent class for agentic RAG.

    Manages state, memory, reasoning, and tool execution.
    """

    def __init__(
        self,
        name: str = "RAGAgent",
        max_steps: int = 10,
        temperature: float = 0.7,
        verbose: bool = False,
    ):
        """
        Initialize agent.

        Args:
            name: Agent identifier
            max_steps: Maximum reasoning steps per query
            temperature: Randomness for reasoning (0-1)
            verbose: Enable verbose logging
        """
        self.name = name
        self.max_steps = max_steps
        self.temperature = temperature
        self.verbose = verbose

        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self.tools: Dict[str, Callable] = {}
        self._step_count = 0
        self._goal: Optional[str] = None

        self.logger = logging.getLogger(f"Agent.{name}")
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def register_tool(self, name: str, tool: Callable) -> None:
        """
        Register a tool for agent to use.

        Args:
            name: Tool name/identifier
            tool: Callable tool function
        """
        self.tools[name] = tool
        self._log(f"Registered tool: {name}")

    def _log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            self.logger.debug(message)

    def _transition_state(self, new_state: AgentState) -> None:
        """Transition to new state."""
        old_state = self.state
        self.state = new_state
        self._log(f"State transition: {old_state.value} -> {new_state.value}")

    def plan(self, goal: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Create action plan for goal.

        Args:
            goal: User query or goal
            context: Optional additional context

        Returns:
            List of planned steps
        """
        self._goal = goal
        self._step_count = 0
        self._transition_state(AgentState.THINKING)

        self.memory.add_message("user", goal)

        # Simple goal decomposition
        plan_steps = self._decompose_goal(goal, context or {})
        self._log(f"Created plan with {len(plan_steps)} steps")

        return plan_steps

    def _decompose_goal(self, goal: str, context: Dict[str, Any]) -> List[str]:
        """
        Decompose goal into steps.

        Args:
            goal: Goal to decompose
            context: Context information

        Returns:
            List of steps
        """
        # Simple heuristic decomposition
        steps = [
            "Understand the query and identify key concepts",
            "Retrieve relevant documents",
            "Analyze retrieved information",
            "Synthesize answer from analysis",
        ]

        if "how" in goal.lower():
            steps.insert(2, "Plan approach for 'how' question")
        elif "why" in goal.lower():
            steps.insert(2, "Identify causation factors")

        return steps

    def act(self, action: Action) -> Optional[Observation]:
        """
        Execute a tool action.

        Args:
            action: Action to execute

        Returns:
            Observation from tool execution
        """
        self._transition_state(AgentState.ACTING)
        self._step_count += 1

        if self._step_count > self.max_steps:
            self._transition_state(AgentState.ERROR)
            return Observation(
                action_id=action.action_id,
                tool_name=action.tool_name,
                result=None,
                success=False,
                error=f"Exceeded max steps ({self.max_steps})",
            )

        if action.tool_name not in self.tools:
            return Observation(
                action_id=action.action_id,
                tool_name=action.tool_name,
                result=None,
                success=False,
                error=f"Tool not found: {action.tool_name}",
            )

        self.memory.add_message(
            "agent",
            f"Executing: {action.tool_name}",
            metadata={"action": action.to_dict()},
        )

        try:
            import time

            start_time = time.time()
            tool = self.tools[action.tool_name]
            result = tool(**action.tool_input)
            latency = (time.time() - start_time) * 1000

            observation = Observation(
                action_id=action.action_id,
                tool_name=action.tool_name,
                result=result,
                success=True,
                latency_ms=latency,
            )

            self._transition_state(AgentState.OBSERVING)
            self.memory.add_observation(observation)
            self.memory.add_message(
                "observation",
                str(result)[:500],  # Limit observation length
                metadata={"observation": observation.to_dict()},
            )

            return observation

        except Exception as e:
            error_msg = str(e)
            observation = Observation(
                action_id=action.action_id,
                tool_name=action.tool_name,
                result=None,
                success=False,
                error=error_msg,
            )
            self.memory.add_observation(observation)
            return observation

    def reflect(self) -> str:
        """
        Reflect on current state and observations.

        Returns:
            Reflection text
        """
        last_obs = self.memory.get_last_observation()
        if not last_obs:
            return "No observations yet."

        reflection = f"Tool '{last_obs.tool_name}' {'succeeded' if last_obs.success else 'failed'}"
        if last_obs.error:
            reflection += f" with error: {last_obs.error}"
        else:
            reflection += f" in {last_obs.latency_ms:.1f}ms"

        self.memory.add_message("agent", f"Reflection: {reflection}")
        return reflection

    def finalize(self, response: str) -> str:
        """
        Finalize agent response.

        Args:
            response: Final response text

        Returns:
            Formatted final response
        """
        self._transition_state(AgentState.COMPLETE)
        self.memory.add_message("agent", response)

        return response

    def reset(self) -> None:
        """Reset agent state for new conversation."""
        self._transition_state(AgentState.IDLE)
        self.memory.clear()
        self._step_count = 0
        self._goal = None

    def get_state_summary(self) -> Dict[str, Any]:
        """Get current agent state summary."""
        return {
            "name": self.name,
            "state": self.state.value,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "goal": self._goal,
            "num_messages": len(self.memory.messages),
            "num_observations": len(self.memory.observations),
            "num_tools": len(self.tools),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export agent state to dictionary."""
        return {
            "name": self.name,
            "state": self.state.value,
            "step_count": self._step_count,
            "goal": self._goal,
            "memory": self.memory.to_dict(),
        }
