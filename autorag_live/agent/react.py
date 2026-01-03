"""
ReAct (Reasoning + Acting) Agent Framework.

Implements the canonical ReAct loop with proper Thought → Action → Observation
cycle and LLM-driven decision making.

Based on: "ReAct: Synergizing Reasoning and Acting in Language Models"
(Yao et al., 2022) https://arxiv.org/abs/2210.03629

Key Features:
1. Proper thought-action-observation cycle
2. LLM-driven action selection
3. Tool execution with observation feedback
4. Configurable stopping conditions
5. Full conversation history tracking
6. Async-first design

Example:
    >>> agent = ReActAgent(llm=my_llm, tools=[search_tool, calc_tool])
    >>> answer = await agent.run("What is the population of France?")
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from autorag_live.core.protocols import BaseLLM, Message, ToolCall
from autorag_live.llm.tool_calling import ToolDefinition, ToolManager

logger = logging.getLogger(__name__)


# =============================================================================
# ReAct Specific Types
# =============================================================================


class StopReason(str, Enum):
    """Reasons for stopping agent execution."""

    FINAL_ANSWER = "final_answer"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"
    USER_INTERRUPT = "user_interrupt"
    NO_PROGRESS = "no_progress"


@dataclass
class Thought:
    """
    Agent's reasoning at a step.

    Attributes:
        content: The reasoning text
        confidence: Confidence in this reasoning (0-1)
        step: Step number
        timestamp: When thought was generated
    """

    content: str
    confidence: float = 1.0
    step: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __str__(self) -> str:
        return f"Thought: {self.content}"


@dataclass
class Action:
    """
    Agent's action decision.

    Attributes:
        type: Action type (tool_call or final_answer)
        tool_name: Tool to call (if tool_call)
        tool_args: Arguments for tool
        answer: Final answer (if final_answer)
        reasoning: Why this action was chosen
    """

    type: str  # "tool_call" or "final_answer"
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[str] = None
    reasoning: str = ""

    @property
    def is_final(self) -> bool:
        """Check if this is a final answer action."""
        return self.type == "final_answer"

    def __str__(self) -> str:
        if self.is_final:
            return f"Action: Final Answer - {self.answer}"
        return f"Action: {self.tool_name}({json.dumps(self.tool_args)})"


@dataclass
class Observation:
    """
    Observation from tool execution.

    Attributes:
        content: Observation content
        tool_name: Tool that produced this
        success: Whether tool succeeded
        error: Error message if failed
    """

    content: str
    tool_name: str
    success: bool = True
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.success:
            return f"Observation: {self.content}"
        return f"Observation (Error): {self.error}"


@dataclass
class ReActStep:
    """
    Single step in ReAct loop.

    Contains thought, action, and observation.
    """

    step_num: int
    thought: Thought
    action: Action
    observation: Optional[Observation] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step_num,
            "thought": self.thought.content,
            "action": {
                "type": self.action.type,
                "tool_name": self.action.tool_name,
                "tool_args": self.action.tool_args,
                "answer": self.action.answer,
            },
            "observation": (
                {
                    "content": self.observation.content,
                    "success": self.observation.success,
                }
                if self.observation
                else None
            ),
            "latency_ms": self.latency_ms,
        }


@dataclass
class ReActTrace:
    """
    Complete trace of ReAct execution.

    Attributes:
        query: Original user query
        steps: List of ReAct steps
        final_answer: Final answer if completed
        stop_reason: Why execution stopped
        total_latency_ms: Total execution time
    """

    query: str
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    stop_reason: StopReason = StopReason.MAX_ITERATIONS
    total_latency_ms: float = 0.0

    def add_step(self, step: ReActStep) -> None:
        """Add a step to the trace."""
        self.steps.append(step)

    def get_formatted_trace(self) -> str:
        """Get human-readable trace."""
        lines = [f"Query: {self.query}\n"]
        for step in self.steps:
            lines.append(f"Step {step.step_num}:")
            lines.append(f"  {step.thought}")
            lines.append(f"  {step.action}")
            if step.observation:
                lines.append(f"  {step.observation}")
            lines.append("")
        if self.final_answer:
            lines.append(f"Final Answer: {self.final_answer}")
        lines.append(f"\nStop Reason: {self.stop_reason.value}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "stop_reason": self.stop_reason.value,
            "total_latency_ms": self.total_latency_ms,
        }


# =============================================================================
# ReAct Prompts
# =============================================================================

REACT_SYSTEM_PROMPT = """You are a helpful AI assistant that uses tools to answer questions.

You follow the ReAct pattern: Think step-by-step, then decide on an action.

For each step, you must output:
1. THOUGHT: Your reasoning about what to do next
2. ACTION: Either a tool call or final answer

Available tools:
{tools_description}

Output format (follow EXACTLY):
```
THOUGHT: [Your reasoning here]
ACTION: [tool_name](arg1="value1", arg2="value2")
```

OR for final answer:
```
THOUGHT: [Your reasoning here]
ACTION: FINAL_ANSWER(answer="Your complete answer here")
```

Important:
- Think before acting
- Use tools when you need information
- Give final answer when you have enough information
- Be concise but thorough
"""

REACT_USER_TEMPLATE = """Question: {query}

{history}

Now continue the reasoning. Output your THOUGHT and ACTION."""


# =============================================================================
# ReAct Agent
# =============================================================================


class ReActAgent:
    """
    ReAct Agent implementing the Thought-Action-Observation loop.

    This agent:
    1. Thinks about the current state and what to do
    2. Decides on an action (tool call or final answer)
    3. Executes the action and observes the result
    4. Repeats until a final answer is reached

    Example:
        >>> llm = MyLLM()
        >>> tools = ToolManager()
        >>> tools.register(search_tool)
        >>>
        >>> agent = ReActAgent(llm, tools)
        >>> answer = await agent.run("What is 2+2?")
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[ToolManager] = None,
        *,
        max_iterations: int = 10,
        temperature: float = 0.7,
        verbose: bool = False,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize ReAct agent.

        Args:
            llm: Language model for reasoning
            tools: Tool manager with registered tools
            max_iterations: Maximum reasoning steps
            temperature: LLM sampling temperature
            verbose: Enable verbose logging
            system_prompt: Override default system prompt
        """
        self.llm = llm
        self.tools = tools or ToolManager()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.verbose = verbose
        self.custom_system_prompt = system_prompt

        # State
        self._current_trace: Optional[ReActTrace] = None

    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        if self.custom_system_prompt:
            return self.custom_system_prompt

        # Build tools description
        tools_desc = []
        for name in self.tools.list_tools():
            tool = self.tools.get_tool(name)
            if tool:
                params_desc = ", ".join(
                    f'{p.name}: {p.type_hint.__name__ if hasattr(p.type_hint, "__name__") else str(p.type_hint)}'
                    for p in tool.parameters
                )
                tools_desc.append(f"- {name}({params_desc}): {tool.description}")

        return REACT_SYSTEM_PROMPT.format(
            tools_description="\n".join(tools_desc) or "No tools available"
        )

    def _build_history(self, steps: List[ReActStep]) -> str:
        """Build history string from previous steps."""
        if not steps:
            return ""

        lines = ["Previous steps:"]
        for step in steps:
            lines.append(f"THOUGHT: {step.thought.content}")
            lines.append(str(step.action))
            if step.observation:
                lines.append(str(step.observation))
            lines.append("")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> tuple[Thought, Action]:
        """
        Parse LLM response into Thought and Action.

        Args:
            response: Raw LLM response

        Returns:
            (Thought, Action) tuple
        """
        # Extract thought
        thought_match = re.search(
            r"THOUGHT:\s*(.+?)(?=ACTION:|$)", response, re.DOTALL | re.IGNORECASE
        )
        thought_content = thought_match.group(1).strip() if thought_match else response

        thought = Thought(content=thought_content)

        # Extract action
        action_match = re.search(
            r"ACTION:\s*(.+?)(?=THOUGHT:|$)", response, re.DOTALL | re.IGNORECASE
        )

        if not action_match:
            # Default to final answer if no action found
            return thought, Action(
                type="final_answer",
                answer=thought_content,
                reasoning="No explicit action found",
            )

        action_str = action_match.group(1).strip()

        # Check for final answer
        final_match = re.search(
            r"FINAL_ANSWER\s*\(\s*answer\s*=\s*[\"'](.+?)[\"']\s*\)",
            action_str,
            re.DOTALL | re.IGNORECASE,
        )
        if final_match:
            return thought, Action(
                type="final_answer",
                answer=final_match.group(1),
                reasoning=thought_content,
            )

        # Also check for simpler final answer format
        simple_final = re.search(
            r"FINAL[_\s]?ANSWER[:\s]+(.+)", action_str, re.DOTALL | re.IGNORECASE
        )
        if simple_final:
            return thought, Action(
                type="final_answer",
                answer=simple_final.group(1).strip(),
                reasoning=thought_content,
            )

        # Parse tool call: tool_name(arg1="value1", arg2="value2")
        tool_match = re.search(r"(\w+)\s*\((.+?)\)", action_str, re.DOTALL)
        if tool_match:
            tool_name = tool_match.group(1)
            args_str = tool_match.group(2)

            # Parse arguments
            args = {}
            # Match key="value" or key=value patterns
            arg_pattern = r'(\w+)\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|([^\s,\)]+))'
            for match in re.finditer(arg_pattern, args_str):
                key = match.group(1)
                value = match.group(2) or match.group(3) or match.group(4)
                # Try to parse as JSON for numbers/booleans
                try:
                    args[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    args[key] = value

            return thought, Action(
                type="tool_call",
                tool_name=tool_name,
                tool_args=args,
                reasoning=thought_content,
            )

        # Fallback: treat as final answer
        return thought, Action(
            type="final_answer",
            answer=action_str,
            reasoning=thought_content,
        )

    async def step(
        self,
        query: str,
        history: List[ReActStep],
    ) -> ReActStep:
        """
        Execute one ReAct step.

        Args:
            query: User query
            history: Previous steps

        Returns:
            New ReActStep with thought, action, and observation
        """
        start_time = time.time()
        step_num = len(history) + 1

        # Build messages
        messages = [
            Message.system(self._build_system_prompt()),
            Message.user(
                REACT_USER_TEMPLATE.format(
                    query=query,
                    history=self._build_history(history),
                )
            ),
        ]

        # Generate thought and action
        result = await self.llm.generate(
            messages,
            temperature=self.temperature,
        )

        thought, action = self._parse_response(result.content)
        thought.step = step_num

        if self.verbose:
            logger.info(f"Step {step_num}: {thought}")
            logger.info(f"Step {step_num}: {action}")

        # Execute action if it's a tool call
        observation = None
        if not action.is_final and action.tool_name:
            tool_call = ToolCall(
                id=f"call_{step_num}",
                name=action.tool_name,
                arguments=action.tool_args,
            )
            tool_result = await self.tools.execute_tool_call(tool_call)

            observation = Observation(
                content=str(tool_result.result) if tool_result.success else "",
                tool_name=action.tool_name,
                success=tool_result.success,
                error=tool_result.error,
            )

            if self.verbose:
                logger.info(f"Step {step_num}: {observation}")

        latency_ms = (time.time() - start_time) * 1000

        return ReActStep(
            step_num=step_num,
            thought=thought,
            action=action,
            observation=observation,
            latency_ms=latency_ms,
        )

    async def run(
        self,
        query: str,
        *,
        max_iterations: Optional[int] = None,
        on_step: Optional[Callable[[ReActStep], None]] = None,
    ) -> str:
        """
        Run agent to completion.

        Args:
            query: User query
            max_iterations: Override max iterations
            on_step: Callback for each step

        Returns:
            Final answer string
        """
        max_iter = max_iterations or self.max_iterations
        start_time = time.time()

        # Initialize trace
        trace = ReActTrace(query=query)
        self._current_trace = trace

        try:
            for i in range(max_iter):
                # Execute step
                step = await self.step(query, trace.steps)
                trace.add_step(step)

                # Call step callback
                if on_step:
                    on_step(step)

                # Check for final answer
                if step.action.is_final:
                    trace.final_answer = step.action.answer
                    trace.stop_reason = StopReason.FINAL_ANSWER
                    break

                # Check for stuck state (no observation)
                if not step.observation or not step.observation.success:
                    # Continue but maybe we're stuck
                    if i > 2 and all(
                        not s.observation or not s.observation.success for s in trace.steps[-3:]
                    ):
                        trace.stop_reason = StopReason.NO_PROGRESS
                        break

            else:
                # Max iterations reached
                trace.stop_reason = StopReason.MAX_ITERATIONS

                # Try to extract answer from last thought
                if trace.steps:
                    trace.final_answer = trace.steps[-1].thought.content

        except Exception as e:
            logger.exception("ReAct execution error")
            trace.stop_reason = StopReason.ERROR
            trace.final_answer = f"Error: {str(e)}"

        trace.total_latency_ms = (time.time() - start_time) * 1000

        if self.verbose:
            logger.info(f"Completed in {trace.total_latency_ms:.0f}ms")
            logger.info(f"Stop reason: {trace.stop_reason.value}")

        return trace.final_answer or "Unable to provide an answer."

    def get_trace(self) -> Optional[ReActTrace]:
        """Get the trace from the last run."""
        return self._current_trace

    async def run_with_trace(
        self,
        query: str,
        **kwargs: Any,
    ) -> tuple[str, ReActTrace]:
        """
        Run agent and return both answer and trace.

        Args:
            query: User query
            **kwargs: Additional arguments for run()

        Returns:
            (answer, trace) tuple
        """
        answer = await self.run(query, **kwargs)
        return answer, self._current_trace or ReActTrace(query=query)


# =============================================================================
# ReAct Agent Builder
# =============================================================================


class ReActAgentBuilder:
    """
    Builder pattern for constructing ReAct agents.

    Example:
        >>> agent = (
        ...     ReActAgentBuilder()
        ...     .with_llm(my_llm)
        ...     .with_tool(search_tool)
        ...     .with_tool(calc_tool)
        ...     .with_max_iterations(15)
        ...     .verbose()
        ...     .build()
        ... )
    """

    def __init__(self):
        """Initialize builder."""
        self._llm: Optional[BaseLLM] = None
        self._tools = ToolManager()
        self._max_iterations = 10
        self._temperature = 0.7
        self._verbose = False
        self._system_prompt: Optional[str] = None

    def with_llm(self, llm: BaseLLM) -> "ReActAgentBuilder":
        """Set the LLM."""
        self._llm = llm
        return self

    def with_tool(
        self,
        tool: Union[ToolDefinition, Callable[..., Any]],
        name: Optional[str] = None,
    ) -> "ReActAgentBuilder":
        """Add a tool."""
        self._tools.register(tool, name=name)
        return self

    def with_tools(self, tools: ToolManager) -> "ReActAgentBuilder":
        """Set the tool manager."""
        self._tools = tools
        return self

    def with_max_iterations(self, max_iter: int) -> "ReActAgentBuilder":
        """Set max iterations."""
        self._max_iterations = max_iter
        return self

    def with_temperature(self, temp: float) -> "ReActAgentBuilder":
        """Set LLM temperature."""
        self._temperature = temp
        return self

    def with_system_prompt(self, prompt: str) -> "ReActAgentBuilder":
        """Set custom system prompt."""
        self._system_prompt = prompt
        return self

    def verbose(self, enabled: bool = True) -> "ReActAgentBuilder":
        """Enable verbose mode."""
        self._verbose = enabled
        return self

    def build(self) -> ReActAgent:
        """Build the agent."""
        if not self._llm:
            raise ValueError("LLM is required. Call with_llm() first.")

        return ReActAgent(
            llm=self._llm,
            tools=self._tools,
            max_iterations=self._max_iterations,
            temperature=self._temperature,
            verbose=self._verbose,
            system_prompt=self._system_prompt,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_react_agent(
    llm: BaseLLM,
    tools: Optional[List[Union[ToolDefinition, Callable[..., Any]]]] = None,
    **kwargs: Any,
) -> ReActAgent:
    """
    Create a ReAct agent with given tools.

    Args:
        llm: Language model
        tools: List of tools to register
        **kwargs: Additional ReActAgent arguments

    Returns:
        Configured ReActAgent
    """
    tool_manager = ToolManager()
    if tools:
        for tool in tools:
            tool_manager.register(tool)

    return ReActAgent(llm=llm, tools=tool_manager, **kwargs)
