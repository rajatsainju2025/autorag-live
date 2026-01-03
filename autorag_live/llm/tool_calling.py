"""
Native LLM tool calling with structured outputs.

This module provides a unified interface for tool/function calling across
different LLM providers (OpenAI, Anthropic, etc.) with automatic schema
generation, validation, and structured output parsing.

Key Features:
1. Automatic JSON schema generation from Python functions
2. OpenAI and Anthropic tool format compatibility
3. Structured output parsing with validation
4. Tool execution with error handling
5. Parallel tool call support

Example:
    >>> @tool(description="Search the web for information")
    ... def web_search(query: str, num_results: int = 5) -> List[str]:
    ...     return ["result1", "result2"]
    ...
    >>> tools = ToolManager()
    >>> tools.register(web_search)
    >>> schema = tools.get_openai_tools()
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from autorag_live.core.protocols import ToolCall, ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# Type Mapping
# =============================================================================


class JSONType(str, Enum):
    """JSON Schema types."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"


# Python type to JSON Schema type mapping
PYTHON_TO_JSON_TYPE: Dict[type, JSONType] = {
    str: JSONType.STRING,
    int: JSONType.INTEGER,
    float: JSONType.NUMBER,
    bool: JSONType.BOOLEAN,
    list: JSONType.ARRAY,
    dict: JSONType.OBJECT,
    type(None): JSONType.NULL,
}


def python_type_to_json_schema(python_type: Any) -> Dict[str, Any]:
    """
    Convert Python type hint to JSON Schema.

    Args:
        python_type: Python type or type hint

    Returns:
        JSON Schema dictionary
    """
    # Handle None type
    if python_type is type(None):
        return {"type": "null"}

    # Handle ForwardRef (string annotations)
    if isinstance(python_type, (str, ForwardRef)):
        return {"type": "string"}

    # Get origin for generic types (List[X], Dict[K,V], Optional[X], etc.)
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle Union types (including Optional)
    if origin is Union:
        # Check if it's Optional (Union with None)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            # Optional[X] -> X with nullable
            schema = python_type_to_json_schema(non_none_args[0])
            return schema
        # General Union
        return {"anyOf": [python_type_to_json_schema(arg) for arg in args]}

    # Handle List/Sequence types
    if origin in (list, List):
        if args:
            return {"type": "array", "items": python_type_to_json_schema(args[0])}
        return {"type": "array"}

    # Handle Dict types
    if origin in (dict, Dict):
        schema: Dict[str, Any] = {"type": "object"}
        if len(args) >= 2:
            schema["additionalProperties"] = python_type_to_json_schema(args[1])
        return schema

    # Handle basic types
    if python_type in PYTHON_TO_JSON_TYPE:
        return {"type": PYTHON_TO_JSON_TYPE[python_type].value}

    # Handle Enum types
    if isinstance(python_type, type) and issubclass(python_type, Enum):
        return {"type": "string", "enum": [e.value for e in python_type]}

    # Default to string for unknown types
    return {"type": "string"}


# =============================================================================
# Tool Definition
# =============================================================================


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type_hint: Any
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema = python_type_to_json_schema(self.type_hint)
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """
    Complete tool definition with schema and implementation.

    Attributes:
        name: Tool name for invocation
        description: Tool description for LLM
        parameters: Parameter definitions
        func: Implementation function
        is_async: Whether function is async
        metadata: Additional tool metadata
    """

    name: str
    description: str
    parameters: List[ToolParameter]
    func: Callable[..., Any]
    is_async: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_openai_schema(self) -> Dict[str, Any]:
        """
        Get OpenAI function calling schema.

        Returns:
            OpenAI-compatible tool definition
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def get_anthropic_schema(self) -> Dict[str, Any]:
        """
        Get Anthropic tool use schema.

        Returns:
            Anthropic-compatible tool definition
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    async def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool with given arguments.

        Args:
            **kwargs: Tool arguments

        Returns:
            Tool execution result
        """
        if self.is_async:
            return await self.func(**kwargs)
        else:
            return self.func(**kwargs)


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    **metadata: Any,
) -> Callable[[Callable[..., Any]], ToolDefinition]:
    """
    Decorator to create a tool from a function.

    Automatically extracts parameter types and descriptions from
    function signature and docstring.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to docstring)
        **metadata: Additional tool metadata

    Returns:
        Decorator that creates ToolDefinition

    Example:
        >>> @tool(description="Search for documents")
        ... def search(query: str, limit: int = 10) -> List[str]:
        ...     '''Search the knowledge base.
        ...
        ...     Args:
        ...         query: Search query string
        ...         limit: Maximum results to return
        ...     '''
        ...     return ["doc1", "doc2"]
    """

    def decorator(func: Callable[..., Any]) -> ToolDefinition:
        # Get function metadata
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip().split("\n")[0]

        # Get type hints
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        # Parse docstring for parameter descriptions
        param_descriptions = _parse_docstring_params(func.__doc__ or "")

        # Extract parameters from signature
        sig = inspect.signature(func)
        parameters: List[ToolParameter] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, str)
            param_desc = param_descriptions.get(param_name, "")
            is_required = param.default is inspect.Parameter.empty
            default = None if is_required else param.default

            parameters.append(
                ToolParameter(
                    name=param_name,
                    type_hint=param_type,
                    description=param_desc,
                    required=is_required,
                    default=default,
                )
            )

        return ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            func=func,
            is_async=asyncio.iscoroutinefunction(func),
            metadata=metadata,
        )

    return decorator


def _parse_docstring_params(docstring: str) -> Dict[str, str]:
    """
    Parse parameter descriptions from docstring (Google/NumPy style).

    Args:
        docstring: Function docstring

    Returns:
        Dictionary mapping parameter names to descriptions
    """
    params = {}
    lines = docstring.split("\n")
    in_args_section = False
    current_param = None
    current_desc = []

    for line in lines:
        stripped = line.strip()

        # Check for Args section start
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args_section = True
            continue

        # Check for section end
        if stripped.lower() in ("returns:", "raises:", "yields:", "examples:"):
            if current_param:
                params[current_param] = " ".join(current_desc).strip()
            in_args_section = False
            continue

        if in_args_section:
            # Check for new parameter (name: description)
            if ":" in stripped and not stripped.startswith(" "):
                # Save previous parameter
                if current_param:
                    params[current_param] = " ".join(current_desc).strip()

                parts = stripped.split(":", 1)
                # Handle "name (type): description" format
                param_part = parts[0].strip()
                if "(" in param_part:
                    param_part = param_part.split("(")[0].strip()
                current_param = param_part
                current_desc = [parts[1].strip()] if len(parts) > 1 else []
            elif current_param and stripped:
                # Continuation of description
                current_desc.append(stripped)

    # Don't forget last parameter
    if current_param:
        params[current_param] = " ".join(current_desc).strip()

    return params


# =============================================================================
# Tool Manager
# =============================================================================


class ToolManager:
    """
    Manages tool registration, schema generation, and execution.

    Provides a unified interface for working with tools across different
    LLM providers.

    Example:
        >>> manager = ToolManager()
        >>> manager.register(my_tool)
        >>> schemas = manager.get_openai_tools()
        >>> result = await manager.execute_tool_call(tool_call)
    """

    def __init__(self):
        """Initialize tool manager."""
        self._tools: Dict[str, ToolDefinition] = {}
        self._call_history: List[Dict[str, Any]] = []

    def register(
        self,
        tool_or_func: Union[ToolDefinition, Callable[..., Any]],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ToolDefinition:
        """
        Register a tool.

        Args:
            tool_or_func: ToolDefinition or function to register
            name: Override tool name
            description: Override tool description

        Returns:
            Registered ToolDefinition
        """
        if isinstance(tool_or_func, ToolDefinition):
            tool_def = tool_or_func
        else:
            # Create tool from function
            tool_def = tool(name=name, description=description)(tool_or_func)

        if name:
            tool_def.name = name
        if description:
            tool_def.description = description

        self._tools[tool_def.name] = tool_def
        logger.debug(f"Registered tool: {tool_def.name}")
        return tool_def

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool.

        Args:
            name: Tool name

        Returns:
            True if tool was removed
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List registered tool names."""
        return list(self._tools.keys())

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools in OpenAI format.

        Returns:
            List of OpenAI tool definitions
        """
        return [tool.get_openai_schema() for tool in self._tools.values()]

    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools in Anthropic format.

        Returns:
            List of Anthropic tool definitions
        """
        return [tool.get_anthropic_schema() for tool in self._tools.values()]

    async def execute_tool_call(
        self,
        tool_call: ToolCall,
        *,
        timeout: float = 30.0,
    ) -> ToolResult:
        """
        Execute a single tool call.

        Args:
            tool_call: Tool call to execute
            timeout: Execution timeout in seconds

        Returns:
            Tool execution result
        """
        tool_def = self._tools.get(tool_call.name)
        if not tool_def:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                success=False,
                error=f"Unknown tool: {tool_call.name}",
            )

        # Parse arguments
        args = tool_call.get_arguments()

        # Execute with timeout
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                tool_def.execute(**args),
                timeout=timeout,
            )
            latency_ms = (time.time() - start_time) * 1000

            # Record call
            self._call_history.append(
                {
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "args": args,
                    "success": True,
                    "latency_ms": latency_ms,
                }
            )

            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=result,
                success=True,
                latency_ms=latency_ms,
            )

        except asyncio.TimeoutError:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                success=False,
                error=f"Tool execution timed out after {timeout}s",
                latency_ms=timeout * 1000,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.exception(f"Tool execution failed: {tool_call.name}")

            # Record failed call
            self._call_history.append(
                {
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "args": args,
                    "success": False,
                    "error": str(e),
                    "latency_ms": latency_ms,
                }
            )

            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )

    async def execute_parallel(
        self,
        tool_calls: List[ToolCall],
        *,
        timeout: float = 30.0,
    ) -> List[ToolResult]:
        """
        Execute multiple tool calls in parallel.

        Args:
            tool_calls: Tool calls to execute
            timeout: Per-call timeout

        Returns:
            List of results in same order as calls
        """
        tasks = [self.execute_tool_call(tc, timeout=timeout) for tc in tool_calls]
        return await asyncio.gather(*tasks)

    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get tool call history."""
        return self._call_history.copy()

    def clear_history(self) -> None:
        """Clear call history."""
        self._call_history.clear()


# =============================================================================
# Tool Call Parser
# =============================================================================


class ToolCallParser:
    """
    Parses tool calls from LLM responses.

    Handles different formats from OpenAI, Anthropic, and raw JSON.
    """

    @staticmethod
    def parse_openai_response(response: Dict[str, Any]) -> List[ToolCall]:
        """
        Parse tool calls from OpenAI response.

        Args:
            response: OpenAI API response

        Returns:
            List of ToolCall objects
        """
        tool_calls = []

        # Check for tool_calls in message
        message = response.get("choices", [{}])[0].get("message", {})
        raw_calls = message.get("tool_calls", [])

        for call in raw_calls:
            if call.get("type") == "function":
                func = call.get("function", {})
                tool_calls.append(
                    ToolCall(
                        id=call.get("id", str(uuid.uuid4())[:8]),
                        name=func.get("name", ""),
                        arguments=func.get("arguments", "{}"),
                    )
                )

        return tool_calls

    @staticmethod
    def parse_anthropic_response(response: Dict[str, Any]) -> List[ToolCall]:
        """
        Parse tool calls from Anthropic response.

        Args:
            response: Anthropic API response

        Returns:
            List of ToolCall objects
        """
        tool_calls = []

        # Anthropic uses content blocks with tool_use type
        content = response.get("content", [])
        for block in content:
            if block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", str(uuid.uuid4())[:8]),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )

        return tool_calls

    @staticmethod
    def parse_json_string(text: str) -> List[ToolCall]:
        """
        Parse tool calls from raw JSON string.

        Handles both single tool call and array formats.

        Args:
            text: JSON string with tool call(s)

        Returns:
            List of ToolCall objects
        """
        tool_calls = []

        try:
            # Try to find JSON in the text
            import re

            json_pattern = r"\{[^{}]*\}|\[[^\[\]]*\]"
            matches = re.findall(json_pattern, text, re.DOTALL)

            for match in matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, list):
                        for item in data:
                            if "name" in item or "tool" in item or "function" in item:
                                tc = ToolCallParser._dict_to_tool_call(item)
                                if tc:
                                    tool_calls.append(tc)
                    elif isinstance(data, dict):
                        if "name" in data or "tool" in data or "function" in data:
                            tc = ToolCallParser._dict_to_tool_call(data)
                            if tc:
                                tool_calls.append(tc)
                except json.JSONDecodeError:
                    continue

        except Exception:
            pass

        return tool_calls

    @staticmethod
    def _dict_to_tool_call(data: Dict[str, Any]) -> Optional[ToolCall]:
        """Convert dictionary to ToolCall."""
        name = data.get("name") or data.get("tool") or data.get("function")
        if not name:
            return None

        args = (
            data.get("arguments")
            or data.get("args")
            or data.get("input")
            or data.get("parameters")
            or {}
        )

        return ToolCall(
            id=data.get("id", str(uuid.uuid4())[:8]),
            name=name,
            arguments=args,
        )


# =============================================================================
# Built-in RAG Tools
# =============================================================================


def create_retrieval_tool(
    retriever_func: Callable[[str, int], List[Dict[str, Any]]],
) -> ToolDefinition:
    """
    Create a retrieval tool from a retriever function.

    Args:
        retriever_func: Function that takes (query, k) and returns documents

    Returns:
        ToolDefinition for retrieval
    """

    @tool(
        name="retrieve_documents",
        description="Search and retrieve relevant documents from the knowledge base",
    )
    async def retrieve_documents(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents relevant to the query.

        Args:
            query: Search query to find relevant documents
            num_results: Maximum number of documents to retrieve (1-20)
        """
        if asyncio.iscoroutinefunction(retriever_func):
            return await retriever_func(query, num_results)
        return retriever_func(query, num_results)

    return retrieve_documents


def create_answer_tool() -> ToolDefinition:
    """
    Create a final answer tool for agent termination.

    Returns:
        ToolDefinition for final answer
    """

    @tool(
        name="final_answer",
        description="Provide the final answer to the user's question",
    )
    def final_answer(answer: str, confidence: float = 1.0) -> Dict[str, Any]:
        """
        Submit the final answer.

        Args:
            answer: The complete answer to the user's question
            confidence: Confidence in the answer (0.0 to 1.0)
        """
        return {"answer": answer, "confidence": confidence, "is_final": True}

    return final_answer


# =============================================================================
# Convenience Functions
# =============================================================================


def get_tool_manager() -> ToolManager:
    """Get a global tool manager instance."""
    if not hasattr(get_tool_manager, "_instance"):
        get_tool_manager._instance = ToolManager()
    return get_tool_manager._instance


def register_tool(
    func: Callable[..., Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ToolDefinition:
    """Register a tool with the global manager."""
    return get_tool_manager().register(func, name=name, description=description)
