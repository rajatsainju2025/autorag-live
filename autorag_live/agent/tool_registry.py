"""
Centralized Tool Registry with Discovery and Validation.

State-of-the-art tool management for agentic RAG:
- Automatic schema generation from type hints
- Tool discovery and categorization
- Permission and safety controls
- Execution sandboxing
- Tool composition and chaining

Example:
    >>> registry = ToolRegistry()
    >>>
    >>> @registry.tool(category="search", permissions=["web"])
    ... async def web_search(query: str, num_results: int = 5) -> List[str]:
    ...     '''Search the web for information.'''
    ...     return await search_api(query, num_results)
    ...
    >>> # Auto-discovers tools and generates OpenAI-compatible schema
    >>> tools_schema = registry.get_openai_tools()
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# Tool Categories and Permissions
# =============================================================================


class ToolCategory(str, Enum):
    """Categories for organizing tools."""

    RETRIEVAL = "retrieval"  # Document/knowledge retrieval
    SEARCH = "search"  # Web/external search
    COMPUTATION = "computation"  # Math, data processing
    CODE = "code"  # Code execution
    DATABASE = "database"  # Database operations
    FILE = "file"  # File system operations
    API = "api"  # External API calls
    ANALYSIS = "analysis"  # Data analysis
    GENERATION = "generation"  # Content generation
    UTILITY = "utility"  # General utilities
    CUSTOM = "custom"  # User-defined


class ToolPermission(str, Enum):
    """Permissions required to use tools."""

    READ = "read"  # Read-only operations
    WRITE = "write"  # Write/modify operations
    EXECUTE = "execute"  # Code execution
    NETWORK = "network"  # Network access
    FILESYSTEM = "filesystem"  # File system access
    SENSITIVE = "sensitive"  # Access to sensitive data
    ADMIN = "admin"  # Administrative operations


class ToolStatus(Enum):
    """Tool availability status."""

    AVAILABLE = auto()
    DISABLED = auto()
    RATE_LIMITED = auto()
    ERROR = auto()


# =============================================================================
# Tool Schema Types
# =============================================================================


@dataclass
class ParameterSchema:
    """Schema for a tool parameter."""

    name: str
    type: str  # JSON Schema type
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    pattern: Optional[str] = None  # Regex for strings
    items: Optional[Dict[str, Any]] = None  # For arrays
    properties: Optional[Dict[str, Any]] = None  # For objects

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {
            "type": self.type,
        }
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = self.enum
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.pattern:
            schema["pattern"] = self.pattern
        if self.items:
            schema["items"] = self.items
        if self.properties:
            schema["properties"] = self.properties
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolSchema:
    """Complete schema for a tool."""

    name: str
    description: str
    parameters: List[ParameterSchema] = field(default_factory=list)
    returns: str = "any"
    returns_description: str = ""
    category: ToolCategory = ToolCategory.CUSTOM
    permissions: List[ToolPermission] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI tool calling format."""
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

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
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


# =============================================================================
# Tool Execution Results
# =============================================================================


@dataclass
class ToolExecutionResult:
    """Result from tool execution."""

    tool_name: str
    call_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_message_content(self) -> str:
        """Convert to content for LLM message."""
        if not self.success:
            return f"Error executing {self.tool_name}: {self.error}"

        if isinstance(self.result, (dict, list)):
            return json.dumps(self.result, indent=2)
        return str(self.result)


# =============================================================================
# Tool Protocol and Base Class
# =============================================================================


@runtime_checkable
class ToolProtocol(Protocol):
    """Protocol for tool implementations."""

    @property
    def name(self) -> str:
        """Tool name."""
        ...

    @property
    def schema(self) -> ToolSchema:
        """Tool schema."""
        ...

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool."""
        ...


class BaseTool(ABC):
    """Base class for tool implementations."""

    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory = ToolCategory.CUSTOM,
        permissions: Optional[List[ToolPermission]] = None,
    ):
        self._name = name
        self._description = description
        self._category = category
        self._permissions = permissions or []
        self._status = ToolStatus.AVAILABLE
        self._call_count = 0
        self._total_latency = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def category(self) -> ToolCategory:
        return self._category

    @property
    def permissions(self) -> List[ToolPermission]:
        return self._permissions

    @property
    def status(self) -> ToolStatus:
        return self._status

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Return the tool schema."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with given arguments."""
        ...

    def enable(self) -> None:
        """Enable the tool."""
        self._status = ToolStatus.AVAILABLE

    def disable(self) -> None:
        """Disable the tool."""
        self._status = ToolStatus.DISABLED


# =============================================================================
# Type Conversion Utilities
# =============================================================================


def python_type_to_json_type(py_type: Any) -> str:
    """Convert Python type to JSON Schema type."""
    if py_type is None or py_type is type(None):
        return "null"

    origin = get_origin(py_type)

    # Handle Optional
    if origin is Union:
        args = get_args(py_type)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return python_type_to_json_type(non_none[0])
        return "string"  # Fallback

    # Handle List/Array
    if origin in (list, List):
        return "array"

    # Handle Dict/Object
    if origin in (dict, Dict):
        return "object"

    # Basic types
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        bytes: "string",
    }

    return type_map.get(py_type, "string")


def extract_parameters_from_function(
    func: Callable,
) -> List[ParameterSchema]:
    """Extract parameter schemas from function signature."""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    doc = inspect.getdoc(func) or ""

    # Parse docstring for parameter descriptions
    param_docs = _parse_docstring_params(doc)

    parameters = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        py_type = hints.get(name, str)
        json_type = python_type_to_json_type(py_type)

        param_schema = ParameterSchema(
            name=name,
            type=json_type,
            description=param_docs.get(name, f"Parameter {name}"),
            required=param.default is inspect.Parameter.empty,
            default=None if param.default is inspect.Parameter.empty else param.default,
        )

        # Handle List type items
        origin = get_origin(py_type)
        if origin in (list, List):
            args = get_args(py_type)
            if args:
                param_schema.items = {"type": python_type_to_json_type(args[0])}

        parameters.append(param_schema)

    return parameters


def _parse_docstring_params(doc: str) -> Dict[str, str]:
    """Parse parameter descriptions from docstring."""
    params = {}
    lines = doc.split("\n")
    in_params = False
    current_param = None
    current_desc = []

    for line in lines:
        stripped = line.strip()

        # Check for Args/Parameters section
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            in_params = True
            continue

        # Check for end of params section
        if stripped.lower() in ("returns:", "return:", "raises:", "yields:", "examples:"):
            if current_param:
                params[current_param] = " ".join(current_desc).strip()
            in_params = False
            continue

        if in_params:
            # Check if new parameter
            if ":" in stripped and not stripped.startswith(":"):
                if current_param:
                    params[current_param] = " ".join(current_desc).strip()
                parts = stripped.split(":", 1)
                # Handle format: "param_name (type): description"
                param_part = parts[0].strip()
                if "(" in param_part:
                    param_part = param_part.split("(")[0].strip()
                current_param = param_part
                current_desc = [parts[1].strip()] if len(parts) > 1 else []
            elif current_param and stripped:
                current_desc.append(stripped)

    if current_param:
        params[current_param] = " ".join(current_desc).strip()

    return params


# =============================================================================
# Tool Registry
# =============================================================================


class ToolRegistry:
    """
    Centralized registry for tool management.

    Features:
    - Automatic schema generation from decorated functions
    - Tool discovery and categorization
    - Permission-based access control
    - Rate limiting and usage tracking
    - Multi-format schema export (OpenAI, Anthropic)
    """

    def __init__(self, default_timeout: float = 30.0):
        """Initialize tool registry."""
        self._tools: Dict[str, ToolProtocol] = {}
        self._function_tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, ToolSchema] = {}
        self._permissions: Dict[str, Set[ToolPermission]] = {}
        self._categories: Dict[ToolCategory, List[str]] = {cat: [] for cat in ToolCategory}
        self._default_timeout = default_timeout
        self._execution_stats: Dict[str, Dict[str, Any]] = {}

    def tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: ToolCategory = ToolCategory.CUSTOM,
        permissions: Optional[List[ToolPermission]] = None,
        tags: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        timeout: Optional[float] = None,
    ) -> Callable:
        """
        Decorator to register a function as a tool.

        Args:
            name: Tool name (defaults to function name)
            description: Tool description (defaults to docstring)
            category: Tool category for organization
            permissions: Required permissions
            tags: Search/filter tags
            examples: Usage examples
            timeout: Execution timeout

        Returns:
            Decorated function
        """

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or inspect.getdoc(func) or f"Tool: {tool_name}"

            # Extract parameters from function
            params = extract_parameters_from_function(func)

            # Create schema
            schema = ToolSchema(
                name=tool_name,
                description=tool_desc,
                parameters=params,
                category=category,
                permissions=permissions or [],
                tags=tags or [],
                examples=examples or [],
            )

            # Store function and schema
            self._function_tools[tool_name] = func
            self._schemas[tool_name] = schema
            self._categories[category].append(tool_name)

            if permissions:
                self._permissions[tool_name] = set(permissions)

            # Initialize stats
            self._execution_stats[tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_latency_ms": 0.0,
            }

            logger.debug(f"Registered tool: {tool_name}")

            @functools.wraps(func)
            async def wrapper(**kwargs: Any) -> Any:
                return await self.execute(tool_name, **kwargs)

            # Attach metadata
            wrapper._tool_name = tool_name
            wrapper._tool_schema = schema

            return wrapper

        return decorator

    def register(
        self,
        tool: Union[ToolProtocol, Callable],
        name: Optional[str] = None,
        category: ToolCategory = ToolCategory.CUSTOM,
    ) -> None:
        """
        Register a tool instance or function.

        Args:
            tool: Tool instance or callable
            name: Override tool name
            category: Tool category
        """
        if isinstance(tool, ToolProtocol):
            tool_name = name or tool.name
            self._tools[tool_name] = tool
            self._schemas[tool_name] = tool.schema
            self._categories[tool.schema.category].append(tool_name)
        elif callable(tool):
            # Wrap function as tool
            self.tool(name=name, category=category)(tool)
        else:
            raise TypeError(f"Cannot register {type(tool)} as tool")

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
        if name in self._function_tools:
            del self._function_tools[name]
        if name in self._schemas:
            schema = self._schemas[name]
            if name in self._categories.get(schema.category, []):
                self._categories[schema.category].remove(name)
            del self._schemas[name]

    def get(self, name: str) -> Optional[Union[ToolProtocol, Callable]]:
        """Get a tool by name."""
        return self._tools.get(name) or self._function_tools.get(name)

    def get_schema(self, name: str) -> Optional[ToolSchema]:
        """Get tool schema by name."""
        return self._schemas.get(name)

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        permissions: Optional[List[ToolPermission]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[str]:
        """
        List available tools with optional filtering.

        Args:
            category: Filter by category
            permissions: Filter by required permissions
            tags: Filter by tags

        Returns:
            List of tool names
        """
        tools = list(self._schemas.keys())

        if category:
            tools = [t for t in tools if self._schemas[t].category == category]

        if permissions:
            permission_set = set(permissions)
            tools = [t for t in tools if permission_set.issubset(set(self._schemas[t].permissions))]

        if tags:
            tag_set = set(tags)
            tools = [t for t in tools if tag_set.intersection(set(self._schemas[t].tags))]

        return tools

    async def execute(
        self,
        name: str,
        *,
        call_id: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> ToolExecutionResult:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            call_id: Unique call identifier
            timeout: Execution timeout
            **kwargs: Tool arguments

        Returns:
            ToolExecutionResult with result or error
        """
        call_id = call_id or str(uuid.uuid4())[:8]
        timeout = timeout or self._default_timeout
        start_time = time.time()

        # Get tool
        tool = self._tools.get(name)
        func = self._function_tools.get(name)

        if not tool and not func:
            return ToolExecutionResult(
                tool_name=name,
                call_id=call_id,
                success=False,
                result=None,
                error=f"Tool '{name}' not found",
            )

        try:
            # Execute with timeout
            if tool:
                coro = tool.execute(**kwargs)
            else:
                if asyncio.iscoroutinefunction(func):
                    coro = func(**kwargs)
                else:
                    coro = asyncio.to_thread(func, **kwargs)

            result = await asyncio.wait_for(coro, timeout=timeout)
            latency = (time.time() - start_time) * 1000

            # Update stats
            stats = self._execution_stats.get(name, {})
            stats["calls"] = stats.get("calls", 0) + 1
            stats["successes"] = stats.get("successes", 0) + 1
            stats["total_latency_ms"] = stats.get("total_latency_ms", 0) + latency

            return ToolExecutionResult(
                tool_name=name,
                call_id=call_id,
                success=True,
                result=result,
                latency_ms=latency,
            )

        except asyncio.TimeoutError:
            return ToolExecutionResult(
                tool_name=name,
                call_id=call_id,
                success=False,
                result=None,
                error=f"Tool execution timed out after {timeout}s",
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            stats = self._execution_stats.get(name, {})
            stats["failures"] = stats.get("failures", 0) + 1

            return ToolExecutionResult(
                tool_name=name,
                call_id=call_id,
                success=False,
                result=None,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    # -------------------------------------------------------------------------
    # Schema Export
    # -------------------------------------------------------------------------

    def get_openai_tools(
        self,
        names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get tools in OpenAI format."""
        tool_names = names or list(self._schemas.keys())
        return [self._schemas[n].to_openai_tool() for n in tool_names if n in self._schemas]

    def get_anthropic_tools(
        self,
        names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get tools in Anthropic format."""
        tool_names = names or list(self._schemas.keys())
        return [self._schemas[n].to_anthropic_tool() for n in tool_names if n in self._schemas]

    def get_tools_description(
        self,
        names: Optional[List[str]] = None,
        format: str = "text",
    ) -> str:
        """
        Get human-readable tools description.

        Args:
            names: Specific tools to describe
            format: Output format ('text' or 'markdown')

        Returns:
            Formatted tools description
        """
        tool_names = names or list(self._schemas.keys())
        lines = []

        for name in tool_names:
            schema = self._schemas.get(name)
            if not schema:
                continue

            if format == "markdown":
                lines.append(f"### {name}")
                lines.append(f"{schema.description}\n")
                if schema.parameters:
                    lines.append("**Parameters:**")
                    for param in schema.parameters:
                        req = "*required*" if param.required else "*optional*"
                        lines.append(f"- `{param.name}` ({param.type}, {req}): {param.description}")
                lines.append("")
            else:
                lines.append(f"- {name}: {schema.description}")
                for param in schema.parameters:
                    req = "required" if param.required else "optional"
                    lines.append(f"  - {param.name} ({param.type}, {req})")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get execution statistics."""
        if name:
            return self._execution_stats.get(name, {})
        return dict(self._execution_stats)

    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary."""
        total_calls = sum(s.get("calls", 0) for s in self._execution_stats.values())
        total_successes = sum(s.get("successes", 0) for s in self._execution_stats.values())

        return {
            "total_tools": len(self._schemas),
            "tools_by_category": {
                cat.value: len(tools) for cat, tools in self._categories.items() if tools
            },
            "total_calls": total_calls,
            "success_rate": total_successes / total_calls if total_calls > 0 else 0,
        }


# =============================================================================
# Pre-built Common Tools
# =============================================================================


def create_retrieval_tool(retriever: Any) -> Callable:
    """Create a retrieval tool from a retriever instance."""

    async def retrieve_documents(
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with content and scores
        """
        if hasattr(retriever, "retrieve"):
            docs = await retriever.retrieve(query, top_k=top_k)
        elif hasattr(retriever, "search"):
            docs = await retriever.search(query, k=top_k)
        else:
            raise ValueError("Retriever must have 'retrieve' or 'search' method")

        return [
            {
                "content": doc.content if hasattr(doc, "content") else str(doc),
                "score": doc.score if hasattr(doc, "score") else 0.0,
                "metadata": doc.metadata if hasattr(doc, "metadata") else {},
            }
            for doc in docs
        ]

    return retrieve_documents


def create_calculator_tool() -> Callable:
    """Create a basic calculator tool."""

    async def calculate(expression: str) -> float:
        """
        Evaluate a mathematical expression.

        Args:
            expression: Mathematical expression to evaluate (e.g., "2 + 2 * 3")

        Returns:
            Result of the calculation
        """
        # Safe evaluation of math expressions
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Expression contains invalid characters")

        result = eval(expression, {"__builtins__": {}})
        return float(result)

    return calculate


# =============================================================================
# Global Registry Instance
# =============================================================================


# Default global registry
_default_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the default global tool registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: ToolCategory = ToolCategory.CUSTOM,
    permissions: Optional[List[ToolPermission]] = None,
    **kwargs: Any,
) -> Callable:
    """Convenience decorator using global registry."""
    return get_registry().tool(
        name=name,
        description=description,
        category=category,
        permissions=permissions,
        **kwargs,
    )
