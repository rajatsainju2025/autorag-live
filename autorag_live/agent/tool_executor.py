"""
Tool Execution Framework for Agentic RAG.

Provides a flexible framework for agents to execute external tools,
including web search, code execution, API calls, and custom tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


class ToolCategory(str, Enum):
    """Categories of tools available to agents."""

    RETRIEVAL = "retrieval"
    COMPUTATION = "computation"
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    API_CALL = "api_call"
    FILE_OPERATION = "file_operation"
    DATABASE = "database"
    CUSTOM = "custom"


class ToolExecutionStatus(str, Enum):
    """Status of tool execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    param_type: type
    description: str
    required: bool = True
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a parameter value."""
        if value is None:
            if self.required:
                return False, f"Parameter '{self.name}' is required"
            return True, None

        if not isinstance(value, self.param_type):
            return False, (f"Parameter '{self.name}' must be of type {self.param_type.__name__}")

        if self.validator and not self.validator(value):
            return False, f"Parameter '{self.name}' failed validation"

        return True, None


@dataclass
class ToolDefinition:
    """Complete definition of a tool."""

    name: str
    description: str
    category: ToolCategory
    parameters: list[ToolParameter] = field(default_factory=list)
    return_type: type = str
    requires_confirmation: bool = False
    timeout_seconds: float = 30.0
    max_retries: int = 3
    tags: list[str] = field(default_factory=list)

    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON schema for LLM function calling."""
        properties = {}
        required = []

        for param in self.parameters:
            param_schema = {"description": param.description}

            type_mapping = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }
            param_schema["type"] = type_mapping.get(param.param_type, "string")

            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class ToolExecutionResult:
    """Result of a tool execution."""

    tool_name: str
    status: ToolExecutionStatus
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolExecutionStatus.SUCCESS


@dataclass
class ToolCall:
    """A requested tool call from an agent."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = ""
    requester: str = "agent"

    def __post_init__(self):
        """Generate call ID if not provided."""
        if not self.call_id:
            import uuid

            self.call_id = str(uuid.uuid4())[:8]


class BaseTool(ABC):
    """Base class for all tools."""

    def __init__(self, definition: ToolDefinition):
        """Initialize tool with its definition."""
        self.definition = definition

    @property
    def name(self) -> str:
        """Get tool name."""
        return self.definition.name

    def validate_arguments(self, arguments: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate arguments against tool parameters."""
        errors = []

        for param in self.definition.parameters:
            value = arguments.get(param.name, param.default)
            is_valid, error = param.validate(value)
            if not is_valid:
                errors.append(error)

        return len(errors) == 0, errors

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        pass

    def __call__(self, **kwargs) -> ToolExecutionResult:
        """Execute tool and wrap result."""
        import time

        start_time = time.time()

        is_valid, errors = self.validate_arguments(kwargs)
        if not is_valid:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.FAILED,
                error=f"Validation failed: {'; '.join(errors)}",
            )

        try:
            result = self.execute(**kwargs)
            execution_time = (time.time() - start_time) * 1000

            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.SUCCESS,
                result=result,
                execution_time_ms=execution_time,
            )

        except TimeoutError:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.TIMEOUT,
                error=f"Tool execution timed out after {self.definition.timeout_seconds}s",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return ToolExecutionResult(
                tool_name=self.name,
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )


class WebSearchTool(BaseTool):
    """Tool for web search operations."""

    def __init__(self, search_provider: str = "google"):
        """Initialize web search tool."""
        definition = ToolDefinition(
            name="web_search",
            description="Search the web for information on a given query",
            category=ToolCategory.WEB_SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    param_type=str,
                    description="The search query",
                    required=True,
                ),
                ToolParameter(
                    name="num_results",
                    param_type=int,
                    description="Number of results to return",
                    required=False,
                    default=5,
                ),
            ],
            timeout_seconds=15.0,
            tags=["search", "web", "retrieval"],
        )
        super().__init__(definition)
        self.search_provider = search_provider

    def execute(self, **kwargs) -> list[dict[str, Any]]:
        """Execute web search."""
        query = kwargs.get("query", "")
        num_results = kwargs.get("num_results", 5)
        """Execute web search."""
        return [
            {
                "title": f"Result {i} for: {query}",
                "url": f"https://example.com/result{i}",
                "snippet": f"This is a mock search result for query: {query}",
            }
            for i in range(num_results)
        ]


class CalculatorTool(BaseTool):
    """Tool for mathematical calculations."""

    def __init__(self):
        """Initialize calculator tool."""
        definition = ToolDefinition(
            name="calculator",
            description="Perform mathematical calculations safely",
            category=ToolCategory.COMPUTATION,
            parameters=[
                ToolParameter(
                    name="expression",
                    param_type=str,
                    description="The mathematical expression to evaluate",
                    required=True,
                ),
            ],
            timeout_seconds=5.0,
            tags=["math", "calculation", "computation"],
        )
        super().__init__(definition)

    def execute(self, **kwargs) -> float:
        """Execute calculation safely."""
        expression = kwargs.get("expression", "")
        import ast
        import operator

        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def _eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                op = allowed_operators.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                return op(_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                op = allowed_operators.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                return op(_eval(node.operand))
            else:
                raise ValueError(f"Unsupported expression type: {type(node).__name__}")

        tree = ast.parse(expression, mode="eval")
        return float(_eval(tree.body))


class CodeExecutorTool(BaseTool):
    """Tool for safe code execution in a sandbox."""

    def __init__(self, allowed_modules: Optional[list[str]] = None):
        """Initialize code executor tool."""
        definition = ToolDefinition(
            name="code_executor",
            description="Execute Python code in a safe sandbox environment",
            category=ToolCategory.CODE_EXECUTION,
            parameters=[
                ToolParameter(
                    name="code",
                    param_type=str,
                    description="The Python code to execute",
                    required=True,
                ),
                ToolParameter(
                    name="timeout",
                    param_type=float,
                    description="Maximum execution time in seconds",
                    required=False,
                    default=5.0,
                ),
            ],
            timeout_seconds=10.0,
            requires_confirmation=True,
            tags=["code", "python", "execution"],
        )
        super().__init__(definition)
        self.allowed_modules = allowed_modules or ["math", "statistics", "datetime"]

    def execute(self, **kwargs) -> dict[str, Any]:
        """Execute code in sandbox."""
        code = kwargs.get("code", "")
        import io
        import sys

        safe_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "bool": bool,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
        }

        safe_globals = {"__builtins__": safe_builtins}

        for module_name in self.allowed_modules:
            try:
                safe_globals[module_name] = __import__(module_name)
            except ImportError:
                pass

        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            exec(code, safe_globals)
            output = captured_output.getvalue()

            return {
                "success": True,
                "output": output,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "output": captured_output.getvalue(),
                "error": str(e),
            }

        finally:
            sys.stdout = old_stdout


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        """Initialize tool registry."""
        self._tools: dict[str, BaseTool] = {}
        self._execution_history: list[ToolExecutionResult] = []

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, tool_name: str) -> None:
        """Unregister a tool."""
        if tool_name in self._tools:
            del self._tools[tool_name]

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)

    def list_tools(self, category: Optional[ToolCategory] = None) -> list[ToolDefinition]:
        """List available tools, optionally filtered by category."""
        tools = self._tools.values()
        if category:
            tools = [t for t in tools if t.definition.category == category]
        return [t.definition for t in tools]

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all tools (for LLM function calling)."""
        return [tool.definition.to_schema() for tool in self._tools.values()]

    def execute(
        self, tool_call: ToolCall, require_confirmation: bool = True
    ) -> ToolExecutionResult:
        """Execute a tool call."""
        tool = self.get(tool_call.tool_name)
        if tool is None:
            result = ToolExecutionResult(
                tool_name=tool_call.tool_name,
                status=ToolExecutionStatus.FAILED,
                error=f"Tool '{tool_call.tool_name}' not found",
            )
            self._execution_history.append(result)
            return result

        if require_confirmation and tool.definition.requires_confirmation:
            pass

        result = tool(**tool_call.arguments)
        self._execution_history.append(result)
        return result

    @property
    def execution_history(self) -> list[ToolExecutionResult]:
        """Get execution history."""
        return self._execution_history.copy()


class ToolExecutor:
    """Executor for handling tool calls from agents."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        """Initialize tool executor."""
        self.registry = registry or ToolRegistry()
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default tools."""
        self.registry.register(WebSearchTool())
        self.registry.register(CalculatorTool())
        self.registry.register(CodeExecutorTool())

    def add_tool(self, tool: BaseTool) -> None:
        """Add a custom tool."""
        self.registry.register(tool)

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool."""
        self.registry.unregister(tool_name)

    def get_available_tools(self, category: Optional[ToolCategory] = None) -> list[ToolDefinition]:
        """Get list of available tools."""
        return self.registry.list_tools(category)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas for LLM function calling."""
        return self.registry.get_schemas()

    def execute_tool(self, tool_call: ToolCall) -> ToolExecutionResult:
        """Execute a single tool call."""
        return self.registry.execute(tool_call)

    def execute_tools(
        self, tool_calls: list[ToolCall], parallel: bool = False
    ) -> list[ToolExecutionResult]:
        """Execute multiple tool calls."""
        if parallel:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.execute_tool, call) for call in tool_calls]
                return [f.result() for f in concurrent.futures.as_completed(futures)]
        else:
            return [self.execute_tool(call) for call in tool_calls]

    def parse_llm_tool_calls(self, llm_response: dict[str, Any]) -> list[ToolCall]:
        """Parse tool calls from LLM response."""
        tool_calls = []

        if "tool_calls" in llm_response:
            for tc in llm_response["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        tool_name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", {}),
                        call_id=tc.get("id", ""),
                    )
                )

        return tool_calls

    def format_results_for_llm(self, results: list[ToolExecutionResult]) -> list[dict[str, Any]]:
        """Format tool results for LLM consumption."""
        return [
            {
                "tool_name": r.tool_name,
                "status": r.status.value,
                "result": r.result if r.is_success else None,
                "error": r.error,
            }
            for r in results
        ]


def create_custom_tool(
    name: str,
    description: str,
    category: ToolCategory,
    parameters: list[dict[str, Any]],
    executor: Callable[..., Any],
    **kwargs,
) -> BaseTool:
    """Create a custom tool from a function."""
    params = [
        ToolParameter(
            name=p["name"],
            param_type=p.get("type", str),
            description=p.get("description", ""),
            required=p.get("required", True),
            default=p.get("default"),
        )
        for p in parameters
    ]

    definition = ToolDefinition(
        name=name,
        description=description,
        category=category,
        parameters=params,
        **kwargs,
    )

    class CustomTool(BaseTool):
        def execute(self, **kw):
            return executor(**kw)

    return CustomTool(definition)


__all__ = [
    "ToolCategory",
    "ToolExecutionStatus",
    "ToolParameter",
    "ToolDefinition",
    "ToolExecutionResult",
    "ToolCall",
    "BaseTool",
    "WebSearchTool",
    "CalculatorTool",
    "CodeExecutorTool",
    "ToolRegistry",
    "ToolExecutor",
    "create_custom_tool",
]
