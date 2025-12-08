"""
Tool registry and execution engine for agents.

Provides tool discovery, schema validation, and safe execution.
"""

from __future__ import annotations

import inspect
import logging
import signal
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from autorag_live.types import ValidationError


@dataclass
class ToolParameter:
    """Tool parameter definition."""

    name: str
    param_type: str  # "string", "integer", "array", "object", "boolean"
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None

    def validate(self, value: Any) -> None:
        """
        Validate parameter value.

        Args:
            value: Value to validate

        Raises:
            ValidationError: If validation fails
        """
        if value is None:
            if self.required:
                raise ValidationError(
                    f"Parameter '{self.name}' is required", context={"param": self.name}
                )
            return

        # Type validation
        type_mapping = {
            "string": str,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected_type = type_mapping.get(self.param_type)
        if expected_type and not isinstance(value, expected_type):
            raise ValidationError(
                f"Parameter '{self.name}' must be {self.param_type}",
                context={
                    "param": self.name,
                    "expected": self.param_type,
                    "got": type(value).__name__,
                },
            )

        # Enum validation
        if self.enum and value not in self.enum:
            raise ValidationError(
                f"Parameter '{self.name}' must be one of {self.enum}",
                context={"param": self.name, "allowed": self.enum},
            )

        # Numeric range validation
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                raise ValidationError(
                    f"Parameter '{self.name}' must be >= {self.min_value}",
                    context={"param": self.name, "min": self.min_value},
                )
            if self.max_value is not None and value > self.max_value:
                raise ValidationError(
                    f"Parameter '{self.name}' must be <= {self.max_value}",
                    context={"param": self.name, "max": self.max_value},
                )


@dataclass
class ToolSchema:
    """Complete tool schema with metadata."""

    name: str
    description: str
    parameters: List[ToolParameter]
    returns: str = "Any"  # Return type description
    examples: Optional[List[Dict[str, Any]]] = field(default=None)
    tags: Optional[List[str]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.param_type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum,
                    "min": p.min_value,
                    "max": p.max_value,
                }
                for p in self.parameters
            ],
            "returns": self.returns,
            "examples": self.examples or [],
            "tags": self.tags or [],
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Validate tool inputs against schema.

        Args:
            inputs: Input dictionary

        Raises:
            ValidationError: If validation fails
        """
        for param in self.parameters:
            if param.name in inputs:
                param.validate(inputs[param.name])
            elif param.required:
                raise ValidationError(
                    f"Required parameter '{param.name}' not provided",
                    context={"tool": self.name, "param": param.name},
                )


class ToolRegistry:
    """
    Registry for managing tools available to agents.

    Handles registration, discovery, validation, and execution of tools.
    """

    def __init__(self):
        """Initialize tool registry."""
        self.tools: Dict[str, Callable] = {}
        self.schemas: Dict[str, ToolSchema] = {}
        self.logger = logging.getLogger("ToolRegistry")

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Optional[List[ToolParameter]] = None,
        tags: Optional[List[str]] = None,
    ) -> ToolSchema:
        """
        Register a tool in the registry.

        Args:
            name: Tool name
            func: Callable function
            description: Tool description
            parameters: List of parameter definitions
            tags: Tool tags for categorization

        Returns:
            Tool schema
        """
        if name in self.tools:
            self.logger.warning(f"Tool '{name}' already registered, overwriting")

        self.tools[name] = func

        # Auto-extract schema from function if not provided
        if parameters is None:
            parameters = self._extract_parameters(func)

        schema = ToolSchema(
            name=name,
            description=description or (func.__doc__ or "").split("\n")[0],
            parameters=parameters,
            returns=self._extract_return_type(func),
            tags=tags or [],
        )

        self.schemas[name] = schema
        self.logger.info(f"Registered tool: {name}")

        return schema

    def _extract_parameters(self, func: Callable) -> List[ToolParameter]:
        """Extract parameters from function signature."""
        sig = inspect.signature(func)
        parameters = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Determine type from annotation
            annotation = param.annotation
            param_type = "string"  # default
            if annotation != inspect.Parameter.empty:
                if annotation in (int, float):
                    param_type = "integer"
                elif annotation == bool:
                    param_type = "boolean"
                elif annotation in (list, List):
                    param_type = "array"
                elif annotation in (dict, Dict):
                    param_type = "object"

            parameters.append(
                ToolParameter(
                    name=param_name,
                    param_type=param_type,
                    description=f"Parameter: {param_name}",
                    required=param.default == inspect.Parameter.empty,
                    default=None if param.default == inspect.Parameter.empty else param.default,
                )
            )

        return parameters

    def _extract_return_type(self, func: Callable) -> str:
        """Extract return type from function."""
        sig = inspect.signature(func)
        if sig.return_annotation != inspect.Signature.empty:
            return str(sig.return_annotation)
        return "Any"

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.

        Args:
            name: Tool name

        Returns:
            True if tool was registered and removed
        """
        if name in self.tools:
            del self.tools[name]
            del self.schemas[name]
            self.logger.info(f"Unregistered tool: {name}")
            return True
        return False

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get tool function by name."""
        return self.tools.get(name)

    def get_schema(self, name: str) -> Optional[ToolSchema]:
        """Get tool schema by name."""
        return self.schemas.get(name)

    def list_tools(
        self, tag: Optional[str] = None, as_dict: bool = False
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        List available tools.

        Args:
            tag: Optional filter by tag
            as_dict: Return schemas instead of names

        Returns:
            List of tool names or schemas
        """
        if tag:
            names = [name for name, schema in self.schemas.items() if tag in (schema.tags or [])]
        else:
            names = list(self.tools.keys())

        if as_dict:
            return [self.schemas[name].to_dict() for name in names]
        return names

    def execute_tool(
        self,
        name: str,
        inputs: Dict[str, Any],
        timeout_seconds: Optional[float] = None,
    ) -> Any:
        """
        Execute a tool with validation.

        Args:
            name: Tool name
            inputs: Tool inputs
            timeout_seconds: Optional execution timeout

        Returns:
            Tool result

        Raises:
            ValidationError: If inputs invalid or tool not found
        """
        if name not in self.tools:
            raise ValidationError(
                f"Tool '{name}' not found", context={"available_tools": list(self.tools.keys())}
            )

        schema = self.schemas[name]
        schema.validate_inputs(inputs)

        tool = self.tools[name]

        try:
            if timeout_seconds:

                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Tool execution exceeded {timeout_seconds}s")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout_seconds))

            result = tool(**inputs)

            if timeout_seconds:
                signal.alarm(0)

            return result

        except TimeoutError as e:
            raise ValidationError(
                f"Tool execution timeout: {str(e)}",
                context={"tool": name, "timeout": timeout_seconds},
            )
        except Exception as e:
            raise ValidationError(
                f"Tool execution failed: {str(e)}", context={"tool": name, "error": str(e)}
            )

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of registered tools."""
        return {
            "num_tools": len(self.tools),
            "tools": [
                {
                    "name": name,
                    "num_params": len(schema.parameters),
                    "tags": schema.tags,
                }
                for name, schema in self.schemas.items()
            ],
        }


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_builtin_tools() -> None:
    """Register common builtin tools for RAG agents."""
    registry = get_tool_registry()

    # Retrieval tool
    def retrieve_documents(query: str, k: int = 5) -> List[str]:
        """Retrieve relevant documents for a query."""
        from autorag_live.retrievers import hybrid

        try:
            corpus = [
                "The sky is blue.",
                "The sun is bright.",
                "Machine learning is powerful.",
                "Python is a programming language.",
            ]
            results = hybrid.hybrid_retrieve(query, corpus, k=k)
            return [doc for doc, score in results]
        except Exception:
            return []

    registry.register_tool(
        "retrieve_documents",
        retrieve_documents,
        description="Retrieve relevant documents for a query",
        parameters=[
            ToolParameter("query", "string", "Search query", required=True),
            ToolParameter(
                "k",
                "integer",
                "Number of documents",
                required=False,
                default=5,
                min_value=1,
                max_value=20,
            ),
        ],
        tags=["retrieval", "rag"],
    )

    # Synthesis tool
    def synthesize_answer(query: str, documents: List[str]) -> str:
        """Synthesize an answer from documents."""
        return f"Based on {len(documents)} documents, the answer to '{query}' is synthesized from available information."

    registry.register_tool(
        "synthesize_answer",
        synthesize_answer,
        description="Synthesize an answer from retrieved documents",
        parameters=[
            ToolParameter("query", "string", "Original query", required=True),
            ToolParameter("documents", "array", "Retrieved documents", required=True),
        ],
        tags=["synthesis", "rag"],
    )

    # Query refinement tool
    def refine_query(query: str) -> str:
        """Refine a query for better retrieval."""
        # Simple refinement: expand with synonyms/related terms
        expansions = {
            "what": "definition explanation concept",
            "how": "process method steps",
            "why": "reason cause factor",
        }

        refined = query
        for key, expansion in expansions.items():
            if key in query.lower():
                refined = f"{query} {expansion}"
                break

        return refined

    registry.register_tool(
        "refine_query",
        refine_query,
        description="Refine a query for better results",
        parameters=[ToolParameter("query", "string", "Query to refine", required=True)],
        tags=["query_processing", "rag"],
    )
