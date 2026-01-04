"""
Agent Orchestration DSL Module.

Implements a declarative domain-specific language for defining
agent workflows using YAML/JSON configuration.

Key Features:
1. Declarative workflow definition
2. Node-based execution graph
3. Conditional branching and loops
4. Parallel execution support
5. Variable interpolation and state management

Example:
    >>> workflow = Workflow.from_yaml("workflow.yaml")
    >>> result = await workflow.execute({"query": "Hello"})
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import yaml

from autorag_live.core.protocols import BaseLLM, Document, Message

logger = logging.getLogger(__name__)


# =============================================================================
# DSL Types
# =============================================================================


class NodeType(str, Enum):
    """Available workflow node types."""

    START = "start"
    END = "end"
    LLM = "llm"  # LLM generation
    RETRIEVAL = "retrieval"  # Document retrieval
    CONDITION = "condition"  # Conditional branching
    PARALLEL = "parallel"  # Parallel execution
    LOOP = "loop"  # Iteration
    TRANSFORM = "transform"  # Data transformation
    TOOL = "tool"  # Tool execution
    SUBFLOW = "subflow"  # Nested workflow


class ExecutionStatus(str, Enum):
    """Node execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionContext:
    """
    Context for workflow execution.

    Attributes:
        variables: Current variable state
        history: Execution history
        current_node: Current node being executed
        errors: Accumulated errors
        metadata: Additional metadata
    """

    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_node: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get variable value."""
        return self.variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set variable value."""
        self.variables[key] = value

    def interpolate(self, template: str) -> str:
        """Interpolate variables in template."""
        result = template
        for key, value in self.variables.items():
            pattern = f"${{{key}}}"
            if pattern in result:
                result = result.replace(pattern, str(value))
        return result

    def add_history(
        self,
        node_id: str,
        status: ExecutionStatus,
        output: Any = None,
    ) -> None:
        """Add execution history entry."""
        self.history.append(
            {
                "node_id": node_id,
                "status": status.value,
                "output": output,
            }
        )


@dataclass
class NodeResult:
    """
    Result of node execution.

    Attributes:
        status: Execution status
        output: Node output
        next_node: Next node to execute
        error: Error message if failed
    """

    status: ExecutionStatus
    output: Any = None
    next_node: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# Node Definitions
# =============================================================================


@dataclass
class NodeConfig:
    """
    Configuration for a workflow node.

    Attributes:
        id: Unique node identifier
        type: Node type
        name: Human-readable name
        inputs: Input variable mappings
        outputs: Output variable mappings
        config: Type-specific configuration
        next: Default next node
        on_error: Error handling node
    """

    id: str
    type: NodeType
    name: str = ""
    inputs: Dict[str, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    next: Optional[str] = None
    on_error: Optional[str] = None

    @classmethod
    def from_dict(cls, node_id: str, data: Dict[str, Any]) -> NodeConfig:
        """Create from dictionary."""
        return cls(
            id=node_id,
            type=NodeType(data.get("type", "transform")),
            name=data.get("name", node_id),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            config=data.get("config", {}),
            next=data.get("next"),
            on_error=data.get("on_error"),
        )


class BaseNode(ABC):
    """Base class for workflow nodes."""

    def __init__(self, config: NodeConfig):
        """Initialize node."""
        self.config = config

    @abstractmethod
    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute node logic."""
        pass

    def resolve_inputs(self, context: ExecutionContext) -> Dict[str, Any]:
        """Resolve input variables."""
        resolved = {}
        for param, var_ref in self.config.inputs.items():
            if var_ref.startswith("${") and var_ref.endswith("}"):
                key = var_ref[2:-1]
                resolved[param] = context.get(key)
            else:
                resolved[param] = context.interpolate(var_ref)
        return resolved

    def store_outputs(
        self,
        context: ExecutionContext,
        result: Any,
    ) -> None:
        """Store outputs in context."""
        for param, var_name in self.config.outputs.items():
            if isinstance(result, dict) and param in result:
                context.set(var_name, result[param])
            elif param == "_result":
                context.set(var_name, result)


class StartNode(BaseNode):
    """Starting node for workflow."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Initialize workflow execution."""
        return NodeResult(
            status=ExecutionStatus.COMPLETED,
            next_node=self.config.next,
        )


class EndNode(BaseNode):
    """Ending node for workflow."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Finalize workflow execution."""
        output = self.resolve_inputs(context).get("result")
        return NodeResult(
            status=ExecutionStatus.COMPLETED,
            output=output,
        )


class LLMNode(BaseNode):
    """LLM generation node."""

    def __init__(
        self,
        config: NodeConfig,
        llm: Optional[BaseLLM] = None,
    ):
        """Initialize LLM node."""
        super().__init__(config)
        self.llm = llm

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute LLM generation."""
        _ = self.resolve_inputs(context)  # Resolve but use context interpolation

        # Build prompt from config
        prompt_template = self.config.config.get("prompt", "")
        prompt = context.interpolate(prompt_template)

        # Add system message if configured
        system = self.config.config.get("system", "")
        if system:
            system = context.interpolate(system)

        if not self.llm:
            return NodeResult(
                status=ExecutionStatus.FAILED,
                error="LLM not configured",
            )

        try:
            messages = []
            if system:
                messages.append(Message.system(system))
            messages.append(Message.user(prompt))

            result = await self.llm.generate(
                messages,
                temperature=self.config.config.get("temperature", 0.7),
                max_tokens=self.config.config.get("max_tokens", 1000),
            )

            output = {"response": result.content}
            self.store_outputs(context, output)

            return NodeResult(
                status=ExecutionStatus.COMPLETED,
                output=output,
                next_node=self.config.next,
            )

        except Exception as e:
            return NodeResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
                next_node=self.config.on_error,
            )


class RetrievalNode(BaseNode):
    """Document retrieval node."""

    def __init__(
        self,
        config: NodeConfig,
        retriever: Optional[Callable[[str, int], List[Document]]] = None,
    ):
        """Initialize retrieval node."""
        super().__init__(config)
        self.retriever = retriever

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute document retrieval."""
        inputs = self.resolve_inputs(context)
        query = inputs.get("query", context.get("query", ""))
        k = self.config.config.get("k", 5)

        if not self.retriever:
            return NodeResult(
                status=ExecutionStatus.FAILED,
                error="Retriever not configured",
            )

        try:
            if asyncio.iscoroutinefunction(self.retriever):
                docs = await self.retriever(query, k)
            else:
                docs = self.retriever(query, k)

            output = {
                "documents": docs,
                "context": "\n\n".join(
                    d.content if isinstance(d, Document) else str(d) for d in docs
                ),
            }
            self.store_outputs(context, output)

            return NodeResult(
                status=ExecutionStatus.COMPLETED,
                output=output,
                next_node=self.config.next,
            )

        except Exception as e:
            return NodeResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
                next_node=self.config.on_error,
            )


class ConditionNode(BaseNode):
    """Conditional branching node."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Evaluate condition and branch."""
        _ = self.resolve_inputs(context)  # Resolve inputs for side effects

        # Get condition expression
        condition = self.config.config.get("condition", "")
        condition = context.interpolate(condition)

        # Evaluate condition safely
        try:
            # Simple expression evaluation
            result = self._evaluate_condition(condition, context.variables)

            branches = self.config.config.get("branches", {})
            if result:
                next_node = branches.get("true", self.config.next)
            else:
                next_node = branches.get("false", self.config.next)

            return NodeResult(
                status=ExecutionStatus.COMPLETED,
                output={"condition_result": result},
                next_node=next_node,
            )

        except Exception as e:
            return NodeResult(
                status=ExecutionStatus.FAILED,
                error=f"Condition evaluation failed: {e}",
                next_node=self.config.on_error,
            )

    def _evaluate_condition(
        self,
        expression: str,
        variables: Dict[str, Any],
    ) -> bool:
        """Safely evaluate condition expression."""
        # Handle simple comparisons
        comparison_ops = ["==", "!=", ">=", "<=", ">", "<", " in ", " not in "]

        for op in comparison_ops:
            if op in expression:
                parts = expression.split(op, 1)
                if len(parts) == 2:
                    left = self._resolve_value(parts[0].strip(), variables)
                    right = self._resolve_value(parts[1].strip(), variables)

                    if op == "==":
                        return left == right
                    elif op == "!=":
                        return left != right
                    elif op == ">=":
                        return left >= right
                    elif op == "<=":
                        return left <= right
                    elif op == ">":
                        return left > right
                    elif op == "<":
                        return left < right
                    elif op == " in ":
                        return left in right
                    elif op == " not in ":
                        return left not in right

        # Handle boolean checks
        value = self._resolve_value(expression, variables)
        return bool(value)

    def _resolve_value(
        self,
        expr: str,
        variables: Dict[str, Any],
    ) -> Any:
        """Resolve expression value."""
        expr = expr.strip()

        # Check for variable reference
        if expr in variables:
            return variables[expr]

        # Check for quoted string
        if (expr.startswith('"') and expr.endswith('"')) or (
            expr.startswith("'") and expr.endswith("'")
        ):
            return expr[1:-1]

        # Check for number
        try:
            if "." in expr:
                return float(expr)
            return int(expr)
        except ValueError:
            pass

        # Check for boolean
        if expr.lower() == "true":
            return True
        if expr.lower() == "false":
            return False

        return expr


class ParallelNode(BaseNode):
    """Parallel execution node."""

    def __init__(
        self,
        config: NodeConfig,
        workflow: Optional[Workflow] = None,
    ):
        """Initialize parallel node."""
        super().__init__(config)
        self.workflow = workflow

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute branches in parallel."""
        branches = self.config.config.get("branches", [])

        if not branches:
            return NodeResult(
                status=ExecutionStatus.COMPLETED,
                next_node=self.config.next,
            )

        if not self.workflow:
            return NodeResult(
                status=ExecutionStatus.FAILED,
                error="Workflow reference not set",
            )

        # Execute each branch
        tasks = []
        for branch_start in branches:
            branch_context = ExecutionContext(
                variables=context.variables.copy(),
                metadata=context.metadata.copy(),
            )
            tasks.append(self.workflow._execute_from(branch_start, branch_context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results
        outputs = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                context.errors.append(f"Branch {i} failed: {result}")
            elif isinstance(result, dict):
                outputs[f"branch_{i}"] = result

        self.store_outputs(context, outputs)

        return NodeResult(
            status=ExecutionStatus.COMPLETED,
            output=outputs,
            next_node=self.config.next,
        )


class LoopNode(BaseNode):
    """Loop/iteration node."""

    def __init__(
        self,
        config: NodeConfig,
        workflow: Optional[Workflow] = None,
    ):
        """Initialize loop node."""
        super().__init__(config)
        self.workflow = workflow

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute loop iterations."""
        _ = self.resolve_inputs(context)  # Resolve inputs for side effects

        # Get loop config
        items_var = self.config.config.get("items", "")
        item_var = self.config.config.get("item_var", "item")
        index_var = self.config.config.get("index_var", "index")
        max_iterations = self.config.config.get("max_iterations", 100)
        body_node = self.config.config.get("body")

        # Get items to iterate
        items = context.get(items_var, [])
        if not isinstance(items, (list, tuple)):
            items = [items]

        # Limit iterations
        items = items[:max_iterations]

        if not self.workflow or not body_node:
            return NodeResult(
                status=ExecutionStatus.COMPLETED,
                next_node=self.config.next,
            )

        # Execute body for each item
        results = []
        for i, item in enumerate(items):
            context.set(item_var, item)
            context.set(index_var, i)

            result = await self.workflow._execute_from(body_node, context)
            results.append(result)

        output = {"loop_results": results, "iterations": len(items)}
        self.store_outputs(context, output)

        return NodeResult(
            status=ExecutionStatus.COMPLETED,
            output=output,
            next_node=self.config.next,
        )


class TransformNode(BaseNode):
    """Data transformation node."""

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute data transformation."""
        inputs = self.resolve_inputs(context)
        transform_type = self.config.config.get("transform", "passthrough")

        try:
            if transform_type == "passthrough":
                output = inputs

            elif transform_type == "concat":
                items = inputs.get("items", [])
                separator = self.config.config.get("separator", "\n")
                output = {"result": separator.join(str(i) for i in items)}

            elif transform_type == "split":
                text = inputs.get("text", "")
                separator = self.config.config.get("separator", "\n")
                output = {"items": text.split(separator)}

            elif transform_type == "format":
                template = self.config.config.get("template", "")
                output = {"result": context.interpolate(template)}

            elif transform_type == "filter":
                items = inputs.get("items", [])
                condition = self.config.config.get("filter", "")
                # Simple filter - keep non-empty items
                if condition == "non_empty":
                    output = {"items": [i for i in items if i]}
                else:
                    output = {"items": items}

            elif transform_type == "select":
                data = inputs.get("data", {})
                keys = self.config.config.get("keys", [])
                output = {k: data.get(k) for k in keys if k in data}

            else:
                output = inputs

            self.store_outputs(context, output)

            return NodeResult(
                status=ExecutionStatus.COMPLETED,
                output=output,
                next_node=self.config.next,
            )

        except Exception as e:
            return NodeResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
                next_node=self.config.on_error,
            )


class ToolNode(BaseNode):
    """Tool execution node."""

    def __init__(
        self,
        config: NodeConfig,
        tools: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize tool node."""
        super().__init__(config)
        self.tools = tools or {}

    async def execute(self, context: ExecutionContext) -> NodeResult:
        """Execute tool."""
        inputs = self.resolve_inputs(context)
        tool_name = self.config.config.get("tool", "")

        if tool_name not in self.tools:
            return NodeResult(
                status=ExecutionStatus.FAILED,
                error=f"Tool not found: {tool_name}",
                next_node=self.config.on_error,
            )

        try:
            tool_func = self.tools[tool_name]

            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**inputs)
            else:
                result = tool_func(**inputs)

            output = {"result": result}
            self.store_outputs(context, output)

            return NodeResult(
                status=ExecutionStatus.COMPLETED,
                output=output,
                next_node=self.config.next,
            )

        except Exception as e:
            return NodeResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
                next_node=self.config.on_error,
            )


# =============================================================================
# Workflow Engine
# =============================================================================


@dataclass
class WorkflowConfig:
    """
    Workflow configuration.

    Attributes:
        name: Workflow name
        description: Workflow description
        version: Workflow version
        nodes: Node configurations
        entry_point: Starting node ID
        variables: Default variables
    """

    name: str = "workflow"
    description: str = ""
    version: str = "1.0"
    nodes: Dict[str, NodeConfig] = field(default_factory=dict)
    entry_point: str = "start"
    variables: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowConfig:
        """Create from dictionary."""
        nodes = {}
        for node_id, node_data in data.get("nodes", {}).items():
            nodes[node_id] = NodeConfig.from_dict(node_id, node_data)

        return cls(
            name=data.get("name", "workflow"),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            nodes=nodes,
            entry_point=data.get("entry_point", "start"),
            variables=data.get("variables", {}),
        )


class Workflow:
    """
    Workflow execution engine.

    Executes declaratively defined workflows with support for:
    - Sequential and parallel execution
    - Conditional branching
    - Loops and iteration
    - Variable interpolation

    Example:
        >>> workflow = Workflow.from_yaml("rag_workflow.yaml")
        >>> result = await workflow.execute({"query": "What is RAG?"})
    """

    def __init__(
        self,
        config: WorkflowConfig,
        *,
        llm: Optional[BaseLLM] = None,
        retriever: Optional[Callable] = None,
        tools: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize workflow.

        Args:
            config: Workflow configuration
            llm: LLM for generation nodes
            retriever: Retriever for retrieval nodes
            tools: Tools for tool nodes
        """
        self.config = config
        self.llm = llm
        self.retriever = retriever
        self.tools = tools or {}

        # Build nodes
        self.nodes: Dict[str, BaseNode] = {}
        for node_id, node_config in config.nodes.items():
            self.nodes[node_id] = self._create_node(node_config)

    def _create_node(self, config: NodeConfig) -> BaseNode:
        """Create node instance from config."""
        if config.type == NodeType.START:
            return StartNode(config)
        elif config.type == NodeType.END:
            return EndNode(config)
        elif config.type == NodeType.LLM:
            return LLMNode(config, self.llm)
        elif config.type == NodeType.RETRIEVAL:
            return RetrievalNode(config, self.retriever)
        elif config.type == NodeType.CONDITION:
            return ConditionNode(config)
        elif config.type == NodeType.PARALLEL:
            node = ParallelNode(config)
            node.workflow = self
            return node
        elif config.type == NodeType.LOOP:
            node = LoopNode(config)
            node.workflow = self
            return node
        elif config.type == NodeType.TRANSFORM:
            return TransformNode(config)
        elif config.type == NodeType.TOOL:
            return ToolNode(config, self.tools)
        else:
            return TransformNode(config)

    @classmethod
    def from_yaml(
        cls,
        yaml_content: str,
        **kwargs: Any,
    ) -> Workflow:
        """
        Create workflow from YAML.

        Args:
            yaml_content: YAML string or file path
            **kwargs: Additional arguments (llm, retriever, tools)

        Returns:
            Workflow instance
        """
        # Try to load as file path first
        try:
            with open(yaml_content) as f:
                data = yaml.safe_load(f)
        except (FileNotFoundError, OSError):
            # Parse as YAML string
            data = yaml.safe_load(yaml_content)

        config = WorkflowConfig.from_dict(data)
        return cls(config, **kwargs)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        **kwargs: Any,
    ) -> Workflow:
        """
        Create workflow from dictionary.

        Args:
            data: Workflow definition
            **kwargs: Additional arguments

        Returns:
            Workflow instance
        """
        config = WorkflowConfig.from_dict(data)
        return cls(config, **kwargs)

    async def execute(
        self,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute workflow.

        Args:
            inputs: Input variables

        Returns:
            Workflow output
        """
        # Initialize context
        context = ExecutionContext(
            variables={
                **self.config.variables,
                **(inputs or {}),
            }
        )

        # Execute from entry point
        result = await self._execute_from(self.config.entry_point, context)

        return {
            "output": result,
            "variables": context.variables,
            "history": context.history,
            "errors": context.errors,
        }

    async def _execute_from(
        self,
        start_node: str,
        context: ExecutionContext,
    ) -> Any:
        """
        Execute workflow from a specific node.

        Args:
            start_node: Starting node ID
            context: Execution context

        Returns:
            Final output
        """
        current_node = start_node
        max_steps = 1000  # Prevent infinite loops
        steps = 0

        while current_node and steps < max_steps:
            steps += 1
            context.current_node = current_node

            node = self.nodes.get(current_node)
            if not node:
                context.errors.append(f"Node not found: {current_node}")
                break

            logger.debug(f"Executing node: {current_node}")

            try:
                result = await node.execute(context)
                context.add_history(current_node, result.status, result.output)

                if result.status == ExecutionStatus.FAILED:
                    if result.error:
                        context.errors.append(f"{current_node}: {result.error}")
                    current_node = result.next_node
                    if not current_node:
                        break
                else:
                    # Check for end condition
                    if node.config.type == NodeType.END:
                        return result.output
                    current_node = result.next_node

            except Exception as e:
                context.errors.append(f"{current_node}: {e}")
                if node.config.on_error:
                    current_node = node.config.on_error
                else:
                    break

        # Return last output or variables
        if context.history:
            return context.history[-1].get("output")
        return context.variables


# =============================================================================
# Workflow Builder
# =============================================================================


class WorkflowBuilder:
    """
    Fluent builder for creating workflows programmatically.

    Example:
        >>> builder = WorkflowBuilder("rag_pipeline")
        >>> workflow = (builder
        ...     .add_retrieval("retrieve", query="${query}", k=5)
        ...     .add_llm("generate", prompt="Answer: ${context}")
        ...     .build())
    """

    def __init__(
        self,
        name: str = "workflow",
        description: str = "",
    ):
        """Initialize builder."""
        self.name = name
        self.description = description
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.variables: Dict[str, Any] = {}
        self.entry_point = "start"
        self._last_node: Optional[str] = None

        # Add start node
        self.nodes["start"] = {"type": "start"}
        self._last_node = "start"

    def add_variable(self, name: str, default: Any = None) -> WorkflowBuilder:
        """Add workflow variable."""
        self.variables[name] = default
        return self

    def add_node(
        self,
        node_id: str,
        node_type: str,
        *,
        inputs: Optional[Dict[str, str]] = None,
        outputs: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        next_node: Optional[str] = None,
    ) -> WorkflowBuilder:
        """Add generic node."""
        self.nodes[node_id] = {
            "type": node_type,
            "inputs": inputs or {},
            "outputs": outputs or {},
            "config": config or {},
        }

        # Link from previous node
        if self._last_node and self._last_node in self.nodes:
            self.nodes[self._last_node]["next"] = node_id

        self._last_node = node_id

        if next_node:
            self.nodes[node_id]["next"] = next_node

        return self

    def add_llm(
        self,
        node_id: str,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        outputs: Optional[Dict[str, str]] = None,
    ) -> WorkflowBuilder:
        """Add LLM generation node."""
        return self.add_node(
            node_id,
            "llm",
            outputs=outputs or {"response": "response"},
            config={
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
            },
        )

    def add_retrieval(
        self,
        node_id: str,
        query: str = "${query}",
        k: int = 5,
        *,
        outputs: Optional[Dict[str, str]] = None,
    ) -> WorkflowBuilder:
        """Add retrieval node."""
        return self.add_node(
            node_id,
            "retrieval",
            inputs={"query": query},
            outputs=outputs or {"context": "context", "documents": "documents"},
            config={"k": k},
        )

    def add_condition(
        self,
        node_id: str,
        condition: str,
        true_branch: str,
        false_branch: str,
    ) -> WorkflowBuilder:
        """Add conditional branch."""
        return self.add_node(
            node_id,
            "condition",
            config={
                "condition": condition,
                "branches": {
                    "true": true_branch,
                    "false": false_branch,
                },
            },
        )

    def add_parallel(
        self,
        node_id: str,
        branches: List[str],
        *,
        outputs: Optional[Dict[str, str]] = None,
    ) -> WorkflowBuilder:
        """Add parallel execution node."""
        return self.add_node(
            node_id,
            "parallel",
            outputs=outputs or {},
            config={"branches": branches},
        )

    def add_loop(
        self,
        node_id: str,
        items: str,
        body_node: str,
        *,
        item_var: str = "item",
        max_iterations: int = 100,
    ) -> WorkflowBuilder:
        """Add loop node."""
        return self.add_node(
            node_id,
            "loop",
            config={
                "items": items,
                "item_var": item_var,
                "body": body_node,
                "max_iterations": max_iterations,
            },
        )

    def add_transform(
        self,
        node_id: str,
        transform: str,
        *,
        inputs: Optional[Dict[str, str]] = None,
        outputs: Optional[Dict[str, str]] = None,
        **config: Any,
    ) -> WorkflowBuilder:
        """Add transformation node."""
        return self.add_node(
            node_id,
            "transform",
            inputs=inputs or {},
            outputs=outputs or {},
            config={"transform": transform, **config},
        )

    def add_tool(
        self,
        node_id: str,
        tool_name: str,
        *,
        inputs: Optional[Dict[str, str]] = None,
        outputs: Optional[Dict[str, str]] = None,
    ) -> WorkflowBuilder:
        """Add tool execution node."""
        return self.add_node(
            node_id,
            "tool",
            inputs=inputs or {},
            outputs=outputs or {"result": "tool_result"},
            config={"tool": tool_name},
        )

    def add_end(
        self,
        node_id: str = "end",
        result: str = "${response}",
    ) -> WorkflowBuilder:
        """Add end node."""
        return self.add_node(
            node_id,
            "end",
            inputs={"result": result},
        )

    def build(
        self,
        *,
        llm: Optional[BaseLLM] = None,
        retriever: Optional[Callable] = None,
        tools: Optional[Dict[str, Callable]] = None,
    ) -> Workflow:
        """Build workflow instance."""
        data = {
            "name": self.name,
            "description": self.description,
            "nodes": self.nodes,
            "entry_point": self.entry_point,
            "variables": self.variables,
        }
        return Workflow.from_dict(data, llm=llm, retriever=retriever, tools=tools)

    def to_yaml(self) -> str:
        """Export workflow as YAML."""
        data = {
            "name": self.name,
            "description": self.description,
            "version": "1.0",
            "entry_point": self.entry_point,
            "variables": self.variables,
            "nodes": self.nodes,
        }
        return yaml.dump(data, default_flow_style=False, sort_keys=False)


# =============================================================================
# Pre-built Workflows
# =============================================================================


def create_simple_rag_workflow(
    llm: BaseLLM,
    retriever: Callable,
) -> Workflow:
    """
    Create simple RAG workflow.

    Query → Retrieve → Generate → Output
    """
    return (
        WorkflowBuilder("simple_rag", "Basic RAG pipeline")
        .add_variable("query", "")
        .add_retrieval("retrieve", query="${query}", k=5)
        .add_llm(
            "generate",
            prompt="""Answer the question based on the context.

Context: ${context}

Question: ${query}

Answer:""",
        )
        .add_end("end", result="${response}")
        .build(llm=llm, retriever=retriever)
    )


def create_agentic_rag_workflow(
    llm: BaseLLM,
    retriever: Callable,
) -> Workflow:
    """
    Create agentic RAG workflow with routing.

    Query → Analyze → Branch (simple/complex) → Generate → Output
    """
    builder = WorkflowBuilder("agentic_rag", "RAG with query analysis")
    builder.add_variable("query", "")

    # Analyze query
    builder.add_llm(
        "analyze",
        prompt="""Classify this query complexity:

Query: ${query}

Is this a simple factual question (simple) or requires reasoning (complex)?
Answer with one word: simple or complex""",
        outputs={"response": "complexity"},
    )

    # Branch based on complexity
    builder.add_condition(
        "route",
        condition="complexity == 'simple'",
        true_branch="retrieve_simple",
        false_branch="retrieve_deep",
    )

    # Simple retrieval path
    builder.nodes["retrieve_simple"] = {
        "type": "retrieval",
        "inputs": {"query": "${query}"},
        "outputs": {"context": "context"},
        "config": {"k": 3},
        "next": "generate",
    }

    # Deep retrieval path
    builder.nodes["retrieve_deep"] = {
        "type": "retrieval",
        "inputs": {"query": "${query}"},
        "outputs": {"context": "context"},
        "config": {"k": 10},
        "next": "generate",
    }

    # Generate response
    builder.nodes["generate"] = {
        "type": "llm",
        "outputs": {"response": "response"},
        "config": {
            "prompt": """Answer based on context.

Context: ${context}

Question: ${query}

Detailed answer:""",
            "temperature": 0.7,
        },
        "next": "end",
    }

    builder.add_end("end", result="${response}")

    return builder.build(llm=llm, retriever=retriever)
