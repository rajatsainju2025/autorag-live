"""
Comprehensive monitoring and distributed tracing for agents.

Tracks latency, token usage, tool execution statistics, and reasoning traces.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LatencyMetric:
    """Latency measurement."""

    operation: str
    start_time: float
    end_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.end_time == 0:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def finish(self) -> None:
        """Mark operation as finished."""
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Export metric as dictionary."""
        return {
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class TokenUsageMetric:
    """Token usage tracking."""

    component: str  # e.g., "retrieval", "reasoning", "synthesis"
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = field(init=False)

    def __post_init__(self):
        """Calculate total tokens."""
        self.tokens_total = self.tokens_input + self.tokens_output

    def to_dict(self) -> Dict[str, Any]:
        """Export metric as dictionary."""
        return {
            "component": self.component,
            "input": self.tokens_input,
            "output": self.tokens_output,
            "total": self.tokens_total,
        }


@dataclass
class ToolExecutionMetric:
    """Tool execution statistics."""

    tool_name: str
    num_calls: int = 0
    num_successes: int = 0
    num_failures: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = field(init=False)

    def __post_init__(self):
        """Calculate average latency."""
        if self.num_calls > 0:
            self.avg_latency_ms = self.total_latency_ms / self.num_calls
        else:
            self.avg_latency_ms = 0.0

    def record_execution(self, success: bool, latency_ms: float) -> None:
        """Record tool execution."""
        self.num_calls += 1
        if success:
            self.num_successes += 1
        else:
            self.num_failures += 1
        self.total_latency_ms += latency_ms

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.num_calls == 0:
            return 0.0
        return self.num_successes / self.num_calls

    def to_dict(self) -> Dict[str, Any]:
        """Export metric as dictionary."""
        return {
            "tool": self.tool_name,
            "calls": self.num_calls,
            "successes": self.num_successes,
            "failures": self.num_failures,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
        }


class PerformanceMonitor:
    """Monitor performance metrics for agent execution."""

    def __init__(self):
        """Initialize performance monitor."""
        self.latency_metrics: List[LatencyMetric] = []
        self.token_metrics: Dict[str, TokenUsageMetric] = {}
        self.tool_metrics: Dict[str, ToolExecutionMetric] = {}
        self.logger = logging.getLogger("PerformanceMonitor")

    def start_operation(
        self, operation: str, metadata: Optional[Dict[str, Any]] = None
    ) -> LatencyMetric:
        """
        Start timing an operation.

        Args:
            operation: Operation name
            metadata: Optional metadata

        Returns:
            Latency metric object
        """
        metric = LatencyMetric(
            operation=operation,
            start_time=time.time(),
            metadata=metadata or {},
        )
        self.latency_metrics.append(metric)
        return metric

    def finish_operation(self, metric: LatencyMetric) -> None:
        """
        Finish timing an operation.

        Args:
            metric: Latency metric object
        """
        metric.finish()
        self.logger.debug(f"{metric.operation}: {metric.duration_ms:.2f}ms")

    def track_tokens(self, component: str, tokens_input: int = 0, tokens_output: int = 0) -> None:
        """
        Track token usage.

        Args:
            component: Component name
            tokens_input: Input tokens
            tokens_output: Output tokens
        """
        if component not in self.token_metrics:
            self.token_metrics[component] = TokenUsageMetric(component=component)

        metric = self.token_metrics[component]
        metric.tokens_input += tokens_input
        metric.tokens_output += tokens_output

    def record_tool_execution(self, tool_name: str, success: bool, latency_ms: float) -> None:
        """
        Record tool execution.

        Args:
            tool_name: Tool name
            success: Whether execution was successful
            latency_ms: Execution time in milliseconds
        """
        if tool_name not in self.tool_metrics:
            self.tool_metrics[tool_name] = ToolExecutionMetric(tool_name=tool_name)

        metric = self.tool_metrics[tool_name]
        metric.record_execution(success, latency_ms)

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_latency = sum(m.duration_ms for m in self.latency_metrics)
        avg_latency = total_latency / len(self.latency_metrics) if self.latency_metrics else 0

        total_tokens = sum(m.tokens_total for m in self.token_metrics.values())

        return {
            "total_operations": len(self.latency_metrics),
            "total_latency_ms": total_latency,
            "average_latency_ms": avg_latency,
            "total_tokens": total_tokens,
            "num_tools": len(self.tool_metrics),
            "tools": {name: metric.to_dict() for name, metric in self.tool_metrics.items()},
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics as dictionary."""
        return {
            "latency_metrics": [m.to_dict() for m in self.latency_metrics],
            "token_metrics": {k: v.to_dict() for k, v in self.token_metrics.items()},
            "tool_metrics": {k: v.to_dict() for k, v in self.tool_metrics.items()},
            "summary": self.get_summary(),
        }


@dataclass
class TraceSpan:
    """Single span in distributed trace."""

    span_id: str
    operation_name: str
    start_time: float
    end_time: float = 0.0
    parent_span_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Get span duration."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def add_tag(self, key: str, value: Any) -> None:
        """Add tag to span."""
        self.tags[key] = value

    def log_event(self, event: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log event in span."""
        self.logs.append({"timestamp": time.time(), "event": event, "metadata": metadata or {}})

    def finish(self) -> None:
        """Mark span as finished."""
        self.end_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Export span as dictionary."""
        return {
            "span_id": self.span_id,
            "operation": self.operation_name,
            "duration_ms": self.duration_ms,
            "parent_span_id": self.parent_span_id,
            "tags": self.tags,
            "logs": self.logs,
        }


class DistributedTracer:
    """Distributed tracing for agent execution."""

    def __init__(self, trace_id: str = ""):
        """
        Initialize distributed tracer.

        Args:
            trace_id: Trace identifier
        """
        self.trace_id = trace_id or self._generate_trace_id()
        self.spans: Dict[str, TraceSpan] = {}
        self.current_span_stack: List[str] = []
        self.logger = logging.getLogger("DistributedTracer")

    @staticmethod
    def _generate_trace_id() -> str:
        """Generate unique trace ID."""
        import uuid

        return str(uuid.uuid4())[:8]

    def start_span(
        self,
        operation_name: str,
        tags: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        """
        Start new trace span.

        Args:
            operation_name: Operation name
            tags: Optional initial tags

        Returns:
            Trace span
        """
        import uuid

        span_id = str(uuid.uuid4())[:8]
        parent_span_id = self.current_span_stack[-1] if self.current_span_stack else None

        span = TraceSpan(
            span_id=span_id,
            operation_name=operation_name,
            start_time=time.time(),
            parent_span_id=parent_span_id,
            tags=tags or {},
        )

        self.spans[span_id] = span
        self.current_span_stack.append(span_id)

        self.logger.debug(f"Started span: {operation_name} ({span_id})")

        return span

    def finish_span(self, span: TraceSpan) -> None:
        """
        Finish trace span.

        Args:
            span: Trace span to finish
        """
        span.finish()
        if self.current_span_stack and self.current_span_stack[-1] == span.span_id:
            self.current_span_stack.pop()

        self.logger.debug(
            f"Finished span: {span.operation_name} ({span.span_id}) - {span.duration_ms:.2f}ms"
        )

    def get_trace(self) -> Dict[str, Any]:
        """Get complete trace."""
        # Build trace tree
        root_spans = [s for s in self.spans.values() if s.parent_span_id is None]

        def build_tree(span: TraceSpan) -> Dict[str, Any]:
            children = [
                build_tree(self.spans[child_id])
                for child_id, child in self.spans.items()
                if child.parent_span_id == span.span_id
            ]

            return {
                "span": span.to_dict(),
                "children": children,
            }

        return {
            "trace_id": self.trace_id,
            "spans": [build_tree(s) for s in root_spans],
        }

    def print_trace(self) -> str:
        """Print formatted trace."""
        lines = [f"Trace ID: {self.trace_id}"]

        def print_span(span: TraceSpan, indent: int = 0) -> None:
            prefix = "  " * indent
            lines.append(f"{prefix}├─ {span.operation_name} ({span.duration_ms:.1f}ms)")

            for child_id, child in self.spans.items():
                if child.parent_span_id == span.span_id:
                    print_span(child, indent + 1)

        for span in self.spans.values():
            if span.parent_span_id is None:
                print_span(span)

        return "\n".join(lines)


class AgentMetricsCollector:
    """Collects all metrics for agent execution."""

    def __init__(self):
        """Initialize metrics collector."""
        self.performance_monitor = PerformanceMonitor()
        self.tracer = DistributedTracer()
        self.logger = logging.getLogger("AgentMetricsCollector")

    def get_metrics_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report."""
        return {
            "trace_id": self.tracer.trace_id,
            "performance": self.performance_monitor.get_summary(),
            "trace_tree": self.tracer.get_trace(),
        }

    def print_report(self) -> str:
        """Print human-readable metrics report."""
        lines = []

        # Performance summary
        summary = self.performance_monitor.get_summary()
        lines.append("=" * 50)
        lines.append("PERFORMANCE METRICS")
        lines.append("=" * 50)
        lines.append(f"Total Operations: {summary['total_operations']}")
        lines.append(f"Total Latency: {summary['total_latency_ms']:.1f}ms")
        lines.append(f"Average Latency: {summary['average_latency_ms']:.1f}ms")
        lines.append(f"Total Tokens: {summary['total_tokens']}")
        lines.append("")

        # Tool statistics
        if summary["tools"]:
            lines.append("TOOL STATISTICS")
            lines.append("-" * 50)
            for tool_name, tool_metrics in summary["tools"].items():
                lines.append(
                    f"{tool_name}: {tool_metrics['calls']} calls, "
                    f"{tool_metrics['success_rate']:.0%} success rate, "
                    f"{tool_metrics['avg_latency_ms']:.1f}ms avg"
                )
            lines.append("")

        # Trace
        lines.append("EXECUTION TRACE")
        lines.append("-" * 50)
        lines.append(self.tracer.print_trace())

        return "\n".join(lines)
