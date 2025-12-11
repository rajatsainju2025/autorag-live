"""Pipeline Profiler for AutoRAG-Live.

Profile and optimize pipeline performance:
- Execution timing
- Memory tracking
- Bottleneck detection
- Performance reports
"""

from __future__ import annotations

import functools
import gc
import logging
import statistics
import threading
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generator, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class ProfileLevel(Enum):
    """Profiling detail levels."""
    
    NONE = "none"
    BASIC = "basic"  # Only timing
    STANDARD = "standard"  # Timing + call counts
    DETAILED = "detailed"  # All metrics including memory


@dataclass
class TimingStats:
    """Statistics for timing data."""
    
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    last_ms: float = 0.0
    samples: list[float] = field(default_factory=list)
    
    def add_sample(self, duration_ms: float, keep_samples: int = 1000) -> None:
        """Add a timing sample.
        
        Args:
            duration_ms: Duration in milliseconds
            keep_samples: Maximum samples to keep
        """
        self.count += 1
        self.total_ms += duration_ms
        self.last_ms = duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)
        
        self.samples.append(duration_ms)
        if len(self.samples) > keep_samples:
            self.samples = self.samples[-keep_samples:]
        
        self._update_statistics()
    
    def _update_statistics(self) -> None:
        """Update derived statistics."""
        if not self.samples:
            return
        
        self.mean_ms = statistics.mean(self.samples)
        if len(self.samples) > 1:
            self.std_ms = statistics.stdev(self.samples)
        
        sorted_samples = sorted(self.samples)
        n = len(sorted_samples)
        self.p50_ms = sorted_samples[n // 2]
        self.p95_ms = sorted_samples[int(n * 0.95)]
        self.p99_ms = sorted_samples[int(n * 0.99)]


@dataclass
class SpanInfo:
    """Information about a profiling span."""
    
    name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    parent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    children: list[str] = field(default_factory=list)


@dataclass
class ProfileSnapshot:
    """Snapshot of profiling data."""
    
    timestamp: datetime
    timings: dict[str, TimingStats]
    active_spans: list[str]
    total_operations: int
    total_errors: int
    memory_usage_mb: float | None = None


class Span:
    """Context manager for timing a code span."""
    
    def __init__(
        self,
        profiler: "PipelineProfiler",
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize span.
        
        Args:
            profiler: Parent profiler
            name: Span name
            metadata: Optional metadata
        """
        self.profiler = profiler
        self.name = name
        self.metadata = metadata or {}
        self.start_time = 0.0
        self.end_time: float | None = None
        self.error: str | None = None
    
    def __enter__(self) -> "Span":
        """Start the span."""
        self.start_time = time.perf_counter()
        self.profiler._start_span(self.name, self.metadata)
        return self
    
    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> None:
        """End the span."""
        self.end_time = time.perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        if exc_val:
            self.error = str(exc_val)
        
        self.profiler._end_span(self.name, duration_ms, self.error)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to span."""
        self.metadata[key] = value


class PipelineProfiler:
    """Main profiler for pipeline operations."""
    
    _instance: "PipelineProfiler | None" = None
    _lock = threading.Lock()
    
    def __init__(
        self,
        level: ProfileLevel = ProfileLevel.STANDARD,
        max_samples: int = 1000,
    ) -> None:
        """Initialize profiler.
        
        Args:
            level: Profiling detail level
            max_samples: Maximum samples per metric
        """
        self.level = level
        self.max_samples = max_samples
        
        self._timings: dict[str, TimingStats] = defaultdict(TimingStats)
        self._active_spans: dict[str, SpanInfo] = {}
        self._span_stack: list[str] = []
        self._error_counts: dict[str, int] = defaultdict(int)
        self._total_operations = 0
        self._total_errors = 0
        self._start_time = time.time()
        self._lock = threading.Lock()
        
        # Memory tracking
        self._track_memory = level == ProfileLevel.DETAILED
        self._memory_samples: list[tuple[float, float]] = []
    
    @classmethod
    def get_instance(cls) -> "PipelineProfiler":
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance."""
        with cls._lock:
            cls._instance = None
    
    def span(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        """Create a profiling span.
        
        Args:
            name: Span name
            metadata: Optional metadata
            
        Returns:
            Span context manager
        """
        return Span(self, name, metadata)
    
    def _start_span(self, name: str, metadata: dict[str, Any]) -> None:
        """Start a profiling span."""
        if self.level == ProfileLevel.NONE:
            return
        
        with self._lock:
            parent = self._span_stack[-1] if self._span_stack else None
            
            span_info = SpanInfo(
                name=name,
                start_time=time.perf_counter(),
                parent=parent,
                metadata=metadata,
            )
            
            self._active_spans[name] = span_info
            self._span_stack.append(name)
            
            if parent and parent in self._active_spans:
                self._active_spans[parent].children.append(name)
    
    def _end_span(
        self,
        name: str,
        duration_ms: float,
        error: str | None,
    ) -> None:
        """End a profiling span."""
        if self.level == ProfileLevel.NONE:
            return
        
        with self._lock:
            self._timings[name].add_sample(duration_ms, self.max_samples)
            self._total_operations += 1
            
            if error:
                self._error_counts[name] += 1
                self._total_errors += 1
            
            if name in self._active_spans:
                self._active_spans[name].end_time = time.perf_counter()
                self._active_spans[name].duration_ms = duration_ms
                self._active_spans[name].error = error
                del self._active_spans[name]
            
            if self._span_stack and self._span_stack[-1] == name:
                self._span_stack.pop()
    
    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        """Context manager for measuring timing.
        
        Args:
            name: Metric name
            
        Yields:
            None
        """
        if self.level == ProfileLevel.NONE:
            yield
            return
        
        start = time.perf_counter()
        error: str | None = None
        
        try:
            yield
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            with self._lock:
                self._timings[name].add_sample(duration_ms, self.max_samples)
                self._total_operations += 1
                if error:
                    self._error_counts[name] += 1
                    self._total_errors += 1
    
    def record(self, name: str, duration_ms: float) -> None:
        """Record a timing measurement directly.
        
        Args:
            name: Metric name
            duration_ms: Duration in milliseconds
        """
        if self.level == ProfileLevel.NONE:
            return
        
        with self._lock:
            self._timings[name].add_sample(duration_ms, self.max_samples)
            self._total_operations += 1
    
    def get_timing(self, name: str) -> TimingStats | None:
        """Get timing statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Timing statistics or None
        """
        with self._lock:
            if name in self._timings:
                return self._timings[name]
        return None
    
    def get_all_timings(self) -> dict[str, TimingStats]:
        """Get all timing statistics."""
        with self._lock:
            return dict(self._timings)
    
    def get_snapshot(self) -> ProfileSnapshot:
        """Get current profiling snapshot."""
        with self._lock:
            memory_mb: float | None = None
            if self._track_memory:
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                except ImportError:
                    pass
            
            return ProfileSnapshot(
                timestamp=datetime.now(),
                timings=dict(self._timings),
                active_spans=list(self._active_spans.keys()),
                total_operations=self._total_operations,
                total_errors=self._total_errors,
                memory_usage_mb=memory_mb,
            )
    
    def reset(self) -> None:
        """Reset all profiling data."""
        with self._lock:
            self._timings.clear()
            self._active_spans.clear()
            self._span_stack.clear()
            self._error_counts.clear()
            self._total_operations = 0
            self._total_errors = 0
            self._start_time = time.time()
            self._memory_samples.clear()
    
    def get_report(self, top_n: int = 20) -> str:
        """Generate profiling report.
        
        Args:
            top_n: Number of top metrics to show
            
        Returns:
            Formatted report
        """
        snapshot = self.get_snapshot()
        
        lines = [
            "# Pipeline Performance Report",
            f"\nGenerated: {snapshot.timestamp.isoformat()}",
            f"Total Operations: {snapshot.total_operations:,}",
            f"Total Errors: {snapshot.total_errors:,}",
            f"Uptime: {(time.time() - self._start_time):.1f}s",
        ]
        
        if snapshot.memory_usage_mb:
            lines.append(f"Memory Usage: {snapshot.memory_usage_mb:.1f} MB")
        
        # Sort by total time
        sorted_timings = sorted(
            snapshot.timings.items(),
            key=lambda x: x[1].total_ms,
            reverse=True,
        )[:top_n]
        
        if sorted_timings:
            lines.extend([
                "\n## Top Operations by Total Time",
                "",
                "| Operation | Count | Total (ms) | Mean (ms) | P95 (ms) | P99 (ms) |",
                "|-----------|-------|------------|-----------|----------|----------|",
            ])
            
            for name, stats in sorted_timings:
                lines.append(
                    f"| {name[:30]} | {stats.count:,} | {stats.total_ms:,.1f} | "
                    f"{stats.mean_ms:.2f} | {stats.p95_ms:.2f} | {stats.p99_ms:.2f} |"
                )
        
        # Error summary
        if self._error_counts:
            lines.extend([
                "\n## Errors by Operation",
                "",
            ])
            for name, count in sorted(
                self._error_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]:
                lines.append(f"- {name}: {count} errors")
        
        return "\n".join(lines)
    
    def identify_bottlenecks(
        self,
        threshold_ms: float = 100,
    ) -> list[tuple[str, TimingStats]]:
        """Identify performance bottlenecks.
        
        Args:
            threshold_ms: P95 threshold for bottleneck detection
            
        Returns:
            List of (name, stats) for bottlenecks
        """
        bottlenecks: list[tuple[str, TimingStats]] = []
        
        with self._lock:
            for name, stats in self._timings.items():
                if stats.p95_ms > threshold_ms:
                    bottlenecks.append((name, stats))
        
        return sorted(bottlenecks, key=lambda x: x[1].p95_ms, reverse=True)


def profile(
    name: str | None = None,
    level: ProfileLevel = ProfileLevel.STANDARD,
) -> Callable[[F], F]:
    """Decorator to profile a function.
    
    Args:
        name: Custom metric name (uses function name if None)
        level: Minimum profile level to enable
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        metric_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler = PipelineProfiler.get_instance()
            
            if profiler.level.value < level.value:
                return func(*args, **kwargs)
            
            with profiler.span(metric_name):
                return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


def async_profile(
    name: str | None = None,
    level: ProfileLevel = ProfileLevel.STANDARD,
) -> Callable[[F], F]:
    """Decorator to profile an async function.
    
    Args:
        name: Custom metric name
        level: Minimum profile level
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        metric_name = name or func.__name__
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler = PipelineProfiler.get_instance()
            
            if profiler.level.value < level.value:
                return await func(*args, **kwargs)
            
            start = time.perf_counter()
            error: str | None = None
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                profiler.record(metric_name, duration_ms)
                if error:
                    profiler._error_counts[metric_name] += 1
                    profiler._total_errors += 1
        
        return wrapper  # type: ignore
    
    return decorator


class ScopedProfiler:
    """Profiler for specific scope with isolation."""
    
    def __init__(self, scope_name: str) -> None:
        """Initialize scoped profiler.
        
        Args:
            scope_name: Name prefix for this scope
        """
        self.scope_name = scope_name
        self._profiler = PipelineProfiler.get_instance()
    
    def span(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Span:
        """Create a scoped span.
        
        Args:
            name: Span name (will be prefixed with scope)
            metadata: Optional metadata
            
        Returns:
            Span context manager
        """
        full_name = f"{self.scope_name}.{name}"
        return self._profiler.span(full_name, metadata)
    
    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        """Scoped measurement context.
        
        Args:
            name: Metric name (will be prefixed)
            
        Yields:
            None
        """
        full_name = f"{self.scope_name}.{name}"
        with self._profiler.measure(full_name):
            yield
    
    def get_scope_timings(self) -> dict[str, TimingStats]:
        """Get timings for this scope only."""
        all_timings = self._profiler.get_all_timings()
        prefix = f"{self.scope_name}."
        return {
            k[len(prefix):]: v
            for k, v in all_timings.items()
            if k.startswith(prefix)
        }


class TraceContext:
    """Context for distributed tracing."""
    
    def __init__(
        self,
        trace_id: str | None = None,
        span_id: str | None = None,
    ) -> None:
        """Initialize trace context.
        
        Args:
            trace_id: Trace ID (generated if None)
            span_id: Span ID (generated if None)
        """
        import uuid
        self.trace_id = trace_id or str(uuid.uuid4())
        self.span_id = span_id or str(uuid.uuid4())[:8]
        self.spans: list[dict[str, Any]] = []
        self._start_time = time.perf_counter()
    
    def add_span(
        self,
        name: str,
        duration_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a span to the trace.
        
        Args:
            name: Span name
            duration_ms: Duration in milliseconds
            metadata: Optional metadata
        """
        self.spans.append({
            "name": name,
            "duration_ms": duration_ms,
            "timestamp": time.perf_counter() - self._start_time,
            "metadata": metadata or {},
        })
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "total_duration_ms": (time.perf_counter() - self._start_time) * 1000,
            "spans": self.spans,
        }


# Global profiler access
def get_profiler() -> PipelineProfiler:
    """Get global profiler instance."""
    return PipelineProfiler.get_instance()


def configure_profiler(
    level: ProfileLevel = ProfileLevel.STANDARD,
    max_samples: int = 1000,
) -> PipelineProfiler:
    """Configure and return profiler.
    
    Args:
        level: Profiling level
        max_samples: Maximum samples per metric
        
    Returns:
        Configured profiler
    """
    PipelineProfiler.reset_instance()
    profiler = PipelineProfiler(level=level, max_samples=max_samples)
    PipelineProfiler._instance = profiler
    return profiler
