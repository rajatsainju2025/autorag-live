"""Detailed query performance profiler.

Provides breakdown of query execution time across different stages
(embedding, retrieval, reranking) with visualization support.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProfileEntry:
    """Single profiling entry."""

    stage: str
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryProfiler:
    """Profile query execution stages.

    Tracks time spent in each stage of query processing.

    Example:
        >>> profiler = QueryProfiler()
        >>> with profiler.stage("embedding"):
        ...     generate_embeddings(query)
        >>> with profiler.stage("retrieval"):
        ...     retrieve_docs(query)
        >>> report = profiler.get_report()
        >>> print(report)
    """

    def __init__(self):
        """Initialize profiler."""
        self.entries: List[ProfileEntry] = []
        self._stage_stack: List[tuple] = []

    def stage(self, name: str, **metadata):
        """Context manager for profiling a stage."""
        return _StageContext(self, name, metadata)

    def record(self, stage: str, duration: float, **metadata) -> None:
        """Record profiling entry."""
        self.entries.append(ProfileEntry(stage, duration, metadata))

    def get_report(self) -> Dict[str, Any]:
        """Get profiling report."""
        if not self.entries:
            return {"total_time": 0.0, "stages": {}}

        stage_times = defaultdict(list)
        for entry in self.entries:
            stage_times[entry.stage].append(entry.duration)

        stages = {}
        for stage, times in stage_times.items():
            stages[stage] = {
                "total": sum(times),
                "count": len(times),
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            }

        return {
            "total_time": sum(e.duration for e in self.entries),
            "stages": stages,
        }

    def reset(self) -> None:
        """Reset profiler."""
        self.entries.clear()
        self._stage_stack.clear()


class _StageContext:
    """Context manager for profiling stages."""

    def __init__(self, profiler: QueryProfiler, name: str, metadata: Dict):
        self.profiler = profiler
        self.name = name
        self.metadata = metadata
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.profiler.record(self.name, duration, **self.metadata)


__all__ = ["QueryProfiler", "ProfileEntry"]
