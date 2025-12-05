"""Convenience functions for performance monitoring in production."""

from contextlib import contextmanager
from typing import Any, Dict, Optional

from .performance import monitor_performance, performance_monitor, profile_function


def enable_monitoring() -> None:
    """Enable global performance monitoring.

    Example:
        >>> from autorag_live.utils import monitoring
        >>> monitoring.enable_monitoring()
        >>> # All monitored operations will now be tracked
    """
    performance_monitor.enable()


def disable_monitoring() -> None:
    """Disable global performance monitoring.

    Example:
        >>> monitoring.disable_monitoring()
        >>> # Monitoring overhead is eliminated
    """
    performance_monitor.disable()


def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics summary.

    Returns:
        Dictionary with performance statistics including:
        - Total operations monitored
        - Average latencies by operation
        - Peak memory usage

    Example:
        >>> stats = monitoring.get_performance_stats()
        >>> print(f"Operations: {list(stats.keys())}")
    """
    return performance_monitor.get_metrics()


def reset_performance_stats() -> None:
    """Reset all accumulated performance statistics.

    Useful for benchmarking specific code sections.

    Example:
        >>> monitoring.reset_performance_stats()
        >>> # Run code to benchmark
        >>> stats = monitoring.get_performance_stats()
    """
    performance_monitor.reset()


@contextmanager
def track_operation(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Simple context manager for tracking an operation.

    Args:
        name: Name of the operation to track
        metadata: Optional metadata to attach

    Example:
        >>> with monitoring.track_operation("database_query", {"table": "users"}):
        ...     # Perform database operation
        ...     results = db.query("SELECT * FROM users")
    """
    with monitor_performance(name, metadata):
        yield


def track_function(name: Optional[str] = None):
    """Decorator to automatically track function performance.

    Args:
        name: Optional custom name for the operation

    Example:
        >>> @monitoring.track_function("user_retrieval")
        ... def get_user(user_id: int):
        ...     return db.query(f"SELECT * FROM users WHERE id={user_id}")
    """
    return profile_function(name)


def print_performance_report() -> None:
    """Print a formatted performance report to stdout.

    Shows all tracked operations with timing and memory stats.

    Example:
        >>> monitoring.print_performance_report()
        Performance Report:
        ==================
        Operation: dense_retrieval
          Calls: 100
          Avg Duration: 0.123s
          Total Duration: 12.3s
    """
    stats = get_performance_stats()

    print("\nPerformance Report:")
    print("=" * 50)

    if stats:
        for op_name, metrics in stats.items():
            print(f"\nOperation: {op_name}")
            print(f"  Calls: {metrics.get('calls', 0)}")
            print(f"  Avg Duration: {metrics.get('avg_duration', 0):.4f}s")
            print(f"  Total Duration: {metrics.get('total_duration', 0):.4f}s")
            if "peak_memory_mb" in metrics:
                print(f"  Peak Memory: {metrics['peak_memory_mb']:.2f} MB")
    else:
        print("No performance data collected yet.")

    print("=" * 50)


__all__ = [
    "enable_monitoring",
    "disable_monitoring",
    "get_performance_stats",
    "reset_performance_stats",
    "track_operation",
    "track_function",
    "print_performance_report",
]
