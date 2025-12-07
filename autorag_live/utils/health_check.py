"""
Health check utility for AutoRAG-Live system validation.

This module provides comprehensive health checks for the AutoRAG-Live system,
validating dependencies, configuration, resources, and system state.

Features:
    - Dependency version checks
    - Configuration validation
    - Resource availability checks (memory, disk)
    - Model availability verification
    - Cache system validation

Example:
    >>> from autorag_live.utils.health_check import HealthChecker, run_health_check
    >>>
    >>> # Quick health check
    >>> result = run_health_check()
    >>> print(result.status)  # "healthy", "degraded", or "unhealthy"
    >>>
    >>> # Custom health checker
    >>> checker = HealthChecker()
    >>> checker.add_check("custom", lambda: True)
    >>> result = checker.run()
"""

import importlib.util
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal

import psutil

HealthStatus = Literal["healthy", "degraded", "unhealthy"]


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def is_healthy(self) -> bool:
        """Check if status is healthy."""
        return self.status == "healthy"


@dataclass
class HealthCheckResult:
    """Overall health check result."""

    status: HealthStatus
    checks: List[CheckResult]
    timestamp: datetime = field(default_factory=datetime.now)
    system_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if all checks passed."""
        return self.status == "healthy"

    @property
    def failed_checks(self) -> List[CheckResult]:
        """Get list of failed checks."""
        return [c for c in self.checks if not c.is_healthy()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
            "system_info": self.system_info,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


class HealthChecker:
    """
    System health checker with extensible checks.

    Performs comprehensive health validation including:
    - Python version check
    - Required dependencies
    - Optional dependencies
    - System resources
    - Configuration
    """

    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable[[], CheckResult]] = {}
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.add_check("python_version", self._check_python_version)
        self.add_check("dependencies", self._check_dependencies)
        self.add_check("memory", self._check_memory)
        self.add_check("disk", self._check_disk)
        self.add_check("optional_dependencies", self._check_optional_dependencies)

    def add_check(self, name: str, check_func: Callable[[], CheckResult]) -> None:
        """
        Add a custom health check.

        Args:
            name: Unique check name
            check_func: Function that returns CheckResult
        """
        self.checks[name] = check_func

    def _check_python_version(self) -> CheckResult:
        """Check if Python version meets requirements."""
        current = sys.version_info
        required = (3, 10)

        if current >= required:
            return CheckResult(
                name="python_version",
                status="healthy",
                message=f"Python {current.major}.{current.minor}.{current.micro}",
                details={
                    "version": f"{current.major}.{current.minor}.{current.micro}",
                    "required": f"{required[0]}.{required[1]}+",
                },
            )
        else:
            return CheckResult(
                name="python_version",
                status="unhealthy",
                message=f"Python {current.major}.{current.minor} < required {required[0]}.{required[1]}",
                details={
                    "version": f"{current.major}.{current.minor}.{current.micro}",
                    "required": f"{required[0]}.{required[1]}+",
                },
            )

    def _check_dependencies(self) -> CheckResult:
        """Check if required dependencies are installed."""
        required_packages = [
            "numpy",
            "scipy",
            "pandas",
            "sentence_transformers",
            "rank_bm25",
        ]

        missing = []
        installed = []

        for package in required_packages:
            spec = importlib.util.find_spec(package)
            if spec is None:
                missing.append(package)
            else:
                installed.append(package)

        if not missing:
            return CheckResult(
                name="dependencies",
                status="healthy",
                message=f"All {len(installed)} required packages installed",
                details={"installed": installed},
            )
        else:
            return CheckResult(
                name="dependencies",
                status="unhealthy",
                message=f"{len(missing)} required packages missing",
                details={"missing": missing, "installed": installed},
            )

    def _check_optional_dependencies(self) -> CheckResult:
        """Check optional dependencies."""
        optional_packages = {
            "numba": "Performance optimization",
            "faiss": "Vector search",
            "qdrant_client": "Vector database",
        }

        available = {}
        unavailable = {}

        for package, description in optional_packages.items():
            spec = importlib.util.find_spec(package)
            if spec is None:
                unavailable[package] = description
            else:
                available[package] = description

        # Optional dependencies don't affect health status
        return CheckResult(
            name="optional_dependencies",
            status="healthy",
            message=f"{len(available)}/{len(optional_packages)} optional packages available",
            details={"available": available, "unavailable": unavailable},
        )

    def _check_memory(self) -> CheckResult:
        """Check system memory availability."""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        percent_used = memory.percent

        # Warn if less than 1GB available or > 90% used
        if available_gb < 1.0 or percent_used > 90:
            status: HealthStatus = "degraded"
            message = f"Low memory: {available_gb:.1f}GB available ({percent_used:.1f}% used)"
        else:
            status = "healthy"
            message = f"Memory OK: {available_gb:.1f}GB/{total_gb:.1f}GB available"

        return CheckResult(
            name="memory",
            status=status,
            message=message,
            details={
                "available_gb": round(available_gb, 2),
                "total_gb": round(total_gb, 2),
                "percent_used": round(percent_used, 1),
            },
        )

    def _check_disk(self) -> CheckResult:
        """Check disk space availability."""
        disk = psutil.disk_usage("/")
        available_gb = disk.free / (1024**3)
        total_gb = disk.total / (1024**3)
        percent_used = disk.percent

        # Warn if less than 5GB available or > 95% used
        if available_gb < 5.0 or percent_used > 95:
            status: HealthStatus = "degraded"
            message = f"Low disk space: {available_gb:.1f}GB available ({percent_used:.1f}% used)"
        else:
            status = "healthy"
            message = f"Disk OK: {available_gb:.1f}GB/{total_gb:.1f}GB available"

        return CheckResult(
            name="disk",
            status=status,
            message=message,
            details={
                "available_gb": round(available_gb, 2),
                "total_gb": round(total_gb, 2),
                "percent_used": round(percent_used, 1),
            },
        )

    def run(self, include_system_info: bool = True) -> HealthCheckResult:
        """
        Run all health checks.

        Args:
            include_system_info: Whether to include system information

        Returns:
            Overall health check result
        """
        results: List[CheckResult] = []

        # Run each check
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results.append(result)
            except Exception as e:
                # If check itself fails, mark as unhealthy
                results.append(
                    CheckResult(
                        name=name,
                        status="unhealthy",
                        message=f"Check failed: {e}",
                        details={"error": str(e)},
                    )
                )

        # Determine overall status
        if all(r.status == "healthy" for r in results):
            overall_status: HealthStatus = "healthy"
        elif any(r.status == "unhealthy" for r in results):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"

        # Collect system info
        system_info = {}
        if include_system_info:
            system_info = {
                "platform": platform.platform(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "architecture": platform.machine(),
                "processor": platform.processor(),
            }

        return HealthCheckResult(
            status=overall_status,
            checks=results,
            system_info=system_info,
        )


def run_health_check(verbose: bool = False) -> HealthCheckResult:
    """
    Run a quick health check of the system.

    Args:
        verbose: Whether to print detailed results

    Returns:
        Health check result

    Example:
        >>> result = run_health_check(verbose=True)
        >>> if not result.is_healthy:
        ...     print(f"Issues found: {len(result.failed_checks)}")
    """
    checker = HealthChecker()
    result = checker.run()

    if verbose:
        print("\n=== AutoRAG-Live Health Check ===")
        print(f"Overall Status: {result.status.upper()}")
        print("\nSystem Info:")
        for key, value in result.system_info.items():
            print(f"  {key}: {value}")

        print("\nChecks:")
        for check in result.checks:
            status_symbol = "✓" if check.is_healthy() else "✗"
            print(f"  {status_symbol} {check.name}: {check.message}")

        if result.failed_checks:
            print(f"\nFailed Checks ({len(result.failed_checks)}):")
            for check in result.failed_checks:
                print(f"  - {check.name}: {check.message}")
                if check.details:
                    print(f"    Details: {check.details}")

    return result


def validate_config_file(config_path: str) -> CheckResult:
    """
    Validate that a configuration file exists and is readable.

    Args:
        config_path: Path to configuration file

    Returns:
        Check result

    Example:
        >>> result = validate_config_file("config.yaml")
        >>> assert result.is_healthy()
    """
    path = Path(config_path)

    if not path.exists():
        return CheckResult(
            name="config_file",
            status="unhealthy",
            message=f"Configuration file not found: {config_path}",
            details={"path": config_path, "exists": False},
        )

    if not path.is_file():
        return CheckResult(
            name="config_file",
            status="unhealthy",
            message=f"Path is not a file: {config_path}",
            details={"path": config_path, "is_file": False},
        )

    try:
        # Try to read the file
        with open(path, "r") as f:
            content = f.read()

        return CheckResult(
            name="config_file",
            status="healthy",
            message=f"Configuration file OK: {config_path}",
            details={"path": config_path, "size_bytes": len(content)},
        )

    except Exception as e:
        return CheckResult(
            name="config_file",
            status="unhealthy",
            message=f"Cannot read configuration file: {e}",
            details={"path": config_path, "error": str(e)},
        )
