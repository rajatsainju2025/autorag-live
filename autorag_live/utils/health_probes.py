"""Production Health Probes for AutoRAG-Live.

Kubernetes-style health probes for production deployment:
- Liveness probes (is the service running?)
- Readiness probes (is the service ready to handle requests?)
- Startup probes (has the service finished starting?)
- Dependency monitoring
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ProbeStatus(Enum):
    """Probe status values."""
    
    UP = "up"
    DOWN = "down"
    STARTING = "starting"
    UNKNOWN = "unknown"


@dataclass
class ProbeResult:
    """Result of a probe check."""
    
    name: str
    status: ProbeStatus
    message: str
    latency_ms: float
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_up(self) -> bool:
        """Check if probe is up."""
        return self.status == ProbeStatus.UP


@dataclass
class ProbeConfig:
    """Configuration for a probe."""
    
    initial_delay_seconds: int = 0
    period_seconds: int = 10
    timeout_seconds: int = 1
    success_threshold: int = 1
    failure_threshold: int = 3


class HealthProbe(ABC):
    """Abstract base class for health probes."""
    
    def __init__(
        self,
        name: str,
        config: ProbeConfig | None = None,
    ) -> None:
        """Initialize probe.
        
        Args:
            name: Probe name
            config: Probe configuration
        """
        self.name = name
        self.config = config or ProbeConfig()
        self._consecutive_successes = 0
        self._consecutive_failures = 0
        self._last_result: ProbeResult | None = None
    
    @abstractmethod
    def check(self) -> ProbeResult:
        """Perform probe check.
        
        Returns:
            Probe result
        """
        pass
    
    def execute(self) -> ProbeResult:
        """Execute probe with tracking."""
        start_time = time.perf_counter()
        
        try:
            result = self.check()
            latency_ms = (time.perf_counter() - start_time) * 1000
            result.latency_ms = latency_ms
            
            # Track consecutive results
            if result.is_up:
                self._consecutive_successes += 1
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                self._consecutive_successes = 0
            
            self._last_result = result
            return result
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            
            result = ProbeResult(
                name=self.name,
                status=ProbeStatus.DOWN,
                message=f"Probe failed: {str(e)}",
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                details={"error": str(e)},
            )
            self._last_result = result
            return result
    
    def is_passing(self) -> bool:
        """Check if probe meets success threshold."""
        return self._consecutive_successes >= self.config.success_threshold
    
    def is_failing(self) -> bool:
        """Check if probe meets failure threshold."""
        return self._consecutive_failures >= self.config.failure_threshold


class LivenessProbe(HealthProbe):
    """Basic liveness probe - service is running."""
    
    def __init__(self, config: ProbeConfig | None = None) -> None:
        """Initialize liveness probe."""
        super().__init__("liveness", config)
        self._start_time = time.time()
    
    def check(self) -> ProbeResult:
        """Check if service is alive."""
        uptime = time.time() - self._start_time
        
        return ProbeResult(
            name=self.name,
            status=ProbeStatus.UP,
            message="Service is alive",
            latency_ms=0.0,
            timestamp=datetime.now(),
            details={"uptime_seconds": uptime},
        )


class ReadinessProbe(HealthProbe):
    """Readiness probe - service is ready to handle requests."""
    
    def __init__(
        self,
        ready_fn: Callable[[], bool] | None = None,
        config: ProbeConfig | None = None,
    ) -> None:
        """Initialize readiness probe.
        
        Args:
            ready_fn: Function that returns True if ready
            config: Probe configuration
        """
        super().__init__("readiness", config)
        self._ready_fn = ready_fn or (lambda: True)
        self._ready = False
    
    def set_ready(self, ready: bool = True) -> None:
        """Set readiness state."""
        self._ready = ready
    
    def check(self) -> ProbeResult:
        """Check if service is ready."""
        is_ready = self._ready or self._ready_fn()
        
        if is_ready:
            return ProbeResult(
                name=self.name,
                status=ProbeStatus.UP,
                message="Service is ready",
                latency_ms=0.0,
                timestamp=datetime.now(),
            )
        else:
            return ProbeResult(
                name=self.name,
                status=ProbeStatus.DOWN,
                message="Service is not ready",
                latency_ms=0.0,
                timestamp=datetime.now(),
            )


class StartupProbe(HealthProbe):
    """Startup probe - service has completed initialization."""
    
    def __init__(
        self,
        startup_fn: Callable[[], bool] | None = None,
        config: ProbeConfig | None = None,
    ) -> None:
        """Initialize startup probe.
        
        Args:
            startup_fn: Function that returns True when started
            config: Probe configuration
        """
        super().__init__("startup", config)
        self._startup_fn = startup_fn
        self._started = False
    
    def mark_started(self) -> None:
        """Mark service as started."""
        self._started = True
    
    def check(self) -> ProbeResult:
        """Check if service has started."""
        is_started = self._started
        
        if self._startup_fn:
            is_started = is_started or self._startup_fn()
        
        if is_started:
            return ProbeResult(
                name=self.name,
                status=ProbeStatus.UP,
                message="Service startup complete",
                latency_ms=0.0,
                timestamp=datetime.now(),
            )
        else:
            return ProbeResult(
                name=self.name,
                status=ProbeStatus.STARTING,
                message="Service is starting",
                latency_ms=0.0,
                timestamp=datetime.now(),
            )


class HTTPProbe(HealthProbe):
    """HTTP endpoint probe."""
    
    def __init__(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        config: ProbeConfig | None = None,
    ) -> None:
        """Initialize HTTP probe.
        
        Args:
            name: Probe name
            url: URL to probe
            expected_status: Expected HTTP status code
            config: Probe configuration
        """
        super().__init__(name, config)
        self.url = url
        self.expected_status = expected_status
    
    def check(self) -> ProbeResult:
        """Probe HTTP endpoint."""
        import urllib.request
        import urllib.error
        
        try:
            request = urllib.request.Request(self.url, method="GET")
            request.add_header("User-Agent", "HealthProbe/1.0")
            
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                status_code = response.status
            
            if status_code == self.expected_status:
                return ProbeResult(
                    name=self.name,
                    status=ProbeStatus.UP,
                    message=f"HTTP {status_code}",
                    latency_ms=0.0,
                    timestamp=datetime.now(),
                    details={"url": self.url, "status_code": status_code},
                )
            else:
                return ProbeResult(
                    name=self.name,
                    status=ProbeStatus.DOWN,
                    message=f"Unexpected HTTP {status_code}",
                    latency_ms=0.0,
                    timestamp=datetime.now(),
                    details={
                        "url": self.url,
                        "status_code": status_code,
                        "expected": self.expected_status,
                    },
                )
        except Exception as e:
            return ProbeResult(
                name=self.name,
                status=ProbeStatus.DOWN,
                message=str(e),
                latency_ms=0.0,
                timestamp=datetime.now(),
                details={"url": self.url, "error": str(e)},
            )


class TCPProbe(HealthProbe):
    """TCP socket probe."""
    
    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        config: ProbeConfig | None = None,
    ) -> None:
        """Initialize TCP probe.
        
        Args:
            name: Probe name
            host: Host to connect
            port: Port to connect
            config: Probe configuration
        """
        super().__init__(name, config)
        self.host = host
        self.port = port
    
    def check(self) -> ProbeResult:
        """Probe TCP socket."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout_seconds)
            sock.connect((self.host, self.port))
            sock.close()
            
            return ProbeResult(
                name=self.name,
                status=ProbeStatus.UP,
                message=f"Connected to {self.host}:{self.port}",
                latency_ms=0.0,
                timestamp=datetime.now(),
                details={"host": self.host, "port": self.port},
            )
        except Exception as e:
            return ProbeResult(
                name=self.name,
                status=ProbeStatus.DOWN,
                message=str(e),
                latency_ms=0.0,
                timestamp=datetime.now(),
                details={"host": self.host, "port": self.port, "error": str(e)},
            )


class ExecProbe(HealthProbe):
    """Command execution probe."""
    
    def __init__(
        self,
        name: str,
        command: list[str],
        config: ProbeConfig | None = None,
    ) -> None:
        """Initialize exec probe.
        
        Args:
            name: Probe name
            command: Command to execute
            config: Probe configuration
        """
        super().__init__(name, config)
        self.command = command
    
    def check(self) -> ProbeResult:
        """Execute command and check return code."""
        import subprocess
        
        try:
            result = subprocess.run(
                self.command,
                capture_output=True,
                timeout=self.config.timeout_seconds,
                text=True,
            )
            
            if result.returncode == 0:
                return ProbeResult(
                    name=self.name,
                    status=ProbeStatus.UP,
                    message="Command succeeded",
                    latency_ms=0.0,
                    timestamp=datetime.now(),
                    details={
                        "command": self.command,
                        "return_code": result.returncode,
                        "stdout": result.stdout[:200] if result.stdout else None,
                    },
                )
            else:
                return ProbeResult(
                    name=self.name,
                    status=ProbeStatus.DOWN,
                    message=f"Command failed with code {result.returncode}",
                    latency_ms=0.0,
                    timestamp=datetime.now(),
                    details={
                        "command": self.command,
                        "return_code": result.returncode,
                        "stderr": result.stderr[:200] if result.stderr else None,
                    },
                )
        except Exception as e:
            return ProbeResult(
                name=self.name,
                status=ProbeStatus.DOWN,
                message=str(e),
                latency_ms=0.0,
                timestamp=datetime.now(),
                details={"command": self.command, "error": str(e)},
            )


class DependencyProbe(HealthProbe):
    """Probe for checking dependency availability."""
    
    def __init__(
        self,
        name: str,
        check_fn: Callable[[], tuple[bool, str]],
        config: ProbeConfig | None = None,
    ) -> None:
        """Initialize dependency probe.
        
        Args:
            name: Dependency name
            check_fn: Function returning (is_available, message)
            config: Probe configuration
        """
        super().__init__(f"dependency:{name}", config)
        self.check_fn = check_fn
    
    def check(self) -> ProbeResult:
        """Check dependency."""
        is_available, message = self.check_fn()
        
        if is_available:
            return ProbeResult(
                name=self.name,
                status=ProbeStatus.UP,
                message=message,
                latency_ms=0.0,
                timestamp=datetime.now(),
            )
        else:
            return ProbeResult(
                name=self.name,
                status=ProbeStatus.DOWN,
                message=message,
                latency_ms=0.0,
                timestamp=datetime.now(),
            )


class ProbeManager:
    """Manager for health probes."""
    
    def __init__(self) -> None:
        """Initialize probe manager."""
        self._probes: dict[str, HealthProbe] = {}
        self._liveness = LivenessProbe()
        self._readiness = ReadinessProbe()
        self._startup = StartupProbe()
        self._background_task: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        
        # Register core probes
        self._probes["liveness"] = self._liveness
        self._probes["readiness"] = self._readiness
        self._probes["startup"] = self._startup
    
    def register_probe(self, probe: HealthProbe) -> None:
        """Register a custom probe.
        
        Args:
            probe: Probe to register
        """
        with self._lock:
            self._probes[probe.name] = probe
        logger.info(f"Registered probe: {probe.name}")
    
    def unregister_probe(self, name: str) -> None:
        """Unregister a probe.
        
        Args:
            name: Probe name
        """
        with self._lock:
            self._probes.pop(name, None)
    
    def set_ready(self, ready: bool = True) -> None:
        """Set readiness state."""
        self._readiness.set_ready(ready)
    
    def mark_started(self) -> None:
        """Mark service as started."""
        self._startup.mark_started()
    
    def get_liveness(self) -> ProbeResult:
        """Get liveness status."""
        return self._liveness.execute()
    
    def get_readiness(self) -> ProbeResult:
        """Get readiness status."""
        return self._readiness.execute()
    
    def get_startup(self) -> ProbeResult:
        """Get startup status."""
        return self._startup.execute()
    
    def run_probe(self, name: str) -> ProbeResult | None:
        """Run a specific probe.
        
        Args:
            name: Probe name
            
        Returns:
            Probe result or None
        """
        with self._lock:
            probe = self._probes.get(name)
        
        if probe:
            return probe.execute()
        return None
    
    def run_all_probes(self) -> dict[str, ProbeResult]:
        """Run all registered probes.
        
        Returns:
            Dictionary of probe results
        """
        results: dict[str, ProbeResult] = {}
        
        with self._lock:
            probes = list(self._probes.items())
        
        for name, probe in probes:
            results[name] = probe.execute()
        
        return results
    
    def get_health_status(self) -> dict[str, Any]:
        """Get overall health status.
        
        Returns:
            Health status dictionary
        """
        liveness = self.get_liveness()
        readiness = self.get_readiness()
        startup = self.get_startup()
        
        # Determine overall status
        if not startup.is_up:
            overall = "starting"
        elif not liveness.is_up:
            overall = "unhealthy"
        elif not readiness.is_up:
            overall = "not_ready"
        else:
            overall = "healthy"
        
        return {
            "status": overall,
            "timestamp": datetime.now().isoformat(),
            "probes": {
                "liveness": liveness.is_up,
                "readiness": readiness.is_up,
                "startup": startup.is_up,
            },
            "details": {
                "liveness": {"message": liveness.message, "latency_ms": liveness.latency_ms},
                "readiness": {"message": readiness.message, "latency_ms": readiness.latency_ms},
                "startup": {"message": startup.message, "latency_ms": startup.latency_ms},
            },
        }
    
    def start_background_monitoring(
        self,
        interval_seconds: float = 10.0,
    ) -> None:
        """Start background probe monitoring.
        
        Args:
            interval_seconds: Check interval
        """
        if self._running:
            return
        
        self._running = True
        
        def monitor_loop() -> None:
            while self._running:
                try:
                    results = self.run_all_probes()
                    
                    # Log any failures
                    for name, result in results.items():
                        if not result.is_up:
                            logger.warning(f"Probe {name} is DOWN: {result.message}")
                    
                except Exception as e:
                    logger.error(f"Probe monitoring error: {e}")
                
                time.sleep(interval_seconds)
        
        self._background_task = threading.Thread(
            target=monitor_loop,
            daemon=True,
            name="probe-monitor",
        )
        self._background_task.start()
        logger.info("Started background probe monitoring")
    
    def stop_background_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._background_task:
            self._background_task.join(timeout=5.0)
        logger.info("Stopped background probe monitoring")


# Global probe manager
_probe_manager: ProbeManager | None = None


def get_probe_manager() -> ProbeManager:
    """Get global probe manager."""
    global _probe_manager
    if _probe_manager is None:
        _probe_manager = ProbeManager()
    return _probe_manager


# Convenience functions for HTTP endpoints
def liveness_handler() -> dict[str, Any]:
    """Liveness endpoint handler.
    
    Returns:
        Liveness response
    """
    result = get_probe_manager().get_liveness()
    return {
        "status": "ok" if result.is_up else "fail",
        "timestamp": result.timestamp.isoformat(),
    }


def readiness_handler() -> dict[str, Any]:
    """Readiness endpoint handler.
    
    Returns:
        Readiness response
    """
    result = get_probe_manager().get_readiness()
    return {
        "status": "ok" if result.is_up else "fail",
        "timestamp": result.timestamp.isoformat(),
    }


def startup_handler() -> dict[str, Any]:
    """Startup endpoint handler.
    
    Returns:
        Startup response
    """
    result = get_probe_manager().get_startup()
    return {
        "status": "ok" if result.is_up else "fail",
        "timestamp": result.timestamp.isoformat(),
    }


def health_handler() -> dict[str, Any]:
    """Full health endpoint handler.
    
    Returns:
        Health status response
    """
    return get_probe_manager().get_health_status()
