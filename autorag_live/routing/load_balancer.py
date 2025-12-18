"""
Load balancing module for AutoRAG-Live.

Provides intelligent request distribution across multiple
backends with health checking and failover support.

Features:
- Multiple balancing strategies (round-robin, weighted, least-conn)
- Health checking with circuit breakers
- Automatic failover
- Rate limiting per backend
- Sticky sessions
- Request queuing
- Metrics collection

Example usage:
    >>> balancer = LoadBalancer(strategy="weighted")
    >>> balancer.add_backend("api1", weight=3)
    >>> balancer.add_backend("api2", weight=1)
    >>> 
    >>> backend = balancer.get_backend()
    >>> result = await backend.execute(request)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BalancingStrategy(Enum):
    """Load balancing strategies."""
    
    ROUND_ROBIN = auto()
    WEIGHTED_ROUND_ROBIN = auto()
    LEAST_CONNECTIONS = auto()
    RANDOM = auto()
    WEIGHTED_RANDOM = auto()
    IP_HASH = auto()
    LEAST_RESPONSE_TIME = auto()


class BackendStatus(Enum):
    """Backend health status."""
    
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    MAINTENANCE = auto()


@dataclass
class BackendStats:
    """Statistics for a backend."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Timing
    total_response_time_ms: float = 0.0
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    max_response_time_ms: float = 0.0
    
    # Current state
    active_connections: int = 0
    queued_requests: int = 0
    
    # Health
    consecutive_failures: int = 0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    
    # Rate limiting
    requests_this_window: int = 0
    window_start_time: float = 0.0


@dataclass
class BackendConfig:
    """Configuration for a backend."""
    
    name: str
    url: str
    
    # Capacity
    weight: int = 1
    max_connections: int = 100
    max_queue_size: int = 1000
    
    # Timeouts
    connect_timeout: float = 5.0
    request_timeout: float = 30.0
    
    # Health checking
    health_check_interval: float = 10.0
    health_check_path: str = "/health"
    
    # Circuit breaker
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    
    # Rate limiting
    rate_limit: int = 0  # 0 = unlimited
    rate_window: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Backend:
    """A backend service."""
    
    config: BackendConfig
    status: BackendStatus = BackendStatus.HEALTHY
    stats: BackendStats = field(default_factory=BackendStats)
    
    # Circuit breaker state
    circuit_open: bool = False
    circuit_open_time: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def is_available(self) -> bool:
        """Check if backend is available for requests."""
        if self.status in (BackendStatus.UNHEALTHY, BackendStatus.MAINTENANCE):
            return False
        
        if self.circuit_open:
            # Check if recovery timeout passed
            if time.time() - self.circuit_open_time > self.config.recovery_timeout:
                return True  # Allow probe request
            return False
        
        # Check connection limit
        if self.stats.active_connections >= self.config.max_connections:
            return False
        
        # Check rate limit
        if self.config.rate_limit > 0:
            if self._is_rate_limited():
                return False
        
        return True
    
    def _is_rate_limited(self) -> bool:
        """Check if rate limited."""
        now = time.time()
        
        # Reset window if needed
        if now - self.stats.window_start_time > self.config.rate_window:
            self.stats.requests_this_window = 0
            self.stats.window_start_time = now
        
        return self.stats.requests_this_window >= self.config.rate_limit
    
    def record_request_start(self) -> None:
        """Record request start."""
        self.stats.active_connections += 1
        self.stats.total_requests += 1
        self.stats.requests_this_window += 1
    
    def record_request_end(
        self,
        success: bool,
        response_time_ms: float,
    ) -> None:
        """Record request completion."""
        self.stats.active_connections = max(0, self.stats.active_connections - 1)
        
        if success:
            self.stats.successful_requests += 1
            self.stats.consecutive_failures = 0
            self.stats.last_success_time = time.time()
            
            # Close circuit on success
            if self.circuit_open:
                self.circuit_open = False
                logger.info(f"Circuit closed for backend {self.name}")
        else:
            self.stats.failed_requests += 1
            self.stats.consecutive_failures += 1
            self.stats.last_failure_time = time.time()
            
            # Check circuit breaker
            if self.stats.consecutive_failures >= self.config.failure_threshold:
                self.circuit_open = True
                self.circuit_open_time = time.time()
                logger.warning(f"Circuit opened for backend {self.name}")
        
        # Update response time stats
        self.stats.total_response_time_ms += response_time_ms
        self.stats.avg_response_time_ms = (
            self.stats.total_response_time_ms / self.stats.total_requests
        )
        self.stats.min_response_time_ms = min(
            self.stats.min_response_time_ms, response_time_ms
        )
        self.stats.max_response_time_ms = max(
            self.stats.max_response_time_ms, response_time_ms
        )


class BaseBalancer(ABC):
    """Base class for load balancers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass
    
    @abstractmethod
    def select(
        self,
        backends: List[Backend],
        request_key: Optional[str] = None,
    ) -> Optional[Backend]:
        """Select a backend for the request."""
        pass


class RoundRobinBalancer(BaseBalancer):
    """Round-robin load balancing."""
    
    def __init__(self):
        """Initialize balancer."""
        self._index = 0
    
    @property
    def name(self) -> str:
        return "round_robin"
    
    def select(
        self,
        backends: List[Backend],
        request_key: Optional[str] = None,
    ) -> Optional[Backend]:
        """Select next available backend."""
        available = [b for b in backends if b.is_available]
        
        if not available:
            return None
        
        backend = available[self._index % len(available)]
        self._index += 1
        
        return backend


class WeightedRoundRobinBalancer(BaseBalancer):
    """Weighted round-robin load balancing."""
    
    def __init__(self):
        """Initialize balancer."""
        self._current_weight = 0
        self._index = 0
        self._gcd = 1
        self._max_weight = 0
    
    @property
    def name(self) -> str:
        return "weighted_round_robin"
    
    def _compute_gcd(self, weights: List[int]) -> int:
        """Compute GCD of weights."""
        from math import gcd
        from functools import reduce
        return reduce(gcd, weights)
    
    def select(
        self,
        backends: List[Backend],
        request_key: Optional[str] = None,
    ) -> Optional[Backend]:
        """Select backend based on weights."""
        available = [b for b in backends if b.is_available]
        
        if not available:
            return None
        
        weights = [b.config.weight for b in available]
        self._gcd = self._compute_gcd(weights)
        self._max_weight = max(weights)
        
        while True:
            self._index = (self._index + 1) % len(available)
            
            if self._index == 0:
                self._current_weight -= self._gcd
                if self._current_weight <= 0:
                    self._current_weight = self._max_weight
            
            if available[self._index].config.weight >= self._current_weight:
                return available[self._index]


class LeastConnectionsBalancer(BaseBalancer):
    """Least connections load balancing."""
    
    @property
    def name(self) -> str:
        return "least_connections"
    
    def select(
        self,
        backends: List[Backend],
        request_key: Optional[str] = None,
    ) -> Optional[Backend]:
        """Select backend with fewest connections."""
        available = [b for b in backends if b.is_available]
        
        if not available:
            return None
        
        return min(
            available,
            key=lambda b: b.stats.active_connections
        )


class RandomBalancer(BaseBalancer):
    """Random load balancing."""
    
    @property
    def name(self) -> str:
        return "random"
    
    def select(
        self,
        backends: List[Backend],
        request_key: Optional[str] = None,
    ) -> Optional[Backend]:
        """Select random backend."""
        available = [b for b in backends if b.is_available]
        
        if not available:
            return None
        
        return random.choice(available)


class WeightedRandomBalancer(BaseBalancer):
    """Weighted random load balancing."""
    
    @property
    def name(self) -> str:
        return "weighted_random"
    
    def select(
        self,
        backends: List[Backend],
        request_key: Optional[str] = None,
    ) -> Optional[Backend]:
        """Select random backend based on weights."""
        available = [b for b in backends if b.is_available]
        
        if not available:
            return None
        
        weights = [b.config.weight for b in available]
        total_weight = sum(weights)
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for backend, weight in zip(available, weights):
            cumulative += weight
            if r <= cumulative:
                return backend
        
        return available[-1]


class IPHashBalancer(BaseBalancer):
    """IP hash load balancing for sticky sessions."""
    
    @property
    def name(self) -> str:
        return "ip_hash"
    
    def select(
        self,
        backends: List[Backend],
        request_key: Optional[str] = None,
    ) -> Optional[Backend]:
        """Select backend based on request key hash."""
        available = [b for b in backends if b.is_available]
        
        if not available:
            return None
        
        if not request_key:
            return random.choice(available)
        
        hash_val = int(hashlib.md5(request_key.encode()).hexdigest(), 16)
        return available[hash_val % len(available)]


class LeastResponseTimeBalancer(BaseBalancer):
    """Least response time load balancing."""
    
    @property
    def name(self) -> str:
        return "least_response_time"
    
    def select(
        self,
        backends: List[Backend],
        request_key: Optional[str] = None,
    ) -> Optional[Backend]:
        """Select backend with lowest average response time."""
        available = [b for b in backends if b.is_available]
        
        if not available:
            return None
        
        # Prefer backends with data
        with_data = [
            b for b in available
            if b.stats.total_requests > 0
        ]
        
        if not with_data:
            return random.choice(available)
        
        return min(
            with_data,
            key=lambda b: b.stats.avg_response_time_ms
        )


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer."""
    
    strategy: BalancingStrategy = BalancingStrategy.ROUND_ROBIN
    
    # Health checking
    health_check_enabled: bool = True
    health_check_interval: float = 10.0
    
    # Request handling
    retry_count: int = 3
    retry_delay: float = 1.0
    request_timeout: float = 30.0
    
    # Queue
    queue_enabled: bool = True
    max_queue_size: int = 1000
    queue_timeout: float = 60.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """Health checker for backends."""
    
    def __init__(
        self,
        check_interval: float = 10.0,
        timeout: float = 5.0,
    ):
        """
        Initialize health checker.
        
        Args:
            check_interval: Check interval in seconds
            timeout: Health check timeout
        """
        self.check_interval = check_interval
        self.timeout = timeout
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def check_backend(self, backend: Backend) -> bool:
        """Check backend health."""
        try:
            # Simulate health check
            # Real implementation would make HTTP request
            await asyncio.sleep(0.01)
            
            # Update status based on recent performance
            if backend.stats.consecutive_failures > 10:
                backend.status = BackendStatus.UNHEALTHY
                return False
            elif backend.stats.consecutive_failures > 5:
                backend.status = BackendStatus.DEGRADED
                return True
            else:
                backend.status = BackendStatus.HEALTHY
                return True
                
        except Exception as e:
            logger.error(f"Health check failed for {backend.name}: {e}")
            backend.status = BackendStatus.UNHEALTHY
            return False
    
    async def start(self, backends: List[Backend]) -> None:
        """Start health checking."""
        self._running = True
        
        while self._running:
            for backend in backends:
                if backend.status != BackendStatus.MAINTENANCE:
                    await self.check_backend(backend)
            
            await asyncio.sleep(self.check_interval)
    
    def stop(self) -> None:
        """Stop health checking."""
        self._running = False


class RequestQueue:
    """Queue for pending requests."""
    
    def __init__(
        self,
        max_size: int = 1000,
        timeout: float = 60.0,
    ):
        """
        Initialize queue.
        
        Args:
            max_size: Maximum queue size
            timeout: Request timeout
        """
        self.max_size = max_size
        self.timeout = timeout
        self._queue: Deque[Tuple[float, Any]] = deque()
    
    def enqueue(self, request: Any) -> bool:
        """
        Enqueue a request.
        
        Returns:
            True if enqueued, False if full
        """
        if len(self._queue) >= self.max_size:
            return False
        
        self._queue.append((time.time(), request))
        return True
    
    def dequeue(self) -> Optional[Any]:
        """
        Dequeue a request.
        
        Returns:
            Request or None if empty/expired
        """
        while self._queue:
            timestamp, request = self._queue.popleft()
            
            if time.time() - timestamp < self.timeout:
                return request
        
        return None
    
    @property
    def size(self) -> int:
        """Get queue size."""
        return len(self._queue)


class LoadBalancer:
    """
    Main load balancer interface.
    
    Example:
        >>> # Create balancer
        >>> balancer = LoadBalancer(strategy="weighted_round_robin")
        >>> 
        >>> # Add backends
        >>> balancer.add_backend("primary", url="http://api1:8000", weight=3)
        >>> balancer.add_backend("secondary", url="http://api2:8000", weight=1)
        >>> 
        >>> # Get backend for request
        >>> backend = balancer.get_backend()
        >>> if backend:
        ...     result = await make_request(backend.config.url)
        ...     balancer.record_result(backend, success=True, time_ms=100)
    """
    
    STRATEGIES = {
        'round_robin': RoundRobinBalancer,
        'weighted_round_robin': WeightedRoundRobinBalancer,
        'least_connections': LeastConnectionsBalancer,
        'random': RandomBalancer,
        'weighted_random': WeightedRandomBalancer,
        'ip_hash': IPHashBalancer,
        'least_response_time': LeastResponseTimeBalancer,
    }
    
    def __init__(
        self,
        strategy: Union[str, BalancingStrategy] = "round_robin",
        health_check_enabled: bool = True,
        retry_count: int = 3,
    ):
        """
        Initialize load balancer.
        
        Args:
            strategy: Balancing strategy
            health_check_enabled: Enable health checking
            retry_count: Number of retries
        """
        self.retry_count = retry_count
        self.health_check_enabled = health_check_enabled
        
        # Create strategy
        self._balancer = self._create_balancer(strategy)
        
        # Backends
        self._backends: Dict[str, Backend] = {}
        
        # Health checker
        self._health_checker = HealthChecker() if health_check_enabled else None
        
        # Queue
        self._queue = RequestQueue()
    
    def _create_balancer(
        self,
        strategy: Union[str, BalancingStrategy],
    ) -> BaseBalancer:
        """Create balancer instance."""
        strategy_name = (
            strategy.name.lower()
            if isinstance(strategy, BalancingStrategy)
            else strategy.lower()
        )
        
        if strategy_name not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return self.STRATEGIES[strategy_name]()
    
    def add_backend(
        self,
        name: str,
        url: str = "",
        weight: int = 1,
        max_connections: int = 100,
        **kwargs,
    ) -> Backend:
        """
        Add a backend.
        
        Args:
            name: Backend name
            url: Backend URL
            weight: Backend weight
            max_connections: Max concurrent connections
            **kwargs: Additional config
            
        Returns:
            Backend instance
        """
        config = BackendConfig(
            name=name,
            url=url,
            weight=weight,
            max_connections=max_connections,
            **kwargs,
        )
        
        backend = Backend(config=config)
        self._backends[name] = backend
        
        logger.info(f"Added backend: {name} (weight={weight})")
        
        return backend
    
    def remove_backend(self, name: str) -> bool:
        """
        Remove a backend.
        
        Args:
            name: Backend name
            
        Returns:
            True if removed
        """
        if name in self._backends:
            del self._backends[name]
            logger.info(f"Removed backend: {name}")
            return True
        return False
    
    def get_backend(
        self,
        request_key: Optional[str] = None,
    ) -> Optional[Backend]:
        """
        Get a backend for request.
        
        Args:
            request_key: Optional key for sticky sessions
            
        Returns:
            Backend or None if none available
        """
        backends = list(self._backends.values())
        
        if not backends:
            return None
        
        return self._balancer.select(backends, request_key)
    
    def record_result(
        self,
        backend: Backend,
        success: bool,
        time_ms: float,
    ) -> None:
        """
        Record request result.
        
        Args:
            backend: Backend used
            success: Whether request succeeded
            time_ms: Response time in milliseconds
        """
        backend.record_request_end(success, time_ms)
    
    def set_maintenance(
        self,
        name: str,
        enabled: bool = True,
    ) -> None:
        """Set backend maintenance mode."""
        if name in self._backends:
            self._backends[name].status = (
                BackendStatus.MAINTENANCE if enabled
                else BackendStatus.HEALTHY
            )
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all backends."""
        stats = {}
        
        for name, backend in self._backends.items():
            stats[name] = {
                'status': backend.status.name,
                'total_requests': backend.stats.total_requests,
                'successful_requests': backend.stats.successful_requests,
                'failed_requests': backend.stats.failed_requests,
                'active_connections': backend.stats.active_connections,
                'avg_response_time_ms': backend.stats.avg_response_time_ms,
                'circuit_open': backend.circuit_open,
                'weight': backend.config.weight,
            }
        
        return stats
    
    @property
    def backends(self) -> List[Backend]:
        """Get all backends."""
        return list(self._backends.values())
    
    @property
    def available_backends(self) -> List[Backend]:
        """Get available backends."""
        return [b for b in self._backends.values() if b.is_available]


# Convenience functions

def create_load_balancer(
    backends: List[Dict[str, Any]],
    strategy: str = "round_robin",
) -> LoadBalancer:
    """
    Create load balancer with backends.
    
    Args:
        backends: List of backend configs
        strategy: Balancing strategy
        
    Returns:
        LoadBalancer instance
    """
    balancer = LoadBalancer(strategy=strategy)
    
    for config in backends:
        balancer.add_backend(**config)
    
    return balancer


def round_robin_select(
    items: List[T],
    index: int,
) -> Tuple[T, int]:
    """
    Simple round-robin selection.
    
    Args:
        items: Items to select from
        index: Current index
        
    Returns:
        Tuple of (selected item, next index)
    """
    if not items:
        raise ValueError("No items to select from")
    
    selected = items[index % len(items)]
    return selected, index + 1
