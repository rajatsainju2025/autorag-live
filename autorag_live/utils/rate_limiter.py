"""
Rate limiting utilities for AutoRAG-Live.

Provides multiple rate limiting algorithms for controlling
API call rates and preventing quota exhaustion.

Algorithms:
- Token Bucket: Smooth burst handling
- Sliding Window: Precise rate limiting
- Fixed Window: Simple time-based limiting
- Adaptive: Dynamic rate adjustment

Example usage:
    >>> limiter = TokenBucketLimiter(rate=100, capacity=200)
    >>> if limiter.acquire():
    ...     make_api_call()
    
    >>> # With async support
    >>> async with limiter:
    ...     await make_api_call()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after


class LimiterType(str, Enum):
    """Rate limiter types."""
    
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiters."""
    
    # Requests per time window
    requests_per_second: float = 10.0
    requests_per_minute: float = 600.0
    
    # Token bucket settings
    bucket_capacity: int = 100
    refill_rate: float = 10.0  # tokens per second
    
    # Sliding window settings
    window_size: float = 60.0  # seconds
    
    # Retry settings
    max_wait_time: float = 30.0
    blocking: bool = True
    
    # Adaptive settings
    target_utilization: float = 0.8
    adjustment_interval: float = 10.0


@dataclass
class RateLimitStats:
    """Statistics for rate limiter."""
    
    total_requests: int = 0
    allowed_requests: int = 0
    denied_requests: int = 0
    total_wait_time: float = 0.0
    
    @property
    def denial_rate(self) -> float:
        """Calculate denial rate."""
        if self.total_requests == 0:
            return 0.0
        return self.denied_requests / self.total_requests
    
    @property
    def avg_wait_time(self) -> float:
        """Calculate average wait time."""
        if self.allowed_requests == 0:
            return 0.0
        return self.total_wait_time / self.allowed_requests


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""
    
    @abstractmethod
    def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if acquired, False otherwise
        """
        pass
    
    @abstractmethod
    def wait_time(self, tokens: int = 1) -> float:
        """
        Get wait time until tokens available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Wait time in seconds
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the limiter state."""
        pass
    
    @property
    @abstractmethod
    def stats(self) -> RateLimitStats:
        """Get limiter statistics."""
        pass
    
    def acquire_or_wait(
        self,
        tokens: int = 1,
        max_wait: Optional[float] = None,
    ) -> bool:
        """
        Acquire tokens, waiting if necessary.
        
        Args:
            tokens: Number of tokens
            max_wait: Maximum wait time
            
        Returns:
            True if acquired, False if timeout
        """
        wait = self.wait_time(tokens)
        
        if wait <= 0:
            return self.acquire(tokens)
        
        if max_wait is not None and wait > max_wait:
            return False
        
        time.sleep(wait)
        return self.acquire(tokens)
    
    async def acquire_async(
        self,
        tokens: int = 1,
        max_wait: Optional[float] = None,
    ) -> bool:
        """
        Acquire tokens asynchronously.
        
        Args:
            tokens: Number of tokens
            max_wait: Maximum wait time
            
        Returns:
            True if acquired, False if timeout
        """
        wait = self.wait_time(tokens)
        
        if wait <= 0:
            return self.acquire(tokens)
        
        if max_wait is not None and wait > max_wait:
            return False
        
        await asyncio.sleep(wait)
        return self.acquire(tokens)
    
    def __enter__(self) -> "RateLimiter":
        """Context manager entry."""
        if not self.acquire_or_wait():
            raise RateLimitExceeded(retry_after=self.wait_time())
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        pass
    
    async def __aenter__(self) -> "RateLimiter":
        """Async context manager entry."""
        if not await self.acquire_async():
            raise RateLimitExceeded(retry_after=self.wait_time())
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        pass


class TokenBucketLimiter(RateLimiter):
    """
    Token bucket rate limiter.
    
    Allows bursts up to bucket capacity while maintaining
    a steady average rate.
    
    Example:
        >>> limiter = TokenBucketLimiter(rate=10, capacity=20)
        >>> # Can burst up to 20 requests
        >>> for _ in range(20):
        ...     assert limiter.acquire()
        >>> # 21st request needs to wait
        >>> assert not limiter.acquire()
    """
    
    def __init__(
        self,
        rate: float = 10.0,
        capacity: Optional[int] = None,
    ):
        """
        Initialize token bucket limiter.
        
        Args:
            rate: Token refill rate (tokens per second)
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity or int(rate * 2)
        
        self._tokens = float(self.capacity)
        self._last_update = time.time()
        self._lock = threading.Lock()
        self._stats = RateLimitStats()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        
        self._tokens = min(
            self.capacity,
            self._tokens + elapsed * self.rate,
        )
        self._last_update = now
    
    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens."""
        with self._lock:
            self._refill()
            self._stats.total_requests += 1
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats.allowed_requests += 1
                return True
            
            self._stats.denied_requests += 1
            return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """Get wait time until tokens available."""
        with self._lock:
            self._refill()
            
            if self._tokens >= tokens:
                return 0.0
            
            needed = tokens - self._tokens
            return needed / self.rate
    
    def reset(self) -> None:
        """Reset the limiter."""
        with self._lock:
            self._tokens = float(self.capacity)
            self._last_update = time.time()
    
    @property
    def stats(self) -> RateLimitStats:
        """Get statistics."""
        return self._stats
    
    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self._lock:
            self._refill()
            return self._tokens


class SlidingWindowLimiter(RateLimiter):
    """
    Sliding window rate limiter.
    
    Tracks request timestamps in a sliding time window
    for precise rate limiting.
    
    Example:
        >>> limiter = SlidingWindowLimiter(max_requests=100, window_seconds=60)
        >>> # Allow up to 100 requests per minute
    """
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
    ):
        """
        Initialize sliding window limiter.
        
        Args:
            max_requests: Maximum requests in window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        self._timestamps: deque = deque()
        self._lock = threading.Lock()
        self._stats = RateLimitStats()
    
    def _cleanup(self) -> None:
        """Remove expired timestamps."""
        cutoff = time.time() - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens."""
        with self._lock:
            self._cleanup()
            self._stats.total_requests += 1
            
            if len(self._timestamps) + tokens <= self.max_requests:
                now = time.time()
                for _ in range(tokens):
                    self._timestamps.append(now)
                self._stats.allowed_requests += 1
                return True
            
            self._stats.denied_requests += 1
            return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """Get wait time until tokens available."""
        with self._lock:
            self._cleanup()
            
            if len(self._timestamps) + tokens <= self.max_requests:
                return 0.0
            
            # Wait until oldest request expires
            if self._timestamps:
                oldest = self._timestamps[0]
                return max(0.0, oldest + self.window_seconds - time.time())
            
            return 0.0
    
    def reset(self) -> None:
        """Reset the limiter."""
        with self._lock:
            self._timestamps.clear()
    
    @property
    def stats(self) -> RateLimitStats:
        """Get statistics."""
        return self._stats
    
    @property
    def current_count(self) -> int:
        """Get current request count in window."""
        with self._lock:
            self._cleanup()
            return len(self._timestamps)


class FixedWindowLimiter(RateLimiter):
    """
    Fixed window rate limiter.
    
    Simple and efficient, but can allow bursts at window boundaries.
    
    Example:
        >>> limiter = FixedWindowLimiter(max_requests=100, window_seconds=60)
        >>> # Reset count every minute
    """
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
    ):
        """
        Initialize fixed window limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        self._count = 0
        self._window_start = time.time()
        self._lock = threading.Lock()
        self._stats = RateLimitStats()
    
    def _check_window(self) -> None:
        """Check and reset window if needed."""
        now = time.time()
        if now - self._window_start >= self.window_seconds:
            self._count = 0
            self._window_start = now
    
    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens."""
        with self._lock:
            self._check_window()
            self._stats.total_requests += 1
            
            if self._count + tokens <= self.max_requests:
                self._count += tokens
                self._stats.allowed_requests += 1
                return True
            
            self._stats.denied_requests += 1
            return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """Get wait time until tokens available."""
        with self._lock:
            self._check_window()
            
            if self._count + tokens <= self.max_requests:
                return 0.0
            
            # Wait until window resets
            elapsed = time.time() - self._window_start
            return max(0.0, self.window_seconds - elapsed)
    
    def reset(self) -> None:
        """Reset the limiter."""
        with self._lock:
            self._count = 0
            self._window_start = time.time()
    
    @property
    def stats(self) -> RateLimitStats:
        """Get statistics."""
        return self._stats
    
    @property
    def remaining(self) -> int:
        """Get remaining requests in current window."""
        with self._lock:
            self._check_window()
            return max(0, self.max_requests - self._count)


class LeakyBucketLimiter(RateLimiter):
    """
    Leaky bucket rate limiter.
    
    Smooths out bursts by processing requests at a fixed rate.
    
    Example:
        >>> limiter = LeakyBucketLimiter(rate=10, capacity=50)
        >>> # Process at 10 requests/second, queue up to 50
    """
    
    def __init__(
        self,
        rate: float = 10.0,
        capacity: int = 50,
    ):
        """
        Initialize leaky bucket limiter.
        
        Args:
            rate: Leak rate (requests per second)
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        
        self._queue: deque = deque()
        self._last_leak = time.time()
        self._lock = threading.Lock()
        self._stats = RateLimitStats()
    
    def _leak(self) -> None:
        """Remove requests based on leak rate."""
        now = time.time()
        elapsed = now - self._last_leak
        
        # Number of requests to leak
        to_leak = int(elapsed * self.rate)
        
        for _ in range(min(to_leak, len(self._queue))):
            self._queue.popleft()
        
        if to_leak > 0:
            self._last_leak = now
    
    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to add to bucket."""
        with self._lock:
            self._leak()
            self._stats.total_requests += 1
            
            if len(self._queue) + tokens <= self.capacity:
                for _ in range(tokens):
                    self._queue.append(time.time())
                self._stats.allowed_requests += 1
                return True
            
            self._stats.denied_requests += 1
            return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """Get wait time until space available."""
        with self._lock:
            self._leak()
            
            space_needed = len(self._queue) + tokens - self.capacity
            if space_needed <= 0:
                return 0.0
            
            return space_needed / self.rate
    
    def reset(self) -> None:
        """Reset the limiter."""
        with self._lock:
            self._queue.clear()
            self._last_leak = time.time()
    
    @property
    def stats(self) -> RateLimitStats:
        """Get statistics."""
        return self._stats
    
    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        with self._lock:
            self._leak()
            return len(self._queue)


class AdaptiveLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts based on response.
    
    Increases rate when successful, decreases on failures.
    
    Example:
        >>> limiter = AdaptiveLimiter(initial_rate=50, min_rate=10, max_rate=200)
        >>> if limiter.acquire():
        ...     try:
        ...         response = make_api_call()
        ...         limiter.report_success()
        ...     except RateLimitError:
        ...         limiter.report_failure()
    """
    
    def __init__(
        self,
        initial_rate: float = 50.0,
        min_rate: float = 10.0,
        max_rate: float = 200.0,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.5,
    ):
        """
        Initialize adaptive limiter.
        
        Args:
            initial_rate: Starting rate
            min_rate: Minimum rate
            max_rate: Maximum rate
            increase_factor: Rate increase on success
            decrease_factor: Rate decrease on failure
        """
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        
        self._current_rate = initial_rate
        self._bucket = TokenBucketLimiter(
            rate=initial_rate,
            capacity=int(initial_rate * 2),
        )
        self._lock = threading.Lock()
        self._stats = RateLimitStats()
        
        # Tracking
        self._successes = 0
        self._failures = 0
        self._last_adjustment = time.time()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens."""
        self._stats.total_requests += 1
        result = self._bucket.acquire(tokens)
        
        if result:
            self._stats.allowed_requests += 1
        else:
            self._stats.denied_requests += 1
        
        return result
    
    def wait_time(self, tokens: int = 1) -> float:
        """Get wait time until tokens available."""
        return self._bucket.wait_time(tokens)
    
    def reset(self) -> None:
        """Reset the limiter."""
        with self._lock:
            self._bucket.reset()
            self._successes = 0
            self._failures = 0
    
    @property
    def stats(self) -> RateLimitStats:
        """Get statistics."""
        return self._stats
    
    def report_success(self) -> None:
        """Report a successful request."""
        with self._lock:
            self._successes += 1
            self._maybe_adjust()
    
    def report_failure(self) -> None:
        """Report a failed/rate-limited request."""
        with self._lock:
            self._failures += 1
            
            # Immediate decrease on failure
            self._current_rate = max(
                self.min_rate,
                self._current_rate * self.decrease_factor,
            )
            self._update_bucket()
    
    def _maybe_adjust(self) -> None:
        """Maybe adjust rate based on recent history."""
        now = time.time()
        
        # Adjust every 10 seconds
        if now - self._last_adjustment < 10.0:
            return
        
        total = self._successes + self._failures
        if total < 10:
            return
        
        success_rate = self._successes / total
        
        if success_rate > 0.95:
            # Increase rate
            self._current_rate = min(
                self.max_rate,
                self._current_rate * self.increase_factor,
            )
        elif success_rate < 0.8:
            # Decrease rate
            self._current_rate = max(
                self.min_rate,
                self._current_rate * self.decrease_factor,
            )
        
        self._update_bucket()
        self._successes = 0
        self._failures = 0
        self._last_adjustment = now
    
    def _update_bucket(self) -> None:
        """Update underlying bucket with new rate."""
        self._bucket = TokenBucketLimiter(
            rate=self._current_rate,
            capacity=int(self._current_rate * 2),
        )
    
    @property
    def current_rate(self) -> float:
        """Get current rate."""
        return self._current_rate


class MultiResourceLimiter:
    """
    Rate limiter for multiple resources (requests, tokens, etc.).
    
    Example:
        >>> limiter = MultiResourceLimiter()
        >>> limiter.add_limiter("requests", TokenBucketLimiter(rate=100))
        >>> limiter.add_limiter("tokens", TokenBucketLimiter(rate=10000))
        >>> 
        >>> if limiter.acquire(requests=1, tokens=500):
        ...     make_api_call()
    """
    
    def __init__(self):
        """Initialize multi-resource limiter."""
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.Lock()
    
    def add_limiter(self, name: str, limiter: RateLimiter) -> None:
        """Add a limiter for a resource."""
        with self._lock:
            self._limiters[name] = limiter
    
    def remove_limiter(self, name: str) -> None:
        """Remove a limiter."""
        with self._lock:
            self._limiters.pop(name, None)
    
    def acquire(self, **resources: int) -> bool:
        """
        Acquire multiple resources.
        
        Args:
            **resources: Resource name to amount mapping
            
        Returns:
            True if all acquired, False otherwise
        """
        with self._lock:
            # Check all resources first
            for name, amount in resources.items():
                if name not in self._limiters:
                    continue
                if not self._limiters[name].acquire(amount):
                    return False
            
            return True
    
    def wait_time(self, **resources: int) -> float:
        """Get maximum wait time across all resources."""
        max_wait = 0.0
        
        with self._lock:
            for name, amount in resources.items():
                if name not in self._limiters:
                    continue
                wait = self._limiters[name].wait_time(amount)
                max_wait = max(max_wait, wait)
        
        return max_wait
    
    def reset(self) -> None:
        """Reset all limiters."""
        with self._lock:
            for limiter in self._limiters.values():
                limiter.reset()
    
    @property
    def stats(self) -> Dict[str, RateLimitStats]:
        """Get stats for all limiters."""
        with self._lock:
            return {
                name: limiter.stats
                for name, limiter in self._limiters.items()
            }


class RateLimitedExecutor:
    """
    Execute functions with rate limiting.
    
    Example:
        >>> executor = RateLimitedExecutor(limiter=TokenBucketLimiter(rate=10))
        >>> 
        >>> @executor.limit
        >>> def api_call():
        ...     return requests.get("https://api.example.com")
        >>> 
        >>> result = api_call()  # Automatically rate limited
    """
    
    def __init__(
        self,
        limiter: RateLimiter,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize rate limited executor.
        
        Args:
            limiter: Rate limiter to use
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
        """
        self.limiter = limiter
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        tokens: int = 1,
        **kwargs: Any,
    ) -> T:
        """
        Execute function with rate limiting.
        
        Args:
            func: Function to execute
            *args: Function arguments
            tokens: Tokens to acquire
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RateLimitExceeded: If rate limit cannot be acquired
        """
        for attempt in range(self.max_retries):
            if self.limiter.acquire_or_wait(tokens):
                return func(*args, **kwargs)
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        raise RateLimitExceeded(
            f"Rate limit exceeded after {self.max_retries} attempts",
            retry_after=self.limiter.wait_time(tokens),
        )
    
    async def execute_async(
        self,
        func: Callable[..., T],
        *args: Any,
        tokens: int = 1,
        **kwargs: Any,
    ) -> T:
        """Execute function asynchronously with rate limiting."""
        for attempt in range(self.max_retries):
            if await self.limiter.acquire_async(tokens):
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        raise RateLimitExceeded(
            f"Rate limit exceeded after {self.max_retries} attempts",
            retry_after=self.limiter.wait_time(tokens),
        )
    
    def limit(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to rate limit a function."""
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.execute(func, *args, **kwargs)
        return wrapper
    
    def limit_async(
        self, func: Callable[..., T]
    ) -> Callable[..., T]:
        """Decorator to rate limit an async function."""
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await self.execute_async(func, *args, **kwargs)
        return wrapper


# Factory function
def create_limiter(
    limiter_type: Union[str, LimiterType] = LimiterType.TOKEN_BUCKET,
    **kwargs: Any,
) -> RateLimiter:
    """
    Create a rate limiter of the specified type.
    
    Args:
        limiter_type: Type of limiter to create
        **kwargs: Limiter-specific configuration
        
    Returns:
        RateLimiter instance
    """
    if isinstance(limiter_type, str):
        limiter_type = LimiterType(limiter_type)
    
    factories = {
        LimiterType.TOKEN_BUCKET: TokenBucketLimiter,
        LimiterType.SLIDING_WINDOW: SlidingWindowLimiter,
        LimiterType.FIXED_WINDOW: FixedWindowLimiter,
        LimiterType.LEAKY_BUCKET: LeakyBucketLimiter,
        LimiterType.ADAPTIVE: AdaptiveLimiter,
    }
    
    factory = factories.get(limiter_type)
    if factory is None:
        raise ValueError(f"Unknown limiter type: {limiter_type}")
    
    return factory(**kwargs)


# Global limiter instance
_default_limiter: Optional[RateLimiter] = None


def get_default_limiter(
    rate: float = 10.0,
    limiter_type: LimiterType = LimiterType.TOKEN_BUCKET,
) -> RateLimiter:
    """Get or create the default rate limiter."""
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = create_limiter(limiter_type, rate=rate)
    return _default_limiter


def rate_limit(tokens: int = 1) -> bool:
    """Convenience function to acquire rate limit tokens."""
    return get_default_limiter().acquire(tokens)
