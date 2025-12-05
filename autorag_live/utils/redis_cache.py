"""Redis-based distributed cache backend.

Provides a Redis cache backend for distributed AutoRAG deployments,
enabling cache sharing across multiple instances.
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-based cache with TTL support.

    Args:
        host: Redis host
        port: Redis port
        db: Redis database number
        password: Redis password (optional)
        prefix: Key prefix for namespacing
        default_ttl: Default TTL in seconds

    Example:
        >>> cache = RedisCache(host="localhost", prefix="autorag:")
        >>> cache.put("query:123", results, ttl=3600)
        >>> results = cache.get("query:123")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "autorag:",
        default_ttl: int = 3600,
    ):
        """Initialize Redis cache."""
        try:
            import redis  # type: ignore

            self._redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,
            )
            self._redis.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except ImportError:
            raise ImportError(
                "redis package required for RedisCache. Install with: pip install redis"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")

        self.prefix = prefix
        self.default_ttl = default_ttl

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            full_key = self._make_key(key)
            data = self._redis.get(full_key)
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis get failed for {key}: {e}")
            return None

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put value in cache with TTL."""
        try:
            full_key = self._make_key(key)
            data = json.dumps(value)
            ttl_seconds = ttl if ttl is not None else self.default_ttl
            self._redis.setex(full_key, ttl_seconds, data)
        except Exception as e:
            logger.warning(f"Redis put failed for {key}: {e}")

    def clear(self) -> None:
        """Clear all keys with prefix."""
        try:
            pattern = f"{self.prefix}*"
            for key in self._redis.scan_iter(match=pattern):
                self._redis.delete(key)
            logger.info(f"Cleared cache with prefix {self.prefix}")
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")

    def delete(self, key: str) -> None:
        """Delete specific key."""
        try:
            full_key = self._make_key(key)
            self._redis.delete(full_key)
        except Exception as e:
            logger.warning(f"Redis delete failed for {key}: {e}")


__all__ = ["RedisCache"]
