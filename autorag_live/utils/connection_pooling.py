"""Connection pooling for network-based retrievers."""

from typing import Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session_with_pooling(
    pool_connections: int = 10,
    pool_maxsize: int = 20,
    max_retries: int = 3,
) -> requests.Session:
    """
    Create requests session with connection pooling.

    Args:
        pool_connections: Number of connection pools to cache
        pool_maxsize: Maximum number of connections in each pool
        max_retries: Number of retries for failed requests

    Returns:
        Configured requests session
    """
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


class ConnectionPool:
    """Manages a pool of connections for efficient reuse."""

    def __init__(self, max_size: int = 10):
        """Initialize connection pool."""
        self._pool: Dict[str, requests.Session] = {}
        self._max_size = max_size

    def get_session(self, endpoint: str) -> requests.Session:
        """Get or create session for endpoint."""
        if endpoint not in self._pool:
            if len(self._pool) >= self._max_size:
                # Remove oldest connection
                oldest_key = next(iter(self._pool))
                del self._pool[oldest_key]

            self._pool[endpoint] = create_session_with_pooling()

        return self._pool[endpoint]

    def clear(self):
        """Close all connections in pool."""
        for session in self._pool.values():
            session.close()
        self._pool.clear()
