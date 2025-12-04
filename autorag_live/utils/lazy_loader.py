"""Lazy loading utilities for heavy dependencies to improve startup time."""

import threading
from typing import Any, Callable, Dict, Optional


class LazyLoader:
    """Thread-safe lazy loader for modules with caching."""

    def __init__(self):
        """Initialize the lazy loader with empty cache and thread lock."""
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, module_path: str, attr_name: Optional[str] = None) -> Any:
        """
        Lazy load and cache a module attribute with thread safety.

        Args:
            module_path: Full module path (e.g., "transformers")
            attr_name: Optional attribute to extract (e.g., "AutoTokenizer")

        Returns:
            The loaded module or attribute

        Raises:
            ImportError: If module cannot be imported
        """
        cache_key = f"{module_path}.{attr_name}" if attr_name else module_path

        # Fast path: check cache without lock
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Slow path: load with thread safety
        with self._lock:
            # Double-check after acquiring lock
            if cache_key in self._cache:
                return self._cache[cache_key]

            from importlib import import_module

            module = import_module(module_path)

            # Get the attribute if specified
            if attr_name:
                result = getattr(module, attr_name)
            else:
                result = module

            # Cache and return
            self._cache[cache_key] = result
            return result

    def get_callable_wrapper(self, module_path: str, attr_name: str) -> Callable:
        """
        Return a callable wrapper that lazy-loads the actual function.

        This allows functions to be "imported" without loading dependencies.

        Args:
            module_path: Full module path
            attr_name: Attribute name to load

        Returns:
            Wrapper function that loads and calls the actual function
        """

        def wrapper(*args, **kwargs):
            func = self.get(module_path, attr_name)
            return func(*args, **kwargs)

        wrapper.__name__ = f"{attr_name}_lazy"
        wrapper.__doc__ = f"Lazy-loaded wrapper for {module_path}.{attr_name}"
        return wrapper

    def clear_cache(self):
        """Clear the cache to force reloading on next access."""
        with self._lock:
            self._cache.clear()


# Global lazy loader instance
_global_loader = LazyLoader()


def lazy_import(module_path: str, attr_name: str):
    """
    Decorator to create lazy-loaded imports for functions.

    Usage:
        @lazy_import("transformers", "AutoTokenizer")
        def tokenize(text, tokenizer_class=None):
            if tokenizer_class is None:
                # Will be injected automatically
                pass
    """

    def decorator(func: Callable) -> Callable:
        original_func = func

        def wrapper(*args, **kwargs):
            # Lazy load dependency if not provided
            if attr_name not in kwargs or kwargs[attr_name] is None:
                kwargs[attr_name] = _global_loader.get(module_path, attr_name)
            return original_func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


# Pre-configured loaders for common heavy dependencies
def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load and get SentenceTransformer model.

    This function delays the import of sentence-transformers until needed,
    reducing startup time and memory usage.

    Args:
        model_name: Name of the model to load. Defaults to "all-MiniLM-L6-v2".
            Popular options include:
            - "all-MiniLM-L6-v2": Fast, small model (80MB)
            - "all-mpnet-base-v2": Higher quality (420MB)
            - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual support

    Returns:
        SentenceTransformer: Loaded model instance ready for encoding.

    Raises:
        ImportError: If sentence-transformers package is not installed.

    Example:
        >>> model = get_sentence_transformer()
        >>> embeddings = model.encode(["Hello world", "Another sentence"])
        >>> print(embeddings.shape)  # (2, 384)
    """
    try:
        transformer_class = _global_loader.get("sentence_transformers", "SentenceTransformer")
        return transformer_class(model_name)
    except ImportError as e:
        raise ImportError(
            "sentence-transformers not installed. Install with: pip install sentence-transformers"
        ) from e


def get_elasticsearch_client(hosts: Optional[list] = None, **kwargs):
    """Lazy-load and create Elasticsearch client.

    Args:
        hosts: List of Elasticsearch host addresses.
            Defaults to ["localhost:9200"].
        **kwargs: Additional keyword arguments passed to Elasticsearch client.
            Common options:
            - timeout: Request timeout in seconds
            - http_auth: (username, password) tuple for authentication
            - use_ssl: Whether to use SSL/TLS

    Returns:
        Elasticsearch: Configured client instance.

    Raises:
        ImportError: If elasticsearch package is not installed.

    Example:
        >>> client = get_elasticsearch_client(["localhost:9200"])
        >>> client.ping()  # Test connection
        True
        >>> # With authentication
        >>> client = get_elasticsearch_client(
        ...     hosts=["https://elastic.example.com:9200"],
        ...     http_auth=("user", "pass"),
        ...     use_ssl=True
        ... )
    """
    if hosts is None:
        hosts = ["localhost:9200"]
    try:
        client_class = _global_loader.get("elasticsearch", "Elasticsearch")
        return client_class(hosts, **kwargs)
    except ImportError as e:
        raise ImportError(
            "elasticsearch not installed. Install with: pip install elasticsearch"
        ) from e


def get_qdrant_client(url: Optional[str] = None, **kwargs):
    """Lazy-load and create Qdrant client.

    Args:
        url: Qdrant server URL. Defaults to "http://localhost:6333".
        **kwargs: Additional keyword arguments passed to QdrantClient.
            Common options:
            - api_key: API key for authentication
            - timeout: Request timeout in seconds
            - prefer_grpc: Whether to use gRPC protocol

    Returns:
        QdrantClient: Configured client instance.

    Raises:
        ImportError: If qdrant-client package is not installed.

    Example:
        >>> client = get_qdrant_client()
        >>> # Check if collection exists
        >>> collections = client.get_collections()
        >>> # With cloud deployment
        >>> client = get_qdrant_client(
        ...     url="https://xyz.aws.qdrant.cloud:6333",
        ...     api_key="your-api-key"
        ... )
    """
    if url is None:
        url = "http://localhost:6333"
    try:
        client_class = _global_loader.get("qdrant_client", "QdrantClient")
        return client_class(url=url, **kwargs)
    except ImportError as e:
        raise ImportError(
            "qdrant-client not installed. Install with: pip install qdrant-client"
        ) from e
