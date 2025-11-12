"""Lazy loading utilities for heavy dependencies to improve startup time."""

from typing import Any, Callable, Dict, Optional


class LazyLoader:
    """Lazy loads modules and caches results for subsequent accesses."""

    def __init__(self):
        """Initialize the lazy loader with empty cache."""
        self._cache: Dict[str, Any] = {}

    def get(self, module_path: str, attr_name: str) -> Any:
        """
        Lazy load and cache a module attribute.

        Args:
            module_path: Full module path (e.g., "transformers.AutoTokenizer")
            attr_name: Optional attribute to extract (e.g., "AutoTokenizer")

        Returns:
            The loaded module or attribute

        Raises:
            ImportError: If module cannot be imported
        """
        cache_key = f"{module_path}.{attr_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Extract module and attribute
        parts = module_path.rsplit(".", 1)
        if len(parts) == 2:
            module_name, default_attr = parts
        else:
            module_name = module_path
            default_attr = attr_name or None

        from importlib import import_module

        module = import_module(module_name)

        # Get the attribute if specified
        if attr_name:
            result = getattr(module, attr_name)
        elif default_attr:
            result = getattr(module, default_attr)
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
    """Lazy-load and get SentenceTransformer model."""
    try:
        transformer_class = _global_loader.get("sentence_transformers", "SentenceTransformer")
        return transformer_class(model_name)
    except ImportError as e:
        raise ImportError(
            "sentence-transformers not installed. Install with: pip install sentence-transformers"
        ) from e


def get_elasticsearch_client(hosts: Optional[list] = None, **kwargs):
    """Lazy-load and create Elasticsearch client."""
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
    """Lazy-load and create Qdrant client."""
    if url is None:
        url = "http://localhost:6333"
    try:
        client_class = _global_loader.get("qdrant_client", "QdrantClient")
        return client_class(url=url, **kwargs)
    except ImportError as e:
        raise ImportError(
            "qdrant-client not installed. Install with: pip install qdrant-client"
        ) from e
