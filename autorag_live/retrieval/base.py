"""Base retriever protocol and composable mixins for unified sync/async interface."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set

from autorag_live.types.types import Document


class AsyncRetriever(ABC):
    """Protocol for async retriever implementations."""

    @abstractmethod
    async def aretrieve(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Asynchronously retrieve the k most relevant documents for a query.

        Args:
            query: Query to find relevant documents for
            k: Number of documents to retrieve (default: 5)
            **kwargs: Additional retriever-specific parameters

        Returns:
            List of documents, sorted by relevance (highest first)

        Raises:
            ValueError: If k < 1 or retrieval fails
        """


class SyncRetriever(ABC):
    """Protocol for sync retriever implementations."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Synchronously retrieve the k most relevant documents for a query.

        Args:
            query: Query to find relevant documents for
            k: Number of documents to retrieve (default: 5)
            **kwargs: Additional retriever-specific parameters

        Returns:
            List of documents, sorted by relevance (highest first)

        Raises:
            ValueError: If k < 1 or retrieval fails
        """


class CacheMixin:
    """Mixin for in-memory query result caching."""

    def __init__(self, cache_size: int = 1000, ttl_seconds: Optional[int] = None):
        """
        Initialize cache mixin.

        Args:
            cache_size: Maximum number of cached results
            ttl_seconds: Time-to-live for cache entries in seconds, None for permanent
        """
        self._cache: Dict[str, List[Document]] = {}
        self._cache_size = cache_size
        self._ttl_seconds = ttl_seconds
        self._timestamps: Dict[str, float] = {}

    def _get_cache_key(self, query: str, k: int, **kwargs: Any) -> str:
        """Generate a cache key from query parameters."""

        key_parts = [query, str(k)] + sorted(f"{k}={v}" for k, v in kwargs.items())
        return "|".join(key_parts)

    def _get_cached(
        self,
        query: str,
        k: int,
        **kwargs: Any,
    ) -> Optional[List[Document]]:
        """Retrieve cached results if available and not expired."""
        import time

        key = self._get_cache_key(query, k, **kwargs)
        if key not in self._cache:
            return None

        if self._ttl_seconds:
            age = time.time() - self._timestamps.get(key, 0)
            if age > self._ttl_seconds:
                del self._cache[key]
                del self._timestamps[key]
                return None

        return self._cache[key]

    def _set_cached(
        self,
        query: str,
        k: int,
        results: List[Document],
        **kwargs: Any,
    ) -> None:
        """Store results in cache, evicting oldest if needed."""
        import time

        key = self._get_cache_key(query, k, **kwargs)

        if len(self._cache) >= self._cache_size and key not in self._cache:
            # Evict oldest entry (FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]

        self._cache[key] = results
        self._timestamps[key] = time.time()


class DeduplicationMixin:
    """Mixin for automatic result deduplication."""

    def __init__(self, dedup_field: str = "content"):
        """
        Initialize deduplication mixin.

        Args:
            dedup_field: Document field to use for deduplication
        """
        self.dedup_field = dedup_field

    def _deduplicate(
        self,
        docs: List[Document],
    ) -> List[Document]:
        """Remove duplicate documents based on dedup_field."""
        seen: Set[str] = set()
        deduplicated = []

        for doc in docs:
            field_value = getattr(doc, self.dedup_field, "")
            if field_value not in seen:
                seen.add(field_value)
                deduplicated.append(doc)

        return deduplicated


class FilterMixin:
    """Mixin for filtering retrieved documents."""

    def __init__(self):
        """Initialize filter mixin."""
        self._filters: List[Callable[[Document], bool]] = []

    def add_filter(self, filter_fn: Callable[[Document], bool]) -> None:
        """
        Add a filter function.

        Args:
            filter_fn: Function that takes Document and returns bool
        """
        self._filters.append(filter_fn)

    def _apply_filters(self, docs: List[Document]) -> List[Document]:
        """Apply all registered filters to documents."""
        result = docs
        for filter_fn in self._filters:
            result = [doc for doc in result if filter_fn(doc)]
        return result


class ComposableRetriever(SyncRetriever, CacheMixin, DeduplicationMixin, FilterMixin):
    """
    Abstract base class combining sync retriever with composable mixins.

    Provides caching, deduplication, and filtering out of the box.
    """

    def __init__(
        self,
        cache_size: int = 1000,
        ttl_seconds: Optional[int] = None,
        dedup_field: str = "content",
        enable_cache: bool = True,
        enable_dedup: bool = True,
    ):
        """
        Initialize composable retriever.

        Args:
            cache_size: Maximum cached results
            ttl_seconds: Cache TTL in seconds
            dedup_field: Field to use for deduplication
            enable_cache: Whether to use caching
            enable_dedup: Whether to deduplicate results
        """
        CacheMixin.__init__(self, cache_size, ttl_seconds)
        DeduplicationMixin.__init__(self, dedup_field)
        FilterMixin.__init__(self)
        self.enable_cache = enable_cache
        self.enable_dedup = enable_dedup

    def retrieve(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Retrieve documents with caching and deduplication.

        Args:
            query: Query string
            k: Number of documents to retrieve
            **kwargs: Additional parameters

        Returns:
            List of retrieved documents
        """
        # Check cache
        if self.enable_cache:
            cached = self._get_cached(query, k, **kwargs)
            if cached is not None:
                return cached

        # Retrieve from implementation
        results = self._retrieve_impl(query, k, **kwargs)

        # Apply deduplication
        if self.enable_dedup:
            results = self._deduplicate(results)

        # Apply filters
        results = self._apply_filters(results)

        # Store in cache
        if self.enable_cache:
            self._set_cached(query, k, results, **kwargs)

        return results

    @abstractmethod
    def _retrieve_impl(
        self,
        query: str,
        k: int,
        **kwargs: Any,
    ) -> List[Document]:
        """Implementation-specific retrieval logic."""


class AsyncComposableRetriever(AsyncRetriever, CacheMixin, DeduplicationMixin, FilterMixin):
    """
    Abstract base class combining async retriever with composable mixins.

    Provides caching, deduplication, and filtering out of the box.
    """

    def __init__(
        self,
        cache_size: int = 1000,
        ttl_seconds: Optional[int] = None,
        dedup_field: str = "content",
        enable_cache: bool = True,
        enable_dedup: bool = True,
    ):
        """
        Initialize async composable retriever.

        Args:
            cache_size: Maximum cached results
            ttl_seconds: Cache TTL in seconds
            dedup_field: Field to use for deduplication
            enable_cache: Whether to use caching
            enable_dedup: Whether to deduplicate results
        """
        CacheMixin.__init__(self, cache_size, ttl_seconds)
        DeduplicationMixin.__init__(self, dedup_field)
        FilterMixin.__init__(self)
        self.enable_cache = enable_cache
        self.enable_dedup = enable_dedup

    async def aretrieve(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Asynchronously retrieve documents with caching and deduplication.

        Args:
            query: Query string
            k: Number of documents to retrieve
            **kwargs: Additional parameters

        Returns:
            List of retrieved documents
        """
        # Check cache
        if self.enable_cache:
            cached = self._get_cached(query, k, **kwargs)
            if cached is not None:
                return cached

        # Retrieve from implementation
        results = await self._aretrieve_impl(query, k, **kwargs)

        # Apply deduplication
        if self.enable_dedup:
            results = self._deduplicate(results)

        # Apply filters
        results = self._apply_filters(results)

        # Store in cache
        if self.enable_cache:
            self._set_cached(query, k, results, **kwargs)

        return results

    @abstractmethod
    async def _aretrieve_impl(
        self,
        query: str,
        k: int,
        **kwargs: Any,
    ) -> List[Document]:
        """Implementation-specific async retrieval logic."""


class SyncAsyncAdapter(SyncRetriever):
    """Adapter to use AsyncRetriever as SyncRetriever via event loop."""

    def __init__(self, async_retriever: AsyncRetriever):
        """
        Initialize adapter.

        Args:
            async_retriever: Async retriever to adapt
        """
        self.async_retriever = async_retriever

    def retrieve(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Retrieve using the async retriever in event loop.

        Args:
            query: Query string
            k: Number of documents to retrieve
            **kwargs: Additional parameters

        Returns:
            List of retrieved documents
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create new one
            return asyncio.run(self.async_retriever.aretrieve(query, k, **kwargs))

        # Already in async context, use run_in_executor
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                self.async_retriever.aretrieve(query, k, **kwargs),
            )
            return future.result()


class AsyncSyncAdapter(AsyncRetriever):
    """Adapter to use SyncRetriever as AsyncRetriever."""

    def __init__(self, sync_retriever: SyncRetriever):
        """
        Initialize adapter.

        Args:
            sync_retriever: Sync retriever to adapt
        """
        self.sync_retriever = sync_retriever

    async def aretrieve(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Asynchronously call the sync retriever.

        Args:
            query: Query string
            k: Number of documents to retrieve
            **kwargs: Additional parameters

        Returns:
            List of retrieved documents
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.sync_retriever.retrieve(query, k, **kwargs),
        )
