"""Tests for retriever protocol refactoring and composable mixins."""

import pytest

from autorag_live.retrieval.base import (
    AsyncComposableRetriever,
    AsyncSyncAdapter,
    CacheMixin,
    ComposableRetriever,
    DeduplicationMixin,
    FilterMixin,
    SyncAsyncAdapter,
)
from autorag_live.types.types import Document


class SimpleRetriever(ComposableRetriever):
    """Simple test retriever implementation."""

    def _retrieve_impl(self, query: str, k: int = 5, **kwargs) -> list[Document]:
        """Mock retrieval: return k dummy documents."""
        return [
            Document(
                id=f"doc-{i}",
                text=f"Doc {i}: {query}",
                metadata={"index": i, "query": query},
            )
            for i in range(min(k, 3))
        ]


class SimpleAsyncRetriever(AsyncComposableRetriever):
    """Simple async test retriever implementation."""

    async def _aretrieve_impl(self, query: str, k: int = 5, **kwargs) -> list[Document]:
        """Mock async retrieval: return k dummy documents."""
        return [
            Document(
                id=f"doc-{i}",
                text=f"Doc {i}: {query}",
                metadata={"index": i, "query": query},
            )
            for i in range(min(k, 3))
        ]


def test_composable_retriever_basic():
    """Test basic retrieval without mixins."""
    retriever = SimpleRetriever(enable_cache=False, enable_dedup=False)
    docs = retriever.retrieve("test", k=2)

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].text == "Doc 0: test"


def test_composable_retriever_caching():
    """Test caching mixin integration."""
    retriever = SimpleRetriever(cache_size=10, enable_cache=True, enable_dedup=False)

    # First call
    docs1 = retriever.retrieve("query1", k=2)
    assert len(docs1) == 2

    # Second call should be cached (same results object)
    docs2 = retriever.retrieve("query1", k=2)
    assert len(docs2) == 2
    # Both should have same content due to cache
    assert docs1[0].id == docs2[0].id


def test_composable_retriever_deduplication():
    """Test deduplication mixin integration."""

    class DuplicateRetriever(ComposableRetriever):
        def _retrieve_impl(self, query: str, k: int = 5, **kwargs) -> list[Document]:
            # Return duplicates based on text field
            return [
                Document(id="1", text="duplicate", metadata={"idx": 1}),
                Document(id="2", text="duplicate", metadata={"idx": 2}),
                Document(id="3", text="unique", metadata={"idx": 3}),
            ]

    retriever = DuplicateRetriever(enable_cache=False, enable_dedup=True)
    docs = retriever.retrieve("test")

    # Should deduplicate by text field, removing id=2, keeping id=1 and id=3
    assert len(docs) == 2
    assert docs[0].id == "1" and docs[0].text == "duplicate"
    assert docs[1].id == "3" and docs[1].text == "unique"


def test_composable_retriever_filtering():
    """Test filter mixin integration."""
    retriever = SimpleRetriever(enable_cache=False, enable_dedup=False)

    # Add filter to exclude docs with index > 0
    retriever.add_filter(lambda doc: doc.metadata.get("index", 0) == 0)

    docs = retriever.retrieve("test", k=3)
    assert len(docs) == 1
    assert docs[0].metadata["index"] == 0


def test_cache_mixin_expiry():
    """Test cache TTL expiry."""
    import time

    mixin = CacheMixin(cache_size=10, ttl_seconds=1)
    docs = [Document(id="test-1", text="test", metadata={})]

    # Store in cache
    mixin._set_cached("query", 5, docs)
    assert mixin._get_cached("query", 5) is not None

    # Wait for expiry
    time.sleep(1.1)
    assert mixin._get_cached("query", 5) is None


def test_deduplication_mixin():
    """Test deduplication mixin directly."""
    mixin = DeduplicationMixin(dedup_field="text")
    docs = [
        Document(id="1", text="A", metadata={}),
        Document(id="2", text="A", metadata={}),
        Document(id="3", text="B", metadata={}),
    ]

    deduplicated = mixin._deduplicate(docs)
    assert len(deduplicated) == 2
    assert deduplicated[0].text == "A"
    assert deduplicated[1].text == "B"


def test_filter_mixin():
    """Test filter mixin directly."""
    mixin = FilterMixin()
    docs = [
        Document(id="1", text="short", metadata={}),
        Document(id="2", text="longer content", metadata={}),
        Document(id="3", text="x", metadata={}),
    ]

    # Add filter for minimum length - should filter out 'short' and 'x'
    mixin.add_filter(lambda doc: len(doc.text) > 5)

    filtered = mixin._apply_filters(docs)
    assert len(filtered) == 1
    assert filtered[0].text == "longer content"


@pytest.mark.asyncio
async def test_async_composable_retriever_basic():
    """Test basic async retrieval."""
    retriever = SimpleAsyncRetriever(enable_cache=False, enable_dedup=False)
    docs = await retriever.aretrieve("test", k=2)

    assert len(docs) == 2
    assert docs[0].text == "Doc 0: test"


@pytest.mark.asyncio
async def test_async_composable_retriever_caching():
    """Test async retriever with caching."""
    retriever = SimpleAsyncRetriever(cache_size=10, enable_cache=True)

    docs1 = await retriever.aretrieve("query1", k=2)
    docs2 = await retriever.aretrieve("query1", k=2)

    # Should have same content (cache working)
    assert len(docs1) == len(docs2)
    assert docs1[0].id == docs2[0].id


@pytest.mark.asyncio
async def test_sync_async_adapter():
    """Test adapter to use sync retriever as async."""
    sync_retriever = SimpleRetriever(enable_cache=False, enable_dedup=False)
    async_adapter = AsyncSyncAdapter(sync_retriever)

    docs = await async_adapter.aretrieve("test", k=2)
    assert len(docs) == 2
    assert docs[0].text == "Doc 0: test"


@pytest.mark.asyncio
async def test_async_sync_adapter():
    """Test adapter to use async retriever as sync."""
    async_retriever = SimpleAsyncRetriever(enable_cache=False, enable_dedup=False)
    sync_adapter = SyncAsyncAdapter(async_retriever)

    # This should work if called outside async context
    docs = sync_adapter.retrieve("test", k=2)
    assert len(docs) == 2
