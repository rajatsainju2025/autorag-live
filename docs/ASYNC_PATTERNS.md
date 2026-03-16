# Async Patterns in AutoRAG-Live

This document explains the async/await patterns used throughout AutoRAG-Live for non-blocking I/O, concurrent operations, and efficient resource utilization.

## Table of Contents

1. [Core Async Concepts](#core-async-concepts)
2. [Sync-to-Async Adaptation Pattern](#sync-to-async-adaptation-pattern)
3. [Async Retrievers](#async-retrievers)
4. [Async Rerankers](#async-rerankers)
5. [Multi-Agent Async Orchestration](#multi-agent-async-orchestration)
6. [Best Practices](#best-practices)
7. [Performance Considerations](#performance-considerations)

## Core Async Concepts

AutoRAG-Live uses Python's `asyncio` library for concurrent operations without multithreading overhead. Key concepts:

- **`async/await`**: Syntactic sugar for coroutines that yield control back to the event loop when waiting for I/O
- **Event Loop**: Single-threaded scheduler that manages concurrent coroutines
- **Coroutines**: Functions defined with `async def` that can be paused and resumed
- **Tasks**: Wrapped coroutines scheduled for concurrent execution

### Example: Basic Coroutine

```python
import asyncio

async def fetch_document(doc_id: str) -> str:
    """Simulate fetching a document."""
    await asyncio.sleep(1)  # Simulate I/O wait
    return f"Content of {doc_id}"

async def main():
    """Run multiple fetches concurrently."""
    results = await asyncio.gather(
        fetch_document("doc1"),
        fetch_document("doc2"),
        fetch_document("doc3"),
    )
    print(results)

asyncio.run(main())
```

## Sync-to-Async Adaptation Pattern

Many retrieval systems (Qdrant, Elasticsearch, FAISS, ColBERT) provide synchronous APIs. AutoRAG-Live adapts them for async operations using `loop.run_in_executor()`:

### The Problem

- Sync functions block the event loop when performing I/O
- Multiple sync calls cannot run concurrently in the main thread
- Solution: Run sync code in a thread pool without blocking the event loop

### The Solution Pattern

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncRetrieverAdapter:
    """Wraps a synchronous retriever for async operation."""

    def __init__(self, sync_retriever, max_workers: int = 4):
        self.sync_retriever = sync_retriever
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """Asynchronously retrieve documents.

        The actual retrieval happens in a thread pool, allowing other
        coroutines to run while this operation completes.
        """
        loop = asyncio.get_event_loop()

        # Run sync method in thread pool without blocking the loop
        documents = await loop.run_in_executor(
            self._executor,
            self.sync_retriever.retrieve,  # Sync function
            query,                          # Function argument
            top_k                           # Function argument
        )

        return documents
```

### How It Works

1. `asyncio.get_event_loop()` gets the current event loop
2. `loop.run_in_executor(executor, func, *args)` schedules `func(*args)` in the thread pool
3. The coroutine yields control to the event loop
4. Other coroutines can execute while waiting for the result
5. When the thread finishes, the event loop resumes the coroutine with the result

### Performance Benefits

- **Concurrency**: 3 retrievers with 1-second latency each: **~1 second total** (instead of 3 seconds sequential)
- **Scalability**: Event loop can manage hundreds of concurrent async retrievers
- **No GIL Contention**: Work distributed across thread pool avoids Python GIL bottlenecks for I/O

## Async Retrievers

AutoRAG-Live provides async wrappers for major retrieval systems:

### AsyncQdrantWrapper

```python
from autorag_live.retrievers.async_qdrant import AsyncQdrantWrapper
from qdrant_client import QdrantClient

# Create async wrapper around sync Qdrant client
client = QdrantClient("http://localhost:6333")
async_retriever = AsyncQdrantWrapper(client)

# Use in async context
async def search_documents():
    results = await async_retriever.retrieve(
        query="machine learning",
        top_k=10
    )
    for doc in results:
        print(doc.content)

asyncio.run(search_documents())
```

### AsyncElasticsearchWrapper

```python
from autorag_live.retrievers.async_elasticsearch import AsyncElasticsearchWrapper
from elasticsearch import Elasticsearch

client = Elasticsearch(["http://localhost:9200"])
async_retriever = AsyncElasticsearchWrapper(client)

results = await async_retriever.retrieve("neural networks")
```

### AsyncFAISSRetriever

```python
from autorag_live.retrievers.async_faiss import AsyncFAISSRetriever
import faiss

# Create FAISS index
index = faiss.read_index("embeddings.index")
async_retriever = AsyncFAISSRetriever(index)

results = await async_retriever.retrieve("deep learning")
```

### AsyncColBERTRetriever

```python
from autorag_live.retrievers.async_colbert import AsyncColBERTRetriever
from colbert.retrieval import Retriever

retriever = Retriever("path/to/checkpoint")
async_retriever = AsyncColBERTRetriever(retriever)

results = await async_retriever.retrieve("question")
```

## Async Rerankers

Rerankers also benefit from async execution, especially when applied to large result sets:

### AsyncMMRReranker

Maximal Marginal Relevance (MMR) reranking with async support:

```python
from autorag_live.rerank.async_mmr import AsyncMMRReranker

reranker = AsyncMMRReranker(lambda_mult=0.5)  # Balance relevance and diversity

async def rerank_results():
    documents = [...]  # Retrieved documents
    query = "what is machine learning?"

    reranked = await reranker.rerank(
        documents=documents,
        query=query,
        top_k=5
    )
    return reranked

results = asyncio.run(rerank_results())
```

**Parameters**:
- `lambda_mult` (0.0-1.0): Trade-off between relevance (1.0) and diversity (0.0)
  - 0.5: Balanced approach
  - 0.9: Prioritize relevance
  - 0.1: Prioritize diversity

### StreamingReranker

Progressive reranking with streaming results:

```python
from autorag_live.rerank.streaming_reranker import StreamingReranker

reranker = StreamingReranker(batch_size=5)

async def stream_reranked_results():
    documents = [...]
    query = "..."

    async for batch in reranker.rerank_stream(
        documents=documents,
        query=query
    ):
        for doc in batch:
            print(f"Score: {doc.score}, Content: {doc.content}")
            # Can update UI or send to client in real-time

        yield batch  # For streaming responses

# Use with async generators
async for batch in stream_reranked_results():
    pass
```

## Multi-Agent Async Orchestration

The multi-agent framework uses async for concurrent specialist agent execution:

### Synchronous Orchestration (Sequential)

```python
from autorag_live.multi_agent.collaboration import (
    MultiAgentOrchestrator,
    create_default_agents
)

orchestrator = MultiAgentOrchestrator()

for agent in create_default_agents():
    orchestrator.register_agent(agent)

# Synchronous: agents process sequentially
result = orchestrator.orchestrate_query("What is RAG?")
print(result["consensus"]["chosen_proposal"])
```

### Asynchronous Orchestration (Concurrent)

```python
import asyncio
from autorag_live.multi_agent.collaboration import (
    MultiAgentOrchestrator,
    create_default_agents
)

async def concurrent_orchestration():
    orchestrator = MultiAgentOrchestrator()

    for agent in create_default_agents():
        orchestrator.register_agent(agent)

    # Async: agents process in parallel
    # All 5 agents execute simultaneously
    result = await orchestrator.orchestrate_query_async(
        "What is RAG?"
    )

    return result

result = asyncio.run(concurrent_orchestration())
print(result["consensus"]["chosen_proposal"])
```

### Performance Comparison

With 5 agents, each taking ~100ms to process:

| Method | Time | Speedup |
|--------|------|---------|
| Synchronous (`orchestrate_query`) | ~500ms | 1x |
| Asynchronous (`orchestrate_query_async`) | ~100ms | 5x |

### Implementation Details

The async orchestration pattern:

```python
async def orchestrate_query_async(self, query: str):
    """Orchestrate with parallel agent execution."""
    loop = asyncio.get_event_loop()

    # Phase 1: Parallel agent processing
    async def _run_agent(agent_id, agent):
        result = await loop.run_in_executor(
            self._executor,
            agent.process_query,
            query
        )
        return agent_id, result

    # Create tasks for all agents concurrently
    tasks = [
        _run_agent(aid, ag)
        for aid, ag in self.agents.items()
    ]

    # Wait for all tasks concurrently
    results_list = await asyncio.gather(*tasks)
    results = dict(results_list)

    # Phase 2-3: Consensus building (lightweight, stays sync)
    ...
```

## Best Practices

### 1. Always Use Async Context Managers

```python
# Bad: Resource might leak if exception occurs
client = Elasticsearch()
results = await async_retriever.retrieve(query)
client.close()  # Might not execute

# Good: Automatic cleanup
async with Elasticsearch() as client:
    async_retriever = AsyncElasticsearchWrapper(client)
    results = await async_retriever.retrieve(query)
    # Cleanup guaranteed
```

### 2. Use asyncio.gather() for Concurrent Operations

```python
# Bad: Sequential execution
result1 = await retriever1.retrieve(query)
result2 = await retriever2.retrieve(query)
result3 = await retriever3.retrieve(query)

# Good: Concurrent execution
results = await asyncio.gather(
    retriever1.retrieve(query),
    retriever2.retrieve(query),
    retriever3.retrieve(query),
)
```

### 3. Handle Errors Properly

```python
# Good: Catch errors in concurrent tasks
try:
    results = await asyncio.gather(
        retriever1.retrieve(query),
        retriever2.retrieve(query),
        return_exceptions=True  # Don't stop on first error
    )

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Retriever failed: {result}")
        else:
            process_result(result)

except asyncio.TimeoutError:
    logger.error("Retrieval timed out")
```

### 4. Use Timeouts for Reliability

```python
async def retrieve_with_timeout(retriever, query, timeout=5.0):
    """Retrieve with timeout protection."""
    try:
        results = await asyncio.wait_for(
            retriever.retrieve(query),
            timeout=timeout
        )
        return results
    except asyncio.TimeoutError:
        logger.warning(f"Retrieval timeout after {timeout}s")
        return []
```

### 5. Monitor Executor Thread Pool

```python
class MonitoredAsyncRetriever:
    """Track executor thread pool usage."""

    def __init__(self, max_workers=4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._active_tasks = 0

    async def retrieve(self, query, top_k=10):
        self._active_tasks += 1
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._sync_retrieve,
                query,
                top_k
            )
            return result
        finally:
            self._active_tasks -= 1

    def get_status(self):
        return {"active_tasks": self._active_tasks}
```

## Performance Considerations

### Executor Thread Pool Sizing

```python
import multiprocessing

# Good: Match CPU core count for CPU-bound, more for I/O-bound
num_cores = multiprocessing.cpu_count()
executor = ThreadPoolExecutor(max_workers=num_cores * 2)  # For I/O
```

### Event Loop Efficiency

- **Coroutines**: ~1000-10000 coroutines per event loop
- **Threads**: ~100-1000 threads before OS scheduling overhead

### Concurrent Retriever Limits

| Retriever | Latency | Ideal Concurrency | Speedup |
|-----------|---------|-------------------|---------|
| Qdrant | 50-100ms | 4-8 | 4-8x |
| Elasticsearch | 100-500ms | 8-16 | 8-16x |
| FAISS (local) | 10-50ms | 2-4 | 2-4x |
| ColBERT | 500-2000ms | 8-16 | 8-16x |

### Memory Usage

- Async coroutines: ~1KB each
- Threads: ~8MB each
- Use async for high concurrency, threads only when necessary

## Common Patterns

### Pattern 1: Parallel Retrieval with Fallback

```python
async def retrieve_with_fallback(primary, fallback, query):
    """Try primary retriever, fallback to secondary on failure."""
    try:
        results = await asyncio.wait_for(
            primary.retrieve(query),
            timeout=2.0
        )
        if results:
            return results
    except (asyncio.TimeoutError, Exception):
        pass

    return await fallback.retrieve(query)
```

### Pattern 2: Concurrent Reranking

```python
async def rerank_with_ensemble(documents, query, rerankers):
    """Combine scores from multiple rerankers."""
    reranked_sets = await asyncio.gather(
        *[r.rerank(documents, query) for r in rerankers]
    )

    # Combine scores
    combined = {}
    for doc_set in reranked_sets:
        for doc in doc_set:
            combined[doc.id] = combined.get(doc.id, 0) + doc.score

    return sorted(
        combined.items(),
        key=lambda x: x[1],
        reverse=True
    )
```

### Pattern 3: Streaming Pipeline

```python
async def rag_pipeline_streaming(query):
    """RAG pipeline with streaming reranking."""
    # Parallel retrieval
    results = await asyncio.gather(
        retriever1.retrieve(query),
        retriever2.retrieve(query),
    )
    all_docs = sum(results, [])

    # Stream reranked results
    async for batch in reranker.rerank_stream(all_docs, query):
        for doc in batch:
            yield doc  # Stream to client

        # Can stop early if confidence is high
        if batch[0].score > 0.95:
            break
```

## Debugging Async Code

### Enable Debug Mode

```python
import asyncio
import logging

logging.basicConfig(level=logging.DEBUG)
asyncio.run(main(), debug=True)  # Enable asyncio debugging
```

### Monitor Event Loop

```python
import asyncio

async def monitor_event_loop():
    """Monitor event loop health."""
    loop = asyncio.get_event_loop()

    while True:
        # Get all running tasks
        tasks = asyncio.all_tasks(loop)
        print(f"Active tasks: {len(tasks)}")

        # Warn if too many pending
        if len(tasks) > 100:
            logger.warning("Event loop overloaded")

        await asyncio.sleep(1)

# Run alongside main logic
asyncio.create_task(monitor_event_loop())
```

### Timeout Debugging

```python
async def debug_timeout(coro, timeout=5.0):
    """Debug which operations timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(
            f"Operation timed out after {timeout}s",
            extra={"stacktrace": traceback.format_stack()}
        )
        raise
```

## Conclusion

Async/await patterns in AutoRAG-Live enable:

- **Scalability**: Handle hundreds of concurrent operations
- **Efficiency**: Better CPU utilization with non-blocking I/O
- **Responsiveness**: Main application thread never blocks
- **Resource Usage**: Lower memory footprint than threads

Use `asyncio.gather()` for concurrent operations, `loop.run_in_executor()` for sync-to-async adaptation, and `async for` for streaming results.
