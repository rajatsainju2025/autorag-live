"""
Async-first agentic RAG pipeline for state-of-the-art performance.

Provides non-blocking I/O, concurrent retrieval, and streaming support
for maximum throughput and minimal latency.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from autorag_live.agent.base import Agent
from autorag_live.agent.memory import ConversationMemory
from autorag_live.agent.reasoning import Reasoner
from autorag_live.agent.tools import ToolRegistry, get_tool_registry


@dataclass
class AsyncRetrievalResult:
    """Async retrieval result with metadata."""

    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: float = 0.0


@dataclass
class AsyncRAGResponse:
    """Async RAG response with streaming support."""

    query: str
    answer: str
    sources: List[AsyncRetrievalResult] = field(default_factory=list)
    reasoning_trace: str = ""
    confidence: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncQueryRefinement:
    """Async query refinement engine."""

    def __init__(self):
        self.logger = logging.getLogger("AsyncQueryRefinement")

    async def refine(
        self, query: str, previous_results: Optional[List[AsyncRetrievalResult]] = None
    ) -> str:
        """
        Refine query based on previous results.

        Args:
            query: Original query
            previous_results: Previous retrieval results

        Returns:
            Refined query
        """
        if not previous_results or not previous_results:
            return query

        # Simulate async refinement
        await asyncio.sleep(0.01)

        # Extract low-confidence terms
        low_conf_docs = [r for r in previous_results if r.score < 0.5]
        if low_conf_docs:
            return f"{query} (focus on specifics)"

        return query

    async def expand_query(self, query: str) -> List[str]:
        """
        Expand query into multiple variants.

        Args:
            query: Original query

        Returns:
            List of query variants
        """
        await asyncio.sleep(0.01)

        # Generate variants
        variants = [
            query,
            f"What is {query}?",
            f"Explain {query}",
        ]
        return variants


class AsyncIterativeRetriever:
    """Async iterative retriever with parallel execution."""

    def __init__(self, max_iterations: int = 3, top_k: int = 5):
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.logger = logging.getLogger("AsyncIterativeRetriever")

    async def retrieve(
        self, query: str, corpus: Optional[List[str]] = None
    ) -> List[AsyncRetrievalResult]:
        """
        Perform async iterative retrieval.

        Args:
            query: Search query
            corpus: Optional document corpus

        Returns:
            List of retrieval results
        """
        if corpus is None:
            corpus = self._get_default_corpus()

        all_results = []
        current_query = query

        for iteration in range(self.max_iterations):
            self.logger.debug(f"Iteration {iteration + 1}: {current_query}")

            # Simulate async retrieval
            results = await self._retrieve_batch(current_query, corpus)
            all_results.extend(results)

            # Check if we have enough high-quality results
            high_quality = [r for r in results if r.score > 0.7]
            if len(high_quality) >= self.top_k:
                break

            # Refine query for next iteration
            if iteration < self.max_iterations - 1:
                current_query = await self._refine_for_next_iteration(current_query, results)

        # Deduplicate and sort
        unique_results = self._deduplicate(all_results)
        return sorted(unique_results, key=lambda r: r.score, reverse=True)[: self.top_k]

    async def _retrieve_batch(self, query: str, corpus: List[str]) -> List[AsyncRetrievalResult]:
        """Retrieve batch of documents asynchronously."""
        await asyncio.sleep(0.02)  # Simulate I/O

        # Simple scoring (in production, use embeddings)
        results = []
        query_terms = set(query.lower().split())

        for idx, doc in enumerate(corpus):
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / max(len(query_terms), 1)

            results.append(
                AsyncRetrievalResult(
                    text=doc,
                    score=score,
                    metadata={"corpus_idx": idx},
                    source=f"doc_{idx}",
                )
            )

        return results

    async def _refine_for_next_iteration(
        self, query: str, results: List[AsyncRetrievalResult]
    ) -> str:
        """Refine query for next iteration."""
        await asyncio.sleep(0.01)

        # If we got low scores, expand the query
        avg_score = sum(r.score for r in results) / max(len(results), 1)
        if avg_score < 0.3:
            return f"{query} details"

        return query

    def _deduplicate(self, results: List[AsyncRetrievalResult]) -> List[AsyncRetrievalResult]:
        """Remove duplicate documents."""
        seen = set()
        unique = []

        for result in results:
            if result.text not in seen:
                seen.add(result.text)
                unique.append(result)

        return unique

    def _get_default_corpus(self) -> List[str]:
        """Get default document corpus."""
        return [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Python is a popular programming language for data science.",
            "Natural language processing helps computers understand human language.",
            "Transformers revolutionized NLP with attention mechanisms.",
            "RAG combines retrieval with generation for better answers.",
            "Vector databases store embeddings for semantic search.",
            "LLMs are large language models trained on massive text data.",
        ]


class AsyncAnswerSynthesizer:
    """Async answer synthesizer with streaming support."""

    def __init__(self):
        self.logger = logging.getLogger("AsyncAnswerSynthesizer")

    async def synthesize(self, query: str, documents: List[AsyncRetrievalResult]) -> str:
        """
        Synthesize answer from documents asynchronously.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            Synthesized answer
        """
        await asyncio.sleep(0.02)  # Simulate LLM call

        if not documents:
            return "I don't have enough information to answer that question."

        # Build context
        context = "\n".join([f"- {doc.text}" for doc in documents[:3]])

        # Synthesize answer
        answer = f"Based on the available information:\n\n{context}\n\nIn summary, {query.lower()} relates to the concepts mentioned above."

        return answer

    async def synthesize_stream(
        self, query: str, documents: List[AsyncRetrievalResult]
    ) -> AsyncIterator[str]:
        """
        Stream answer tokens asynchronously.

        Args:
            query: User query
            documents: Retrieved documents

        Yields:
            Answer tokens
        """
        # Synthesize full answer
        answer = await self.synthesize(query, documents)

        # Stream tokens
        words = answer.split()
        for word in words:
            await asyncio.sleep(0.01)  # Simulate token generation
            yield word + " "


class AsyncAgenticRAGPipeline:
    """
    Async-first agentic RAG pipeline for state-of-the-art performance.

    Key features:
    - Non-blocking I/O throughout
    - Concurrent retrieval and refinement
    - Streaming token generation
    - Parallel query expansion
    - Efficient resource utilization
    """

    def __init__(
        self,
        agent: Optional[Agent] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_iterations: int = 3,
        enable_streaming: bool = True,
    ):
        """
        Initialize async RAG pipeline.

        Args:
            agent: Optional agent instance
            tool_registry: Optional tool registry
            max_iterations: Max retrieval iterations
            enable_streaming: Enable streaming responses
        """
        self.agent = agent or Agent(name="AsyncRAGAgent", max_steps=10)
        self.tool_registry = tool_registry or get_tool_registry()
        self.max_iterations = max_iterations
        self.enable_streaming = enable_streaming

        # Initialize async components
        self.refinement_engine = AsyncQueryRefinement()
        self.retriever = AsyncIterativeRetriever(max_iterations=max_iterations)
        self.synthesizer = AsyncAnswerSynthesizer()
        self.memory = ConversationMemory(max_context_tokens=2048)
        self.reasoner = Reasoner(verbose=False)

        self.logger = logging.getLogger("AsyncAgenticRAGPipeline")

    async def process_query(self, query: str) -> AsyncRAGResponse:
        """
        Process query asynchronously through full pipeline.

        Args:
            query: User query

        Returns:
            RAG response with answer and metadata
        """
        import time

        start_time = time.time()

        # Add to memory
        self.memory.add_message("user", query)

        # Step 1: Concurrent reasoning and query expansion
        reasoning_task = asyncio.create_task(self._reason_about_query(query))
        expansion_task = asyncio.create_task(self.refinement_engine.expand_query(query))

        reasoning_trace, query_variants = await asyncio.gather(reasoning_task, expansion_task)

        # Step 2: Parallel retrieval across query variants
        retrieval_tasks = [self.retriever.retrieve(variant) for variant in query_variants]
        all_results = await asyncio.gather(*retrieval_tasks)

        # Merge and deduplicate results
        merged_results = self._merge_retrieval_results(all_results)

        # Step 3: Synthesize answer
        answer = await self.synthesizer.synthesize(query, merged_results)

        # Add to memory
        self.memory.add_message("assistant", answer)

        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        confidence = self._calculate_confidence(merged_results)

        response = AsyncRAGResponse(
            query=query,
            answer=answer,
            sources=merged_results,
            reasoning_trace=reasoning_trace,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={
                "query_variants": len(query_variants),
                "total_documents": len(merged_results),
            },
        )

        return response

    async def process_query_stream(self, query: str) -> AsyncIterator[str]:
        """
        Process query with streaming response.

        Args:
            query: User query

        Yields:
            Answer tokens
        """
        if not self.enable_streaming:
            response = await self.process_query(query)
            yield response.answer
            return

        # Add to memory
        self.memory.add_message("user", query)

        # Concurrent retrieval and reasoning
        reasoning_task = asyncio.create_task(self._reason_about_query(query))
        retrieval_task = asyncio.create_task(self.retriever.retrieve(query))

        _, documents = await asyncio.gather(reasoning_task, retrieval_task)

        # Stream answer
        answer_parts = []
        async for token in self.synthesizer.synthesize_stream(query, documents):
            answer_parts.append(token)
            yield token

        # Add complete answer to memory
        full_answer = "".join(answer_parts)
        self.memory.add_message("assistant", full_answer)

    async def _reason_about_query(self, query: str) -> str:
        """Async reasoning about query."""
        # Simulate async reasoning
        await asyncio.sleep(0.01)

        reasoning = self.reasoner.reason_about_query(query)
        return reasoning.get_reasoning_chain()

    def _merge_retrieval_results(
        self, results_list: List[List[AsyncRetrievalResult]]
    ) -> List[AsyncRetrievalResult]:
        """Merge and deduplicate retrieval results."""
        seen = set()
        merged = []

        for results in results_list:
            for result in results:
                if result.text not in seen:
                    seen.add(result.text)
                    merged.append(result)

        # Sort by score
        return sorted(merged, key=lambda r: r.score, reverse=True)[:10]

    def _calculate_confidence(self, documents: List[AsyncRetrievalResult]) -> float:
        """Calculate confidence from retrieval scores."""
        if not documents:
            return 0.0

        avg_score = sum(d.score for d in documents) / len(documents)
        return min(1.0, avg_score * 1.5)

    async def multi_turn_conversation(self, queries: List[str]) -> List[AsyncRAGResponse]:
        """
        Process multiple queries in conversation asynchronously.

        Args:
            queries: List of user queries

        Returns:
            List of responses
        """
        responses = []
        for query in queries:
            response = await self.process_query(query)
            responses.append(response)

        return responses

    def reset(self) -> None:
        """Reset pipeline state."""
        self.memory = ConversationMemory(max_context_tokens=2048)
        self.logger.info("Pipeline reset")


# Convenience functions
async def create_async_pipeline(**kwargs) -> AsyncAgenticRAGPipeline:
    """Create async RAG pipeline."""
    return AsyncAgenticRAGPipeline(**kwargs)


async def quick_query(query: str) -> AsyncRAGResponse:
    """Quick async query processing."""
    pipeline = AsyncAgenticRAGPipeline()
    return await pipeline.process_query(query)
