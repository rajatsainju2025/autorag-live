"""
Agentic RAG pipeline combining retrieval with agent reasoning.

Implements iterative retrieval, query refinement, and answer synthesis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from autorag_live.agent.base import Agent
from autorag_live.agent.memory import ConversationMemory
from autorag_live.agent.reasoning import Reasoner
from autorag_live.agent.tools import ToolRegistry, get_tool_registry


@dataclass
class RetrievalResult:
    """Single retrieval result."""

    document: str
    score: float
    source: str = ""  # e.g., "bm25", "dense", "hybrid"
    iterations: int = 0  # which iteration found this


@dataclass
class RAGResponse:
    """Response from agentic RAG pipeline."""

    query: str
    answer: str
    sources: List[RetrievalResult] = field(default_factory=list)
    reasoning_trace: Optional[str] = None
    iterations: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Export response as dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [
                {"document": s.document, "score": s.score, "source": s.source} for s in self.sources
            ],
            "iterations": self.iterations,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class QueryRefinementEngine:
    """Refines queries for better retrieval."""

    def __init__(self):
        """Initialize query refinement engine."""
        self.refinement_history: List[Tuple[str, str]] = []

    def refine_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Refine query based on context.

        Args:
            query: Original query
            context: Optional context from previous iterations

        Returns:
            Refined query
        """
        context = context or {}

        # Strategy 1: Expand with question keywords
        refined = query
        keywords = ["definition", "explanation", "example", "benefits", "process"]

        for keyword in keywords:
            if keyword in query.lower():
                refined = query
                break
        else:
            # Add relevant keyword
            refined = f"{query} explanation"

        # Strategy 2: Remove ambiguity
        if "it" in query.lower() or "that" in query.lower():
            if "previous_topic" in context:
                refined += f" {context['previous_topic']}"

        # Strategy 3: Add synonyms if retrieval was poor
        if context.get("retrieval_score", 0) < 0.5:
            refined = f"{query} similar related"

        self.refinement_history.append((query, refined))
        return refined


class IterativeRetriever:
    """Performs iterative retrieval with refinement."""

    def __init__(self, max_iterations: int = 3):
        """
        Initialize iterative retriever.

        Args:
            max_iterations: Max retrieval iterations
        """
        self.max_iterations = max_iterations
        self.logger = logging.getLogger("IterativeRetriever")

    def retrieve_iteratively(
        self,
        initial_query: str,
        refinement_engine: QueryRefinementEngine,
        retriever_fn,
        k: int = 5,
    ) -> Tuple[List[RetrievalResult], str]:
        """
        Perform iterative retrieval.

        Args:
            initial_query: Starting query
            refinement_engine: Query refinement engine
            retriever_fn: Retrieval function
            k: Top-k results per iteration

        Returns:
            (list of unique results, final query used)
        """
        all_results = {}  # doc -> best RetrievalResult
        current_query = initial_query
        best_score = 0.0

        for iteration in range(self.max_iterations):
            # Retrieve with current query
            try:
                docs = retriever_fn(current_query, k=k)
                for doc, score in docs:
                    doc_hash = hash(doc)
                    if doc_hash not in all_results or score > all_results[doc_hash].score:
                        all_results[doc_hash] = RetrievalResult(
                            document=doc, score=score, iterations=iteration
                        )
                        best_score = max(best_score, score)
                        self.logger.debug(
                            f"Iteration {iteration+1}: Retrieved '{doc[:50]}...' (score: {score:.3f})"
                        )
            except Exception as e:
                self.logger.warning(f"Retrieval failed in iteration {iteration}: {str(e)}")
                break

            # Decide if to refine and continue
            if best_score > 0.8 or iteration == self.max_iterations - 1:
                break

            # Refine query for next iteration
            current_query = refinement_engine.refine_query(
                current_query, {"retrieval_score": best_score}
            )
            self.logger.debug(f"Refined query for iteration {iteration+2}: {current_query}")

        return list(all_results.values()), current_query


class AnswerSynthesizer:
    """Synthesizes final answer from retrieved documents."""

    def __init__(self):
        """Initialize answer synthesizer."""
        self.logger = logging.getLogger("AnswerSynthesizer")

    def synthesize(
        self, query: str, documents: List[RetrievalResult], max_length: int = 500
    ) -> str:
        """
        Synthesize answer from documents.

        Args:
            query: Original query
            documents: Retrieved documents
            max_length: Max answer length

        Returns:
            Synthesized answer
        """
        if not documents:
            return "No relevant information found."

        # Simple synthesis: combine top results with relevance
        answer_parts = []

        # Add introduction
        answer_parts.append(f"Based on the retrieved information about '{query}':")

        # Add top documents
        for doc in documents[:3]:
            if doc.score > 0.5:
                snippet = doc.document[:100]
                answer_parts.append(f"\n- {snippet}...")

        # Add summary statement
        answer_parts.append("\nThese results provide relevant information to answer your question.")

        answer = "\n".join(answer_parts)

        # Limit length
        if len(answer) > max_length:
            answer = answer[:max_length] + "..."

        return answer


class AgenticRAGPipeline:
    """
    Complete agentic RAG pipeline.

    Combines agent reasoning, query refinement, iterative retrieval, and synthesis.
    """

    def __init__(
        self,
        agent: Optional[Agent] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_iterations: int = 3,
    ):
        """
        Initialize agentic RAG pipeline.

        Args:
            agent: Optional custom agent
            tool_registry: Optional custom tool registry
            max_iterations: Max iterations for retrieval
        """
        self.agent = agent or Agent(name="RAGAgent", max_steps=10)
        self.tool_registry = tool_registry or get_tool_registry()
        self.max_iterations = max_iterations

        # Initialize components
        self.refinement_engine = QueryRefinementEngine()
        self.iterative_retriever = IterativeRetriever(max_iterations=max_iterations)
        self.synthesizer = AnswerSynthesizer()
        self.memory = ConversationMemory(max_context_tokens=2048)
        self.reasoner = Reasoner(verbose=False)

        self.logger = logging.getLogger("AgenticRAGPipeline")

    def process_query(self, query: str) -> RAGResponse:
        """
        Process query through full agentic RAG pipeline.

        Args:
            query: User query

        Returns:
            RAG response with answer and sources
        """
        # Add to memory
        self.memory.add_message("user", query)

        # Step 1: Reason about query
        reasoning_trace = self.reasoner.reason_about_query(query)
        trace_str = reasoning_trace.get_reasoning_chain()
        self.logger.info(f"Reasoning:\n{trace_str}")

        # Step 2: Create retrieval action plan
        plan = self.reasoner.generate_action_plan(query)
        planned_actions = plan.get("action_sequence", "retrieve_documents")

        # Step 3: Iterative retrieval
        retrieved_docs = self._execute_retrieval(query)

        # Step 4: Synthesize answer
        answer = self.synthesizer.synthesize(query, retrieved_docs)

        # Add to memory
        self.memory.add_message("assistant", answer)

        # Build response
        response = RAGResponse(
            query=query,
            answer=answer,
            sources=retrieved_docs,
            reasoning_trace=trace_str,
            iterations=self.max_iterations,
            confidence=self._calculate_confidence(retrieved_docs),
            metadata={
                "planned_actions": planned_actions,
                "context_window": self.memory.get_conversation_stats(),
            },
        )

        return response

    def _execute_retrieval(self, query: str) -> List[RetrievalResult]:
        """Execute retrieval phase."""
        try:
            from autorag_live.retrievers import hybrid

            corpus = [
                "The sky is blue.",
                "The sun is bright.",
                "Machine learning enables computers to learn from data.",
                "Artificial intelligence is transforming industries.",
                "Python is a popular programming language.",
            ]

            def retriever_fn(q: str, k: int):
                results = hybrid.hybrid_retrieve(q, corpus, k=k)
                return results

            docs, final_query = self.iterative_retriever.retrieve_iteratively(
                query, self.refinement_engine, retriever_fn, k=5
            )

            return docs

        except Exception as e:
            self.logger.error(f"Retrieval failed: {str(e)}")
            return []

    def _calculate_confidence(self, documents: List[RetrievalResult]) -> float:
        """Calculate confidence based on retrieval scores."""
        if not documents:
            return 0.0

        avg_score = sum(d.score for d in documents) / len(documents)
        # Scale to 0-1
        return min(1.0, avg_score * 1.5)

    def multi_turn_conversation(self, queries: List[str]) -> List[RAGResponse]:
        """
        Process multiple queries in conversation.

        Args:
            queries: List of user queries

        Returns:
            List of responses
        """
        responses = []
        for query in queries:
            response = self.process_query(query)
            responses.append(response)

        return responses

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation so far."""
        return {
            "memory_stats": self.memory.get_conversation_stats(),
            "reasoning_summary": self.reasoner.get_reasoning_summary(),
            "num_queries_processed": len([m for m in self.memory.messages if m.role == "user"]),
        }

    def reset(self) -> None:
        """Reset pipeline for new conversation."""
        self.agent.reset()
        self.memory.clear()
        self.refinement_engine.refinement_history.clear()
