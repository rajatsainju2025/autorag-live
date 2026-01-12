"""
Advanced Retrieval Strategies for State-of-the-Art RAG.

Implements modern retrieval techniques:
- ColBERT-style late interaction
- Speculative retrieval with prefetching
- Iterative retrieval with feedback
- Multi-vector retrieval
- Agentic retrieval with query routing

References:
- ColBERT (Khattab & Zaharia, 2020)
- Speculative RAG (Cho et al., 2024)
- RAPTOR (Sarthi et al., 2024)

Example:
    >>> retriever = AdvancedRetrieverPipeline(
    ...     dense_retriever=dense,
    ...     sparse_retriever=sparse,
    ...     reranker=cross_encoder,
    ... )
    >>> docs = await retriever.retrieve("complex query", strategy="iterative")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Protocol for retriever implementations."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List["RetrievedDocument"]:
        """Retrieve documents for query."""
        ...

    @property
    def name(self) -> str:
        """Retriever identifier."""
        ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> List[float]:
        """Generate single embedding."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings."""
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Protocol for reranking models."""

    async def rerank(
        self,
        query: str,
        documents: List["RetrievedDocument"],
        top_k: int = 10,
    ) -> List["RetrievedDocument"]:
        """Rerank documents for query."""
        ...


@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol for LLM interactions."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response."""
        ...


# =============================================================================
# Data Structures
# =============================================================================


class RetrievalStrategy(Enum):
    """Available retrieval strategies."""

    SINGLE = auto()  # Single retriever
    HYBRID = auto()  # Dense + sparse fusion
    ITERATIVE = auto()  # Multi-round with feedback
    SPECULATIVE = auto()  # Parallel speculation
    AGENTIC = auto()  # LLM-guided routing
    COLBERT = auto()  # Late interaction
    MULTI_VECTOR = auto()  # Multi-representation


@dataclass
class RetrievedDocument:
    """A retrieved document with metadata."""

    id: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    token_embeddings: Optional[List[List[float]]] = None
    source_retriever: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source_retriever,
        }


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""

    query: str
    documents: List[RetrievedDocument]
    strategy: RetrievalStrategy
    latency_ms: float = 0.0
    iterations: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_k(self) -> int:
        """Number of documents returned."""
        return len(self.documents)

    def get_context(self, max_docs: Optional[int] = None) -> str:
        """Get concatenated context string."""
        docs = self.documents[:max_docs] if max_docs else self.documents
        return "\n\n".join(f"[{i+1}] {doc.content}" for i, doc in enumerate(docs))


@dataclass
class QueryAnalysis:
    """Analysis of query for routing."""

    original_query: str
    query_type: str = "factual"
    complexity: str = "simple"
    domains: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    requires_multi_hop: bool = False
    suggested_strategy: RetrievalStrategy = RetrievalStrategy.SINGLE
    expanded_queries: List[str] = field(default_factory=list)


# =============================================================================
# Query Analyzer
# =============================================================================


class QueryAnalyzer:
    """Analyze queries to determine optimal retrieval strategy."""

    def __init__(self, llm: Optional[LLMProtocol] = None):
        """Initialize analyzer."""
        self.llm = llm

    async def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze query characteristics.

        Args:
            query: Input query

        Returns:
            QueryAnalysis with recommendations
        """
        analysis = QueryAnalysis(original_query=query)

        # Basic heuristics
        analysis.query_type = self._detect_type(query)
        analysis.complexity = self._assess_complexity(query)
        analysis.entities = self._extract_entities(query)
        analysis.requires_multi_hop = self._needs_multi_hop(query)

        # Select strategy based on analysis
        analysis.suggested_strategy = self._recommend_strategy(analysis)

        # Generate expanded queries if LLM available
        if self.llm and analysis.complexity in ("moderate", "complex"):
            analysis.expanded_queries = await self._expand_query(query)

        return analysis

    def _detect_type(self, query: str) -> str:
        """Detect query type."""
        query_lower = query.lower()
        if any(w in query_lower for w in ["how", "explain", "why"]):
            return "explanatory"
        if any(w in query_lower for w in ["compare", "difference", "vs"]):
            return "comparative"
        if any(w in query_lower for w in ["list", "what are", "examples"]):
            return "enumerative"
        return "factual"

    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity."""
        words = query.split()
        if len(words) > 20 or query.count(",") > 2:
            return "complex"
        if len(words) > 10:
            return "moderate"
        return "simple"

    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential named entities."""
        # Simple heuristic: capitalized words
        words = query.split()
        entities = [w for w in words if w[0].isupper() and len(w) > 2]
        return entities

    def _needs_multi_hop(self, query: str) -> bool:
        """Check if query requires multi-hop reasoning."""
        indicators = ["and", "then", "after", "before", "relationship", "connected"]
        return any(ind in query.lower() for ind in indicators)

    def _recommend_strategy(self, analysis: QueryAnalysis) -> RetrievalStrategy:
        """Recommend retrieval strategy."""
        if analysis.requires_multi_hop:
            return RetrievalStrategy.ITERATIVE
        if analysis.complexity == "complex":
            return RetrievalStrategy.SPECULATIVE
        if analysis.query_type == "comparative":
            return RetrievalStrategy.HYBRID
        return RetrievalStrategy.SINGLE

    async def _expand_query(self, query: str) -> List[str]:
        """Expand query into multiple search queries."""
        if not self.llm:
            return []

        prompt = f"""Generate 3 alternative search queries for: "{query}"
Return only the queries, one per line."""

        response = await self.llm.generate(prompt, temperature=0.7)
        queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
        return queries[:3]


# =============================================================================
# ColBERT-style Late Interaction Retriever
# =============================================================================


class ColBERTRetriever:
    """
    ColBERT-style retriever with late interaction scoring.

    Uses token-level embeddings and MaxSim scoring for fine-grained matching.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        token_embedder: Optional[EmbedderProtocol] = None,
        index: Optional[Any] = None,
    ):
        """Initialize ColBERT retriever."""
        self.embedder = embedder
        self.token_embedder = token_embedder or embedder
        self._documents: Dict[str, RetrievedDocument] = {}
        self._token_embeddings: Dict[str, List[List[float]]] = {}

    @property
    def name(self) -> str:
        return "colbert"

    async def index_document(self, doc: RetrievedDocument) -> None:
        """Index a document with token embeddings."""
        # Generate token-level embeddings
        tokens = doc.content.split()  # Simple tokenization
        token_texts = [" ".join(tokens[i : i + 3]) for i in range(0, len(tokens), 3)]

        if token_texts:
            token_embs = await self.token_embedder.embed_batch(token_texts)
            self._token_embeddings[doc.id] = token_embs
            doc.token_embeddings = token_embs

        self._documents[doc.id] = doc

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievedDocument]:
        """Retrieve using MaxSim scoring."""
        # Generate query token embeddings
        query_tokens = query.split()
        query_token_texts = [
            " ".join(query_tokens[i : i + 3]) for i in range(0, len(query_tokens), 3)
        ]

        if not query_token_texts:
            query_token_texts = [query]

        query_token_embs = await self.token_embedder.embed_batch(query_token_texts)

        # Score all documents
        scored_docs = []
        for doc_id, doc in self._documents.items():
            doc_token_embs = self._token_embeddings.get(doc_id, [])
            if not doc_token_embs:
                continue

            # MaxSim scoring
            score = self._maxsim_score(query_token_embs, doc_token_embs)
            scored_docs.append((score, doc))

        # Sort and return top_k
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored_docs[:top_k]:
            result = RetrievedDocument(
                id=doc.id,
                content=doc.content,
                score=score,
                metadata=doc.metadata,
                source_retriever=self.name,
            )
            results.append(result)

        return results

    def _maxsim_score(
        self,
        query_embs: List[List[float]],
        doc_embs: List[List[float]],
    ) -> float:
        """Compute MaxSim score between query and document."""
        if not query_embs or not doc_embs:
            return 0.0

        total_score = 0.0
        for q_emb in query_embs:
            max_sim = max(self._cosine_similarity(q_emb, d_emb) for d_emb in doc_embs)
            total_score += max_sim

        return total_score / len(query_embs)

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# =============================================================================
# Speculative Retrieval
# =============================================================================


class SpeculativeRetriever:
    """
    Speculative retrieval with parallel query execution.

    Generates multiple query variations and retrieves in parallel,
    then aggregates results.
    """

    def __init__(
        self,
        base_retriever: RetrieverProtocol,
        llm: Optional[LLMProtocol] = None,
        num_speculations: int = 3,
    ):
        """Initialize speculative retriever."""
        self.base_retriever = base_retriever
        self.llm = llm
        self.num_speculations = num_speculations

    @property
    def name(self) -> str:
        return "speculative"

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievedDocument]:
        """Retrieve with speculative queries."""
        # Generate speculative queries
        queries = [query]
        if self.llm:
            speculative = await self._generate_speculations(query)
            queries.extend(speculative)

        # Retrieve in parallel
        tasks = [self.base_retriever.retrieve(q, top_k=top_k) for q in queries]
        results_list = await asyncio.gather(*tasks)

        # Aggregate results using RRF
        aggregated = self._aggregate_rrf(results_list)

        return aggregated[:top_k]

    async def _generate_speculations(self, query: str) -> List[str]:
        """Generate speculative query variations."""
        if not self.llm:
            return []

        prompt = f"""Generate {self.num_speculations} alternative phrasings of this search query.
Each should capture slightly different aspects or phrasings.

Original: {query}

Alternatives (one per line):"""

        response = await self.llm.generate(prompt, temperature=0.8)
        queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
        return queries[: self.num_speculations]

    def _aggregate_rrf(
        self,
        results_list: List[List[RetrievedDocument]],
        k: int = 60,
    ) -> List[RetrievedDocument]:
        """Aggregate results using Reciprocal Rank Fusion."""
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, RetrievedDocument] = {}

        for results in results_list:
            for rank, doc in enumerate(results, 1):
                doc_id = doc.id
                rrf_score = 1.0 / (k + rank)

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_objects[doc_id] = doc

                doc_scores[doc_id] += rrf_score

        # Sort by aggregated score
        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

        results = []
        for doc_id in sorted_ids:
            doc = doc_objects[doc_id]
            results.append(
                RetrievedDocument(
                    id=doc.id,
                    content=doc.content,
                    score=doc_scores[doc_id],
                    metadata=doc.metadata,
                    source_retriever=self.name,
                )
            )

        return results


# =============================================================================
# Iterative Retrieval with Feedback
# =============================================================================


class IterativeRetriever:
    """
    Iterative retrieval with query refinement.

    Performs multiple retrieval rounds, refining the query based on results.
    """

    def __init__(
        self,
        base_retriever: RetrieverProtocol,
        llm: LLMProtocol,
        max_iterations: int = 3,
        docs_per_iteration: int = 5,
    ):
        """Initialize iterative retriever."""
        self.base_retriever = base_retriever
        self.llm = llm
        self.max_iterations = max_iterations
        self.docs_per_iteration = docs_per_iteration

    @property
    def name(self) -> str:
        return "iterative"

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievedDocument]:
        """Perform iterative retrieval with refinement."""
        all_docs: Dict[str, RetrievedDocument] = {}
        current_query = query

        for iteration in range(self.max_iterations):
            # Retrieve with current query
            docs = await self.base_retriever.retrieve(
                current_query,
                top_k=self.docs_per_iteration,
            )

            # Add to collection
            for doc in docs:
                if doc.id not in all_docs:
                    all_docs[doc.id] = doc

            # Check if we have enough diverse results
            if len(all_docs) >= top_k * 2:
                break

            # Refine query based on results
            current_query = await self._refine_query(
                original_query=query,
                current_query=current_query,
                retrieved_docs=docs,
                iteration=iteration,
            )

        # Score and rank all collected documents
        scored_docs = await self._score_documents(query, list(all_docs.values()))

        return scored_docs[:top_k]

    async def _refine_query(
        self,
        original_query: str,
        current_query: str,
        retrieved_docs: List[RetrievedDocument],
        iteration: int,
    ) -> str:
        """Refine query based on retrieved documents."""
        doc_summaries = "\n".join(f"- {doc.content[:200]}..." for doc in retrieved_docs[:3])

        prompt = f"""Based on the original question and retrieved documents, generate a refined search query.

Original question: {original_query}
Current query: {current_query}

Retrieved documents:
{doc_summaries}

Generate a refined query that might find additional relevant information:"""

        response = await self.llm.generate(prompt, temperature=0.5)
        return response.strip()

    async def _score_documents(
        self,
        query: str,
        documents: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """Score documents for relevance to original query."""
        # Simple scoring based on existing scores
        # Could be enhanced with cross-encoder
        scored = sorted(documents, key=lambda d: d.score, reverse=True)

        for i, doc in enumerate(scored):
            doc.source_retriever = self.name

        return scored


# =============================================================================
# Agentic Retrieval with LLM Routing
# =============================================================================


class AgenticRetriever:
    """
    Agentic retrieval with LLM-guided routing and strategy selection.

    Uses an LLM to analyze queries and select optimal retrieval strategy.
    """

    def __init__(
        self,
        retrievers: Dict[str, RetrieverProtocol],
        llm: LLMProtocol,
        reranker: Optional[RerankerProtocol] = None,
    ):
        """Initialize agentic retriever."""
        self.retrievers = retrievers
        self.llm = llm
        self.reranker = reranker
        self.analyzer = QueryAnalyzer(llm)

    @property
    def name(self) -> str:
        return "agentic"

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[RetrievedDocument]:
        """Retrieve using LLM-guided strategy."""
        # Analyze query
        analysis = await self.analyzer.analyze(query)

        # Select retrievers based on analysis
        selected = await self._select_retrievers(analysis)

        # Execute retrieval
        if len(selected) == 1:
            docs = await selected[0].retrieve(query, top_k=top_k * 2)
        else:
            # Parallel retrieval with multiple retrievers
            tasks = [r.retrieve(query, top_k=top_k) for r in selected]
            results_list = await asyncio.gather(*tasks)
            docs = self._merge_results(results_list)

        # Rerank if available
        if self.reranker:
            docs = await self.reranker.rerank(query, docs, top_k=top_k)

        return docs[:top_k]

    async def _select_retrievers(
        self,
        analysis: QueryAnalysis,
    ) -> List[RetrieverProtocol]:
        """Select retrievers based on query analysis."""
        retriever_names = list(self.retrievers.keys())

        prompt = f"""Select the best retriever(s) for this query.

Query: {analysis.original_query}
Query type: {analysis.query_type}
Complexity: {analysis.complexity}

Available retrievers: {retriever_names}

Return retriever names (comma-separated if multiple):"""

        response = await self.llm.generate(prompt, temperature=0.3)

        selected_names = [n.strip() for n in response.strip().split(",")]
        selected = [self.retrievers[n] for n in selected_names if n in self.retrievers]

        # Fallback to first retriever
        if not selected:
            selected = [list(self.retrievers.values())[0]]

        return selected

    def _merge_results(
        self,
        results_list: List[List[RetrievedDocument]],
    ) -> List[RetrievedDocument]:
        """Merge results from multiple retrievers using RRF."""
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, RetrievedDocument] = {}

        for results in results_list:
            for rank, doc in enumerate(results, 1):
                rrf_score = 1.0 / (60 + rank)
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = 0.0
                    doc_objects[doc.id] = doc
                doc_scores[doc.id] += rrf_score

        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

        return [
            RetrievedDocument(
                id=doc_objects[did].id,
                content=doc_objects[did].content,
                score=doc_scores[did],
                metadata=doc_objects[did].metadata,
                source_retriever=self.name,
            )
            for did in sorted_ids
        ]


# =============================================================================
# Advanced Retrieval Pipeline
# =============================================================================


class AdvancedRetrieverPipeline:
    """
    Unified retrieval pipeline supporting multiple strategies.

    Combines different retrievers and provides a single interface
    for state-of-the-art retrieval.
    """

    def __init__(
        self,
        dense_retriever: Optional[RetrieverProtocol] = None,
        sparse_retriever: Optional[RetrieverProtocol] = None,
        reranker: Optional[RerankerProtocol] = None,
        llm: Optional[LLMProtocol] = None,
        embedder: Optional[EmbedderProtocol] = None,
        default_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
    ):
        """Initialize pipeline."""
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker
        self.llm = llm
        self.embedder = embedder
        self.default_strategy = default_strategy

        # Initialize advanced retrievers
        self._speculative: Optional[SpeculativeRetriever] = None
        self._iterative: Optional[IterativeRetriever] = None
        self._agentic: Optional[AgenticRetriever] = None
        self._colbert: Optional[ColBERTRetriever] = None

        self._setup_retrievers()

    def _setup_retrievers(self) -> None:
        """Setup advanced retriever instances."""
        base_retriever = self.dense_retriever or self.sparse_retriever

        if base_retriever and self.llm:
            self._speculative = SpeculativeRetriever(base_retriever, self.llm)
            self._iterative = IterativeRetriever(base_retriever, self.llm)

        retrievers = {}
        if self.dense_retriever:
            retrievers["dense"] = self.dense_retriever
        if self.sparse_retriever:
            retrievers["sparse"] = self.sparse_retriever

        if retrievers and self.llm:
            self._agentic = AgenticRetriever(retrievers, self.llm, self.reranker)

        if self.embedder:
            self._colbert = ColBERTRetriever(self.embedder)

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        strategy: Optional[RetrievalStrategy] = None,
    ) -> RetrievalResult:
        """
        Retrieve documents using specified or default strategy.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            strategy: Retrieval strategy (uses default if None)

        Returns:
            RetrievalResult with documents and metadata
        """
        import time

        start_time = time.time()
        selected_strategy = strategy or self.default_strategy

        documents: List[RetrievedDocument] = []

        if selected_strategy == RetrievalStrategy.SINGLE:
            documents = await self._retrieve_single(query, top_k)

        elif selected_strategy == RetrievalStrategy.HYBRID:
            documents = await self._retrieve_hybrid(query, top_k)

        elif selected_strategy == RetrievalStrategy.SPECULATIVE:
            if self._speculative:
                documents = await self._speculative.retrieve(query, top_k)
            else:
                documents = await self._retrieve_hybrid(query, top_k)

        elif selected_strategy == RetrievalStrategy.ITERATIVE:
            if self._iterative:
                documents = await self._iterative.retrieve(query, top_k)
            else:
                documents = await self._retrieve_hybrid(query, top_k)

        elif selected_strategy == RetrievalStrategy.AGENTIC:
            if self._agentic:
                documents = await self._agentic.retrieve(query, top_k)
            else:
                documents = await self._retrieve_hybrid(query, top_k)

        elif selected_strategy == RetrievalStrategy.COLBERT:
            if self._colbert:
                documents = await self._colbert.retrieve(query, top_k)
            else:
                documents = await self._retrieve_single(query, top_k)

        # Apply reranker if available
        if self.reranker and documents:
            documents = await self.reranker.rerank(query, documents, top_k)

        latency = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            documents=documents,
            strategy=selected_strategy,
            latency_ms=latency,
        )

    async def _retrieve_single(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievedDocument]:
        """Single retriever execution."""
        retriever = self.dense_retriever or self.sparse_retriever
        if retriever:
            return await retriever.retrieve(query, top_k)
        return []

    async def _retrieve_hybrid(
        self,
        query: str,
        top_k: int,
        alpha: float = 0.5,
    ) -> List[RetrievedDocument]:
        """Hybrid retrieval with dense + sparse."""
        results = []

        # Parallel retrieval
        tasks = []
        if self.dense_retriever:
            tasks.append(self.dense_retriever.retrieve(query, top_k))
        if self.sparse_retriever:
            tasks.append(self.sparse_retriever.retrieve(query, top_k))

        if not tasks:
            return []

        results_list = await asyncio.gather(*tasks)

        # RRF fusion
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, RetrievedDocument] = {}

        for results in results_list:
            for rank, doc in enumerate(results, 1):
                rrf_score = 1.0 / (60 + rank)
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = 0.0
                    doc_objects[doc.id] = doc
                doc_scores[doc.id] += rrf_score

        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

        return [
            RetrievedDocument(
                id=doc_objects[did].id,
                content=doc_objects[did].content,
                score=doc_scores[did],
                metadata=doc_objects[did].metadata,
                source_retriever="hybrid",
            )
            for did in sorted_ids[:top_k]
        ]

    async def auto_retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Automatically select and execute best strategy for query.

        Args:
            query: Search query
            top_k: Number of documents

        Returns:
            RetrievalResult with documents
        """
        analyzer = QueryAnalyzer(self.llm)
        analysis = await analyzer.analyze(query)

        return await self.retrieve(
            query,
            top_k=top_k,
            strategy=analysis.suggested_strategy,
        )
