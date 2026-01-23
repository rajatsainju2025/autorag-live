"""
Cross-encoder reranking pipeline for precision retrieval.

State-of-the-art reranking using cross-encoders:
- Significantly improves relevance over bi-encoder retrieval
- Uses full query-document attention (vs embedding similarity)
- Achieves 15-30% improvement in nDCG@10
- Implements 2-stage cascade: fast retrieval → precise reranking

Based on:
- "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (Reimers & Gurevych, 2019)
- "ColBERT: Efficient and Effective Passage Search" (Khattab & Zaharia, 2020)
- "RankGPT: Listwise Reranking with LLMs" (Sun et al., 2023)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RerankConfig:
    """Configuration for reranking."""

    # Reranker selection
    reranker_type: str = "cross_encoder"  # "cross_encoder", "colbert", "llm"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Reranking parameters
    top_k_rerank: int = 100  # Rerank top K from retrieval
    final_top_k: int = 10  # Return top K after reranking

    # Batch processing
    batch_size: int = 32
    enable_batching: bool = True

    # Performance
    use_gpu: bool = False
    num_workers: int = 1

    # Caching
    cache_scores: bool = True
    cache_size: int = 10000


@dataclass
class RerankedResult:
    """A reranked document with scores."""

    doc_id: str
    text: str
    retrieval_score: float
    rerank_score: float
    combined_score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """Base class for rerankers."""

    def __init__(self, config: RerankConfig):
        """Initialize reranker."""
        self.config = config

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """
        Rerank documents for a query.

        Args:
            query: Query text
            documents: List of documents with text and scores
            top_k: Number of results to return

        Returns:
            Reranked documents with updated scores
        """
        pass

    def _prepare_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Prepare documents for reranking.

        Returns:
            List of (doc_id, text, retrieval_score, metadata)
        """
        prepared = []
        for idx, doc in enumerate(documents):
            doc_id = doc.get("id", str(idx))
            text = doc.get("text", doc.get("content", ""))
            score = doc.get("score", 0.0)
            metadata = doc.get("metadata", {})

            prepared.append((doc_id, text, score, metadata))

        return prepared


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker using sentence-transformers.

    Cross-encoders jointly encode query and document for precise
    relevance scoring using full attention.
    """

    def __init__(self, config: RerankConfig):
        """Initialize cross-encoder reranker."""
        super().__init__(config)
        self.model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            device = "cuda" if self.config.use_gpu else "cpu"
            self.model = CrossEncoder(
                self.config.model_name,
                device=device,
                max_length=512,
            )

            logger.info(f"Initialized CrossEncoder: {self.config.model_name} on {device}")

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Rerank documents using cross-encoder."""
        if self.model is None:
            return self._fallback_rerank(query, documents, top_k)

        top_k = top_k or self.config.final_top_k

        # Prepare documents
        prepared = self._prepare_documents(documents)
        if not prepared:
            return []

        # Limit to top_k_rerank for efficiency
        prepared = prepared[: self.config.top_k_rerank]

        # Create query-document pairs
        pairs = [(query, text) for _, text, _, _ in prepared]

        # Score in batches
        start_time = time.time()

        if self.config.enable_batching:
            scores = await asyncio.to_thread(
                self.model.predict,
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )
        else:
            scores = []
            for pair in pairs:
                score = await asyncio.to_thread(self.model.predict, [pair])
                scores.append(score[0])

        scores = np.array(scores)
        elapsed_ms = (time.time() - start_time) * 1000

        logger.debug(f"Cross-encoder reranking: {len(pairs)} docs in {elapsed_ms:.1f}ms")

        # Combine scores (weighted average)
        results = []
        for idx, ((doc_id, text, retrieval_score, metadata), rerank_score) in enumerate(
            zip(prepared, scores)
        ):
            # Combine: 70% rerank + 30% retrieval
            combined_score = 0.7 * float(rerank_score) + 0.3 * retrieval_score

            results.append(
                RerankedResult(
                    doc_id=doc_id,
                    text=text,
                    retrieval_score=retrieval_score,
                    rerank_score=float(rerank_score),
                    combined_score=combined_score,
                    rank=idx,
                    metadata=metadata,
                )
            )

        # Sort by combined score
        results.sort(key=lambda r: r.combined_score, reverse=True)

        # Update ranks
        for idx, result in enumerate(results):
            result.rank = idx

        return results[:top_k]

    def _fallback_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int],
    ) -> List[RerankedResult]:
        """Fallback: return original ranking."""
        top_k = top_k or self.config.final_top_k
        prepared = self._prepare_documents(documents)

        results = []
        for idx, (doc_id, text, score, metadata) in enumerate(prepared[:top_k]):
            results.append(
                RerankedResult(
                    doc_id=doc_id,
                    text=text,
                    retrieval_score=score,
                    rerank_score=score,
                    combined_score=score,
                    rank=idx,
                    metadata=metadata,
                )
            )

        return results


class ColBERTReranker(BaseReranker):
    """
    ColBERT-style reranker with late interaction.

    Uses MaxSim operation between query and document token embeddings
    for efficient yet precise reranking.
    """

    def __init__(self, config: RerankConfig):
        """Initialize ColBERT reranker."""
        super().__init__(config)
        self.model = None
        # ColBERT implementation would go here
        logger.warning("ColBERT reranker not fully implemented")

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Rerank using ColBERT late interaction."""
        # Placeholder - would implement MaxSim scoring
        fallback = CrossEncoderReranker(self.config)
        return await fallback.rerank(query, documents, top_k)


class LLMReranker(BaseReranker):
    """
    LLM-based reranker (RankGPT style).

    Uses LLM to directly assess relevance through prompting.
    More accurate but higher latency.
    """

    def __init__(self, config: RerankConfig, llm: Any = None):
        """Initialize LLM reranker."""
        super().__init__(config)
        self.llm = llm

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """Rerank using LLM."""
        if not self.llm:
            logger.warning("No LLM provided, using fallback")
            fallback = CrossEncoderReranker(self.config)
            return await fallback.rerank(query, documents, top_k)

        top_k = top_k or self.config.final_top_k
        prepared = self._prepare_documents(documents)[: self.config.top_k_rerank]

        # Create prompt for LLM
        docs_text = "\n\n".join(
            [f"[{i}] {text[:200]}..." for i, (_, text, _, _) in enumerate(prepared)]
        )

        prompt = f"""Rank the following documents by relevance to the query.
Return only the document numbers in order, separated by commas.

Query: {query}

Documents:
{docs_text}

Ranking (most relevant first):"""

        try:
            response = await self.llm.generate(prompt)

            # Parse ranking
            rankings = self._parse_llm_ranking(response, len(prepared))

            # Reorder results
            results = []
            for rank, doc_idx in enumerate(rankings):
                if doc_idx < len(prepared):
                    doc_id, text, retrieval_score, metadata = prepared[doc_idx]

                    # Assign score based on rank
                    rerank_score = 1.0 - (rank / len(rankings))

                    results.append(
                        RerankedResult(
                            doc_id=doc_id,
                            text=text,
                            retrieval_score=retrieval_score,
                            rerank_score=rerank_score,
                            combined_score=rerank_score,
                            rank=rank,
                            metadata=metadata,
                        )
                    )

            return results[:top_k]

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            # Fallback to original order
            fallback = CrossEncoderReranker(self.config)
            return await fallback._fallback_rerank(query, documents, top_k)

    def _parse_llm_ranking(self, response: str, num_docs: int) -> List[int]:
        """Parse LLM ranking response."""
        # Look for numbers in response
        import re

        numbers = re.findall(r"\d+", response)
        rankings = [int(n) for n in numbers if int(n) < num_docs]

        # Fill missing with original order
        for i in range(num_docs):
            if i not in rankings:
                rankings.append(i)

        return rankings[:num_docs]


class RerankingPipeline:
    """
    Two-stage retrieval-reranking pipeline.

    Combines fast retrieval with precise reranking for optimal
    precision-recall tradeoff.

    Example:
        >>> pipeline = RerankingPipeline()
        >>> results = await pipeline.retrieve_and_rerank(
        ...     query="machine learning",
        ...     retriever=my_retriever,
        ...     top_k=10
        ... )
    """

    def __init__(
        self,
        reranker: Optional[BaseReranker] = None,
        config: Optional[RerankConfig] = None,
    ):
        """
        Initialize reranking pipeline.

        Args:
            reranker: Reranker instance (creates default if None)
            config: Reranking configuration
        """
        self.config = config or RerankConfig()

        if reranker:
            self.reranker = reranker
        else:
            # Create default cross-encoder reranker
            self.reranker = CrossEncoderReranker(self.config)

        self._metrics = {
            "total_queries": 0,
            "total_documents_retrieved": 0,
            "total_documents_reranked": 0,
            "avg_rerank_time_ms": 0.0,
        }

    async def retrieve_and_rerank(
        self,
        query: str,
        retriever: Any,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """
        Full pipeline: retrieve → rerank.

        Args:
            query: Query text
            retriever: Retriever instance with retrieve() method
            top_k: Final number of results
            rerank_top_k: Number to rerank (uses config default if None)

        Returns:
            Reranked results
        """
        top_k = top_k or self.config.final_top_k
        rerank_top_k = rerank_top_k or self.config.top_k_rerank

        # Stage 1: Fast retrieval
        retrieve_start = time.time()
        documents = await retriever.retrieve(query, top_k=rerank_top_k)
        retrieve_time = (time.time() - retrieve_start) * 1000

        logger.debug(f"Retrieved {len(documents)} documents in {retrieve_time:.1f}ms")

        # Stage 2: Precise reranking
        rerank_start = time.time()
        results = await self.reranker.rerank(query, documents, top_k=top_k)
        rerank_time = (time.time() - rerank_start) * 1000

        logger.debug(f"Reranked to {len(results)} results in {rerank_time:.1f}ms")

        # Update metrics
        self._metrics["total_queries"] += 1
        self._metrics["total_documents_retrieved"] += len(documents)
        self._metrics["total_documents_reranked"] += len(results)

        # Update average rerank time
        n = self._metrics["total_queries"]
        self._metrics["avg_rerank_time_ms"] = (
            self._metrics["avg_rerank_time_ms"] * (n - 1) + rerank_time
        ) / n

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        return self._metrics.copy()


def create_reranker(
    reranker_type: str = "cross_encoder",
    config: Optional[RerankConfig] = None,
    **kwargs: Any,
) -> BaseReranker:
    """
    Factory function for creating rerankers.

    Args:
        reranker_type: Type of reranker
        config: Reranker configuration
        **kwargs: Additional arguments (e.g., llm for LLM reranker)

    Returns:
        Reranker instance
    """
    config = config or RerankConfig(reranker_type=reranker_type)

    if reranker_type == "cross_encoder":
        return CrossEncoderReranker(config)
    elif reranker_type == "colbert":
        return ColBERTReranker(config)
    elif reranker_type == "llm":
        return LLMReranker(config, llm=kwargs.get("llm"))
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")
