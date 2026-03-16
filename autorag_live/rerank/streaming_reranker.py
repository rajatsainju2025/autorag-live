"""Streaming reranker with progressive rank updates."""

import asyncio
from typing import Any, AsyncIterator, Dict, List


class StreamingReranker:
    """Reranker that progressively updates ranks via async streaming."""

    def __init__(self):
        """Initialize streaming reranker."""
        self.candidate_scores: Dict[str, float] = {}

    async def rerank_stream(
        self, query: str, candidates: List[Dict[str, Any]], k: int = 10
    ) -> AsyncIterator[Dict[str, Any]]:
        """Rerank candidates with progressive streaming updates.

        Yields candidates as they are scored, allowing progressive
        refinement and token-by-token rank updates.

        Args:
            query: Query string
            candidates: List of candidate documents
            k: Number of top candidates to stream

        Yields:
            Progressively scored candidate documents
        """
        if not candidates:
            return

        # Reset scores for this batch
        self.candidate_scores = {}

        # Simulate progressive scoring - in real implementation would
        # compute embeddings incrementally and yield as they're computed
        for i, candidate in enumerate(candidates[:k]):
            # Simulate some async work (e.g., embedding computation, similarity)
            await asyncio.sleep(0.001)

            # Assign a dummy score based on position
            score = 1.0 - (i * 0.1)
            self.candidate_scores[candidate.get("id", str(i))] = score

            # Yield with updated score
            result = candidate.copy()
            result["rerank_score"] = score
            yield result

    async def stream_with_callback(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        on_candidate: Any = None,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Rerank with callback on each candidate update.

        Args:
            query: Query string
            candidates: List of candidates
            on_candidate: Callback function for each candidate
            k: Number of results

        Returns:
            Final reranked candidates
        """
        results = []
        async for candidate in self.rerank_stream(query, candidates, k):
            if on_candidate:
                await on_candidate(candidate)
            results.append(candidate)
        return results

    def get_score(self, candidate_id: str) -> float:
        """Get current score for a candidate.

        Args:
            candidate_id: ID of candidate

        Returns:
            Current score or 0.0 if not found
        """
        return self.candidate_scores.get(candidate_id, 0.0)
