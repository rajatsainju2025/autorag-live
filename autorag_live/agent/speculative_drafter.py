"""Speculative answer drafter for agentic RAG.

Runs a fast draft model in parallel with retrieval. If high-quality evidence
arrives before draft finalization, the draft is refined; otherwise the system
falls back to a guarded quick response path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, List

DraftFn = Callable[[str], Awaitable[str]]
RefineFn = Callable[[str, List[str], str], Awaitable[str]]
RetrieveFn = Callable[[str], Awaitable[List[str]]]


@dataclass
class SpeculativeResult:
    answer: str
    used_refinement: bool
    retrieval_count: int
    latency_mode: str


class SpeculativeDrafter:
    """Latency-first drafting with evidence-aware refinement."""

    def __init__(
        self,
        draft_fn: DraftFn,
        retrieve_fn: RetrieveFn,
        refine_fn: RefineFn,
        retrieval_deadline_s: float = 1.5,
    ) -> None:
        self._draft_fn = draft_fn
        self._retrieve_fn = retrieve_fn
        self._refine_fn = refine_fn
        self._deadline = retrieval_deadline_s

    async def generate(self, query: str) -> SpeculativeResult:
        """Generate speculative answer with optional retrieval refinement."""
        draft_task = asyncio.create_task(self._draft_fn(query))
        retrieval_task = asyncio.create_task(self._retrieve_fn(query))

        try:
            passages = await asyncio.wait_for(retrieval_task, timeout=self._deadline)
            draft = await draft_task
            if passages:
                refined = await self._refine_fn(query, passages, draft)
                return SpeculativeResult(
                    answer=refined,
                    used_refinement=True,
                    retrieval_count=len(passages),
                    latency_mode="retrieve_then_refine",
                )
            return SpeculativeResult(
                answer=draft,
                used_refinement=False,
                retrieval_count=0,
                latency_mode="draft_only_no_evidence",
            )
        except TimeoutError:
            # Retrieval too slow: return draft for latency SLO.
            draft = await draft_task
            retrieval_task.cancel()
            return SpeculativeResult(
                answer=draft,
                used_refinement=False,
                retrieval_count=0,
                latency_mode="draft_only_timeout",
            )


def create_speculative_drafter(
    draft_fn: DraftFn,
    retrieve_fn: RetrieveFn,
    refine_fn: RefineFn,
    retrieval_deadline_s: float = 1.5,
) -> SpeculativeDrafter:
    """Factory for `SpeculativeDrafter`."""
    return SpeculativeDrafter(
        draft_fn=draft_fn,
        retrieve_fn=retrieve_fn,
        refine_fn=refine_fn,
        retrieval_deadline_s=retrieval_deadline_s,
    )
