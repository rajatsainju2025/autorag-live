"""Adaptive token budget allocator for agentic RAG.

Distributes a fixed context budget across evidence chunks by maximizing
information value under token constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class BudgetItem:
    """Candidate text unit to include in context."""

    item_id: str
    tokens: int
    relevance: float
    novelty: float = 1.0
    recency: float = 1.0

    @property
    def utility(self) -> float:
        # Weighted utility signal; clipped to sane range.
        val = 0.65 * self.relevance + 0.25 * self.novelty + 0.10 * self.recency
        return max(0.0, min(1.0, val))

    @property
    def density(self) -> float:
        # Utility per token favors concise high-signal chunks.
        return self.utility / max(1, self.tokens)


@dataclass
class AllocationResult:
    selected: List[BudgetItem]
    used_tokens: int
    budget_tokens: int

    @property
    def utilization(self) -> float:
        return self.used_tokens / self.budget_tokens if self.budget_tokens else 0.0


class TokenBudgetAllocator:
    """Greedy+repair allocator for retrieval context packing."""

    def __init__(self, reserve_for_instructions: int = 400) -> None:
        self.reserve_for_instructions = reserve_for_instructions

    def allocate(
        self,
        items: List[BudgetItem],
        max_context_tokens: int,
        min_chunks: int = 3,
    ) -> AllocationResult:
        """Select chunk subset within budget.

        Strategy:
        1) Sort by utility density and greedily pack.
        2) If under-selected, backfill smallest chunks by utility.
        """
        budget = max(0, max_context_tokens - self.reserve_for_instructions)
        if budget == 0 or not items:
            return AllocationResult([], 0, budget)

        ordered = sorted(items, key=lambda x: (x.density, x.utility), reverse=True)
        selected: List[BudgetItem] = []
        used = 0

        for item in ordered:
            if used + item.tokens <= budget:
                selected.append(item)
                used += item.tokens

        if len(selected) < min_chunks:
            remaining = [x for x in items if x.item_id not in {s.item_id for s in selected}]
            remaining.sort(key=lambda x: (x.tokens, -x.utility))
            for item in remaining:
                if len(selected) >= min_chunks:
                    break
                if used + item.tokens <= budget:
                    selected.append(item)
                    used += item.tokens

        selected.sort(key=lambda x: x.utility, reverse=True)
        return AllocationResult(selected=selected, used_tokens=used, budget_tokens=budget)


def create_token_budget_allocator(
    reserve_for_instructions: int = 400,
) -> TokenBudgetAllocator:
    """Factory for `TokenBudgetAllocator`."""
    return TokenBudgetAllocator(reserve_for_instructions=reserve_for_instructions)
