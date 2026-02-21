"""Context knapsack packer.

Packs evidence chunks into a strict token budget using 0/1 knapsack dynamic
programming. Useful when high-value chunks are long and greedy packing fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ContextChunk:
    chunk_id: str
    text: str
    tokens: int
    value: float


@dataclass
class PackedContext:
    chunks: List[ContextChunk]
    total_tokens: int
    total_value: float


class ContextKnapsackPacker:
    """0/1 knapsack packer for RAG context optimization."""

    def pack(
        self,
        chunks: List[ContextChunk],
        budget_tokens: int,
    ) -> PackedContext:
        if budget_tokens <= 0 or not chunks:
            return PackedContext([], 0, 0.0)

        n = len(chunks)
        # DP table on value (float) via scaled integers for efficiency.
        scale = 100
        vals = [max(0, int(round(c.value * scale))) for c in chunks]
        toks = [max(1, c.tokens) for c in chunks]

        # dp[i][b] = max value using first i items with budget b.
        dp = [[0] * (budget_tokens + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            token_cost = toks[i - 1]
            chunk_val = vals[i - 1]
            for b in range(budget_tokens + 1):
                dp[i][b] = dp[i - 1][b]
                if token_cost <= b:
                    cand = dp[i - 1][b - token_cost] + chunk_val
                    if cand > dp[i][b]:
                        dp[i][b] = cand

        # Traceback
        chosen_idx: list[int] = []
        b = budget_tokens
        for i in range(n, 0, -1):
            if dp[i][b] != dp[i - 1][b]:
                chosen_idx.append(i - 1)
                b -= toks[i - 1]
                if b <= 0:
                    break

        selected = [chunks[i] for i in reversed(chosen_idx)]
        total_tokens = sum(c.tokens for c in selected)
        total_value = sum(c.value for c in selected)
        return PackedContext(selected, total_tokens, total_value)


def create_context_knapsack_packer() -> ContextKnapsackPacker:
    """Factory for `ContextKnapsackPacker`."""
    return ContextKnapsackPacker()
