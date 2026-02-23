"""Streaming Token Counter.

Counts tokens on the fly during streaming to enforce budgets and track costs.
Useful for agentic RAG where long generations can exceed token limits or budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncGenerator, Callable, Dict, Optional


@dataclass
class TokenBudgetExceeded(Exception):
    """Raised when the streaming token count exceeds the budget."""

    budget: int
    current_count: int

    def __str__(self) -> str:
        return f"Token budget exceeded: {self.current_count} > {self.budget}"


class StreamingTokenCounter:
    """Counts tokens during streaming and enforces budgets."""

    def __init__(
        self,
        token_counter_fn: Callable[[str], int],
        budget: Optional[int] = None,
    ) -> None:
        self._count_fn = token_counter_fn
        self.budget = budget
        self._total_tokens = 0
        self._total_chunks = 0

    async def stream_with_counting(
        self,
        stream: AsyncGenerator[str, None],
    ) -> AsyncGenerator[str, None]:
        """Wrap an async generator to count tokens and enforce budget."""
        self._total_tokens = 0
        self._total_chunks = 0

        async for chunk in stream:
            self._total_chunks += 1
            tokens = self._count_fn(chunk)
            self._total_tokens += tokens

            if self.budget is not None and self._total_tokens > self.budget:
                raise TokenBudgetExceeded(
                    budget=self.budget,
                    current_count=self._total_tokens,
                )

            yield chunk

    def get_total_tokens(self) -> int:
        """Get the total tokens counted so far."""
        return self._total_tokens

    def stats(self) -> Dict[str, float]:
        """Get statistics about the streaming process."""
        avg_tokens_per_chunk = (
            self._total_tokens / self._total_chunks if self._total_chunks > 0 else 0.0
        )
        return {
            "total_tokens": float(self._total_tokens),
            "total_chunks": float(self._total_chunks),
            "avg_tokens_per_chunk": round(avg_tokens_per_chunk, 2),
            "budget": float(self.budget) if self.budget is not None else -1.0,
        }


def create_streaming_token_counter(
    token_counter_fn: Callable[[str], int],
    budget: Optional[int] = None,
) -> StreamingTokenCounter:
    """Factory for `StreamingTokenCounter`."""
    return StreamingTokenCounter(token_counter_fn=token_counter_fn, budget=budget)
