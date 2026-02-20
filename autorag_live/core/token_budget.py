"""
Dynamic Token Budget Manager for Agentic RAG.

Enforces a hard token budget across the entire pipeline — query, retrieved
context, chain-of-thought scratchpad, and generated answer — using a
priority-based greedy allocation strategy.

Key behaviours:
1. **Priority-ranked allocation** — system prompt > query > high-score chunks
   > low-score chunks > CoT scratchpad. If the budget is tight, low-priority
   content is trimmed or dropped first.
2. **Compression fallback** — chunks that are over-budget are sentence-truncated
   rather than dropped entirely, preserving partial signal.
3. **Tiktoken-accurate counting** — uses ``tiktoken`` when available, falls back
   to the 4-chars-per-token heuristic for environments without it.
4. **Headroom reservation** — always reserves ``generation_reserve`` tokens for
   the model to actually generate an answer.
5. **Pipeline context integration** — attaches budget metadata to the pipeline
   context so downstream stages can see remaining headroom.

References:
- "Lost in the Middle" (Liu et al., 2023) — placement of relevant content
  matters; this manager puts the best chunks at the start.
- "LLMLingua" (Jiang et al., 2023) — coarse token pruning before LLM call.
- OpenAI token counting best practices (tiktoken docs, 2024).

Example:
    >>> manager = TokenBudgetManager(total_budget=4096, generation_reserve=512)
    >>> allocation = manager.allocate(
    ...     query="What is RAG?",
    ...     chunks=[("RAG stands for...", 0.95), ("Unrelated doc", 0.3)],
    ...     system_prompt="You are a helpful assistant.",
    ... )
    >>> print(allocation.context_str)  # trimmed context ready for LLM
    >>> print(allocation.remaining_tokens)  # headroom left for generation
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

try:
    import tiktoken

    _DEFAULT_ENCODING = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        """Accurate token count using tiktoken cl100k_base (GPT-4 / Claude)."""
        return len(_DEFAULT_ENCODING.encode(text))

    _USING_TIKTOKEN = True

except ImportError:
    _USING_TIKTOKEN = False

    def count_tokens(text: str) -> int:  # type: ignore[misc]
        """Fallback: 1 token ≈ 4 characters."""
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ChunkAllocation:
    """Budget allocation decision for a single chunk."""

    text: str
    score: float
    original_tokens: int
    allocated_tokens: int
    was_trimmed: bool = False
    was_dropped: bool = False
    rank: int = 0  # position in final context (0 = first / highest priority)


@dataclass
class BudgetAllocation:
    """
    Full budget allocation result for a single pipeline call.

    Contains the assembled context string and per-component token counts.
    """

    query: str
    system_prompt: str
    context_str: str
    """The assembled context string ready to be passed to the LLM."""

    chunk_allocations: List[ChunkAllocation] = field(default_factory=list)
    """Per-chunk allocation decisions for debugging / observability."""

    # Token counts
    total_budget: int = 0
    system_tokens: int = 0
    query_tokens: int = 0
    context_tokens: int = 0
    generation_reserve: int = 0
    remaining_tokens: int = 0
    """Tokens left for the model's generation (always ≥ generation_reserve)."""

    # Summary
    chunks_used: int = 0
    chunks_dropped: int = 0
    chunks_trimmed: int = 0

    @property
    def budget_utilisation(self) -> float:
        """Fraction of the total budget consumed by non-generation content."""
        used = self.system_tokens + self.query_tokens + self.context_tokens
        return used / max(1, self.total_budget)

    def to_prompt(self) -> str:
        """
        Assemble the full LLM prompt from allocated components.

        Follows the "best chunks first" placement recommended by Liu et al.
        to mitigate the "lost-in-the-middle" effect.
        """
        parts: List[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        if self.context_str:
            parts.append(f"Context:\n{self.context_str}")
        parts.append(f"Question: {self.query}")
        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise allocation summary for logging / telemetry."""
        return {
            "total_budget": self.total_budget,
            "system_tokens": self.system_tokens,
            "query_tokens": self.query_tokens,
            "context_tokens": self.context_tokens,
            "generation_reserve": self.generation_reserve,
            "remaining_tokens": self.remaining_tokens,
            "budget_utilisation": round(self.budget_utilisation, 3),
            "chunks_used": self.chunks_used,
            "chunks_dropped": self.chunks_dropped,
            "chunks_trimmed": self.chunks_trimmed,
            "using_tiktoken": _USING_TIKTOKEN,
        }


# ---------------------------------------------------------------------------
# Budget Manager
# ---------------------------------------------------------------------------


class TokenBudgetManager:
    """
    Priority-based dynamic token budget allocation.

    Priority order (highest → lowest):
        1. System prompt   (never trimmed)
        2. Query           (never trimmed)
        3. Generation reserve (reserved for model output)
        4. High-relevance chunks (score ≥ high_score_threshold)
        5. Low-relevance chunks (filled in until budget runs out)

    Within each tier, higher-scored chunks are preferred.
    Chunks that exceed remaining space are sentence-truncated, not dropped.

    Args:
        total_budget: Total token budget (default 4096).
        generation_reserve: Tokens reserved for model output (default 512).
        high_score_threshold: Score above which chunks get priority (default 0.7).
        min_chunk_tokens: Minimum tokens to keep a trimmed chunk (default 30).
        sentence_boundary_pattern: Regex for sentence boundaries for clean trimming.
    """

    def __init__(
        self,
        total_budget: int = 4096,
        generation_reserve: int = 512,
        high_score_threshold: float = 0.7,
        min_chunk_tokens: int = 30,
        sentence_boundary_pattern: str = r"(?<=[.!?])\s+",
    ) -> None:
        if generation_reserve >= total_budget:
            raise ValueError("generation_reserve must be < total_budget")
        self.total_budget = total_budget
        self.generation_reserve = generation_reserve
        self.high_score_threshold = high_score_threshold
        self.min_chunk_tokens = min_chunk_tokens
        self._sentence_re = re.compile(sentence_boundary_pattern)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def allocate(
        self,
        query: str,
        chunks: List[Tuple[str, float]],
        system_prompt: str = "",
        scratchpad: str = "",
    ) -> BudgetAllocation:
        """
        Allocate token budget across pipeline components.

        Args:
            query: The user query.
            chunks: List of (text, relevance_score) pairs.
            system_prompt: System/instruction prompt (highest priority).
            scratchpad: Optional CoT scratchpad (lowest priority, trimmed first).

        Returns:
            BudgetAllocation with assembled context and metadata.
        """
        # ── Fixed costs ────────────────────────────────────────────────
        system_tokens = count_tokens(system_prompt)
        query_tokens = count_tokens(query)
        fixed_cost = system_tokens + query_tokens + self.generation_reserve

        if fixed_cost > self.total_budget:
            logger.warning(
                "TokenBudget: fixed cost (%d) exceeds total budget (%d) — "
                "context will be empty.",
                fixed_cost,
                self.total_budget,
            )
            return BudgetAllocation(
                query=query,
                system_prompt=system_prompt,
                context_str="",
                total_budget=self.total_budget,
                system_tokens=system_tokens,
                query_tokens=query_tokens,
                generation_reserve=self.generation_reserve,
                remaining_tokens=self.total_budget - fixed_cost,
            )

        context_budget = self.total_budget - fixed_cost

        # ── Optional scratchpad (lowest priority) ─────────────────────
        scratchpad_tokens = 0
        if scratchpad:
            sp_tokens = count_tokens(scratchpad)
            # Give scratchpad at most 20% of context budget
            scratchpad_budget = int(context_budget * 0.2)
            if sp_tokens <= scratchpad_budget:
                scratchpad_tokens = sp_tokens
            else:
                scratchpad = self._trim_to_tokens(scratchpad, scratchpad_budget)
                scratchpad_tokens = count_tokens(scratchpad)
            context_budget -= scratchpad_tokens

        # ── Chunk allocation ──────────────────────────────────────────
        chunk_allocations, context_parts = self._allocate_chunks(chunks, context_budget)

        # Prepend scratchpad if present
        if scratchpad and scratchpad_tokens > 0:
            context_parts.insert(0, f"[Reasoning]\n{scratchpad}")

        context_str = "\n\n".join(context_parts)
        context_tokens = count_tokens(context_str)
        remaining = self.total_budget - system_tokens - query_tokens - context_tokens

        alloc = BudgetAllocation(
            query=query,
            system_prompt=system_prompt,
            context_str=context_str,
            chunk_allocations=chunk_allocations,
            total_budget=self.total_budget,
            system_tokens=system_tokens,
            query_tokens=query_tokens,
            context_tokens=context_tokens,
            generation_reserve=self.generation_reserve,
            remaining_tokens=max(self.generation_reserve, remaining),
            chunks_used=sum(1 for c in chunk_allocations if not c.was_dropped),
            chunks_dropped=sum(1 for c in chunk_allocations if c.was_dropped),
            chunks_trimmed=sum(1 for c in chunk_allocations if c.was_trimmed),
        )

        logger.debug(
            "TokenBudget: budget=%d used=%d+%d+%d remaining=%d "
            "chunks=%d/%d dropped=%d trimmed=%d",
            self.total_budget,
            system_tokens,
            query_tokens,
            context_tokens,
            alloc.remaining_tokens,
            alloc.chunks_used,
            len(chunks),
            alloc.chunks_dropped,
            alloc.chunks_trimmed,
        )
        return alloc

    def estimate_usage(
        self,
        query: str,
        chunks: List[Tuple[str, float]],
        system_prompt: str = "",
    ) -> Dict[str, int]:
        """
        Estimate token usage without full allocation.

        Useful for pre-flight checks before calling the LLM.

        Returns:
            Dict with 'system', 'query', 'context', 'total', 'over_budget'.
        """
        system_t = count_tokens(system_prompt)
        query_t = count_tokens(query)
        context_t = sum(count_tokens(c) for c, _ in chunks)
        total = system_t + query_t + context_t
        return {
            "system": system_t,
            "query": query_t,
            "context": context_t,
            "total": total,
            "over_budget": max(0, total - (self.total_budget - self.generation_reserve)),
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _allocate_chunks(
        self,
        chunks: List[Tuple[str, float]],
        budget: int,
    ) -> Tuple[List[ChunkAllocation], List[str]]:
        """Greedy priority allocation of chunks within a token budget."""
        # Sort: high-score first, then by text length (shorter = denser signal)
        sorted_chunks = sorted(chunks, key=lambda x: (-x[1], len(x[0])))

        allocations: List[ChunkAllocation] = []
        context_parts: List[str] = []
        remaining = budget

        for rank, (text, score) in enumerate(sorted_chunks):
            tokens = count_tokens(text)
            alloc = ChunkAllocation(
                text=text,
                score=score,
                original_tokens=tokens,
                allocated_tokens=0,
                rank=rank,
            )

            if tokens <= remaining:
                # Fits entirely
                alloc.allocated_tokens = tokens
                remaining -= tokens
                context_parts.append(text)
            elif remaining >= self.min_chunk_tokens:
                # Trim to fit
                trimmed = self._trim_to_tokens(text, remaining)
                trimmed_tokens = count_tokens(trimmed)
                if trimmed_tokens >= self.min_chunk_tokens:
                    alloc.allocated_tokens = trimmed_tokens
                    alloc.was_trimmed = True
                    remaining -= trimmed_tokens
                    context_parts.append(trimmed)
                else:
                    alloc.was_dropped = True
            else:
                alloc.was_dropped = True

            allocations.append(alloc)

            if remaining <= 0:
                # Mark remaining chunks as dropped
                for text2, score2 in sorted_chunks[rank + 1 :]:
                    t2 = count_tokens(text2)
                    allocations.append(
                        ChunkAllocation(
                            text=text2,
                            score=score2,
                            original_tokens=t2,
                            allocated_tokens=0,
                            was_dropped=True,
                            rank=rank + 1,
                        )
                    )
                break

        return allocations, context_parts

    def _trim_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Trim text to at most `max_tokens` tokens, preferring sentence boundaries.

        Strategy:
            1. Split on sentence boundaries.
            2. Accumulate sentences until adding the next would exceed budget.
            3. If no full sentence fits, fall back to character-level truncation.
        """
        sentences = self._sentence_re.split(text)
        result: List[str] = []
        used = 0
        for sent in sentences:
            t = count_tokens(sent)
            if used + t > max_tokens:
                break
            result.append(sent)
            used += t

        if result:
            return " ".join(result)

        # Fallback: raw character truncation (≈4 chars/token)
        char_limit = max_tokens * 4
        return text[:char_limit]
