"""Query Decomposer.

Breaks down complex queries into simpler sub-queries for multi-hop retrieval.
Useful for answering questions that require aggregating information from multiple
distinct sources or reasoning steps.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List

logger = logging.getLogger(__name__)

LLMFn = Callable[[str], Coroutine[Any, Any, str]]


@dataclass
class SubQuery:
    index: int
    text: str


class QueryDecomposer:
    """Decomposes complex queries into simpler sub-queries."""

    _DECOMPOSE_PROMPT = """\
You are an expert at breaking down complex questions into simpler, atomic sub-questions.
Given the following complex question, break it down into a list of simpler sub-questions
that need to be answered to fully address the original question.

Format each sub-question on a new line starting with "SUB-QUESTION N:" where N is the
sub-question number (starting from 1).

=== Complex Question ===
{question}

=== Sub-Questions ===
"""

    def __init__(
        self,
        llm_fn: LLMFn,
        max_sub_queries: int = 5,
    ) -> None:
        self._llm = llm_fn
        self.max_sub_queries = max_sub_queries

    def _parse_sub_queries(self, llm_output: str) -> List[str]:
        """Extract sub-query texts from LLM decomposition output."""
        sub_queries: list[str] = []
        pattern = re.compile(r"SUB-QUESTION\s+(\d+)\s*:\s*(.+)", re.IGNORECASE)
        for m in pattern.finditer(llm_output):
            text = m.group(2).strip()
            if text:
                sub_queries.append(text)

        # Fallback: numbered list lines
        if not sub_queries:
            for line in llm_output.splitlines():
                line = line.strip()
                m2 = re.match(r"^\d+[\.\)]\s+(.+)", line)
                if m2:
                    sub_queries.append(m2.group(1))

        return sub_queries

    async def decompose(self, question: str) -> List[SubQuery]:
        """Decompose a complex question into sub-queries."""
        prompt = self._DECOMPOSE_PROMPT.format(question=question)
        try:
            output = await self._llm(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query decomposition failed: %s", exc)
            return [SubQuery(index=1, text=question)]

        raw_sub_queries = self._parse_sub_queries(output)
        raw_sub_queries = raw_sub_queries[: self.max_sub_queries]

        if not raw_sub_queries:
            logger.warning("No sub-queries extracted, returning original question.")
            return [SubQuery(index=1, text=question)]

        sub_queries: list[SubQuery] = []
        for i, text in enumerate(raw_sub_queries, start=1):
            sub_queries.append(SubQuery(index=i, text=text))

        logger.debug("Decomposed into %d sub-queries", len(sub_queries))
        return sub_queries


def create_query_decomposer(
    llm_fn: LLMFn,
    max_sub_queries: int = 5,
) -> QueryDecomposer:
    """Factory for `QueryDecomposer`."""
    return QueryDecomposer(llm_fn=llm_fn, max_sub_queries=max_sub_queries)
