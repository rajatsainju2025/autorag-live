"""
Interleaved Retrieval with Chain-of-Thought (IR-CoT).

Implements the IR-CoT pattern from:
  "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive
  Multi-Step Questions"  (Trivedi et al., ACL 2023)
  https://arxiv.org/abs/2212.10509

Key idea:
  Instead of retrieving once before reasoning, IR-CoT *interleaves* retrieval
  with each reasoning step. After every CoT sentence, the agent queries the
  retriever to gather supporting evidence for the *next* step.

  Loop:
    1. Generate the next reasoning sentence conditioned on the question,
       existing CoT, and the current retrieved context.
    2. Use that new CoT sentence as a search query to retrieve fresh docs.
    3. Add retrieved docs to the context.
    4. Repeat until a FINISH token / max steps.

This is especially powerful for multi-hop QA where each reasoning step
reveals what to look up next — information not available at query time.

Usage::

    from autorag_live.retrieval.ir_cot import IRCoTAgent

    agent = IRCoTAgent(llm_fn=my_llm, retriever_fn=my_retriever, max_steps=6)
    result = await agent.run("Who was the president of the US when the Berlin Wall fell?")
    print(result.answer)
    for step in result.steps:
        print(step)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, List, Sequence

logger = logging.getLogger(__name__)

LLMFn = Callable[[str], Coroutine[Any, Any, str]]
RetrieverFn = Callable[[str], Coroutine[Any, Any, List[str]]]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IRCoTStep:
    """A single interleaved retrieval-reasoning step."""

    step_num: int
    reasoning: str  # New CoT sentence generated this step
    query: str  # Query used to retrieve docs for next step
    retrieved_docs: List[str]


@dataclass
class IRCoTResult:
    """Full IR-CoT run result."""

    question: str
    answer: str
    steps: List[IRCoTStep]
    full_cot: str  # Complete chain-of-thought
    total_docs_retrieved: int


# ---------------------------------------------------------------------------
# IR-CoT Agent
# ---------------------------------------------------------------------------


class IRCoTAgent:
    """
    Interleaved Retrieval + Chain-of-Thought agent.

    Alternates between LLM reasoning steps and targeted retrieval queries,
    grounding each reasoning step in freshly retrieved evidence.

    Args:
        llm_fn: Async callable ``(prompt: str) -> str``.
        retriever_fn: Async callable ``(query: str) -> List[str]``.
        max_steps: Maximum reasoning/retrieval cycles.
        docs_per_step: Number of documents to retrieve each step.
        finish_token: Token the LLM emits to signal the final answer.

    Example::

        agent = IRCoTAgent(llm_fn=my_llm, retriever_fn=my_retriever)
        result = await agent.run("What year did the inventor of the telephone die?")
    """

    _STEP_PROMPT = """\
Answer the question by reasoning step by step.

Question: {question}

Retrieved Context:
{context}

Chain-of-thought so far:
{cot}

Continue with ONE new reasoning sentence. If you have enough information to \
answer the question fully, start your sentence with "{finish_token}".

Next reasoning sentence:"""

    _ANSWER_PROMPT = """\
Extract only the final answer from the following chain-of-thought.

Question: {question}

Chain-of-thought:
{cot}

Final Answer (concise):"""

    def __init__(
        self,
        llm_fn: LLMFn,
        retriever_fn: RetrieverFn,
        max_steps: int = 6,
        docs_per_step: int = 3,
        finish_token: str = "FINISH:",
    ) -> None:
        self._llm = llm_fn
        self._retriever = retriever_fn
        self.max_steps = max_steps
        self.docs_per_step = docs_per_step
        self.finish_token = finish_token

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run(self, question: str) -> IRCoTResult:
        """
        Execute the IR-CoT loop.

        Args:
            question: Natural-language question to answer.

        Returns:
            IRCoTResult with the answer, per-step trace, and full CoT.
        """
        logger.info("IR-CoT starting for: %s", question[:80])

        steps: List[IRCoTStep] = []
        cot_sentences: List[str] = []
        all_docs: List[str] = []

        # Step 0: initial retrieval based on the raw question
        init_docs = await self._safe_retrieve(question)
        all_docs.extend(init_docs)
        current_context = self._format_docs(init_docs)

        for step_num in range(1, self.max_steps + 1):
            cot_so_far = " ".join(cot_sentences) if cot_sentences else "(none yet)"

            prompt = self._STEP_PROMPT.format(
                question=question,
                context=current_context,
                cot=cot_so_far,
                finish_token=self.finish_token,
            )

            new_sentence = (await self._llm(prompt)).strip()
            cot_sentences.append(new_sentence)

            logger.debug("IR-CoT step %d: %s", step_num, new_sentence[:100])

            # Check for finish token
            if new_sentence.upper().startswith(self.finish_token.rstrip(":").upper()):
                steps.append(
                    IRCoTStep(
                        step_num=step_num,
                        reasoning=new_sentence,
                        query="",
                        retrieved_docs=[],
                    )
                )
                break

            # Use new sentence as retrieval query for next step
            new_docs = await self._safe_retrieve(new_sentence)
            all_docs.extend(new_docs)
            current_context = self._format_docs(
                # Keep recent docs prominent (last N)
                self._deduplicate(all_docs)[-(self.docs_per_step * 3) :]
            )

            steps.append(
                IRCoTStep(
                    step_num=step_num,
                    reasoning=new_sentence,
                    query=new_sentence,
                    retrieved_docs=new_docs,
                )
            )

        full_cot = " ".join(cot_sentences)

        # Extract clean final answer
        answer_prompt = self._ANSWER_PROMPT.format(question=question, cot=full_cot)
        answer = (await self._llm(answer_prompt)).strip()

        logger.info(
            "IR-CoT complete: %d steps, %d total docs retrieved",
            len(steps),
            len(all_docs),
        )

        return IRCoTResult(
            question=question,
            answer=answer,
            steps=steps,
            full_cot=full_cot,
            total_docs_retrieved=len(all_docs),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _safe_retrieve(self, query: str) -> List[str]:
        """Retrieve docs; return empty list on failure."""
        try:
            docs = await self._retriever(query)
            return docs[: self.docs_per_step]
        except Exception as exc:
            logger.warning("IR-CoT retrieval failed for query '%s': %s", query[:60], exc)
            return []

    @staticmethod
    def _deduplicate(docs: Sequence[str]) -> List[str]:
        """Remove exact duplicate documents while preserving order."""
        seen: set[str] = set()
        result: List[str] = []
        for doc in docs:
            if doc not in seen:
                seen.add(doc)
                result.append(doc)
        return result

    @staticmethod
    def _format_docs(docs: Sequence[str]) -> str:
        """Format a list of docs into a single context string."""
        if not docs:
            return "(no context retrieved)"
        return "\n\n---\n\n".join(f"[Doc {i + 1}] {d}" for i, d in enumerate(docs))
