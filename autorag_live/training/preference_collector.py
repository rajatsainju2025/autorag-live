"""
Preference Data Collector for RAG Fine-Tuning.

Collects human or LLM-judged preference pairs (chosen / rejected) suitable
for training reward models and running DPO / PPO-based RLHF fine-tuning of
retrieval and generation components.

Data format follows the Anthropic HH-RLHF convention and is compatible with
TRL (Transformer Reinforcement Learning) ``RewardTrainer`` and ``DPOTrainer``.

Each preference record contains:
  - ``prompt``: The input (question + retrieved context).
  - ``chosen``: The preferred answer (higher quality).
  - ``rejected``: The dis-preferred answer (lower quality).
  - ``metadata``: Source, judge, scores, timestamps.

Collection modes:
  1. **Human labels** — ``record_human_preference()`` for direct annotation.
  2. **LLM judge**    — ``judge_and_record()`` uses an LLM to compare two candidate
                        answers and decide which is better (auto-labelling).
  3. **Score-based**  — ``record_from_scores()`` uses existing eval metrics
                        (e.g. RAGAS faithfulness) to derive preferences.

Usage::

    collector = PreferenceDataCollector(storage_path="prefs.jsonl")

    # LLM auto-labelling
    record = await collector.judge_and_record(
        llm_fn=my_llm,
        question="What is RLHF?",
        context="...",
        answer_a="...",
        answer_b="...",
    )

    # Export for DPO training
    collector.export_jsonl("train_prefs.jsonl")
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

LLMFn = Callable[[str], Coroutine[Any, Any, str]]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PreferenceRecord:
    """A single preference pair for RLHF / DPO training."""

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""  # question + context, formatted for the model
    chosen: str = ""  # Preferred answer
    rejected: str = ""  # Dis-preferred answer
    question: str = ""
    context: str = ""
    judge: str = "unknown"  # "human" | "llm" | "score"
    judge_reasoning: str = ""
    chosen_score: Optional[float] = None
    rejected_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_trl_dict(self) -> Dict[str, str]:
        """Convert to TRL-compatible DPOTrainer format."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class PreferenceDataCollector:
    """
    Collects and persists preference data for RLHF fine-tuning.

    Args:
        storage_path: Optional JSONL file path for auto-persistence.
        prompt_template: Template combining question and context into a model prompt.

    Example::

        collector = PreferenceDataCollector(storage_path="rlhf_data.jsonl")
        await collector.judge_and_record(
            llm_fn=my_llm,
            question="Explain transformers",
            context="[retrieved docs]",
            answer_a="...",
            answer_b="...",
        )
        collector.export_jsonl("dpo_train.jsonl")
    """

    _JUDGE_PROMPT = """\
You are an expert judge evaluating two AI-generated answers to a question.
Your goal is to determine which answer is more accurate, faithful to the \
provided context, complete, and concise.

Question: {question}

Retrieved Context:
{context}

Answer A:
{answer_a}

Answer B:
{answer_b}

Which answer is better? Briefly explain your reasoning (2-3 sentences),
then output exactly: WINNER: A  or  WINNER: B

Reasoning:"""

    _DEFAULT_PROMPT_TEMPLATE = "Question: {question}\n\nContext:\n{context}\n\nAnswer:"

    def __init__(
        self,
        storage_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        self.storage_path = storage_path
        self.prompt_template = prompt_template or self._DEFAULT_PROMPT_TEMPLATE
        self._records: List[PreferenceRecord] = []
        self._lock = threading.Lock()

        # Load existing records from file if it exists
        if storage_path and os.path.exists(storage_path):
            self._load_from_file(storage_path)

    # ------------------------------------------------------------------
    # Public – collection methods
    # ------------------------------------------------------------------

    def record_human_preference(
        self,
        question: str,
        context: str,
        chosen: str,
        rejected: str,
        reasoning: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PreferenceRecord:
        """
        Record a human-annotated preference pair.

        Args:
            question: The input question.
            context: The retrieved context shown to the annotator.
            chosen: The answer the annotator preferred.
            rejected: The answer the annotator disliked.
            reasoning: Optional annotator explanation.
            metadata: Additional metadata (e.g., annotator ID).

        Returns:
            The created PreferenceRecord.
        """
        prompt = self.prompt_template.format(question=question, context=context)
        record = PreferenceRecord(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            question=question,
            context=context,
            judge="human",
            judge_reasoning=reasoning,
            metadata=metadata or {},
        )
        self._add(record)
        return record

    async def judge_and_record(
        self,
        llm_fn: LLMFn,
        question: str,
        context: str,
        answer_a: str,
        answer_b: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[PreferenceRecord]:
        """
        Use an LLM to judge which answer is better, then record the preference.

        Args:
            llm_fn: Async LLM callable.
            question: The input question.
            context: The retrieved context.
            answer_a: First candidate answer.
            answer_b: Second candidate answer.
            metadata: Optional metadata to attach.

        Returns:
            PreferenceRecord if a clear winner was determined, else None.
        """
        prompt = self._JUDGE_PROMPT.format(
            question=question,
            context=context[:2000],
            answer_a=answer_a,
            answer_b=answer_b,
        )
        response = (await llm_fn(prompt)).strip()

        winner = self._parse_winner(response)
        if winner is None:
            logger.warning("LLM judge could not determine a winner. Skipping record.")
            return None

        chosen = answer_a if winner == "A" else answer_b
        rejected = answer_b if winner == "A" else answer_a

        model_prompt = self.prompt_template.format(question=question, context=context)
        record = PreferenceRecord(
            prompt=model_prompt,
            chosen=chosen,
            rejected=rejected,
            question=question,
            context=context,
            judge="llm",
            judge_reasoning=response,
            metadata=metadata or {},
        )
        self._add(record)
        logger.info("LLM judge: winner=%s, record_id=%s", winner, record.record_id)
        return record

    def record_from_scores(
        self,
        question: str,
        context: str,
        answer_a: str,
        score_a: float,
        answer_b: str,
        score_b: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[PreferenceRecord]:
        """
        Create a preference record based on numeric quality scores (e.g., RAGAS).

        Args:
            question: The input question.
            context: The retrieved context.
            answer_a / answer_b: Candidate answers.
            score_a / score_b: Numeric quality scores (higher = better).
            metadata: Optional metadata.

        Returns:
            PreferenceRecord if scores differ, else None (tie → skip).
        """
        if abs(score_a - score_b) < 1e-6:
            logger.debug("Score tie (%.4f == %.4f); skipping.", score_a, score_b)
            return None

        chosen, chosen_score, rejected, rejected_score = (
            (answer_a, score_a, answer_b, score_b)
            if score_a > score_b
            else (answer_b, score_b, answer_a, score_a)
        )

        prompt = self.prompt_template.format(question=question, context=context)
        record = PreferenceRecord(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            question=question,
            context=context,
            judge="score",
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            metadata=metadata or {},
        )
        self._add(record)
        return record

    # ------------------------------------------------------------------
    # Public – export / query
    # ------------------------------------------------------------------

    def export_jsonl(self, path: str, trl_format: bool = True) -> int:
        """
        Export collected preference data to a JSONL file.

        Args:
            path: Output file path.
            trl_format: If True, export in TRL DPOTrainer format (prompt/chosen/rejected only).

        Returns:
            Number of records written.
        """
        with self._lock:
            records = list(self._records)

        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                row = rec.to_trl_dict() if trl_format else asdict(rec)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info("Exported %d preference records to %s", len(records), path)
        return len(records)

    @property
    def records(self) -> List[PreferenceRecord]:
        """Return a snapshot of all collected records."""
        with self._lock:
            return list(self._records)

    def stats(self) -> Dict[str, Any]:
        """Summary statistics about collected preferences."""
        with self._lock:
            n = len(self._records)
            judges = {}
            for rec in self._records:
                judges[rec.judge] = judges.get(rec.judge, 0) + 1
        return {"total_records": n, "by_judge": judges}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add(self, record: PreferenceRecord) -> None:
        with self._lock:
            self._records.append(record)
        if self.storage_path:
            self._append_to_file(record)

    def _append_to_file(self, record: PreferenceRecord) -> None:
        try:
            with open(self.storage_path, "a", encoding="utf-8") as fh:  # type: ignore[arg-type]
                fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("Failed to persist preference record: %s", exc)

    def _load_from_file(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self._records.append(PreferenceRecord(**data))
            logger.info("Loaded %d preference records from %s", len(self._records), path)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Could not load preference data from %s: %s", path, exc)

    @staticmethod
    def _parse_winner(response: str) -> Optional[str]:
        """Extract 'A' or 'B' from the judge response."""
        upper = response.upper()
        for line in reversed(upper.splitlines()):
            if "WINNER:" in line:
                if "WINNER: A" in line or "WINNER:A" in line:
                    return "A"
                if "WINNER: B" in line or "WINNER:B" in line:
                    return "B"
        return None
