"""
RAFT (Retrieval Augmented Fine-Tuning) Training Data Generator.

Generates training data for domain adaptation of LLMs in RAG scenarios.
Creates question-answer pairs with context that teach models to ignore
distractor documents and focus on relevant information.

Key Features:
1. Domain-specific training data generation
2. Distractor document injection
3. Chain-of-thought reasoning generation
4. Oracle document identification
5. Training data quality filtering

References:
- RAFT: Adapting Language Model to Domain Specific RAG (Zhang et al., 2024)
- Gorilla: Large Language Model Connected with Massive APIs

Example:
    >>> generator = RAFTGenerator(llm, retriever)
    >>> training_data = await generator.generate(documents, num_samples=1000)
"""

from __future__ import annotations

import json
import logging
import random
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class LLMProtocol(Protocol):
    """Protocol for LLM interface."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response from prompt."""
        ...


class RetrieverProtocol(Protocol):
    """Protocol for retriever interface."""

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents for query."""
        ...


# =============================================================================
# Data Structures
# =============================================================================


class DatasetFormat(str, Enum):
    """Output format for training data."""

    JSONL = "jsonl"  # JSON Lines format
    JSON = "json"  # Single JSON array
    PARQUET = "parquet"  # Parquet format
    CSV = "csv"  # CSV format


class SampleType(str, Enum):
    """Type of training sample."""

    ORACLE_ONLY = "oracle_only"  # Only oracle document
    ORACLE_WITH_DISTRACTORS = "oracle_with_distractors"  # Oracle + distractors
    DISTRACTORS_ONLY = "distractors_only"  # Only distractors (negative)
    NO_CONTEXT = "no_context"  # No context (baseline)


@dataclass
class RAFTSample:
    """
    A single RAFT training sample.

    Attributes:
        question: Generated question
        answer: Generated answer with reasoning
        oracle_document: The relevant document
        distractor_documents: Irrelevant documents
        chain_of_thought: Reasoning chain
        sample_type: Type of sample
        metadata: Additional metadata
    """

    question: str
    answer: str
    oracle_document: Dict[str, Any]
    distractor_documents: List[Dict[str, Any]] = field(default_factory=list)
    chain_of_thought: str = ""
    sample_type: SampleType = SampleType.ORACLE_WITH_DISTRACTORS
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sample_id(self) -> str:
        """Get unique sample ID."""
        return self.metadata.get("id", str(uuid.uuid4())[:8])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.sample_id,
            "question": self.question,
            "answer": self.answer,
            "chain_of_thought": self.chain_of_thought,
            "oracle_document": self.oracle_document,
            "distractor_documents": self.distractor_documents,
            "sample_type": self.sample_type.value,
            "metadata": self.metadata,
        }

    def to_training_format(self, include_cot: bool = True) -> Dict[str, str]:
        """Convert to training format (instruction, input, output)."""
        # Build context from documents
        all_docs = [self.oracle_document] + self.distractor_documents
        random.shuffle(all_docs)

        context_parts = []
        for i, doc in enumerate(all_docs, 1):
            content = doc.get("content", doc.get("text", ""))
            context_parts.append(f"Document {i}:\n{content[:1000]}")

        context = "\n\n".join(context_parts)

        # Build output
        if include_cot and self.chain_of_thought:
            output = f"Reasoning: {self.chain_of_thought}\n\nAnswer: {self.answer}"
        else:
            output = self.answer

        return {
            "instruction": "Answer the question based on the provided documents.",
            "input": f"Documents:\n{context}\n\nQuestion: {self.question}",
            "output": output,
        }


@dataclass
class RAFTDataset:
    """
    Collection of RAFT training samples.

    Attributes:
        samples: List of training samples
        domain: Domain name
        version: Dataset version
        metadata: Dataset metadata
    """

    samples: List[RAFTSample] = field(default_factory=list)
    domain: str = "general"
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Get dataset size."""
        return len(self.samples)

    def add_sample(self, sample: RAFTSample) -> None:
        """Add a sample to the dataset."""
        self.samples.append(sample)

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Tuple["RAFTDataset", "RAFTDataset", "RAFTDataset"]:
        """Split into train/val/test sets."""
        random.shuffle(self.samples)

        n = len(self.samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train = RAFTDataset(
            samples=self.samples[:train_end],
            domain=self.domain,
            version=self.version,
        )
        val = RAFTDataset(
            samples=self.samples[train_end:val_end],
            domain=self.domain,
            version=self.version,
        )
        test = RAFTDataset(
            samples=self.samples[val_end:],
            domain=self.domain,
            version=self.version,
        )

        return train, val, test

    def save(
        self,
        path: str,
        format: DatasetFormat = DatasetFormat.JSONL,
    ) -> None:
        """Save dataset to file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if format == DatasetFormat.JSONL:
            with open(path_obj, "w") as f:
                for sample in self.samples:
                    f.write(json.dumps(sample.to_dict()) + "\n")

        elif format == DatasetFormat.JSON:
            with open(path_obj, "w") as f:
                json.dump(
                    {
                        "domain": self.domain,
                        "version": self.version,
                        "metadata": self.metadata,
                        "samples": [s.to_dict() for s in self.samples],
                    },
                    f,
                    indent=2,
                )

    @classmethod
    def load(cls, path: str) -> "RAFTDataset":
        """Load dataset from file."""
        path_obj = Path(path)

        if path_obj.suffix == ".jsonl":
            samples = []
            with open(path_obj) as f:
                for line in f:
                    data = json.loads(line)
                    samples.append(
                        RAFTSample(
                            question=data["question"],
                            answer=data["answer"],
                            oracle_document=data["oracle_document"],
                            distractor_documents=data.get("distractor_documents", []),
                            chain_of_thought=data.get("chain_of_thought", ""),
                            sample_type=SampleType(
                                data.get("sample_type", "oracle_with_distractors")
                            ),
                            metadata=data.get("metadata", {}),
                        )
                    )
            return cls(samples=samples)

        elif path_obj.suffix == ".json":
            with open(path_obj) as f:
                data = json.load(f)

            samples = [
                RAFTSample(
                    question=s["question"],
                    answer=s["answer"],
                    oracle_document=s["oracle_document"],
                    distractor_documents=s.get("distractor_documents", []),
                    chain_of_thought=s.get("chain_of_thought", ""),
                    sample_type=SampleType(s.get("sample_type", "oracle_with_distractors")),
                    metadata=s.get("metadata", {}),
                )
                for s in data["samples"]
            ]

            return cls(
                samples=samples,
                domain=data.get("domain", "general"),
                version=data.get("version", "1.0"),
                metadata=data.get("metadata", {}),
            )

        raise ValueError(f"Unsupported format: {path_obj.suffix}")


# =============================================================================
# Question Generator
# =============================================================================


class QuestionGenerator:
    """Generates questions from documents."""

    def __init__(self, llm: LLMProtocol):
        """Initialize generator."""
        self.llm = llm

    async def generate_questions(
        self,
        document: Dict[str, Any],
        num_questions: int = 3,
    ) -> List[str]:
        """
        Generate questions from a document.

        Args:
            document: Source document
            num_questions: Number of questions to generate

        Returns:
            List of generated questions
        """
        content = document.get("content", document.get("text", ""))

        prompt = f"""Generate {num_questions} diverse questions that can be answered using ONLY the information in this document.
The questions should:
1. Be specific and factual
2. Have clear answers in the document
3. Vary in complexity (some simple, some requiring reasoning)
4. Cover different aspects of the document

Document:
{content[:3000]}

Generate {num_questions} questions (one per line, numbered):"""

        try:
            response = await self.llm.generate(prompt)
            questions = self._parse_questions(response)
            return questions[:num_questions]
        except Exception as e:
            logger.warning(f"Question generation failed: {e}")
            return []

    def _parse_questions(self, response: str) -> List[str]:
        """Parse questions from LLM response."""
        questions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Remove numbering
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            if line and "?" in line:
                questions.append(line)
        return questions


# =============================================================================
# Answer Generator
# =============================================================================


class AnswerGenerator:
    """Generates answers with chain-of-thought reasoning."""

    def __init__(self, llm: LLMProtocol):
        """Initialize generator."""
        self.llm = llm

    async def generate_answer(
        self,
        question: str,
        document: Dict[str, Any],
        include_cot: bool = True,
    ) -> Tuple[str, str]:
        """
        Generate answer with chain-of-thought.

        Args:
            question: Question to answer
            document: Source document
            include_cot: Whether to include chain-of-thought

        Returns:
            Tuple of (answer, chain_of_thought)
        """
        content = document.get("content", document.get("text", ""))

        if include_cot:
            prompt = f"""Answer the question using ONLY the information in the document.
First, explain your reasoning step by step, then provide the final answer.

Document:
{content[:3000]}

Question: {question}

Reasoning:
[Think step by step about what information from the document is relevant]

Answer:
[Provide a clear, concise answer]"""
        else:
            prompt = f"""Answer the question using ONLY the information in the document.

Document:
{content[:3000]}

Question: {question}

Answer:"""

        try:
            response = await self.llm.generate(prompt)

            if include_cot:
                cot, answer = self._parse_cot_answer(response)
            else:
                cot = ""
                answer = response.strip()

            return answer, cot
        except Exception as e:
            logger.warning(f"Answer generation failed: {e}")
            return "", ""

    def _parse_cot_answer(self, response: str) -> Tuple[str, str]:
        """Parse chain-of-thought and answer from response."""
        # Try to find explicit sections
        response_lower = response.lower()

        if "answer:" in response_lower:
            parts = response.split("Answer:", 1)
            if len(parts) == 2:
                cot = parts[0].replace("Reasoning:", "").strip()
                answer = parts[1].strip()
                return cot, answer

        # Fallback: use last paragraph as answer
        paragraphs = response.strip().split("\n\n")
        if len(paragraphs) > 1:
            return "\n\n".join(paragraphs[:-1]), paragraphs[-1]

        return "", response.strip()


# =============================================================================
# Distractor Selector
# =============================================================================


class DistractorSelector:
    """Selects distractor documents for training samples."""

    def __init__(
        self,
        retriever: Optional[RetrieverProtocol] = None,
    ):
        """Initialize selector."""
        self.retriever = retriever

    async def select_distractors(
        self,
        oracle_doc: Dict[str, Any],
        all_documents: List[Dict[str, Any]],
        num_distractors: int = 4,
        strategy: str = "random",
    ) -> List[Dict[str, Any]]:
        """
        Select distractor documents.

        Args:
            oracle_doc: The oracle (relevant) document
            all_documents: Pool of all documents
            num_distractors: Number of distractors to select
            strategy: Selection strategy (random, hard, semantic)

        Returns:
            List of distractor documents
        """
        # Filter out oracle document
        oracle_content = oracle_doc.get("content", oracle_doc.get("text", ""))
        candidates = [
            d for d in all_documents if d.get("content", d.get("text", "")) != oracle_content
        ]

        if not candidates:
            return []

        if strategy == "random":
            return self._random_select(candidates, num_distractors)
        elif strategy == "hard":
            return await self._hard_negative_select(oracle_doc, candidates, num_distractors)
        else:
            return self._random_select(candidates, num_distractors)

    def _random_select(
        self,
        candidates: List[Dict[str, Any]],
        num: int,
    ) -> List[Dict[str, Any]]:
        """Random distractor selection."""
        return random.sample(candidates, min(num, len(candidates)))

    async def _hard_negative_select(
        self,
        oracle_doc: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        num: int,
    ) -> List[Dict[str, Any]]:
        """Select hard negatives (similar but irrelevant)."""
        if not self.retriever:
            return self._random_select(candidates, num)

        # Use oracle content as query to find similar docs
        oracle_content = oracle_doc.get("content", oracle_doc.get("text", ""))
        query = oracle_content[:200]  # Use beginning as query

        try:
            similar = await self.retriever.retrieve(query, top_k=num * 2)

            # Filter to candidates only
            candidate_contents = {d.get("content", d.get("text", ""))[:100] for d in candidates}

            hard_negatives = [
                d
                for d in similar
                if d.get("content", d.get("text", ""))[:100] in candidate_contents
            ]

            return hard_negatives[:num]
        except Exception:
            return self._random_select(candidates, num)


# =============================================================================
# Main RAFT Generator
# =============================================================================


class RAFTGenerator:
    """
    RAFT Training Data Generator.

    Generates high-quality training data for domain-specific
    RAG fine-tuning.

    Example:
        >>> generator = RAFTGenerator(llm, retriever)
        >>> dataset = await generator.generate(
        ...     documents=documents,
        ...     num_samples=1000,
        ...     questions_per_doc=3
        ... )
        >>> dataset.save("raft_training.jsonl")
    """

    def __init__(
        self,
        llm: LLMProtocol,
        retriever: Optional[RetrieverProtocol] = None,
        domain: str = "general",
    ):
        """
        Initialize generator.

        Args:
            llm: Language model for generation
            retriever: Retriever for hard negative selection
            domain: Domain name for dataset
        """
        self.llm = llm
        self.retriever = retriever
        self.domain = domain

        self.question_generator = QuestionGenerator(llm)
        self.answer_generator = AnswerGenerator(llm)
        self.distractor_selector = DistractorSelector(retriever)

    async def generate(
        self,
        documents: List[Dict[str, Any]],
        *,
        num_samples: Optional[int] = None,
        questions_per_doc: int = 3,
        num_distractors: int = 4,
        include_cot: bool = True,
        oracle_ratio: float = 0.8,
    ) -> RAFTDataset:
        """
        Generate RAFT training dataset.

        Args:
            documents: Source documents
            num_samples: Maximum samples (None = all possible)
            questions_per_doc: Questions per document
            num_distractors: Distractor documents per sample
            include_cot: Include chain-of-thought
            oracle_ratio: Ratio of samples with oracle document

        Returns:
            RAFTDataset with generated samples
        """
        dataset = RAFTDataset(domain=self.domain)

        for doc_idx, document in enumerate(documents):
            # Generate questions for this document
            questions = await self.question_generator.generate_questions(
                document, num_questions=questions_per_doc
            )

            for q_idx, question in enumerate(questions):
                # Check if we've reached sample limit
                if num_samples and dataset.size >= num_samples:
                    break

                # Determine sample type
                if random.random() < oracle_ratio:
                    sample_type = SampleType.ORACLE_WITH_DISTRACTORS
                else:
                    sample_type = SampleType.DISTRACTORS_ONLY

                # Generate answer
                answer, cot = await self.answer_generator.generate_answer(
                    question, document, include_cot=include_cot
                )

                if not answer:
                    continue

                # Select distractors
                distractors = await self.distractor_selector.select_distractors(
                    document,
                    documents,
                    num_distractors=num_distractors,
                    strategy="hard" if self.retriever else "random",
                )

                # Create sample
                sample = RAFTSample(
                    question=question,
                    answer=answer,
                    oracle_document=document,
                    distractor_documents=distractors,
                    chain_of_thought=cot if include_cot else "",
                    sample_type=sample_type,
                    metadata={
                        "id": f"{doc_idx}_{q_idx}",
                        "source_doc_idx": doc_idx,
                    },
                )

                dataset.add_sample(sample)

            # Early exit if sample limit reached
            if num_samples and dataset.size >= num_samples:
                break

        logger.info(f"Generated {dataset.size} RAFT training samples")
        return dataset

    async def generate_from_qa_pairs(
        self,
        qa_pairs: List[Tuple[str, str, Dict[str, Any]]],
        documents: List[Dict[str, Any]],
        num_distractors: int = 4,
    ) -> RAFTDataset:
        """
        Generate dataset from existing QA pairs.

        Args:
            qa_pairs: List of (question, answer, source_doc) tuples
            documents: Pool for distractor selection
            num_distractors: Distractors per sample

        Returns:
            RAFTDataset
        """
        dataset = RAFTDataset(domain=self.domain)

        for i, (question, answer, source_doc) in enumerate(qa_pairs):
            distractors = await self.distractor_selector.select_distractors(
                source_doc, documents, num_distractors=num_distractors
            )

            sample = RAFTSample(
                question=question,
                answer=answer,
                oracle_document=source_doc,
                distractor_documents=distractors,
                sample_type=SampleType.ORACLE_WITH_DISTRACTORS,
                metadata={"id": str(i)},
            )

            dataset.add_sample(sample)

        return dataset


# =============================================================================
# Quality Filter
# =============================================================================


class QualityFilter:
    """Filters training samples for quality."""

    def __init__(
        self,
        min_question_length: int = 10,
        min_answer_length: int = 20,
        max_answer_length: int = 2000,
    ):
        """Initialize filter."""
        self.min_question_length = min_question_length
        self.min_answer_length = min_answer_length
        self.max_answer_length = max_answer_length

    def filter(self, dataset: RAFTDataset) -> RAFTDataset:
        """
        Filter dataset for quality.

        Args:
            dataset: Input dataset

        Returns:
            Filtered dataset
        """
        filtered_samples = []

        for sample in dataset.samples:
            if self._is_valid(sample):
                filtered_samples.append(sample)

        return RAFTDataset(
            samples=filtered_samples,
            domain=dataset.domain,
            version=dataset.version,
            metadata={
                **dataset.metadata,
                "filtered": True,
                "original_size": dataset.size,
            },
        )

    def _is_valid(self, sample: RAFTSample) -> bool:
        """Check if sample meets quality criteria."""
        # Check question length
        if len(sample.question) < self.min_question_length:
            return False

        # Check answer length
        if len(sample.answer) < self.min_answer_length:
            return False

        if len(sample.answer) > self.max_answer_length:
            return False

        # Check for question mark
        if "?" not in sample.question:
            return False

        return True


# =============================================================================
# Convenience Functions
# =============================================================================


def create_raft_generator(
    llm: LLMProtocol,
    retriever: Optional[RetrieverProtocol] = None,
    domain: str = "general",
) -> RAFTGenerator:
    """
    Create a RAFT generator.

    Args:
        llm: Language model
        retriever: Optional retriever for hard negatives
        domain: Domain name

    Returns:
        RAFTGenerator instance
    """
    return RAFTGenerator(llm=llm, retriever=retriever, domain=domain)


async def generate_raft_dataset(
    documents: List[Dict[str, Any]],
    llm: LLMProtocol,
    num_samples: int = 100,
) -> RAFTDataset:
    """
    Quick RAFT dataset generation.

    Args:
        documents: Source documents
        llm: Language model
        num_samples: Number of samples

    Returns:
        RAFTDataset
    """
    generator = RAFTGenerator(llm=llm)
    return await generator.generate(documents, num_samples=num_samples)
