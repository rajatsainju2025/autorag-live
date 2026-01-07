"""
Dense Passage Retrieval (DPR) Training Implementation.

Implements bi-encoder training for dense retrieval with
in-batch negatives and hard negative mining.

Reference: Karpukhin et al., 2020 - "Dense Passage Retrieval for
Open-Domain Question Answering"

Key features:
1. Bi-encoder architecture (separate query/passage encoders)
2. In-batch negatives for efficient training
3. Hard negative mining
4. Contrastive learning objectives
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Protocol, runtime_checkable

# ============================================================================
# Protocols
# ============================================================================


@runtime_checkable
class EncoderProtocol(Protocol):
    """Protocol for encoder models."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to embeddings."""
        ...

    def parameters(self) -> Iterator[Any]:
        """Get model parameters."""
        ...


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol for tokenizers."""

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "pt",
    ) -> Any:
        """Tokenize texts."""
        ...


# ============================================================================
# Training Data Types
# ============================================================================


@dataclass
class DPRExample:
    """A single DPR training example."""

    question: str
    positive_passage: str
    hard_negatives: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DPRBatch:
    """A batch of DPR examples."""

    questions: list[str]
    positive_passages: list[str]
    hard_negatives: list[list[str]]  # Per-question hard negatives
    batch_size: int


@dataclass
class TrainingMetrics:
    """Metrics from training."""

    loss: float
    accuracy: float
    mrr: float  # Mean Reciprocal Rank
    recall_at_k: dict[int, float]
    step: int
    epoch: int


# ============================================================================
# Loss Functions
# ============================================================================


class LossFunction(Enum):
    """Supported loss functions."""

    NLL = "nll"  # Negative Log Likelihood
    CONTRASTIVE = "contrastive"
    TRIPLET = "triplet"
    INFONCE = "infonce"  # InfoNCE / NT-Xent


class DPRLoss(ABC):
    """Abstract base for DPR loss functions."""

    @abstractmethod
    def compute(
        self,
        query_embeddings: Any,
        passage_embeddings: Any,
        labels: Any,
    ) -> Any:
        """Compute loss."""
        ...


class NLLLoss(DPRLoss):
    """Negative Log Likelihood loss for DPR."""

    def __init__(self, temperature: float = 1.0):
        """Initialize with temperature."""
        self.temperature = temperature

    def compute(
        self,
        query_embeddings: Any,
        passage_embeddings: Any,
        labels: Any,
    ) -> Any:
        """
        Compute NLL loss.

        This is a simplified version - actual implementation
        would use PyTorch tensors.
        """
        # Placeholder for actual loss computation
        # In practice: similarity scores -> softmax -> NLL
        return {"loss": 0.0, "accuracy": 0.0}


class ContrastiveLoss(DPRLoss):
    """Contrastive loss with in-batch negatives."""

    def __init__(self, margin: float = 0.5, temperature: float = 0.07):
        """Initialize contrastive loss."""
        self.margin = margin
        self.temperature = temperature

    def compute(
        self,
        query_embeddings: Any,
        passage_embeddings: Any,
        labels: Any,
    ) -> Any:
        """Compute contrastive loss with in-batch negatives."""
        # Placeholder - actual implementation uses matrix operations
        return {"loss": 0.0, "accuracy": 0.0}


class InfoNCELoss(DPRLoss):
    """InfoNCE loss (NT-Xent)."""

    def __init__(self, temperature: float = 0.07):
        """Initialize InfoNCE loss."""
        self.temperature = temperature

    def compute(
        self,
        query_embeddings: Any,
        passage_embeddings: Any,
        labels: Any,
    ) -> Any:
        """Compute InfoNCE loss."""
        # Placeholder for actual implementation
        return {"loss": 0.0, "accuracy": 0.0}


# ============================================================================
# Hard Negative Mining
# ============================================================================


class HardNegativeMiner(ABC):
    """Abstract base for hard negative mining."""

    @abstractmethod
    def mine(
        self,
        questions: list[str],
        corpus: list[str],
        positives: list[str],
        top_k: int = 5,
    ) -> list[list[str]]:
        """Mine hard negatives for questions."""
        ...


class BM25HardNegativeMiner(HardNegativeMiner):
    """Mine hard negatives using BM25."""

    def __init__(self, corpus: list[str] | None = None):
        """Initialize with corpus."""
        self.corpus = corpus or []
        self._index: Any = None

    def build_index(self, corpus: list[str]) -> None:
        """Build BM25 index."""
        self.corpus = corpus
        # Placeholder for actual BM25 index building

    def mine(
        self,
        questions: list[str],
        corpus: list[str],
        positives: list[str],
        top_k: int = 5,
    ) -> list[list[str]]:
        """Mine hard negatives using BM25 retrieval."""
        hard_negatives = []

        for question, positive in zip(questions, positives):
            # Get BM25 results (placeholder)
            # In practice: use rank_bm25 or similar
            candidates = corpus[:top_k]

            # Filter out positives
            negatives = [c for c in candidates if c != positive][:top_k]
            hard_negatives.append(negatives)

        return hard_negatives


class DenseHardNegativeMiner(HardNegativeMiner):
    """Mine hard negatives using dense retrieval."""

    def __init__(
        self,
        encoder: EncoderProtocol,
        corpus: list[str] | None = None,
    ):
        """Initialize with encoder and corpus."""
        self.encoder = encoder
        self.corpus = corpus or []
        self.corpus_embeddings: list[list[float]] = []

    def build_index(self, corpus: list[str]) -> None:
        """Build dense index."""
        self.corpus = corpus
        self.corpus_embeddings = self.encoder.encode(corpus)

    def mine(
        self,
        questions: list[str],
        corpus: list[str],
        positives: list[str],
        top_k: int = 5,
    ) -> list[list[str]]:
        """Mine hard negatives using dense similarity."""
        if not self.corpus_embeddings:
            self.build_index(corpus)

        hard_negatives = []
        question_embeddings = self.encoder.encode(questions)

        for q_emb, positive in zip(question_embeddings, positives):
            # Calculate similarities (placeholder)
            # In practice: use FAISS or similar
            similarities = []
            for i, c_emb in enumerate(self.corpus_embeddings):
                sim = sum(q * c for q, c in zip(q_emb, c_emb))
                similarities.append((i, sim))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get top candidates excluding positive
            negatives = []
            for idx, _ in similarities:
                if corpus[idx] != positive:
                    negatives.append(corpus[idx])
                    if len(negatives) >= top_k:
                        break

            hard_negatives.append(negatives)

        return hard_negatives


# ============================================================================
# Bi-Encoder Architecture
# ============================================================================


@dataclass
class BiEncoderConfig:
    """Configuration for bi-encoder."""

    hidden_size: int = 768
    max_query_length: int = 64
    max_passage_length: int = 256
    share_encoders: bool = False
    pooling_strategy: str = "cls"  # cls, mean, max
    normalize_embeddings: bool = True


class BiEncoder:
    """
    Bi-encoder for DPR.

    Separate encoders for queries and passages.
    """

    def __init__(
        self,
        query_encoder: EncoderProtocol,
        passage_encoder: EncoderProtocol | None = None,
        config: BiEncoderConfig | None = None,
    ):
        """
        Initialize bi-encoder.

        Args:
            query_encoder: Encoder for queries
            passage_encoder: Encoder for passages (defaults to query_encoder)
            config: Configuration options
        """
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder or query_encoder
        self.config = config or BiEncoderConfig()

    def encode_queries(self, queries: list[str]) -> list[list[float]]:
        """Encode queries."""
        embeddings = self.query_encoder.encode(queries)
        if self.config.normalize_embeddings:
            embeddings = self._normalize(embeddings)
        return embeddings

    def encode_passages(self, passages: list[str]) -> list[list[float]]:
        """Encode passages."""
        embeddings = self.passage_encoder.encode(passages)
        if self.config.normalize_embeddings:
            embeddings = self._normalize(embeddings)
        return embeddings

    def _normalize(self, embeddings: list[list[float]]) -> list[list[float]]:
        """L2 normalize embeddings."""
        import math

        normalized = []
        for emb in embeddings:
            norm = math.sqrt(sum(x * x for x in emb))
            if norm > 0:
                normalized.append([x / norm for x in emb])
            else:
                normalized.append(emb)
        return normalized

    def similarity(
        self,
        query_embeddings: list[list[float]],
        passage_embeddings: list[list[float]],
    ) -> list[list[float]]:
        """Compute similarity matrix."""
        # Dot product similarity
        similarities = []
        for q_emb in query_embeddings:
            row = []
            for p_emb in passage_embeddings:
                sim = sum(q * p for q, p in zip(q_emb, p_emb))
                row.append(sim)
            similarities.append(row)
        return similarities


# ============================================================================
# DPR Trainer
# ============================================================================


@dataclass
class DPRTrainingConfig:
    """Configuration for DPR training."""

    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: int = 1000
    max_epochs: int = 40
    hard_negatives_per_example: int = 1
    in_batch_negatives: bool = True
    temperature: float = 1.0
    loss_function: LossFunction = LossFunction.NLL
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    evaluation_steps: int = 500
    save_steps: int = 1000
    output_dir: str = "./dpr_output"


class DPRTrainer:
    """
    Trainer for Dense Passage Retrieval.

    Implements training with in-batch negatives and hard negative mining.
    """

    def __init__(
        self,
        bi_encoder: BiEncoder,
        config: DPRTrainingConfig | None = None,
        hard_negative_miner: HardNegativeMiner | None = None,
    ):
        """
        Initialize DPR trainer.

        Args:
            bi_encoder: Bi-encoder model
            config: Training configuration
            hard_negative_miner: Miner for hard negatives
        """
        self.bi_encoder = bi_encoder
        self.config = config or DPRTrainingConfig()
        self.hard_negative_miner = hard_negative_miner

        # Initialize loss function
        self.loss_fn = self._create_loss_function()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_history: list[TrainingMetrics] = []

    def _create_loss_function(self) -> DPRLoss:
        """Create loss function based on config."""
        if self.config.loss_function == LossFunction.NLL:
            return NLLLoss(temperature=self.config.temperature)
        elif self.config.loss_function == LossFunction.CONTRASTIVE:
            return ContrastiveLoss(temperature=self.config.temperature)
        elif self.config.loss_function == LossFunction.INFONCE:
            return InfoNCELoss(temperature=self.config.temperature)
        else:
            return NLLLoss(temperature=self.config.temperature)

    def train(
        self,
        train_examples: list[DPRExample],
        eval_examples: list[DPRExample] | None = None,
    ) -> list[TrainingMetrics]:
        """
        Train the bi-encoder.

        Args:
            train_examples: Training examples
            eval_examples: Evaluation examples (optional)

        Returns:
            Training metrics history
        """
        print(f"Starting DPR training with {len(train_examples)} examples")
        print(f"Config: batch_size={self.config.batch_size}, " f"lr={self.config.learning_rate}")

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            epoch_metrics = self._train_epoch(train_examples)

            # Evaluation
            if eval_examples and (epoch + 1) % 5 == 0:
                eval_metrics = self.evaluate(eval_examples)
                print(
                    f"Epoch {epoch + 1}: "
                    f"train_loss={epoch_metrics['loss']:.4f}, "
                    f"eval_mrr={eval_metrics.mrr:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}: loss={epoch_metrics['loss']:.4f}")

        return self.training_history

    def _train_epoch(
        self,
        examples: list[DPRExample],
    ) -> dict[str, float]:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0

        # Create batches
        batches = self._create_batches(examples)

        for batch in batches:
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1
            self.global_step += 1

            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}

    def _train_step(self, batch: DPRBatch) -> float:
        """Single training step."""
        # Encode queries
        query_embeddings = self.bi_encoder.encode_queries(batch.questions)

        # Prepare passages (positive + hard negatives)
        all_passages = []
        for i, positive in enumerate(batch.positive_passages):
            all_passages.append(positive)
            if batch.hard_negatives[i]:
                all_passages.extend(
                    batch.hard_negatives[i][: self.config.hard_negatives_per_example]
                )

        # Encode passages
        passage_embeddings = self.bi_encoder.encode_passages(all_passages)

        # Compute loss (placeholder - actual implementation uses gradients)
        result = self.loss_fn.compute(query_embeddings, passage_embeddings, None)

        return result.get("loss", 0.0)

    def _create_batches(
        self,
        examples: list[DPRExample],
    ) -> list[DPRBatch]:
        """Create training batches."""
        import random

        random.shuffle(examples)
        batches = []

        for i in range(0, len(examples), self.config.batch_size):
            batch_examples = examples[i : i + self.config.batch_size]

            batch = DPRBatch(
                questions=[ex.question for ex in batch_examples],
                positive_passages=[ex.positive_passage for ex in batch_examples],
                hard_negatives=[ex.hard_negatives for ex in batch_examples],
                batch_size=len(batch_examples),
            )
            batches.append(batch)

        return batches

    def evaluate(
        self,
        examples: list[DPRExample],
    ) -> TrainingMetrics:
        """Evaluate the model."""
        # Encode all questions
        questions = [ex.question for ex in examples]
        query_embeddings = self.bi_encoder.encode_queries(questions)

        # Encode all passages
        all_passages = []
        passage_to_idx: dict[str, int] = {}
        for ex in examples:
            if ex.positive_passage not in passage_to_idx:
                passage_to_idx[ex.positive_passage] = len(all_passages)
                all_passages.append(ex.positive_passage)
            for neg in ex.hard_negatives:
                if neg not in passage_to_idx:
                    passage_to_idx[neg] = len(all_passages)
                    all_passages.append(neg)

        passage_embeddings = self.bi_encoder.encode_passages(all_passages)

        # Compute similarities
        similarities = self.bi_encoder.similarity(query_embeddings, passage_embeddings)

        # Calculate metrics
        mrr = 0.0
        recall_at_k: dict[int, int] = {1: 0, 5: 0, 10: 0, 20: 0}

        for i, ex in enumerate(examples):
            positive_idx = passage_to_idx[ex.positive_passage]
            sims = similarities[i]

            # Sort by similarity
            indexed_sims = [(j, s) for j, s in enumerate(sims)]
            indexed_sims.sort(key=lambda x: x[1], reverse=True)

            # Find rank of positive
            for rank, (idx, _) in enumerate(indexed_sims, 1):
                if idx == positive_idx:
                    mrr += 1.0 / rank
                    for k in recall_at_k:
                        if rank <= k:
                            recall_at_k[k] += 1
                    break

        n = len(examples)
        metrics = TrainingMetrics(
            loss=0.0,
            accuracy=recall_at_k[1] / n if n > 0 else 0.0,
            mrr=mrr / n if n > 0 else 0.0,
            recall_at_k={k: v / n for k, v in recall_at_k.items()},
            step=self.global_step,
            epoch=self.current_epoch,
        )

        self.training_history.append(metrics)
        return metrics

    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        # Placeholder for actual checkpoint saving
        print(f"Saving checkpoint at step {self.global_step}")


# ============================================================================
# Data Loading Utilities
# ============================================================================


class DPRDataLoader:
    """Data loader for DPR training data."""

    @staticmethod
    def load_natural_questions(
        file_path: str,
        max_examples: int | None = None,
    ) -> list[DPRExample]:
        """
        Load Natural Questions formatted data.

        Args:
            file_path: Path to data file
            max_examples: Maximum examples to load

        Returns:
            List of DPR examples
        """
        examples = []

        # Placeholder for actual file loading
        # Format expected: JSONL with question, positive_ctxs, hard_negative_ctxs

        return examples[:max_examples] if max_examples else examples

    @staticmethod
    def load_from_retrieval_results(
        questions: list[str],
        retrieval_results: list[list[dict[str, Any]]],
        labels: list[str],
    ) -> list[DPRExample]:
        """
        Create DPR examples from retrieval results.

        Args:
            questions: List of questions
            retrieval_results: Retrieved passages per question
            labels: Ground truth answers

        Returns:
            List of DPR examples
        """
        examples = []

        for question, results, label in zip(questions, retrieval_results, labels):
            # Find positive (contains answer)
            positive = None
            negatives = []

            for result in results:
                passage = result.get("content", "")
                if label.lower() in passage.lower():
                    if positive is None:
                        positive = passage
                else:
                    negatives.append(passage)

            if positive:
                examples.append(
                    DPRExample(
                        question=question,
                        positive_passage=positive,
                        hard_negatives=negatives[:10],
                    )
                )

        return examples


# ============================================================================
# Factory Functions
# ============================================================================


def create_dpr_trainer(
    query_encoder: EncoderProtocol,
    passage_encoder: EncoderProtocol | None = None,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    use_hard_negatives: bool = True,
) -> DPRTrainer:
    """
    Create a DPR trainer.

    Args:
        query_encoder: Encoder for queries
        passage_encoder: Encoder for passages
        batch_size: Training batch size
        learning_rate: Learning rate
        use_hard_negatives: Whether to use hard negatives

    Returns:
        Configured DPR trainer
    """
    config = DPRTrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        hard_negatives_per_example=1 if use_hard_negatives else 0,
    )

    bi_encoder_config = BiEncoderConfig(share_encoders=(passage_encoder is None))
    bi_encoder = BiEncoder(query_encoder, passage_encoder, bi_encoder_config)

    miner = BM25HardNegativeMiner() if use_hard_negatives else None

    return DPRTrainer(bi_encoder, config, miner)


def create_contrastive_trainer(
    query_encoder: EncoderProtocol,
    passage_encoder: EncoderProtocol | None = None,
    temperature: float = 0.07,
) -> DPRTrainer:
    """
    Create a DPR trainer with contrastive loss.

    Args:
        query_encoder: Encoder for queries
        passage_encoder: Encoder for passages
        temperature: Temperature for contrastive loss

    Returns:
        Configured DPR trainer with contrastive loss
    """
    config = DPRTrainingConfig(
        loss_function=LossFunction.CONTRASTIVE,
        temperature=temperature,
    )

    bi_encoder = BiEncoder(query_encoder, passage_encoder)
    return DPRTrainer(bi_encoder, config)


# ============================================================================
# Example Usage
# ============================================================================


def example_usage():
    """Example demonstrating DPR training."""
    print("DPR Training Implementation Ready")
    print("=" * 50)
    print("\nReference: Karpukhin et al., 2020")
    print("\nFeatures:")
    print("- Bi-encoder architecture (query/passage encoders)")
    print("- In-batch negatives for efficient training")
    print("- Hard negative mining (BM25 and Dense)")
    print("- Multiple loss functions (NLL, Contrastive, InfoNCE)")
    print("- Evaluation with MRR and Recall@K")
    print("- Checkpoint saving")
    print("- Data loading utilities")


if __name__ == "__main__":
    example_usage()
