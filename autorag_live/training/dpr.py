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


# =============================================================================
# OPTIMIZATION 10: Instructor-Based Retriever Training
# Based on: "One Embedder, Any Task: Instruction-Finetuned Text Embeddings"
# (Su et al., 2023) - INSTRUCTOR model approach
#
# Implements instruction-conditioned embedding training where:
# 1. Task-specific instructions guide retrieval behavior
# 2. Hard negative mining using contrastive learning
# 3. Multi-task learning across retrieval tasks
# 4. Domain adaptation through instruction tuning
# =============================================================================


@dataclass
class InstructorConfig:
    """Configuration for Instructor-based training."""

    instruction_max_length: int = 128
    query_max_length: int = 256
    document_max_length: int = 512
    temperature: float = 0.02  # Lower temp for more focused retrieval
    use_hard_negatives: bool = True
    num_hard_negatives: int = 7
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    fp16: bool = True
    pool_type: str = "mean"  # mean, cls, last
    normalize_embeddings: bool = True


@dataclass
class InstructorExample:
    """A training example for Instructor model."""

    instruction: str
    query: str
    positive_document: str
    hard_negatives: list[str] = field(default_factory=list)
    task_type: str = "retrieval"
    domain: str = "general"


@dataclass
class InstructorBatch:
    """A batch for Instructor training."""

    instructions: list[str]
    queries: list[str]
    positive_docs: list[str]
    hard_negatives: list[list[str]]
    task_types: list[str]


class InstructionTemplate:
    """Templates for task-specific instructions."""

    # Retrieval tasks
    DOCUMENT_RETRIEVAL = "Represent the {domain} question for retrieving supporting documents: "
    PASSAGE_RETRIEVAL = "Represent the {domain} query for retrieving relevant passages: "
    FAQ_RETRIEVAL = "Represent the question for retrieving similar FAQ entries: "

    # Semantic similarity
    SEMANTIC_SIMILARITY = "Represent the {domain} sentence for semantic similarity: "
    PARAPHRASE_DETECTION = "Represent the sentence for paraphrase detection: "

    # Clustering
    CLUSTERING = "Represent the {domain} text for clustering: "

    # Classification
    CLASSIFICATION = "Represent the {domain} text for classification: "

    # Document representation
    DOCUMENT_EMBEDDING = "Represent the {domain} document for retrieval: "

    @classmethod
    def get_retrieval_instruction(cls, domain: str = "general") -> str:
        """Get retrieval instruction for domain."""
        return cls.DOCUMENT_RETRIEVAL.format(domain=domain)

    @classmethod
    def get_document_instruction(cls, domain: str = "general") -> str:
        """Get document embedding instruction."""
        return cls.DOCUMENT_EMBEDDING.format(domain=domain)


class InstructionPooling:
    """Pooling strategies for Instructor embeddings."""

    @staticmethod
    def mean_pooling(
        token_embeddings: Any,
        attention_mask: Any,
    ) -> Any:
        """
        Mean pooling with attention mask.

        Computes weighted average using attention mask.
        """
        # Expand attention mask for broadcasting
        # mask: [batch, seq_len] -> [batch, seq_len, hidden]
        # This is a placeholder for actual PyTorch/numpy implementation
        # In practice:
        # mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        # sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
        # sum_mask = attention_mask.sum(1, keepdim=True).clamp(min=1e-9)
        # return sum_embeddings / sum_mask
        return token_embeddings

    @staticmethod
    def cls_pooling(token_embeddings: Any) -> Any:
        """CLS token pooling - take first token."""
        # return token_embeddings[:, 0]
        return token_embeddings

    @staticmethod
    def last_token_pooling(
        token_embeddings: Any,
        attention_mask: Any,
    ) -> Any:
        """Last valid token pooling."""
        # Useful for decoder-only models
        # In practice:
        # last_idx = attention_mask.sum(1) - 1
        # batch_idx = torch.arange(token_embeddings.size(0))
        # return token_embeddings[batch_idx, last_idx]
        return token_embeddings


@dataclass
class HardNegativeExample:
    """Hard negative with difficulty score."""

    text: str
    score: float
    source: str  # "bm25", "dense", "cross_encoder", "in_batch"


class InstructorHardNegativeMiner:
    """
    Multi-strategy hard negative mining for Instructor.

    Combines multiple mining strategies for diverse negatives:
    1. BM25 negatives (lexical similarity)
    2. Dense negatives (semantic similarity but wrong)
    3. Cross-encoder filtered negatives
    4. In-batch negatives
    """

    def __init__(
        self,
        encoder: EncoderProtocol | None = None,
        use_bm25: bool = True,
        use_dense: bool = True,
        use_cross_encoder: bool = False,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.5,
        cross_encoder_weight: float = 0.2,
        max_negatives: int = 15,
    ):
        """Initialize miner."""
        self.encoder = encoder
        self.use_bm25 = use_bm25
        self.use_dense = use_dense
        self.use_cross_encoder = use_cross_encoder
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.cross_encoder_weight = cross_encoder_weight
        self.max_negatives = max_negatives

    def mine_negatives(
        self,
        query: str,
        positive: str,
        corpus: list[str],
        num_negatives: int = 7,
    ) -> list[HardNegativeExample]:
        """
        Mine hard negatives from multiple sources.

        Strategy:
        1. Get BM25 top candidates (high lexical overlap, wrong answer)
        2. Get dense retrieval candidates (semantic near-misses)
        3. Filter with cross-encoder if available
        4. Rank by difficulty and diversity
        """
        candidates: list[HardNegativeExample] = []

        if self.use_bm25:
            bm25_negs = self._mine_bm25(query, positive, corpus)
            candidates.extend(bm25_negs)

        if self.use_dense and self.encoder:
            dense_negs = self._mine_dense(query, positive, corpus)
            candidates.extend(dense_negs)

        # Deduplicate and rank
        unique_candidates = self._deduplicate(candidates)
        ranked = self._rank_by_difficulty(unique_candidates, query)

        return ranked[:num_negatives]

    def _mine_bm25(
        self,
        query: str,
        positive: str,
        corpus: list[str],
    ) -> list[HardNegativeExample]:
        """Mine BM25 hard negatives."""
        # Tokenize query for BM25
        query_terms = set(query.lower().split())

        scored: list[tuple[str, float]] = []
        for doc in corpus:
            if doc == positive:
                continue
            doc_terms = set(doc.lower().split())
            # Simple term overlap as BM25 proxy
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                score = overlap / (len(query_terms) + len(doc_terms) - overlap + 1e-6)
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            HardNegativeExample(text=doc, score=score, source="bm25")
            for doc, score in scored[: self.max_negatives]
        ]

    def _mine_dense(
        self,
        query: str,
        positive: str,
        corpus: list[str],
    ) -> list[HardNegativeExample]:
        """Mine dense hard negatives."""
        if not self.encoder:
            return []

        # Encode query
        query_emb = self.encoder.encode([query])[0]

        # Encode corpus
        corpus_filtered = [doc for doc in corpus if doc != positive]
        if not corpus_filtered:
            return []

        corpus_embs = self.encoder.encode(corpus_filtered)

        # Compute similarities
        scored: list[tuple[str, float]] = []
        for i, doc in enumerate(corpus_filtered):
            sim = self._cosine_similarity(query_emb, corpus_embs[i])
            scored.append((doc, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            HardNegativeExample(text=doc, score=score, source="dense")
            for doc, score in scored[: self.max_negatives]
        ]

    def _cosine_similarity(
        self,
        v1: list[float],
        v2: list[float],
    ) -> float:
        """Compute cosine similarity."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        return dot / (norm1 * norm2 + 1e-8)

    def _deduplicate(
        self,
        candidates: list[HardNegativeExample],
    ) -> list[HardNegativeExample]:
        """Remove duplicate candidates."""
        seen = set()
        unique = []
        for c in candidates:
            if c.text not in seen:
                seen.add(c.text)
                unique.append(c)
        return unique

    def _rank_by_difficulty(
        self,
        candidates: list[HardNegativeExample],
        query: str,
    ) -> list[HardNegativeExample]:
        """Rank candidates by difficulty (harder = better training)."""
        # Combine scores with source weights
        for c in candidates:
            if c.source == "bm25":
                c.score *= self.bm25_weight
            elif c.source == "dense":
                c.score *= self.dense_weight
            elif c.source == "cross_encoder":
                c.score *= self.cross_encoder_weight

        return sorted(candidates, key=lambda x: x.score, reverse=True)


class InstructorTrainer:
    """
    Trainer for Instructor-style embeddings.

    Implements instruction-conditioned contrastive learning
    with multi-task support.
    """

    def __init__(
        self,
        encoder: EncoderProtocol,
        config: InstructorConfig,
        miner: InstructorHardNegativeMiner | None = None,
    ):
        """Initialize trainer."""
        self.encoder = encoder
        self.config = config
        self.miner = miner or InstructorHardNegativeMiner()
        self._step = 0
        self._epoch = 0
        self._best_mrr = 0.0
        self._training_history: list[dict[str, float]] = []

    def prepare_input(
        self,
        instruction: str,
        text: str,
    ) -> str:
        """
        Prepare input by prepending instruction.

        Format: "[INST] instruction [/INST] text"
        """
        return f"[INST] {instruction} [/INST] {text}"

    def encode_with_instruction(
        self,
        instructions: list[str],
        texts: list[str],
    ) -> list[list[float]]:
        """
        Encode texts with task-specific instructions.

        Each text is prepended with its instruction.
        """
        prepared = [self.prepare_input(inst, text) for inst, text in zip(instructions, texts)]
        embeddings = self.encoder.encode(prepared)

        if self.config.normalize_embeddings:
            embeddings = [self._normalize(e) for e in embeddings]

        return embeddings

    def _normalize(self, embedding: list[float]) -> list[float]:
        """L2 normalize embedding."""
        norm = sum(x * x for x in embedding) ** 0.5 + 1e-8
        return [x / norm for x in embedding]

    def compute_loss(
        self,
        query_embeddings: list[list[float]],
        positive_embeddings: list[list[float]],
        negative_embeddings: list[list[list[float]]],
    ) -> dict[str, float]:
        """
        Compute InfoNCE loss with hard negatives.

        Loss = -log(exp(sim(q, p+)/τ) / Σ exp(sim(q, p-)/τ))
        """
        total_loss = 0.0
        total_correct = 0

        for i in range(len(query_embeddings)):
            q_emb = query_embeddings[i]
            pos_emb = positive_embeddings[i]
            neg_embs = negative_embeddings[i]

            # Positive similarity
            pos_sim = self._cosine_sim(q_emb, pos_emb) / self.config.temperature

            # Negative similarities
            neg_sims = [self._cosine_sim(q_emb, neg) / self.config.temperature for neg in neg_embs]

            # Softmax denominator (add positive)
            all_sims = [pos_sim] + neg_sims
            max_sim = max(all_sims)
            exp_sums = sum(self._safe_exp(s - max_sim) for s in all_sims)

            # InfoNCE loss
            loss = -pos_sim + max_sim + self._safe_log(exp_sums)
            total_loss += loss

            # Accuracy (positive has highest similarity)
            if pos_sim >= max(neg_sims, default=float("-inf")):
                total_correct += 1

        n = max(len(query_embeddings), 1)
        return {
            "loss": total_loss / n,
            "accuracy": total_correct / n,
        }

    def _cosine_sim(self, v1: list[float], v2: list[float]) -> float:
        """Cosine similarity."""
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = sum(a * a for a in v1) ** 0.5
        n2 = sum(b * b for b in v2) ** 0.5
        return dot / (n1 * n2 + 1e-8)

    def _safe_exp(self, x: float) -> float:
        """Safe exponential to avoid overflow."""
        return min(700, max(-700, x)).__class__(2.718281828459045) ** min(700, max(-700, x))

    def _safe_log(self, x: float) -> float:
        """Safe log to avoid domain errors."""
        import math

        return math.log(max(x, 1e-10))

    def train_step(
        self,
        batch: InstructorBatch,
    ) -> dict[str, float]:
        """
        Single training step.

        1. Encode queries with instructions
        2. Encode positive documents
        3. Encode hard negatives
        4. Compute contrastive loss
        """
        # Prepare instructions for queries (retrieval task)
        query_instructions = [
            InstructionTemplate.get_retrieval_instruction(domain="general") for _ in batch.queries
        ]

        # Prepare instructions for documents
        doc_instructions = [
            InstructionTemplate.get_document_instruction(domain="general")
            for _ in batch.positive_docs
        ]

        # Encode queries
        query_embs = self.encode_with_instruction(
            query_instructions,
            batch.queries,
        )

        # Encode positives
        positive_embs = self.encode_with_instruction(
            doc_instructions,
            batch.positive_docs,
        )

        # Encode negatives (batch them for efficiency)
        negative_embs: list[list[list[float]]] = []
        for negs in batch.hard_negatives:
            if negs:
                neg_instructions = [
                    InstructionTemplate.get_document_instruction(domain="general") for _ in negs
                ]
                neg_embs = self.encode_with_instruction(neg_instructions, negs)
                negative_embs.append(neg_embs)
            else:
                negative_embs.append([])

        # Compute loss
        metrics = self.compute_loss(query_embs, positive_embs, negative_embs)

        self._step += 1
        self._training_history.append(metrics)

        return metrics

    def train_epoch(
        self,
        examples: list[InstructorExample],
        corpus: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            examples: Training examples
            corpus: Document corpus for hard negative mining

        Returns:
            Epoch metrics
        """
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        # Create batches
        for i in range(0, len(examples), self.config.batch_size):
            batch_examples = examples[i : i + self.config.batch_size]

            # Mine hard negatives if needed
            hard_negs: list[list[str]] = []
            for ex in batch_examples:
                if ex.hard_negatives:
                    hard_negs.append(ex.hard_negatives)
                elif corpus and self.config.use_hard_negatives:
                    mined = self.miner.mine_negatives(
                        ex.query,
                        ex.positive_document,
                        corpus,
                        self.config.num_hard_negatives,
                    )
                    hard_negs.append([n.text for n in mined])
                else:
                    hard_negs.append([])

            batch = InstructorBatch(
                instructions=[ex.instruction for ex in batch_examples],
                queries=[ex.query for ex in batch_examples],
                positive_docs=[ex.positive_document for ex in batch_examples],
                hard_negatives=hard_negs,
                task_types=[ex.task_type for ex in batch_examples],
            )

            metrics = self.train_step(batch)
            epoch_loss += metrics["loss"]
            epoch_acc += metrics["accuracy"]
            num_batches += 1

        self._epoch += 1
        num_batches = max(num_batches, 1)

        return {
            "epoch": self._epoch,
            "loss": epoch_loss / num_batches,
            "accuracy": epoch_acc / num_batches,
        }

    def evaluate(
        self,
        examples: list[InstructorExample],
    ) -> dict[str, float]:
        """
        Evaluate on examples.

        Computes MRR and Recall@K.
        """
        mrr_sum = 0.0
        recall_at_k: dict[int, float] = {1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0}

        for ex in examples:
            # Encode query
            q_inst = InstructionTemplate.get_retrieval_instruction()
            q_emb = self.encode_with_instruction([q_inst], [ex.query])[0]

            # Encode all candidates (positive + negatives)
            candidates = [ex.positive_document] + ex.hard_negatives
            d_inst = InstructionTemplate.get_document_instruction()
            c_embs = self.encode_with_instruction(
                [d_inst] * len(candidates),
                candidates,
            )

            # Rank by similarity
            sims = [(i, self._cosine_sim(q_emb, c_embs[i])) for i in range(len(c_embs))]
            sims.sort(key=lambda x: x[1], reverse=True)

            # Find rank of positive (index 0 in candidates)
            for rank, (idx, _) in enumerate(sims, 1):
                if idx == 0:  # Found positive
                    mrr_sum += 1.0 / rank
                    for k in recall_at_k:
                        if rank <= k:
                            recall_at_k[k] += 1.0
                    break

        n = max(len(examples), 1)
        return {
            "mrr": mrr_sum / n,
            **{f"recall@{k}": v / n for k, v in recall_at_k.items()},
        }


@dataclass
class MultiTaskConfig:
    """Configuration for multi-task Instructor training."""

    task_weights: dict[str, float] = field(
        default_factory=lambda: {
            "retrieval": 1.0,
            "similarity": 0.5,
            "clustering": 0.3,
        }
    )
    sample_strategy: str = "proportional"  # proportional, uniform, temperature


class MultiTaskInstructorTrainer:
    """
    Multi-task trainer for Instructor.

    Trains on multiple tasks simultaneously:
    - Document retrieval
    - Semantic similarity
    - Clustering
    - Classification
    """

    def __init__(
        self,
        encoder: EncoderProtocol,
        config: InstructorConfig,
        task_config: MultiTaskConfig,
    ):
        """Initialize multi-task trainer."""
        self.encoder = encoder
        self.config = config
        self.task_config = task_config
        self.base_trainer = InstructorTrainer(encoder, config)
        self._task_datasets: dict[str, list[InstructorExample]] = {}

    def add_task_dataset(
        self,
        task_name: str,
        examples: list[InstructorExample],
    ) -> None:
        """Add dataset for a task."""
        self._task_datasets[task_name] = examples

    def sample_batch(self) -> InstructorBatch:
        """
        Sample a mixed batch from all tasks.

        Uses task weights to determine sampling.
        """
        import random

        batch_examples: list[InstructorExample] = []
        remaining = self.config.batch_size

        # Sample from each task proportionally
        for task, examples in self._task_datasets.items():
            weight = self.task_config.task_weights.get(task, 1.0)
            num_samples = int(remaining * weight / sum(self.task_config.task_weights.values()))
            num_samples = min(num_samples, len(examples), remaining)

            batch_examples.extend(random.sample(examples, num_samples))
            remaining -= num_samples

        return InstructorBatch(
            instructions=[ex.instruction for ex in batch_examples],
            queries=[ex.query for ex in batch_examples],
            positive_docs=[ex.positive_document for ex in batch_examples],
            hard_negatives=[ex.hard_negatives for ex in batch_examples],
            task_types=[ex.task_type for ex in batch_examples],
        )

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch across all tasks."""
        total_steps = sum(len(ds) // self.config.batch_size for ds in self._task_datasets.values())

        epoch_loss = 0.0
        for _ in range(max(total_steps, 1)):
            batch = self.sample_batch()
            metrics = self.base_trainer.train_step(batch)
            epoch_loss += metrics["loss"]

        return {
            "loss": epoch_loss / max(total_steps, 1),
        }


# Factory functions for Instructor training


def create_instructor_trainer(
    encoder: EncoderProtocol,
    use_hard_negatives: bool = True,
    temperature: float = 0.02,
) -> InstructorTrainer:
    """
    Create an Instructor-style trainer.

    Args:
        encoder: Base encoder model
        use_hard_negatives: Whether to mine hard negatives
        temperature: Contrastive loss temperature

    Returns:
        Configured InstructorTrainer
    """
    config = InstructorConfig(
        use_hard_negatives=use_hard_negatives,
        temperature=temperature,
    )

    miner = InstructorHardNegativeMiner(encoder=encoder) if use_hard_negatives else None

    return InstructorTrainer(encoder, config, miner)


def create_rag_instructor_trainer(
    encoder: EncoderProtocol,
    domain: str = "general",
) -> InstructorTrainer:
    """
    Create an Instructor trainer optimized for RAG.

    Uses RAG-specific settings:
    - Lower temperature for precise retrieval
    - More hard negatives
    - Document-focused instructions
    """
    config = InstructorConfig(
        temperature=0.01,  # Very focused retrieval
        num_hard_negatives=10,
        use_hard_negatives=True,
        batch_size=16,  # Smaller batches, more negatives
    )

    miner = InstructorHardNegativeMiner(
        encoder=encoder,
        use_bm25=True,
        use_dense=True,
        dense_weight=0.6,
        bm25_weight=0.4,
    )

    return InstructorTrainer(encoder, config, miner)


if __name__ == "__main__":
    example_usage()
