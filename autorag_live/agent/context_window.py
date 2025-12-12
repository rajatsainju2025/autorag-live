"""Context Window Manager for AutoRAG-Live.

Intelligent context window management with:
- Token counting and estimation
- Smart truncation strategies
- Priority-based document selection
- Context budget allocation
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TruncationStrategy(Enum):
    """Strategies for truncating content."""

    HEAD = "head"  # Keep beginning
    TAIL = "tail"  # Keep ending
    MIDDLE_OUT = "middle_out"  # Remove from middle
    SENTENCE_BOUNDARY = "sentence_boundary"  # Truncate at sentence
    PARAGRAPH_BOUNDARY = "paragraph_boundary"  # Truncate at paragraph
    SEMANTIC = "semantic"  # Keep semantically important parts


class PriorityStrategy(Enum):
    """Strategies for prioritizing documents."""

    SCORE = "score"  # By relevance score
    POSITION = "position"  # By original position
    LENGTH = "length"  # Shorter documents first
    RECENCY = "recency"  # By timestamp
    CUSTOM = "custom"  # Custom priority function


@dataclass
class ContextDocument:
    """Represents a document in the context window."""

    content: str
    doc_id: str
    score: float = 1.0
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int | None = None
    is_truncated: bool = False

    def __post_init__(self) -> None:
        """Estimate tokens if not provided."""
        if self.token_count is None:
            self.token_count = TokenCounter.estimate_tokens(self.content)


@dataclass
class ContextBudget:
    """Budget allocation for context window."""

    total_tokens: int = 4096
    system_prompt_tokens: int = 500
    query_tokens: int = 200
    response_reserve_tokens: int = 500
    document_tokens: int = 0

    def __post_init__(self) -> None:
        """Calculate available document tokens."""
        self.document_tokens = max(
            0,
            self.total_tokens
            - self.system_prompt_tokens
            - self.query_tokens
            - self.response_reserve_tokens,
        )

    @property
    def available_tokens(self) -> int:
        """Get available tokens for documents."""
        return self.document_tokens


class TokenCounter:
    """Token counting utilities."""

    # Approximate characters per token for different models
    CHARS_PER_TOKEN = {
        "gpt-4": 4.0,
        "gpt-3.5-turbo": 4.0,
        "claude": 3.5,
        "llama": 4.0,
        "default": 4.0,
    }

    @classmethod
    def estimate_tokens(
        cls,
        text: str,
        model: str = "default",
    ) -> int:
        """Estimate token count for text.

        Args:
            text: Text to count
            model: Model name for estimation

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        chars_per_token = cls.CHARS_PER_TOKEN.get(model, cls.CHARS_PER_TOKEN["default"])
        return int(len(text) / chars_per_token)

    @classmethod
    def count_tokens_tiktoken(
        cls,
        text: str,
        model: str = "gpt-4",
    ) -> int:
        """Count tokens using tiktoken (if available).

        Args:
            text: Text to count
            model: Model name

        Returns:
            Token count
        """
        try:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))
        except ImportError:
            return cls.estimate_tokens(text, model)

    @classmethod
    def tokens_to_chars(
        cls,
        tokens: int,
        model: str = "default",
    ) -> int:
        """Convert tokens to approximate character count.

        Args:
            tokens: Token count
            model: Model name

        Returns:
            Approximate character count
        """
        chars_per_token = cls.CHARS_PER_TOKEN.get(model, cls.CHARS_PER_TOKEN["default"])
        return int(tokens * chars_per_token)


class BaseTruncator(ABC):
    """Abstract base class for content truncation."""

    @abstractmethod
    def truncate(
        self,
        content: str,
        max_tokens: int,
        model: str = "default",
    ) -> str:
        """Truncate content to fit token limit.

        Args:
            content: Content to truncate
            max_tokens: Maximum tokens
            model: Model name for token estimation

        Returns:
            Truncated content
        """
        pass


class HeadTruncator(BaseTruncator):
    """Truncate from the end, keeping the beginning."""

    def truncate(
        self,
        content: str,
        max_tokens: int,
        model: str = "default",
    ) -> str:
        """Keep the beginning of content."""
        max_chars = TokenCounter.tokens_to_chars(max_tokens, model)

        if len(content) <= max_chars:
            return content

        return content[:max_chars] + "..."


class TailTruncator(BaseTruncator):
    """Truncate from the beginning, keeping the end."""

    def truncate(
        self,
        content: str,
        max_tokens: int,
        model: str = "default",
    ) -> str:
        """Keep the end of content."""
        max_chars = TokenCounter.tokens_to_chars(max_tokens, model)

        if len(content) <= max_chars:
            return content

        return "..." + content[-max_chars:]


class MiddleOutTruncator(BaseTruncator):
    """Remove content from the middle."""

    def truncate(
        self,
        content: str,
        max_tokens: int,
        model: str = "default",
    ) -> str:
        """Remove middle content, keep beginning and end."""
        max_chars = TokenCounter.tokens_to_chars(max_tokens, model)

        if len(content) <= max_chars:
            return content

        half_length = (max_chars - 20) // 2  # Reserve space for separator

        start = content[:half_length]
        end = content[-half_length:]

        return f"{start}\n\n[...content truncated...]\n\n{end}"


class SentenceBoundaryTruncator(BaseTruncator):
    """Truncate at sentence boundaries."""

    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

    def truncate(
        self,
        content: str,
        max_tokens: int,
        model: str = "default",
    ) -> str:
        """Truncate at sentence boundary."""
        max_chars = TokenCounter.tokens_to_chars(max_tokens, model)

        if len(content) <= max_chars:
            return content

        # Split into sentences
        sentences = self.SENTENCE_ENDINGS.split(content)

        result: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence) + 1  # +1 for space

            if current_length + sentence_length > max_chars:
                break

            result.append(sentence)
            current_length += sentence_length

        if not result:
            # Fall back to head truncation
            return HeadTruncator().truncate(content, max_tokens, model)

        return " ".join(result)


class ParagraphBoundaryTruncator(BaseTruncator):
    """Truncate at paragraph boundaries."""

    def truncate(
        self,
        content: str,
        max_tokens: int,
        model: str = "default",
    ) -> str:
        """Truncate at paragraph boundary."""
        max_chars = TokenCounter.tokens_to_chars(max_tokens, model)

        if len(content) <= max_chars:
            return content

        # Split into paragraphs
        paragraphs = re.split(r"\n\s*\n", content)

        result: list[str] = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para) + 2  # +2 for newlines

            if current_length + para_length > max_chars:
                break

            result.append(para)
            current_length += para_length

        if not result:
            # Fall back to sentence truncation
            return SentenceBoundaryTruncator().truncate(content, max_tokens, model)

        return "\n\n".join(result)


class ContextWindowManager:
    """Manages context window allocation and document selection."""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        model: str = "default",
        use_tiktoken: bool = False,
    ) -> None:
        """Initialize context window manager.

        Args:
            budget: Context budget configuration
            model: Model name for token counting
            use_tiktoken: Use tiktoken for accurate counting
        """
        self.budget = budget or ContextBudget()
        self.model = model
        self.use_tiktoken = use_tiktoken

        # Truncation strategies
        self._truncators: dict[TruncationStrategy, BaseTruncator] = {
            TruncationStrategy.HEAD: HeadTruncator(),
            TruncationStrategy.TAIL: TailTruncator(),
            TruncationStrategy.MIDDLE_OUT: MiddleOutTruncator(),
            TruncationStrategy.SENTENCE_BOUNDARY: SentenceBoundaryTruncator(),
            TruncationStrategy.PARAGRAPH_BOUNDARY: ParagraphBoundaryTruncator(),
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        if self.use_tiktoken:
            return TokenCounter.count_tokens_tiktoken(text, self.model)
        return TokenCounter.estimate_tokens(text, self.model)

    def update_budget(
        self,
        system_prompt: str | None = None,
        query: str | None = None,
    ) -> None:
        """Update budget based on actual content.

        Args:
            system_prompt: System prompt text
            query: Query text
        """
        if system_prompt:
            self.budget.system_prompt_tokens = self.count_tokens(system_prompt)

        if query:
            self.budget.query_tokens = self.count_tokens(query)

        # Recalculate document budget
        self.budget.document_tokens = max(
            0,
            self.budget.total_tokens
            - self.budget.system_prompt_tokens
            - self.budget.query_tokens
            - self.budget.response_reserve_tokens,
        )

    def select_documents(
        self,
        documents: list[ContextDocument],
        priority_strategy: PriorityStrategy = PriorityStrategy.SCORE,
        priority_fn: Callable[[ContextDocument], float] | None = None,
    ) -> list[ContextDocument]:
        """Select documents that fit within budget.

        Args:
            documents: Documents to select from
            priority_strategy: Strategy for prioritization
            priority_fn: Custom priority function

        Returns:
            Selected documents
        """
        if not documents:
            return []

        # Sort by priority
        sorted_docs = self._sort_by_priority(documents, priority_strategy, priority_fn)

        # Select documents within budget
        selected: list[ContextDocument] = []
        remaining_tokens = self.budget.available_tokens

        for doc in sorted_docs:
            doc_tokens = doc.token_count or self.count_tokens(doc.content)

            if doc_tokens <= remaining_tokens:
                selected.append(doc)
                remaining_tokens -= doc_tokens
            elif remaining_tokens > 100:  # Minimum useful size
                # Try to fit a truncated version
                truncated = self._truncate_document(doc, remaining_tokens)
                if truncated:
                    selected.append(truncated)
                    break

        return selected

    def _sort_by_priority(
        self,
        documents: list[ContextDocument],
        strategy: PriorityStrategy,
        custom_fn: Callable[[ContextDocument], float] | None = None,
    ) -> list[ContextDocument]:
        """Sort documents by priority."""
        if strategy == PriorityStrategy.SCORE:
            return sorted(documents, key=lambda d: d.score, reverse=True)
        elif strategy == PriorityStrategy.POSITION:
            return sorted(documents, key=lambda d: d.priority)
        elif strategy == PriorityStrategy.LENGTH:
            return sorted(documents, key=lambda d: d.token_count or 0)
        elif strategy == PriorityStrategy.RECENCY:
            return sorted(
                documents,
                key=lambda d: d.metadata.get("timestamp", 0),
                reverse=True,
            )
        elif strategy == PriorityStrategy.CUSTOM and custom_fn:
            return sorted(documents, key=custom_fn, reverse=True)

        return documents

    def _truncate_document(
        self,
        doc: ContextDocument,
        max_tokens: int,
        strategy: TruncationStrategy = TruncationStrategy.SENTENCE_BOUNDARY,
    ) -> ContextDocument | None:
        """Truncate a document to fit token limit.

        Args:
            doc: Document to truncate
            max_tokens: Maximum tokens
            strategy: Truncation strategy

        Returns:
            Truncated document or None
        """
        truncator = self._truncators.get(
            strategy, self._truncators[TruncationStrategy.HEAD]
        )

        truncated_content = truncator.truncate(doc.content, max_tokens, self.model)

        if not truncated_content or len(truncated_content) < 50:
            return None

        return ContextDocument(
            content=truncated_content,
            doc_id=doc.doc_id,
            score=doc.score,
            priority=doc.priority,
            metadata=doc.metadata,
            token_count=self.count_tokens(truncated_content),
            is_truncated=True,
        )

    def truncate(
        self,
        content: str,
        max_tokens: int,
        strategy: TruncationStrategy = TruncationStrategy.SENTENCE_BOUNDARY,
    ) -> str:
        """Truncate content using specified strategy.

        Args:
            content: Content to truncate
            max_tokens: Maximum tokens
            strategy: Truncation strategy

        Returns:
            Truncated content
        """
        truncator = self._truncators.get(
            strategy, self._truncators[TruncationStrategy.HEAD]
        )
        return truncator.truncate(content, max_tokens, self.model)

    def allocate_budget(
        self,
        documents: list[ContextDocument],
        equal_allocation: bool = False,
    ) -> dict[str, int]:
        """Allocate token budget across documents.

        Args:
            documents: Documents to allocate budget for
            equal_allocation: Use equal allocation instead of proportional

        Returns:
            Dictionary of doc_id to token allocation
        """
        if not documents:
            return {}

        total_available = self.budget.available_tokens
        num_docs = len(documents)

        allocations: dict[str, int] = {}

        if equal_allocation:
            per_doc = total_available // num_docs
            for doc in documents:
                allocations[doc.doc_id] = per_doc
        else:
            # Proportional allocation based on scores
            total_score = sum(doc.score for doc in documents)

            if total_score == 0:
                # Fall back to equal allocation
                per_doc = total_available // num_docs
                for doc in documents:
                    allocations[doc.doc_id] = per_doc
            else:
                for doc in documents:
                    proportion = doc.score / total_score
                    allocations[doc.doc_id] = int(total_available * proportion)

        return allocations

    def format_context(
        self,
        documents: list[ContextDocument],
        include_metadata: bool = False,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """Format selected documents into context string.

        Args:
            documents: Documents to format
            include_metadata: Include metadata in output
            separator: Separator between documents

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        formatted_parts: list[str] = []

        for i, doc in enumerate(documents, 1):
            parts = [f"[Document {i}]"]

            if include_metadata:
                if doc.metadata.get("source"):
                    parts.append(f"Source: {doc.metadata['source']}")
                if doc.score:
                    parts.append(f"Relevance: {doc.score:.2f}")
                if doc.is_truncated:
                    parts.append("(truncated)")

            parts.append(doc.content)

            formatted_parts.append("\n".join(parts))

        return separator.join(formatted_parts)

    def get_context_stats(
        self,
        documents: list[ContextDocument],
    ) -> dict[str, Any]:
        """Get statistics about context usage.

        Args:
            documents: Selected documents

        Returns:
            Context statistics
        """
        total_doc_tokens = sum(doc.token_count or 0 for doc in documents)
        truncated_count = sum(1 for doc in documents if doc.is_truncated)

        return {
            "total_budget": self.budget.total_tokens,
            "available_for_docs": self.budget.available_tokens,
            "used_by_docs": total_doc_tokens,
            "remaining": self.budget.available_tokens - total_doc_tokens,
            "utilization": total_doc_tokens / self.budget.available_tokens
            if self.budget.available_tokens > 0
            else 0,
            "document_count": len(documents),
            "truncated_count": truncated_count,
            "system_tokens": self.budget.system_prompt_tokens,
            "query_tokens": self.budget.query_tokens,
            "response_reserve": self.budget.response_reserve_tokens,
        }


# Convenience functions


def create_context_manager(
    total_tokens: int = 4096,
    response_reserve: int = 500,
    model: str = "default",
) -> ContextWindowManager:
    """Create a context window manager.

    Args:
        total_tokens: Total token budget
        response_reserve: Tokens reserved for response
        model: Model name

    Returns:
        Configured context manager
    """
    budget = ContextBudget(
        total_tokens=total_tokens,
        response_reserve_tokens=response_reserve,
    )
    return ContextWindowManager(budget=budget, model=model)


def fit_documents_to_context(
    documents: list[str],
    max_tokens: int = 3000,
    scores: list[float] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Fit documents into context window.

    Args:
        documents: Document contents
        max_tokens: Maximum tokens
        scores: Optional relevance scores

    Returns:
        Tuple of (selected documents, stats)
    """
    # Create context documents
    scores = scores or [1.0] * len(documents)
    ctx_docs = [
        ContextDocument(
            content=content,
            doc_id=f"doc_{i}",
            score=scores[i] if i < len(scores) else 1.0,
        )
        for i, content in enumerate(documents)
    ]

    # Create manager with document budget
    budget = ContextBudget(
        total_tokens=max_tokens + 1000,  # Extra for system/query
        system_prompt_tokens=500,
        query_tokens=500,
    )
    manager = ContextWindowManager(budget=budget)

    # Select documents
    selected = manager.select_documents(ctx_docs)

    return [doc.content for doc in selected], manager.get_context_stats(selected)


def truncate_text(
    text: str,
    max_tokens: int,
    strategy: str = "sentence",
    model: str = "default",
) -> str:
    """Truncate text to fit token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens
        strategy: Truncation strategy name
        model: Model name

    Returns:
        Truncated text
    """
    strategy_map = {
        "head": TruncationStrategy.HEAD,
        "tail": TruncationStrategy.TAIL,
        "middle": TruncationStrategy.MIDDLE_OUT,
        "sentence": TruncationStrategy.SENTENCE_BOUNDARY,
        "paragraph": TruncationStrategy.PARAGRAPH_BOUNDARY,
    }

    truncation_strategy = strategy_map.get(strategy, TruncationStrategy.SENTENCE_BOUNDARY)
    manager = ContextWindowManager(model=model)

    return manager.truncate(text, max_tokens, truncation_strategy)
