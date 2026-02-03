"""
Advanced context window management for optimal token utilization.

Implements state-of-the-art techniques for managing LLM context:
- Dynamic context allocation
- Priority-based token budgeting
- Sliding window with attention
- Context compression strategies
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from autorag_live.utils import get_logger

logger = get_logger(__name__)


class ContextPriority(Enum):
    """Priority levels for context items."""

    CRITICAL = 4  # Must include (query, instructions)
    HIGH = 3  # Should include (recent messages)
    MEDIUM = 2  # Include if space (relevant docs)
    LOW = 1  # Optional (background info)


@dataclass
class ContextItem:
    """Single item in context."""

    content: str
    priority: ContextPriority
    token_count: int
    category: str  # query, instruction, message, document, etc.
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextWindow:
    """Managed context window."""

    items: List[ContextItem] = field(default_factory=list)
    max_tokens: int = 8192
    reserved_tokens: int = 512  # For response generation

    @property
    def available_tokens(self) -> int:
        """Tokens available for context."""
        return self.max_tokens - self.reserved_tokens

    @property
    def used_tokens(self) -> int:
        """Tokens currently used."""
        return sum(item.token_count for item in self.items)

    @property
    def remaining_tokens(self) -> int:
        """Tokens remaining."""
        return self.available_tokens - self.used_tokens

    @property
    def utilization(self) -> float:
        """Context utilization (0-1)."""
        return self.used_tokens / self.available_tokens if self.available_tokens > 0 else 0.0


class TokenBudget:
    """Token budget allocator for context categories."""

    def __init__(self, total_tokens: int):
        self.total_tokens = total_tokens

        # Default allocation percentages
        self.allocations = {
            "query": 0.05,  # 5% for query
            "instruction": 0.10,  # 10% for instructions
            "conversation": 0.20,  # 20% for conversation history
            "documents": 0.55,  # 55% for retrieved documents
            "metadata": 0.10,  # 10% for metadata
        }

    def get_budget(self, category: str) -> int:
        """Get token budget for category."""
        return int(self.total_tokens * self.allocations.get(category, 0.1))

    def adjust_allocation(self, category: str, percentage: float) -> None:
        """Adjust allocation for category."""
        self.allocations[category] = max(0.0, min(1.0, percentage))

        # Normalize to ensure sum = 1.0
        total = sum(self.allocations.values())
        if total > 0:
            for cat in self.allocations:
                self.allocations[cat] /= total


class ContextManager:
    """
    Advanced context window manager.

    Features:
    - Priority-based inclusion
    - Token budget enforcement
    - Sliding window for conversations
    - Smart truncation and summarization
    """

    def __init__(
        self,
        max_tokens: int = 8192,
        reserved_tokens: int = 512,
        enable_compression: bool = True,
    ):
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        self.enable_compression = enable_compression

        self.budget = TokenBudget(max_tokens - reserved_tokens)
        self.window = ContextWindow(max_tokens=max_tokens, reserved_tokens=reserved_tokens)

    def build_context(
        self,
        query: str,
        documents: List[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        instructions: Optional[str] = None,
    ) -> str:
        """
        Build optimized context from components.

        Args:
            query: User query
            documents: Retrieved documents
            conversation_history: Recent conversation
            instructions: System instructions

        Returns:
            Optimized context string
        """
        items: List[ContextItem] = []

        # Add critical items
        if instructions:
            items.append(
                ContextItem(
                    content=instructions,
                    priority=ContextPriority.CRITICAL,
                    token_count=self._count_tokens(instructions),
                    category="instruction",
                )
            )

        items.append(
            ContextItem(
                content=query,
                priority=ContextPriority.CRITICAL,
                token_count=self._count_tokens(query),
                category="query",
            )
        )

        # Add conversation history with sliding window
        if conversation_history:
            conv_items = self._process_conversation(conversation_history)
            items.extend(conv_items)

        # Add documents with priority ranking
        doc_items = self._process_documents(documents, query)
        items.extend(doc_items)

        # Optimize context to fit budget
        optimized_items = self._optimize_context(items)

        # Build final context string
        context = self._format_context(optimized_items)

        logger.info(
            f"Built context: {len(context)} chars, "
            f"{self._count_tokens(context)} tokens, "
            f"{len(optimized_items)} items"
        )

        return context

    def _process_conversation(self, history: List[Dict[str, str]]) -> List[ContextItem]:
        """Process conversation history with sliding window."""
        items = []
        budget = self.budget.get_budget("conversation")
        used_tokens = 0

        # Process from most recent
        for idx, message in enumerate(reversed(history)):
            content = f"{message.get('role', 'user')}: {message.get('content', '')}"
            tokens = self._count_tokens(content)

            if used_tokens + tokens > budget:
                # Budget exceeded, summarize older messages
                if self.enable_compression:
                    remaining = history[:-(idx)]
                    if remaining:
                        summary = self._summarize_conversation(remaining)
                        items.append(
                            ContextItem(
                                content=f"[Earlier conversation: {summary}]",
                                priority=ContextPriority.LOW,
                                token_count=self._count_tokens(summary),
                                category="conversation",
                                metadata={"summarized": True},
                            )
                        )
                break

            # Recent messages have higher priority
            priority = ContextPriority.HIGH if idx < 3 else ContextPriority.MEDIUM

            items.append(
                ContextItem(
                    content=content,
                    priority=priority,
                    token_count=tokens,
                    category="conversation",
                    timestamp=idx,
                )
            )

            used_tokens += tokens

        return list(reversed(items))

    def _process_documents(self, documents: List[str], query: str) -> List[ContextItem]:
        """Process documents with relevance-based priority."""
        items = []
        budget = self.budget.get_budget("documents")
        used_tokens = 0

        for idx, doc in enumerate(documents):
            tokens = self._count_tokens(doc)

            # Calculate relevance score (simple keyword matching)
            relevance = self._calculate_relevance(doc, query)

            # Higher relevance = higher priority
            if relevance > 0.7:
                priority = ContextPriority.HIGH
            elif relevance > 0.4:
                priority = ContextPriority.MEDIUM
            else:
                priority = ContextPriority.LOW

            # Check budget
            if used_tokens + tokens > budget:
                # Try compression
                if self.enable_compression and tokens > 200:
                    compressed = self._compress_document(doc, query)
                    tokens = self._count_tokens(compressed)
                    doc = compressed

                if used_tokens + tokens > budget:
                    # Skip low priority docs
                    if priority == ContextPriority.LOW:
                        continue
                    # Truncate medium priority
                    if priority == ContextPriority.MEDIUM:
                        max_tokens = budget - used_tokens
                        doc = self._truncate_to_tokens(doc, max_tokens)
                        tokens = max_tokens

            items.append(
                ContextItem(
                    content=doc,
                    priority=priority,
                    token_count=tokens,
                    category="document",
                    metadata={"relevance": relevance, "index": idx},
                )
            )

            used_tokens += tokens

        return items

    def _optimize_context(self, items: List[ContextItem]) -> List[ContextItem]:
        """Optimize context to fit within budget."""
        # Sort by priority (descending)
        sorted_items = sorted(items, key=lambda x: x.priority.value, reverse=True)

        selected = []
        used_tokens = 0
        available = self.budget.total_tokens

        for item in sorted_items:
            if used_tokens + item.token_count <= available:
                selected.append(item)
                used_tokens += item.token_count
            elif item.priority == ContextPriority.CRITICAL:
                # Critical items must be included, truncate if needed
                max_tokens = available - used_tokens
                if max_tokens > 0:
                    truncated_content = self._truncate_to_tokens(item.content, max_tokens)
                    item.content = truncated_content
                    item.token_count = max_tokens
                    selected.append(item)
                    used_tokens += max_tokens

        # Restore original order by category
        category_order = ["instruction", "conversation", "document", "query"]

        def sort_key(item: ContextItem) -> int:
            try:
                return category_order.index(item.category)
            except ValueError:
                return 999

        selected.sort(key=sort_key)

        return selected

    def _format_context(self, items: List[ContextItem]) -> str:
        """Format context items into final string."""
        sections = []

        # Group by category
        by_category: Dict[str, List[ContextItem]] = {}
        for item in items:
            if item.category not in by_category:
                by_category[item.category] = []
            by_category[item.category].append(item)

        # Format each category
        if "instruction" in by_category:
            for item in by_category["instruction"]:
                sections.append(item.content)

        if "conversation" in by_category:
            sections.append("Conversation History:")
            for item in by_category["conversation"]:
                sections.append(item.content)

        if "document" in by_category:
            sections.append("\nRelevant Documents:")
            for idx, item in enumerate(by_category["document"], 1):
                sections.append(f"\n[Document {idx}]\n{item.content}")

        if "query" in by_category:
            sections.append("\nQuery:")
            for item in by_category["query"]:
                sections.append(item.content)

        return "\n".join(sections)

    def _count_tokens(self, text: str) -> int:
        """Approximate token count."""
        # Simple approximation: 1 token ≈ 4 characters
        # In production: use tiktoken
        return len(text) // 4

    def _calculate_relevance(self, document: str, query: str) -> float:
        """Calculate document relevance to query."""
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())

        if not query_terms:
            return 0.0

        overlap = len(query_terms & doc_terms)
        return overlap / len(query_terms)

    def _compress_document(self, document: str, query: str) -> str:
        """Compress document while preserving relevance."""
        # Extract sentences containing query terms
        query_terms = set(query.lower().split())
        sentences = re.split(r"[.!?]+", document)

        relevant_sentences = []
        for sent in sentences:
            sent_terms = set(sent.lower().split())
            if sent_terms & query_terms:
                relevant_sentences.append(sent.strip())

        if relevant_sentences:
            return ". ".join(relevant_sentences) + "."

        # Fallback: take first 50% of document
        words = document.split()
        return " ".join(words[: len(words) // 2]) + "..."

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        # Approximate words from tokens
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        return text[:max_chars] + "..."

    def _summarize_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Summarize conversation history."""
        # Simple extraction of key points
        # In production: use LLM for summarization
        topics = set()

        for msg in messages:
            content = msg.get("content", "")
            words = content.split()
            # Extract potential topic words (longer words)
            topics.update([w for w in words if len(w) > 6])

        if topics:
            return f"Discussion about {', '.join(list(topics)[:5])}"

        return f"Earlier conversation ({len(messages)} messages)"


# Example usage
def example_context_management():
    """Example of context management."""
    manager = ContextManager(max_tokens=2000)

    query = "Explain machine learning"

    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
        "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
        "Supervised learning requires labeled training data to learn input-output mappings.",
        "The weather today is sunny and warm.",  # Irrelevant
        "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning.",
    ]

    conversation = [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI is artificial intelligence."},
        {"role": "user", "content": "Tell me more about machine learning"},
    ]

    instructions = "You are a helpful AI assistant. Provide clear and accurate answers."

    context = manager.build_context(
        query=query,
        documents=documents,
        conversation_history=conversation,
        instructions=instructions,
    )

    print("Generated Context:")
    print("=" * 50)
    print(context)
    print("\n" + "=" * 50)
    print(f"Total tokens: {manager._count_tokens(context)}")
    print(f"Max tokens: {manager.max_tokens}")


if __name__ == "__main__":
    example_context_management()
