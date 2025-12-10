"""
Context Window Manager for long conversations.

Manages conversation history with automatic summarization, sliding windows,
and intelligent context selection for optimal LLM usage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class MessageRole(Enum):
    """Roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Single message in conversation."""

    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "role": self.role.value,
            "content": self.content,
        }


@dataclass
class ConversationSummary:
    """Summary of a conversation segment."""

    content: str
    message_count: int
    start_time: float
    end_time: float
    key_topics: List[str] = field(default_factory=list)
    token_count: int = 0


@dataclass
class ContextWindowConfig:
    """Configuration for context window management."""

    max_tokens: int = 4096  # Maximum tokens in context
    reserved_tokens: int = 512  # Reserved for response
    summary_threshold: float = 0.8  # Trigger summarization at 80% capacity
    min_recent_messages: int = 4  # Always keep recent messages
    summarization_batch_size: int = 10  # Messages to summarize at once
    enable_auto_summarization: bool = True


class ContextWindowManager:
    """
    Manages conversation context within token limits.

    Features:
    - Automatic summarization when context grows too large
    - Sliding window with preserved recent messages
    - Intelligent message selection for context
    - Token counting and budget management
    """

    def __init__(
        self,
        config: Optional[ContextWindowConfig] = None,
        summarizer: Optional[Callable[[List[Message]], str]] = None,
    ):
        """
        Initialize context window manager.

        Args:
            config: Configuration options
            summarizer: Custom summarization function
        """
        self.config = config or ContextWindowConfig()
        self.summarizer = summarizer or self._default_summarizer
        self.logger = logging.getLogger("ContextWindowManager")

        self.messages: List[Message] = []
        self.summaries: List[ConversationSummary] = []
        self.total_tokens: int = 0
        self.system_prompt: Optional[str] = None

    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt (always included in context)."""
        self.system_prompt = prompt

    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add a message to the conversation.

        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata

        Returns:
            Created message
        """
        token_count = self._estimate_tokens(content)

        message = Message(
            role=role,
            content=content,
            token_count=token_count,
            metadata=metadata or {},
        )

        self.messages.append(message)
        self.total_tokens += token_count

        # Check if summarization needed
        if self.config.enable_auto_summarization:
            self._check_and_summarize()

        return message

    def get_context(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get optimized context for LLM call.

        Args:
            max_tokens: Override max tokens

        Returns:
            List of message dicts ready for API
        """
        budget = max_tokens or (self.config.max_tokens - self.config.reserved_tokens)
        context = []
        used_tokens = 0

        # Always include system prompt
        if self.system_prompt:
            system_tokens = self._estimate_tokens(self.system_prompt)
            context.append({"role": "system", "content": self.system_prompt})
            used_tokens += system_tokens

        # Include summaries if available
        for summary in self.summaries:
            if used_tokens + summary.token_count <= budget:
                context.append(
                    {
                        "role": "system",
                        "content": f"[Previous conversation summary]: {summary.content}",
                    }
                )
                used_tokens += summary.token_count

        # Add recent messages (most important)
        recent_messages = self.messages[-self.config.min_recent_messages :]
        remaining_budget = budget - used_tokens

        # First, ensure recent messages fit
        recent_tokens = sum(m.token_count for m in recent_messages)
        if recent_tokens <= remaining_budget:
            for msg in recent_messages:
                context.append(msg.to_dict())
            used_tokens += recent_tokens
            remaining_budget -= recent_tokens

            # Add older messages if budget allows
            older_messages = self.messages[: -self.config.min_recent_messages]
            for msg in reversed(older_messages):
                if msg.token_count <= remaining_budget:
                    context.insert(-len(recent_messages), msg.to_dict())
                    remaining_budget -= msg.token_count
        else:
            # Just fit what we can from recent messages
            for msg in recent_messages:
                if msg.token_count <= remaining_budget:
                    context.append(msg.to_dict())
                    remaining_budget -= msg.token_count

        return context

    def _check_and_summarize(self) -> None:
        """Check if summarization is needed and perform it."""
        available_tokens = self.config.max_tokens - self.config.reserved_tokens
        threshold = available_tokens * self.config.summary_threshold

        if self.total_tokens > threshold:
            self._summarize_old_messages()

    def _summarize_old_messages(self) -> None:
        """Summarize older messages to free up context space."""
        # Keep recent messages, summarize the rest
        if len(self.messages) <= self.config.min_recent_messages:
            return

        to_summarize = self.messages[: -self.config.min_recent_messages]
        to_keep = self.messages[-self.config.min_recent_messages :]

        if len(to_summarize) < self.config.summarization_batch_size:
            return

        # Generate summary
        summary_content = self.summarizer(to_summarize)
        summary_tokens = self._estimate_tokens(summary_content)

        # Create summary object
        summary = ConversationSummary(
            content=summary_content,
            message_count=len(to_summarize),
            start_time=to_summarize[0].timestamp,
            end_time=to_summarize[-1].timestamp,
            key_topics=self._extract_key_topics(to_summarize),
            token_count=summary_tokens,
        )

        self.summaries.append(summary)

        # Update messages and token count
        summarized_tokens = sum(m.token_count for m in to_summarize)
        self.messages = to_keep
        self.total_tokens = self.total_tokens - summarized_tokens + summary_tokens

        self.logger.info(
            f"Summarized {len(to_summarize)} messages, "
            f"saved {summarized_tokens - summary_tokens} tokens"
        )

    def _default_summarizer(self, messages: List[Message]) -> str:
        """Default summarization (simple concatenation)."""
        user_messages = [m for m in messages if m.role == MessageRole.USER]
        assistant_messages = [m for m in messages if m.role == MessageRole.ASSISTANT]

        user_summary = (
            "; ".join(m.content[:100] for m in user_messages[-3:])
            if user_messages
            else "No user messages"
        )
        assistant_summary = (
            "; ".join(m.content[:100] for m in assistant_messages[-3:])
            if assistant_messages
            else "No assistant responses"
        )

        return (
            f"User discussed: {user_summary}. " f"Assistant responded about: {assistant_summary}."
        )

    def _extract_key_topics(self, messages: List[Message]) -> List[str]:
        """Extract key topics from messages."""
        topics = []
        for msg in messages:
            if msg.role == MessageRole.USER:
                # Simple keyword extraction
                words = msg.content.lower().split()
                long_words = [w for w in words if len(w) > 5][:3]
                topics.extend(long_words)
        return list(set(topics))[:5]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple estimation: ~4 chars per token
        return len(text) // 4

    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages = []
        self.summaries = []
        self.total_tokens = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get context window statistics."""
        return {
            "total_messages": len(self.messages),
            "total_summaries": len(self.summaries),
            "total_tokens": self.total_tokens,
            "max_tokens": self.config.max_tokens,
            "utilization": round(self.total_tokens / self.config.max_tokens, 3),
            "summarized_message_count": sum(s.message_count for s in self.summaries),
        }

    def get_recent_messages(self, count: int = 5) -> List[Message]:
        """Get recent messages."""
        return self.messages[-count:] if self.messages else []

    def search_history(self, query: str) -> List[Message]:
        """Search conversation history for relevant messages."""
        query_lower = query.lower()
        results = []

        for msg in self.messages:
            if query_lower in msg.content.lower():
                results.append(msg)

        for summary in self.summaries:
            if query_lower in summary.content.lower():
                # Create synthetic message for summary match
                results.append(
                    Message(
                        role=MessageRole.SYSTEM,
                        content=f"[From summary]: {summary.content}",
                        timestamp=summary.end_time,
                    )
                )

        return results


class MultiTurnConversationManager:
    """
    Manages multiple conversation threads with context isolation.

    Useful for chatbots handling multiple users or topics.
    """

    def __init__(self, default_config: Optional[ContextWindowConfig] = None):
        """Initialize multi-turn manager."""
        self.default_config = default_config or ContextWindowConfig()
        self.conversations: Dict[str, ContextWindowManager] = {}
        self.logger = logging.getLogger("MultiTurnConversationManager")

    def get_conversation(self, conversation_id: str) -> ContextWindowManager:
        """Get or create conversation context."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ContextWindowManager(config=self.default_config)
        return self.conversations[conversation_id]

    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add message to specific conversation."""
        conv = self.get_conversation(conversation_id)
        return conv.add_message(role, content, metadata)

    def get_context(
        self, conversation_id: str, max_tokens: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get context for specific conversation."""
        conv = self.get_conversation(conversation_id)
        return conv.get_context(max_tokens)

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear specific conversation."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].clear()

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete conversation entirely."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all conversations."""
        return {conv_id: conv.get_stats() for conv_id, conv in self.conversations.items()}

    def cleanup_old_conversations(self, max_age_seconds: float) -> int:
        """Remove conversations older than specified age."""
        current_time = time.time()
        to_remove = []

        for conv_id, conv in self.conversations.items():
            if conv.messages:
                last_message_time = conv.messages[-1].timestamp
                if current_time - last_message_time > max_age_seconds:
                    to_remove.append(conv_id)

        for conv_id in to_remove:
            del self.conversations[conv_id]

        return len(to_remove)
