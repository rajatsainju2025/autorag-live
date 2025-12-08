"""
Multi-turn conversation memory management.

Handles context window optimization, token counting, and relevance-based summarization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Simple heuristic: ~1 token per 4 characters for English.
    """
    return max(1, len(text) // 4)


@dataclass
class ConversationMessage:
    """Single message in conversation."""

    role: str  # "user", "assistant"
    content: str
    turn_number: int
    tokens: int = field(default=0, init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate tokens after initialization."""
        self.tokens = estimate_tokens(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Export message as dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "turn": self.turn_number,
            "tokens": self.tokens,
        }


@dataclass
class ConversationSummary:
    """Summary of conversation segment."""

    summary_text: str
    turn_start: int
    turn_end: int
    key_topics: List[str] = field(default_factory=list)
    tokens: int = field(default=0, init=False)

    def __post_init__(self):
        """Calculate tokens after initialization."""
        self.tokens = estimate_tokens(self.summary_text)

    def to_dict(self) -> Dict[str, Any]:
        """Export summary as dictionary."""
        return {
            "summary": self.summary_text,
            "turns": f"{self.turn_start}-{self.turn_end}",
            "topics": self.key_topics,
            "tokens": self.tokens,
        }


class ConversationMemory:
    """
    Multi-turn conversation memory with context management.

    Implements context window optimization and relevance-based retrieval.
    """

    def __init__(
        self,
        max_context_tokens: int = 4096,
        max_messages: int = 100,
        summarize_after: int = 10,
    ):
        """
        Initialize conversation memory.

        Args:
            max_context_tokens: Max tokens in active context
            max_messages: Max messages to keep in full form
            summarize_after: Summarize after this many turns
        """
        self.max_context_tokens = max_context_tokens
        self.max_messages = max_messages
        self.summarize_after = summarize_after

        self.messages: List[ConversationMessage] = []
        self.summaries: List[ConversationSummary] = []
        self.turn_count = 0

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationMessage:
        """
        Add message to conversation.

        Args:
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata

        Returns:
            Added message
        """
        self.turn_count += 1
        msg = ConversationMessage(
            role=role,
            content=content,
            turn_number=self.turn_count,
            metadata=metadata or {},
        )
        self.messages.append(msg)

        # Maintain memory size
        self._enforce_memory_limits()

        return msg

    def _enforce_memory_limits(self) -> None:
        """Enforce memory size and context limits."""
        # If too many messages, summarize oldest
        if len(self.messages) > self.max_messages:
            oldest_batch = self.messages[: self.max_messages // 2]
            self._summarize_batch(oldest_batch)
            self.messages = self.messages[self.max_messages // 2 :]

        # If context too large, use sliding window
        total_tokens = sum(m.tokens for m in self.messages)
        if total_tokens > self.max_context_tokens:
            self._apply_sliding_window()

    def _summarize_batch(self, messages: List[ConversationMessage]) -> None:
        """Summarize a batch of messages."""
        if not messages:
            return

        # Extract key topics and create summary
        key_topics = self._extract_topics(messages)
        summary_text = self._create_summary_text(messages, key_topics)

        summary = ConversationSummary(
            summary_text=summary_text,
            turn_start=messages[0].turn_number,
            turn_end=messages[-1].turn_number,
            key_topics=key_topics,
        )

        self.summaries.append(summary)

    def _extract_topics(self, messages: List[ConversationMessage]) -> List[str]:
        """Extract key topics from messages."""
        topics = []
        for msg in messages:
            words = msg.content.lower().split()
            # Simple: take first few significant words
            for word in words:
                if len(word) > 5 and word not in topics:
                    topics.append(word)
                    if len(topics) >= 5:
                        break
            if len(topics) >= 5:
                break
        return topics

    def _create_summary_text(self, messages: List[ConversationMessage], topics: List[str]) -> str:
        """Create summary text for message batch."""
        user_msgs = [m.content for m in messages if m.role == "user"]
        assistant_msgs = [m.content for m in messages if m.role == "assistant"]

        summary = f"Discussed topics: {', '.join(topics[:3])}. "
        if user_msgs:
            summary += f"User asked about: {user_msgs[0][:100]}... "
        if assistant_msgs:
            summary += "Responses covered key points."

        return summary

    def _apply_sliding_window(self) -> None:
        """Apply sliding window to maintain context size."""
        # Keep last N messages that fit in context
        total_tokens = 0
        keep_from = len(self.messages) - 1

        for i in range(len(self.messages) - 1, -1, -1):
            total_tokens += self.messages[i].tokens
            if total_tokens > self.max_context_tokens:
                keep_from = i + 1
                break

        # Summarize dropped messages
        if keep_from > 0:
            dropped = self.messages[:keep_from]
            self._summarize_batch(dropped)

        self.messages = self.messages[keep_from:]

    def get_context(self, max_tokens: Optional[int] = None) -> str:
        """
        Get formatted conversation context.

        Args:
            max_tokens: Optional token limit

        Returns:
            Formatted context string
        """
        max_tokens = max_tokens or self.max_context_tokens
        context_lines = []

        # Add recent summaries
        for summary in self.summaries[-3:]:
            context_lines.append(
                f"[Summary T{summary.turn_start}-{summary.turn_end}]: {summary.summary_text}"
            )

        # Add recent messages
        current_tokens = sum(estimate_tokens(line) for line in context_lines)
        for msg in self.messages:
            msg_line = f"{msg.role.upper()}: {msg.content}"
            msg_tokens = estimate_tokens(msg_line)

            if current_tokens + msg_tokens <= max_tokens:
                context_lines.append(msg_line)
                current_tokens += msg_tokens
            else:
                break

        return "\n".join(context_lines)

    def get_summary_context(self) -> str:
        """Get high-level summary of conversation."""
        if not self.summaries:
            return ""

        summary_lines = []
        all_topics = set()

        for summary in self.summaries:
            all_topics.update(summary.key_topics)

        summary_lines.append(f"Conversation Summary (Topics: {', '.join(list(all_topics)[:5])})")
        for summary in self.summaries[-5:]:
            summary_lines.append(f"- {summary.summary_text[:100]}...")

        return "\n".join(summary_lines)

    def search_messages(
        self, query: str, top_k: int = 3
    ) -> List[Tuple[ConversationMessage, float]]:
        """
        Search messages by relevance.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (message, relevance_score) tuples
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_messages = []
        for msg in self.messages:
            msg_lower = msg.content.lower()
            msg_words = set(msg_lower.split())

            # Simple relevance: word overlap
            overlap = len(query_words & msg_words)
            similarity = (
                overlap / (len(query_words) + len(msg_words) - overlap)
                if query_words or msg_words
                else 0
            )
            scored_messages.append((msg, similarity))

        # Sort by relevance descending
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        return scored_messages[:top_k]

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        total_tokens = sum(m.tokens for m in self.messages)
        summary_tokens = sum(s.tokens for s in self.summaries)

        return {
            "total_turns": self.turn_count,
            "active_messages": len(self.messages),
            "summaries": len(self.summaries),
            "active_tokens": total_tokens,
            "summary_tokens": summary_tokens,
            "context_utilization": f"{min(100, (total_tokens / self.max_context_tokens) * 100):.1f}%",
        }

    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages.clear()
        self.summaries.clear()
        self.turn_count = 0

    def export_conversation(self) -> Dict[str, Any]:
        """Export full conversation for logging/analysis."""
        return {
            "total_turns": self.turn_count,
            "messages": [m.to_dict() for m in self.messages],
            "summaries": [s.to_dict() for s in self.summaries],
            "stats": self.get_conversation_stats(),
        }


class ConversationBuffer:
    """
    Efficient conversation buffer with rolling history.

    Keeps most recent N turns plus summaries of older turns.
    """

    def __init__(self, buffer_size: int = 20):
        """
        Initialize conversation buffer.

        Args:
            buffer_size: Max recent turns to keep in buffer
        """
        self.buffer_size = buffer_size
        self.buffer: List[ConversationMessage] = []
        self.archived_summaries: List[ConversationSummary] = []

    def add(self, msg: ConversationMessage) -> None:
        """Add message to buffer."""
        self.buffer.append(msg)

        # If buffer full, archive oldest
        if len(self.buffer) > self.buffer_size:
            to_archive = self.buffer[: len(self.buffer) - self.buffer_size]
            summary_text = f"Archived {len(to_archive)} messages"
            summary = ConversationSummary(
                summary_text=summary_text,
                turn_start=to_archive[0].turn_number,
                turn_end=to_archive[-1].turn_number,
            )
            self.archived_summaries.append(summary)
            self.buffer = self.buffer[len(self.buffer) - self.buffer_size :]

    def get_buffer_context(self) -> str:
        """Get current buffer as context."""
        lines = []
        for msg in self.buffer:
            lines.append(f"{msg.role.upper()}: {msg.content}")
        return "\n".join(lines)

    def get_full_context(self) -> str:
        """Get full context including archived summaries."""
        lines = []

        # Add archived summaries
        for summary in self.archived_summaries[-3:]:
            lines.append(f"[Archived T{summary.turn_start}-{summary.turn_end}]")

        # Add buffer
        for msg in self.buffer:
            lines.append(f"{msg.role.upper()}: {msg.content}")

        return "\n".join(lines)
