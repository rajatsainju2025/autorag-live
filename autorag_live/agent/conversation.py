"""
Conversation management for multi-turn RAG interactions.

Provides utilities for managing conversation history, context
windows, and memory for stateful RAG interactions.

Features:
- Conversation history storage and retrieval
- Memory management with summarization
- Context window optimization
- Turn-based context tracking
- Entity and topic tracking

Example usage:
    >>> manager = ConversationManager(max_turns=10)
    >>> manager.add_user_message("What is machine learning?")
    >>> manager.add_assistant_message("ML is a subset of AI...")
    >>> context = manager.get_context_for_retrieval()
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Message roles in conversation."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationState(str, Enum):
    """State of the conversation."""
    
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


@dataclass
class Message:
    """A single message in the conversation."""
    
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    
    # Optional fields
    message_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # RAG-specific
    sources: List[str] = field(default_factory=list)
    query_used: Optional[str] = None
    
    def __post_init__(self):
        if self.message_id is None:
            self.message_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique message ID."""
        content = f"{self.role}{self.content}{self.timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "metadata": self.metadata,
            "sources": self.sources,
            "query_used": self.query_used,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """Create from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            message_id=data.get("message_id"),
            metadata=data.get("metadata", {}),
            sources=data.get("sources", []),
            query_used=data.get("query_used"),
        )


@dataclass
class Turn:
    """A conversation turn (user message + assistant response)."""
    
    turn_id: int
    user_message: Message
    assistant_message: Optional[Message] = None
    
    # Context used for this turn
    retrieved_contexts: List[str] = field(default_factory=list)
    retrieval_query: Optional[str] = None
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """Check if turn has both messages."""
        return self.assistant_message is not None


@dataclass
class ConversationSummary:
    """Summary of a conversation or segment."""
    
    summary_text: str
    covered_turns: List[int]
    created_at: float = field(default_factory=time.time)
    
    # Key information extracted
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    key_facts: List[str] = field(default_factory=list)


class BaseMemory(ABC):
    """Base class for conversation memory."""
    
    @abstractmethod
    def add_message(self, message: Message) -> None:
        """Add a message to memory."""
        pass
    
    @abstractmethod
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from memory."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear memory."""
        pass


class BufferMemory(BaseMemory):
    """Simple buffer memory storing all messages."""
    
    def __init__(self, max_messages: int = 100):
        """
        Initialize buffer memory.
        
        Args:
            max_messages: Maximum messages to store
        """
        self.max_messages = max_messages
        self._messages: List[Message] = []
    
    def add_message(self, message: Message) -> None:
        """Add message to buffer."""
        self._messages.append(message)
        
        # Trim if exceeds max
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from buffer."""
        if limit:
            return self._messages[-limit:]
        return self._messages.copy()
    
    def clear(self) -> None:
        """Clear buffer."""
        self._messages.clear()


class SlidingWindowMemory(BaseMemory):
    """Sliding window memory with token limit."""
    
    def __init__(
        self,
        max_tokens: int = 4000,
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        """
        Initialize sliding window memory.
        
        Args:
            max_tokens: Maximum tokens to retain
            token_counter: Function to count tokens
        """
        self.max_tokens = max_tokens
        self.token_counter = token_counter or self._estimate_tokens
        self._messages: List[Message] = []
    
    def add_message(self, message: Message) -> None:
        """Add message, trimming old messages if needed."""
        self._messages.append(message)
        self._trim_to_limit()
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages."""
        if limit:
            return self._messages[-limit:]
        return self._messages.copy()
    
    def clear(self) -> None:
        """Clear memory."""
        self._messages.clear()
    
    def _trim_to_limit(self) -> None:
        """Trim messages to fit token limit."""
        while self._total_tokens() > self.max_tokens and len(self._messages) > 1:
            # Remove oldest non-system message
            for i, msg in enumerate(self._messages):
                if msg.role != MessageRole.SYSTEM:
                    self._messages.pop(i)
                    break
    
    def _total_tokens(self) -> int:
        """Count total tokens in memory."""
        return sum(self.token_counter(m.content) for m in self._messages)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4


class SummarizingMemory(BaseMemory):
    """Memory that summarizes old messages."""
    
    def __init__(
        self,
        summarizer: Callable[[List[Message]], str],
        max_recent_messages: int = 10,
        summarize_threshold: int = 20,
    ):
        """
        Initialize summarizing memory.
        
        Args:
            summarizer: Function to summarize messages
            max_recent_messages: Recent messages to keep unsummarized
            summarize_threshold: When to trigger summarization
        """
        self.summarizer = summarizer
        self.max_recent_messages = max_recent_messages
        self.summarize_threshold = summarize_threshold
        
        self._messages: List[Message] = []
        self._summary: Optional[str] = None
    
    def add_message(self, message: Message) -> None:
        """Add message, summarizing if needed."""
        self._messages.append(message)
        
        if len(self._messages) >= self.summarize_threshold:
            self._summarize()
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages, including summary as system message."""
        messages = []
        
        # Add summary as system message
        if self._summary:
            messages.append(Message(
                role=MessageRole.SYSTEM,
                content=f"Previous conversation summary: {self._summary}",
            ))
        
        # Add recent messages
        recent = self._messages[-self.max_recent_messages:] if limit is None else self._messages[-limit:]
        messages.extend(recent)
        
        return messages
    
    def clear(self) -> None:
        """Clear memory."""
        self._messages.clear()
        self._summary = None
    
    def _summarize(self) -> None:
        """Summarize old messages."""
        # Messages to summarize
        to_summarize = self._messages[:-self.max_recent_messages]
        
        if to_summarize:
            # Create new summary
            new_summary = self.summarizer(to_summarize)
            
            # Combine with existing summary
            if self._summary:
                self._summary = f"{self._summary}\n\n{new_summary}"
            else:
                self._summary = new_summary
            
            # Keep only recent messages
            self._messages = self._messages[-self.max_recent_messages:]


class ConversationManager:
    """
    Main conversation management interface.
    
    Example:
        >>> manager = ConversationManager()
        >>> 
        >>> # Add messages
        >>> manager.add_user_message("What is Python?")
        >>> manager.add_assistant_message(
        ...     "Python is a programming language...",
        ...     sources=["source1", "source2"]
        ... )
        >>> 
        >>> # Get context for retrieval
        >>> query_context = manager.get_context_for_retrieval()
        >>> 
        >>> # Get formatted history for LLM
        >>> history = manager.get_formatted_history()
    """
    
    def __init__(
        self,
        memory: Optional[BaseMemory] = None,
        max_turns: int = 20,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize conversation manager.
        
        Args:
            memory: Memory backend
            max_turns: Maximum turns to track
            conversation_id: Unique conversation ID
            system_prompt: System prompt for the conversation
        """
        self.memory = memory or BufferMemory(max_messages=max_turns * 2)
        self.max_turns = max_turns
        self.conversation_id = conversation_id or self._generate_id()
        
        # Conversation state
        self.state = ConversationState.ACTIVE
        self.created_at = time.time()
        
        # Turn tracking
        self._turns: List[Turn] = []
        self._current_turn: Optional[Turn] = None
        
        # Topic and entity tracking
        self._topics: Set[str] = set()
        self._entities: Set[str] = set()
        
        # Add system prompt if provided
        if system_prompt:
            self._add_system_message(system_prompt)
    
    def _generate_id(self) -> str:
        """Generate unique conversation ID."""
        return hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
    
    def _add_system_message(self, content: str) -> None:
        """Add system message."""
        message = Message(role=MessageRole.SYSTEM, content=content)
        self.memory.add_message(message)
    
    def add_user_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add user message.
        
        Args:
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created message
        """
        message = Message(
            role=MessageRole.USER,
            content=content,
            metadata=metadata or {},
        )
        
        # Create new turn
        turn_id = len(self._turns)
        self._current_turn = Turn(
            turn_id=turn_id,
            user_message=message,
        )
        
        self.memory.add_message(message)
        self._extract_entities(content)
        
        return message
    
    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[str]] = None,
        query_used: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add assistant message.
        
        Args:
            content: Message content
            sources: Sources used to generate response
            query_used: Query used for retrieval
            metadata: Optional metadata
            
        Returns:
            Created message
        """
        message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            sources=sources or [],
            query_used=query_used,
            metadata=metadata or {},
        )
        
        # Complete current turn
        if self._current_turn:
            self._current_turn.assistant_message = message
            if sources:
                self._current_turn.retrieved_contexts = sources
            if query_used:
                self._current_turn.retrieval_query = query_used
            
            self._turns.append(self._current_turn)
            self._current_turn = None
        
        self.memory.add_message(message)
        
        return message
    
    def get_messages(
        self,
        include_system: bool = True,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """
        Get conversation messages.
        
        Args:
            include_system: Include system messages
            limit: Maximum messages to return
            
        Returns:
            List of messages
        """
        messages = self.memory.get_messages(limit)
        
        if not include_system:
            messages = [m for m in messages if m.role != MessageRole.SYSTEM]
        
        return messages
    
    def get_turns(self, limit: Optional[int] = None) -> List[Turn]:
        """
        Get conversation turns.
        
        Args:
            limit: Maximum turns to return
            
        Returns:
            List of turns
        """
        if limit:
            return self._turns[-limit:]
        return self._turns.copy()
    
    def get_context_for_retrieval(
        self,
        num_turns: int = 3,
        include_current: bool = True,
    ) -> str:
        """
        Get context for retrieval query enhancement.
        
        Args:
            num_turns: Number of recent turns to include
            include_current: Include current (incomplete) turn
            
        Returns:
            Context string
        """
        context_parts = []
        
        # Recent turns
        recent_turns = self._turns[-num_turns:] if self._turns else []
        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_message.content}")
            if turn.assistant_message:
                context_parts.append(f"Assistant: {turn.assistant_message.content[:200]}...")
        
        # Current turn
        if include_current and self._current_turn:
            context_parts.append(f"Current query: {self._current_turn.user_message.content}")
        
        return "\n".join(context_parts)
    
    def get_formatted_history(
        self,
        format: str = "chat",
        num_turns: Optional[int] = None,
    ) -> str:
        """
        Get formatted conversation history.
        
        Args:
            format: Output format ('chat', 'prompt', 'markdown')
            num_turns: Number of turns to include
            
        Returns:
            Formatted history string
        """
        messages = self.get_messages(include_system=False)
        
        if num_turns:
            messages = messages[-(num_turns * 2):]
        
        if format == "chat":
            return self._format_chat(messages)
        elif format == "prompt":
            return self._format_prompt(messages)
        elif format == "markdown":
            return self._format_markdown(messages)
        else:
            return self._format_chat(messages)
    
    def _format_chat(self, messages: List[Message]) -> str:
        """Format as chat."""
        lines = []
        for msg in messages:
            role = msg.role.value.capitalize()
            lines.append(f"{role}: {msg.content}")
        return "\n\n".join(lines)
    
    def _format_prompt(self, messages: List[Message]) -> str:
        """Format for prompt inclusion."""
        lines = []
        for msg in messages:
            if msg.role == MessageRole.USER:
                lines.append(f"Human: {msg.content}")
            else:
                lines.append(f"Assistant: {msg.content}")
        return "\n".join(lines)
    
    def _format_markdown(self, messages: List[Message]) -> str:
        """Format as markdown."""
        lines = []
        for msg in messages:
            role = "**User**" if msg.role == MessageRole.USER else "**Assistant**"
            lines.append(f"{role}\n\n{msg.content}")
        return "\n\n---\n\n".join(lines)
    
    def get_last_query(self) -> Optional[str]:
        """Get the last user query."""
        messages = self.get_messages(include_system=False)
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                return msg.content
        return None
    
    def get_last_response(self) -> Optional[str]:
        """Get the last assistant response."""
        messages = self.get_messages(include_system=False)
        for msg in reversed(messages):
            if msg.role == MessageRole.ASSISTANT:
                return msg.content
        return None
    
    def _extract_entities(self, text: str) -> None:
        """Extract entities from text (simple approach)."""
        import re
        
        # Extract capitalized phrases
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter common words
        common = {"I", "The", "This", "That", "What", "How", "Why", "When", "Where"}
        entities = [e for e in entities if e not in common]
        
        self._entities.update(entities)
    
    def get_entities(self) -> List[str]:
        """Get all extracted entities."""
        return list(self._entities)
    
    def get_topics(self) -> List[str]:
        """Get conversation topics."""
        return list(self._topics)
    
    def add_topic(self, topic: str) -> None:
        """Add a topic to the conversation."""
        self._topics.add(topic)
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.memory.clear()
        self._turns.clear()
        self._current_turn = None
        self._topics.clear()
        self._entities.clear()
    
    def end_conversation(self) -> None:
        """End the conversation."""
        self.state = ConversationState.ENDED
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "messages": [m.to_dict() for m in self.get_messages()],
            "topics": list(self._topics),
            "entities": list(self._entities),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConversationManager:
        """Create from dictionary."""
        manager = cls(
            conversation_id=data.get("conversation_id"),
        )
        manager.state = ConversationState(data.get("state", "active"))
        manager.created_at = data.get("created_at", time.time())
        manager._topics = set(data.get("topics", []))
        manager._entities = set(data.get("entities", []))
        
        # Restore messages
        for msg_data in data.get("messages", []):
            message = Message.from_dict(msg_data)
            manager.memory.add_message(message)
        
        return manager


class QueryRewriter:
    """
    Rewrite queries based on conversation context.
    
    Example:
        >>> rewriter = QueryRewriter()
        >>> manager = ConversationManager()
        >>> manager.add_user_message("What is Python?")
        >>> manager.add_assistant_message("Python is a programming language...")
        >>> 
        >>> # Rewrite follow-up question
        >>> rewritten = rewriter.rewrite(
        ...     query="What are its main features?",
        ...     conversation=manager
        ... )
        >>> print(rewritten)  # "What are the main features of Python?"
    """
    
    def __init__(
        self,
        llm_func: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize query rewriter.
        
        Args:
            llm_func: Optional LLM function for advanced rewriting
        """
        self.llm_func = llm_func
    
    def rewrite(
        self,
        query: str,
        conversation: ConversationManager,
    ) -> str:
        """
        Rewrite query using conversation context.
        
        Args:
            query: Current query
            conversation: Conversation manager
            
        Returns:
            Rewritten query
        """
        # Check if query needs rewriting
        if not self._needs_rewriting(query):
            return query
        
        # Try LLM rewriting
        if self.llm_func:
            return self._llm_rewrite(query, conversation)
        
        # Fallback to rule-based rewriting
        return self._rule_based_rewrite(query, conversation)
    
    def _needs_rewriting(self, query: str) -> bool:
        """Check if query needs rewriting."""
        # Check for pronouns and references
        pronouns = ["it", "its", "they", "their", "them", "this", "that", "these", "those"]
        query_lower = query.lower()
        
        for pronoun in pronouns:
            if f" {pronoun} " in f" {query_lower} " or query_lower.startswith(f"{pronoun} "):
                return True
        
        return False
    
    def _llm_rewrite(
        self,
        query: str,
        conversation: ConversationManager,
    ) -> str:
        """Rewrite using LLM."""
        context = conversation.get_context_for_retrieval(num_turns=3)
        
        prompt = f"""Given the conversation context, rewrite the query to be standalone.

Conversation:
{context}

Current query: {query}

Rewritten query (standalone):"""
        
        try:
            return self.llm_func(prompt).strip()
        except Exception as e:
            logger.warning(f"LLM rewriting failed: {e}")
            return self._rule_based_rewrite(query, conversation)
    
    def _rule_based_rewrite(
        self,
        query: str,
        conversation: ConversationManager,
    ) -> str:
        """Simple rule-based rewriting."""
        # Get entities from conversation
        entities = conversation.get_entities()
        
        if not entities:
            return query
        
        # Replace pronouns with most recent entity
        recent_entity = entities[-1] if entities else ""
        
        replacements = {
            r'\bit\b': recent_entity,
            r'\bits\b': f"{recent_entity}'s",
            r'\bthis\b': recent_entity,
            r'\bthat\b': recent_entity,
        }
        
        import re
        rewritten = query
        for pattern, replacement in replacements.items():
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        
        return rewritten


class ConversationStore:
    """
    Store and manage multiple conversations.
    
    Example:
        >>> store = ConversationStore()
        >>> 
        >>> # Create conversation
        >>> conv_id = store.create_conversation()
        >>> conv = store.get_conversation(conv_id)
        >>> 
        >>> # List all conversations
        >>> conversations = store.list_conversations()
    """
    
    def __init__(self):
        """Initialize conversation store."""
        self._conversations: Dict[str, ConversationManager] = {}
    
    def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Create a new conversation.
        
        Args:
            conversation_id: Optional ID
            system_prompt: System prompt
            
        Returns:
            Conversation ID
        """
        manager = ConversationManager(
            conversation_id=conversation_id,
            system_prompt=system_prompt,
        )
        self._conversations[manager.conversation_id] = manager
        return manager.conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationManager]:
        """Get conversation by ID."""
        return self._conversations.get(conversation_id)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations."""
        return [
            {
                "id": conv_id,
                "state": conv.state.value,
                "created_at": conv.created_at,
                "num_turns": len(conv._turns),
            }
            for conv_id, conv in self._conversations.items()
        ]
    
    def clear_all(self) -> None:
        """Clear all conversations."""
        self._conversations.clear()


# Convenience functions

def create_conversation(
    system_prompt: Optional[str] = None,
    max_turns: int = 20,
) -> ConversationManager:
    """
    Create a new conversation manager.
    
    Args:
        system_prompt: Optional system prompt
        max_turns: Maximum turns
        
    Returns:
        ConversationManager
    """
    return ConversationManager(
        max_turns=max_turns,
        system_prompt=system_prompt,
    )


def format_messages_for_llm(
    messages: List[Message],
    format: str = "openai",
) -> List[Dict[str, str]]:
    """
    Format messages for LLM API.
    
    Args:
        messages: List of messages
        format: API format ('openai', 'anthropic')
        
    Returns:
        Formatted message list
    """
    if format == "openai":
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
    elif format == "anthropic":
        formatted = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                continue  # Handle separately for Anthropic
            formatted.append({
                "role": msg.role.value,
                "content": msg.content,
            })
        return formatted
    else:
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
