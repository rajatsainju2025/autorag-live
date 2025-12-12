"""Session Management Module for AutoRAG-Live.

Manage RAG sessions with:
- State persistence
- Conversation history
- Session lifecycle
- Context carryover
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session states."""

    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"
    CLOSED = "closed"


@dataclass
class Message:
    """Represents a conversation message."""

    role: str  # user, assistant, system
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionContext:
    """Context information for a session."""

    # Retrieved documents from current session
    retrieved_documents: list[dict[str, Any]] = field(default_factory=list)

    # Important entities/topics from conversation
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    # User preferences learned during session
    preferences: dict[str, Any] = field(default_factory=dict)

    # Cached query expansions
    query_cache: dict[str, list[str]] = field(default_factory=dict)

    # Session-specific settings
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "retrieved_documents": self.retrieved_documents,
            "entities": self.entities,
            "topics": self.topics,
            "preferences": self.preferences,
            "query_cache": self.query_cache,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionContext":
        """Create from dictionary."""
        return cls(
            retrieved_documents=data.get("retrieved_documents", []),
            entities=data.get("entities", []),
            topics=data.get("topics", []),
            preferences=data.get("preferences", {}),
            query_cache=data.get("query_cache", {}),
            settings=data.get("settings", {}),
        )


@dataclass
class Session:
    """Represents a RAG session."""

    session_id: str
    user_id: str | None = None
    state: SessionState = SessionState.ACTIVE

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None

    # Conversation
    messages: list[Message] = field(default_factory=list)
    turn_count: int = 0

    # Context
    context: SessionContext = field(default_factory=SessionContext)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default expiration."""
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(hours=24)

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state == SessionState.ACTIVE

    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        if self.expires_at and datetime.now() > self.expires_at:
            return True
        return self.state == SessionState.EXPIRED

    @property
    def duration(self) -> timedelta:
        """Get session duration."""
        return self.last_activity - self.created_at

    def add_message(self, role: str, content: str, **metadata: Any) -> Message:
        """Add a message to the session.

        Args:
            role: Message role (user, assistant, system)
            content: Message content
            **metadata: Additional metadata

        Returns:
            Created message
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata,
        )

        self.messages.append(message)
        self.last_activity = datetime.now()

        if role == "user":
            self.turn_count += 1

        return message

    def get_history(
        self,
        max_turns: int | None = None,
        max_tokens: int | None = None,
    ) -> list[Message]:
        """Get conversation history.

        Args:
            max_turns: Maximum number of turns
            max_tokens: Maximum total tokens (approximate)

        Returns:
            List of messages
        """
        messages = self.messages.copy()

        # Filter by turns
        if max_turns:
            # Keep system messages and last N turns
            system_msgs = [m for m in messages if m.role == "system"]
            other_msgs = [m for m in messages if m.role != "system"]

            # Each turn is user + assistant pair
            turn_limit = max_turns * 2
            messages = system_msgs + other_msgs[-turn_limit:]

        # Filter by tokens (approximate)
        if max_tokens:
            total_chars = sum(len(m.content) for m in messages)
            approx_tokens = total_chars / 4  # Rough estimate

            if approx_tokens > max_tokens:
                # Remove oldest non-system messages until within limit
                while (
                    len(messages) > 1
                    and sum(len(m.content) for m in messages) / 4 > max_tokens
                ):
                    # Find first non-system message
                    for i, m in enumerate(messages):
                        if m.role != "system":
                            messages.pop(i)
                            break
                    else:
                        break

        return messages

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "messages": [m.to_dict() for m in self.messages],
            "turn_count": self.turn_count,
            "context": self.context.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create from dictionary."""
        session = cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            state=SessionState(data["state"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
            turn_count=data.get("turn_count", 0),
            metadata=data.get("metadata", {}),
        )

        session.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        session.context = SessionContext.from_dict(data.get("context", {}))

        return session


class BaseSessionStore(ABC):
    """Abstract base class for session storage."""

    @abstractmethod
    def save(self, session: Session) -> bool:
        """Save a session."""
        pass

    @abstractmethod
    def load(self, session_id: str) -> Session | None:
        """Load a session by ID."""
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        pass

    @abstractmethod
    def list_sessions(
        self,
        user_id: str | None = None,
        state: SessionState | None = None,
    ) -> list[str]:
        """List session IDs."""
        pass


class InMemorySessionStore(BaseSessionStore):
    """In-memory session storage."""

    def __init__(self) -> None:
        """Initialize store."""
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def save(self, session: Session) -> bool:
        """Save session to memory."""
        with self._lock:
            self._sessions[session.session_id] = session
        return True

    def load(self, session_id: str) -> Session | None:
        """Load session from memory."""
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Delete session from memory."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False

    def list_sessions(
        self,
        user_id: str | None = None,
        state: SessionState | None = None,
    ) -> list[str]:
        """List session IDs."""
        with self._lock:
            sessions = list(self._sessions.values())

        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]

        if state:
            sessions = [s for s in sessions if s.state == state]

        return [s.session_id for s in sessions]


class FileSessionStore(BaseSessionStore):
    """File-based session storage."""

    def __init__(self, storage_path: str | Path) -> None:
        """Initialize file store.

        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, session_id: str) -> Path:
        """Get file path for session."""
        return self.storage_path / f"{session_id}.json"

    def save(self, session: Session) -> bool:
        """Save session to file."""
        try:
            filepath = self._get_filepath(session.session_id)
            filepath.write_text(json.dumps(session.to_dict(), indent=2))
            return True
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False

    def load(self, session_id: str) -> Session | None:
        """Load session from file."""
        filepath = self._get_filepath(session_id)

        if not filepath.exists():
            return None

        try:
            data = json.loads(filepath.read_text())
            return Session.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def delete(self, session_id: str) -> bool:
        """Delete session file."""
        filepath = self._get_filepath(session_id)

        if filepath.exists():
            try:
                filepath.unlink()
                return True
            except Exception as e:
                logger.error(f"Failed to delete session {session_id}: {e}")

        return False

    def list_sessions(
        self,
        user_id: str | None = None,
        state: SessionState | None = None,
    ) -> list[str]:
        """List session IDs from files."""
        session_ids: list[str] = []

        for filepath in self.storage_path.glob("*.json"):
            session_id = filepath.stem
            session = self.load(session_id)

            if session:
                if user_id and session.user_id != user_id:
                    continue
                if state and session.state != state:
                    continue
                session_ids.append(session_id)

        return session_ids


class SessionManager:
    """Manages RAG sessions."""

    def __init__(
        self,
        store: BaseSessionStore | None = None,
        default_ttl_hours: int = 24,
        max_history_turns: int = 10,
    ) -> None:
        """Initialize session manager.

        Args:
            store: Session storage backend
            default_ttl_hours: Default session TTL in hours
            max_history_turns: Default max conversation turns
        """
        self.store = store or InMemorySessionStore()
        self.default_ttl_hours = default_ttl_hours
        self.max_history_turns = max_history_turns

    def create_session(
        self,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_hours: int | None = None,
    ) -> Session:
        """Create a new session.

        Args:
            user_id: Optional user identifier
            metadata: Session metadata
            ttl_hours: Time to live in hours

        Returns:
            New session
        """
        session_id = str(uuid.uuid4())
        ttl = ttl_hours or self.default_ttl_hours

        session = Session(
            session_id=session_id,
            user_id=user_id,
            expires_at=datetime.now() + timedelta(hours=ttl),
            metadata=metadata or {},
        )

        self.store.save(session)
        logger.info(f"Created session {session_id}")

        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session or None if not found
        """
        session = self.store.load(session_id)

        if session and session.is_expired:
            session.state = SessionState.EXPIRED
            self.store.save(session)

        return session

    def get_or_create_session(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> Session:
        """Get existing session or create new one.

        Args:
            session_id: Optional session ID to retrieve
            user_id: User identifier for new session

        Returns:
            Session (existing or new)
        """
        if session_id:
            session = self.get_session(session_id)
            if session and session.is_active:
                return session

        return self.create_session(user_id=user_id)

    def update_session(self, session: Session) -> bool:
        """Update a session.

        Args:
            session: Session to update

        Returns:
            Success status
        """
        session.last_activity = datetime.now()
        return self.store.save(session)

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        documents: list[dict[str, Any]] | None = None,
    ) -> Session | None:
        """Add a conversation turn.

        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_message: Assistant's response
            documents: Retrieved documents

        Returns:
            Updated session or None
        """
        session = self.get_session(session_id)
        if not session or not session.is_active:
            return None

        # Add messages
        session.add_message("user", user_message)
        session.add_message("assistant", assistant_message)

        # Update context
        if documents:
            session.context.retrieved_documents = documents

        self.store.save(session)
        return session

    def get_conversation_history(
        self,
        session_id: str,
        max_turns: int | None = None,
    ) -> list[dict[str, str]]:
        """Get conversation history for RAG context.

        Args:
            session_id: Session identifier
            max_turns: Maximum turns to include

        Returns:
            List of message dicts
        """
        session = self.get_session(session_id)
        if not session:
            return []

        max_turns = max_turns or self.max_history_turns
        messages = session.get_history(max_turns=max_turns)

        return [{"role": m.role, "content": m.content} for m in messages]

    def close_session(self, session_id: str) -> bool:
        """Close a session.

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session.state = SessionState.CLOSED
        session.last_activity = datetime.now()

        return self.store.save(session)

    def pause_session(self, session_id: str) -> bool:
        """Pause a session.

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session.state = SessionState.PAUSED
        return self.store.save(session)

    def resume_session(self, session_id: str) -> Session | None:
        """Resume a paused session.

        Args:
            session_id: Session identifier

        Returns:
            Resumed session or None
        """
        session = self.get_session(session_id)
        if not session or session.state == SessionState.EXPIRED:
            return None

        session.state = SessionState.ACTIVE
        session.last_activity = datetime.now()
        self.store.save(session)

        return session

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        cleaned = 0

        for session_id in self.store.list_sessions():
            session = self.store.load(session_id)
            if session and session.is_expired:
                if self.store.delete(session_id):
                    cleaned += 1

        logger.info(f"Cleaned up {cleaned} expired sessions")
        return cleaned

    def get_session_stats(
        self,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Get session statistics.

        Args:
            user_id: Optional user filter

        Returns:
            Statistics dictionary
        """
        all_sessions = self.store.list_sessions(user_id=user_id)

        stats = {
            "total_sessions": len(all_sessions),
            "active": 0,
            "paused": 0,
            "expired": 0,
            "closed": 0,
            "total_turns": 0,
            "avg_turns_per_session": 0.0,
        }

        for session_id in all_sessions:
            session = self.store.load(session_id)
            if session:
                if session.state == SessionState.ACTIVE:
                    stats["active"] += 1
                elif session.state == SessionState.PAUSED:
                    stats["paused"] += 1
                elif session.state == SessionState.EXPIRED:
                    stats["expired"] += 1
                elif session.state == SessionState.CLOSED:
                    stats["closed"] += 1

                stats["total_turns"] += session.turn_count

        if stats["total_sessions"] > 0:
            stats["avg_turns_per_session"] = (
                stats["total_turns"] / stats["total_sessions"]
            )

        return stats


# Global session manager
_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get global session manager."""
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager


# Convenience functions


def create_session(
    user_id: str | None = None,
    ttl_hours: int = 24,
) -> Session:
    """Create a new session.

    Args:
        user_id: User identifier
        ttl_hours: Session TTL

    Returns:
        New session
    """
    return get_session_manager().create_session(user_id=user_id, ttl_hours=ttl_hours)


def get_session(session_id: str) -> Session | None:
    """Get a session by ID.

    Args:
        session_id: Session identifier

    Returns:
        Session or None
    """
    return get_session_manager().get_session(session_id)


def add_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> Session | None:
    """Add a conversation turn to a session.

    Args:
        session_id: Session identifier
        user_message: User's message
        assistant_message: Assistant's response

    Returns:
        Updated session
    """
    return get_session_manager().add_turn(
        session_id, user_message, assistant_message
    )


def get_history(
    session_id: str,
    max_turns: int = 10,
) -> list[dict[str, str]]:
    """Get conversation history.

    Args:
        session_id: Session identifier
        max_turns: Maximum turns

    Returns:
        List of messages
    """
    return get_session_manager().get_conversation_history(session_id, max_turns)
