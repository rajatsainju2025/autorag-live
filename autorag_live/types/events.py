"""
Event system module for AutoRAG-Live.

Provides event-driven architecture support for building
reactive and decoupled RAG systems.

Features:
- Event bus with pub/sub patterns
- Typed event system
- Event filtering and routing
- Event persistence and replay
- Event sourcing patterns
- Dead letter queues
- Event tracing

Example usage:
    >>> bus = EventBus()
    >>> 
    >>> @bus.subscribe("document.indexed")
    >>> async def on_document_indexed(event: Event):
    ...     print(f"Document indexed: {event.data['doc_id']}")
    >>> 
    >>> await bus.publish(Event(
    ...     type="document.indexed",
    ...     data={"doc_id": "doc123"}
    ... ))
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class EventPriority(Enum):
    """Event priority levels."""
    
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventStatus(Enum):
    """Event processing status."""
    
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    DEAD_LETTER = auto()


class DeliveryGuarantee(Enum):
    """Event delivery guarantees."""
    
    AT_MOST_ONCE = auto()  # Fire and forget
    AT_LEAST_ONCE = auto()  # With retries
    EXACTLY_ONCE = auto()  # With deduplication


@dataclass
class EventMetadata:
    """Event metadata."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Source information
    source: str = ""
    source_id: str = ""
    
    # Correlation
    correlation_id: str = ""
    causation_id: str = ""
    
    # Delivery
    delivery_count: int = 0
    max_retries: int = 3
    
    # Tracing
    trace_id: str = ""
    span_id: str = ""
    
    # User data
    user_id: str = ""
    tenant_id: str = ""
    
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    """
    Base event class.
    
    Example:
        >>> event = Event(
        ...     type="document.created",
        ...     data={"doc_id": "123", "content": "..."},
        ... )
    """
    
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: EventMetadata = field(default_factory=EventMetadata)
    
    # Priority
    priority: EventPriority = EventPriority.NORMAL
    
    # Status
    status: EventStatus = EventStatus.PENDING
    
    @property
    def event_id(self) -> str:
        """Get event ID."""
        return self.metadata.event_id
    
    @property
    def timestamp(self) -> float:
        """Get event timestamp."""
        return self.metadata.timestamp
    
    @property
    def datetime(self) -> datetime:
        """Get event datetime."""
        return datetime.fromtimestamp(self.metadata.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'type': self.type,
            'data': self.data,
            'timestamp': self.timestamp,
            'priority': self.priority.name,
            'status': self.status.name,
            'metadata': {
                'source': self.metadata.source,
                'correlation_id': self.metadata.correlation_id,
                'trace_id': self.metadata.trace_id,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create from dictionary."""
        metadata = EventMetadata(
            event_id=data.get('event_id', str(uuid.uuid4())),
            timestamp=data.get('timestamp', time.time()),
            source=data.get('metadata', {}).get('source', ''),
            correlation_id=data.get('metadata', {}).get('correlation_id', ''),
            trace_id=data.get('metadata', {}).get('trace_id', ''),
        )
        
        return cls(
            type=data['type'],
            data=data.get('data', {}),
            metadata=metadata,
            priority=EventPriority[data.get('priority', 'NORMAL')],
        )


# Typed events for common RAG operations

@dataclass
class DocumentEvent(Event):
    """Event for document operations."""
    
    def __init__(
        self,
        event_type: str,
        doc_id: str,
        **kwargs,
    ):
        data = {'doc_id': doc_id, **kwargs}
        super().__init__(type=f"document.{event_type}", data=data)


@dataclass
class QueryEvent(Event):
    """Event for query operations."""
    
    def __init__(
        self,
        event_type: str,
        query: str,
        **kwargs,
    ):
        data = {'query': query, **kwargs}
        super().__init__(type=f"query.{event_type}", data=data)


@dataclass
class RetrievalEvent(Event):
    """Event for retrieval operations."""
    
    def __init__(
        self,
        event_type: str,
        query: str,
        results: List[Dict[str, Any]],
        **kwargs,
    ):
        data = {'query': query, 'results': results, **kwargs}
        super().__init__(type=f"retrieval.{event_type}", data=data)


@dataclass
class GenerationEvent(Event):
    """Event for generation operations."""
    
    def __init__(
        self,
        event_type: str,
        response: str,
        **kwargs,
    ):
        data = {'response': response, **kwargs}
        super().__init__(type=f"generation.{event_type}", data=data)


@dataclass
class Subscription:
    """Event subscription."""
    
    subscription_id: str
    pattern: str
    handler: Callable[[Event], Awaitable[None]]
    
    # Filters
    priority_filter: Optional[EventPriority] = None
    source_filter: Optional[str] = None
    
    # Options
    async_handler: bool = True
    max_concurrent: int = 10
    
    # Stats
    events_received: int = 0
    events_processed: int = 0
    errors: int = 0
    
    created_at: float = field(default_factory=time.time)
    
    def matches(self, event: Event) -> bool:
        """Check if event matches subscription."""
        import fnmatch
        
        # Type matching
        if not fnmatch.fnmatch(event.type, self.pattern):
            return False
        
        # Priority filter
        if self.priority_filter and event.priority != self.priority_filter:
            return False
        
        # Source filter
        if self.source_filter and event.metadata.source != self.source_filter:
            return False
        
        return True


class EventStore(ABC):
    """Abstract event store."""
    
    @abstractmethod
    async def append(self, event: Event) -> None:
        """Append event to store."""
        pass
    
    @abstractmethod
    async def get(self, event_id: str) -> Optional[Event]:
        """Get event by ID."""
        pass
    
    @abstractmethod
    async def get_by_type(
        self,
        event_type: str,
        limit: int = 100,
    ) -> List[Event]:
        """Get events by type."""
        pass
    
    @abstractmethod
    async def get_by_correlation(
        self,
        correlation_id: str,
    ) -> List[Event]:
        """Get events by correlation ID."""
        pass


class InMemoryEventStore(EventStore):
    """In-memory event store."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize store.
        
        Args:
            max_size: Maximum events to keep
        """
        self.max_size = max_size
        self._events: Dict[str, Event] = {}
        self._by_type: Dict[str, List[str]] = defaultdict(list)
        self._by_correlation: Dict[str, List[str]] = defaultdict(list)
        self._order: List[str] = []
        self._lock = asyncio.Lock()
    
    async def append(self, event: Event) -> None:
        async with self._lock:
            # Evict if at capacity
            while len(self._events) >= self.max_size:
                oldest_id = self._order.pop(0)
                oldest = self._events.pop(oldest_id, None)
                if oldest:
                    self._by_type[oldest.type].remove(oldest_id)
                    if oldest.metadata.correlation_id:
                        self._by_correlation[oldest.metadata.correlation_id].remove(oldest_id)
            
            # Store event
            self._events[event.event_id] = event
            self._by_type[event.type].append(event.event_id)
            self._order.append(event.event_id)
            
            if event.metadata.correlation_id:
                self._by_correlation[event.metadata.correlation_id].append(
                    event.event_id
                )
    
    async def get(self, event_id: str) -> Optional[Event]:
        return self._events.get(event_id)
    
    async def get_by_type(
        self,
        event_type: str,
        limit: int = 100,
    ) -> List[Event]:
        import fnmatch
        
        result = []
        
        for type_key, event_ids in self._by_type.items():
            if fnmatch.fnmatch(type_key, event_type):
                for event_id in event_ids[-limit:]:
                    event = self._events.get(event_id)
                    if event:
                        result.append(event)
        
        return result[-limit:]
    
    async def get_by_correlation(
        self,
        correlation_id: str,
    ) -> List[Event]:
        result = []
        
        for event_id in self._by_correlation.get(correlation_id, []):
            event = self._events.get(event_id)
            if event:
                result.append(event)
        
        return result


class DeadLetterQueue:
    """Queue for failed events."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize dead letter queue.
        
        Args:
            max_size: Maximum events to keep
        """
        self.max_size = max_size
        self._events: List[Tuple[Event, str]] = []
        self._lock = asyncio.Lock()
    
    async def add(
        self,
        event: Event,
        error: str,
    ) -> None:
        """Add failed event."""
        async with self._lock:
            if len(self._events) >= self.max_size:
                self._events.pop(0)
            
            event.status = EventStatus.DEAD_LETTER
            self._events.append((event, error))
    
    async def get_all(self) -> List[Tuple[Event, str]]:
        """Get all dead letter events."""
        return self._events.copy()
    
    async def replay(
        self,
        bus: "EventBus",
    ) -> int:
        """Replay all events."""
        count = 0
        
        async with self._lock:
            for event, _ in self._events:
                event.status = EventStatus.PENDING
                event.metadata.delivery_count = 0
                await bus.publish(event)
                count += 1
            
            self._events.clear()
        
        return count
    
    @property
    def size(self) -> int:
        """Get queue size."""
        return len(self._events)


class EventBus:
    """
    Main event bus for pub/sub messaging.
    
    Example:
        >>> bus = EventBus()
        >>> 
        >>> # Subscribe to events
        >>> @bus.subscribe("document.*")
        >>> async def handle_document_events(event: Event):
        ...     print(f"Document event: {event.type}")
        >>> 
        >>> # Publish events
        >>> await bus.publish(Event(
        ...     type="document.created",
        ...     data={"doc_id": "123"},
        ... ))
    """
    
    def __init__(
        self,
        delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE,
        max_concurrent_handlers: int = 100,
    ):
        """
        Initialize event bus.
        
        Args:
            delivery_guarantee: Delivery guarantee level
            max_concurrent_handlers: Max concurrent handlers
        """
        self.delivery_guarantee = delivery_guarantee
        self.max_concurrent_handlers = max_concurrent_handlers
        
        self._subscriptions: Dict[str, Subscription] = {}
        self._event_store = InMemoryEventStore()
        self._dead_letter_queue = DeadLetterQueue()
        
        self._processed_ids: Set[str] = set()  # For deduplication
        self._semaphore = asyncio.Semaphore(max_concurrent_handlers)
        
        # Statistics
        self._stats = EventBusStats()
    
    def subscribe(
        self,
        pattern: str,
        priority_filter: Optional[EventPriority] = None,
        source_filter: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to subscribe to events.
        
        Args:
            pattern: Event type pattern (supports wildcards)
            priority_filter: Optional priority filter
            source_filter: Optional source filter
            
        Returns:
            Decorator function
        """
        def decorator(handler: Callable[[Event], Awaitable[None]]) -> Callable:
            subscription = Subscription(
                subscription_id=str(uuid.uuid4()),
                pattern=pattern,
                handler=handler,
                priority_filter=priority_filter,
                source_filter=source_filter,
            )
            
            self._subscriptions[subscription.subscription_id] = subscription
            
            return handler
        
        return decorator
    
    def add_subscription(
        self,
        pattern: str,
        handler: Callable[[Event], Awaitable[None]],
        **kwargs,
    ) -> str:
        """
        Add subscription programmatically.
        
        Args:
            pattern: Event type pattern
            handler: Event handler
            **kwargs: Additional subscription options
            
        Returns:
            Subscription ID
        """
        subscription = Subscription(
            subscription_id=str(uuid.uuid4()),
            pattern=pattern,
            handler=handler,
            **kwargs,
        )
        
        self._subscriptions[subscription.subscription_id] = subscription
        return subscription.subscription_id
    
    def remove_subscription(self, subscription_id: str) -> bool:
        """Remove subscription."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False
    
    async def publish(
        self,
        event: Event,
    ) -> None:
        """
        Publish an event.
        
        Args:
            event: Event to publish
        """
        self._stats.events_published += 1
        
        # Deduplication for exactly-once delivery
        if self.delivery_guarantee == DeliveryGuarantee.EXACTLY_ONCE:
            if event.event_id in self._processed_ids:
                logger.debug(f"Duplicate event ignored: {event.event_id}")
                return
            self._processed_ids.add(event.event_id)
        
        # Store event
        await self._event_store.append(event)
        
        # Find matching subscriptions
        matching = [
            sub for sub in self._subscriptions.values()
            if sub.matches(event)
        ]
        
        if not matching:
            logger.debug(f"No subscribers for event: {event.type}")
            return
        
        # Dispatch to handlers
        tasks = []
        for subscription in matching:
            subscription.events_received += 1
            tasks.append(self._dispatch(event, subscription))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _dispatch(
        self,
        event: Event,
        subscription: Subscription,
    ) -> None:
        """Dispatch event to handler."""
        event.status = EventStatus.PROCESSING
        event.metadata.delivery_count += 1
        
        try:
            async with self._semaphore:
                await subscription.handler(event)
            
            event.status = EventStatus.COMPLETED
            subscription.events_processed += 1
            self._stats.events_delivered += 1
            
        except Exception as e:
            subscription.errors += 1
            self._stats.events_failed += 1
            
            logger.error(
                f"Error handling event {event.event_id}: {e}"
            )
            
            # Retry logic
            if self.delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE:
                if event.metadata.delivery_count < event.metadata.max_retries:
                    event.status = EventStatus.RETRYING
                    await asyncio.sleep(2 ** event.metadata.delivery_count)
                    await self._dispatch(event, subscription)
                else:
                    # Move to dead letter queue
                    await self._dead_letter_queue.add(event, str(e))
            else:
                event.status = EventStatus.FAILED
    
    async def publish_many(
        self,
        events: List[Event],
    ) -> None:
        """Publish multiple events."""
        for event in events:
            await self.publish(event)
    
    async def replay_events(
        self,
        event_type: str,
        limit: int = 100,
    ) -> int:
        """
        Replay events from store.
        
        Args:
            event_type: Event type pattern
            limit: Maximum events
            
        Returns:
            Number of events replayed
        """
        events = await self._event_store.get_by_type(event_type, limit)
        
        for event in events:
            event.status = EventStatus.PENDING
            event.metadata.delivery_count = 0
            await self.publish(event)
        
        return len(events)
    
    @property
    def dead_letter_queue(self) -> DeadLetterQueue:
        """Get dead letter queue."""
        return self._dead_letter_queue
    
    @property
    def event_store(self) -> EventStore:
        """Get event store."""
        return self._event_store
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            'events_published': self._stats.events_published,
            'events_delivered': self._stats.events_delivered,
            'events_failed': self._stats.events_failed,
            'dead_letter_size': self._dead_letter_queue.size,
            'subscriptions': len(self._subscriptions),
        }


@dataclass
class EventBusStats:
    """Event bus statistics."""
    
    events_published: int = 0
    events_delivered: int = 0
    events_failed: int = 0


class EventEmitter:
    """
    Mixin for classes that emit events.
    
    Example:
        >>> class DocumentIndexer(EventEmitter):
        ...     async def index(self, doc):
        ...         # Index document
        ...         await self.emit("document.indexed", {"doc_id": doc.id})
    """
    
    def __init__(self, bus: Optional[EventBus] = None):
        """
        Initialize emitter.
        
        Args:
            bus: Event bus (or creates new one)
        """
        self._bus = bus or EventBus()
        self._source = self.__class__.__name__
    
    async def emit(
        self,
        event_type: str,
        data: Dict[str, Any] = None,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: str = "",
    ) -> Event:
        """
        Emit an event.
        
        Args:
            event_type: Event type
            data: Event data
            priority: Event priority
            correlation_id: Correlation ID
            
        Returns:
            Emitted event
        """
        event = Event(
            type=event_type,
            data=data or {},
            priority=priority,
            metadata=EventMetadata(
                source=self._source,
                correlation_id=correlation_id,
            ),
        )
        
        await self._bus.publish(event)
        return event
    
    def on(
        self,
        pattern: str,
    ) -> Callable:
        """
        Decorator to handle events.
        
        Args:
            pattern: Event type pattern
            
        Returns:
            Decorator
        """
        return self._bus.subscribe(pattern)


class EventHandler:
    """
    Base class for event handlers.
    
    Example:
        >>> class DocumentHandler(EventHandler):
        ...     @EventHandler.handles("document.created")
        ...     async def on_document_created(self, event: Event):
        ...         print(f"Document created: {event.data['doc_id']}")
    """
    
    _handlers: Dict[str, str] = {}
    
    @classmethod
    def handles(cls, pattern: str) -> Callable:
        """Decorator to mark handler method."""
        def decorator(method: Callable) -> Callable:
            cls._handlers[pattern] = method.__name__
            return method
        return decorator
    
    def register(self, bus: EventBus) -> None:
        """Register all handlers with bus."""
        for pattern, method_name in self._handlers.items():
            method = getattr(self, method_name)
            bus.add_subscription(pattern, method)


class EventAggregator:
    """
    Aggregate events for batch processing.
    
    Example:
        >>> aggregator = EventAggregator(
        ...     window_seconds=5,
        ...     max_batch_size=100,
        ... )
        >>> 
        >>> @aggregator.on_batch
        >>> async def process_batch(events: List[Event]):
        ...     print(f"Processing {len(events)} events")
        >>> 
        >>> await aggregator.add(event)
    """
    
    def __init__(
        self,
        window_seconds: float = 5.0,
        max_batch_size: int = 100,
    ):
        """
        Initialize aggregator.
        
        Args:
            window_seconds: Time window for batching
            max_batch_size: Maximum batch size
        """
        self.window_seconds = window_seconds
        self.max_batch_size = max_batch_size
        
        self._buffer: List[Event] = []
        self._batch_handler: Optional[Callable] = None
        self._timer_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    def on_batch(
        self,
        handler: Callable[[List[Event]], Awaitable[None]],
    ) -> Callable:
        """Set batch handler."""
        self._batch_handler = handler
        return handler
    
    async def add(self, event: Event) -> None:
        """Add event to buffer."""
        async with self._lock:
            self._buffer.append(event)
            
            # Start timer if needed
            if self._timer_task is None or self._timer_task.done():
                self._timer_task = asyncio.create_task(
                    self._flush_after_delay()
                )
            
            # Flush if at capacity
            if len(self._buffer) >= self.max_batch_size:
                await self._flush()
    
    async def _flush_after_delay(self) -> None:
        """Flush after time window."""
        await asyncio.sleep(self.window_seconds)
        await self._flush()
    
    async def _flush(self) -> None:
        """Flush buffer to handler."""
        async with self._lock:
            if not self._buffer:
                return
            
            events = self._buffer.copy()
            self._buffer.clear()
        
        if self._batch_handler:
            await self._batch_handler(events)


# Convenience functions

def create_event_bus(
    delivery_guarantee: str = "at_least_once",
) -> EventBus:
    """
    Create event bus.
    
    Args:
        delivery_guarantee: Delivery guarantee level
        
    Returns:
        EventBus instance
    """
    guarantee_map = {
        'at_most_once': DeliveryGuarantee.AT_MOST_ONCE,
        'at_least_once': DeliveryGuarantee.AT_LEAST_ONCE,
        'exactly_once': DeliveryGuarantee.EXACTLY_ONCE,
    }
    
    guarantee = guarantee_map.get(delivery_guarantee, DeliveryGuarantee.AT_LEAST_ONCE)
    
    return EventBus(delivery_guarantee=guarantee)


def create_document_event(
    action: str,
    doc_id: str,
    **kwargs,
) -> Event:
    """
    Create document event.
    
    Args:
        action: Event action (created, updated, deleted)
        doc_id: Document ID
        **kwargs: Additional data
        
    Returns:
        Event instance
    """
    return DocumentEvent(action, doc_id, **kwargs)


def create_query_event(
    action: str,
    query: str,
    **kwargs,
) -> Event:
    """
    Create query event.
    
    Args:
        action: Event action (started, completed, failed)
        query: Query text
        **kwargs: Additional data
        
    Returns:
        Event instance
    """
    return QueryEvent(action, query, **kwargs)
