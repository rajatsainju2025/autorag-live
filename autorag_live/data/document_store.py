"""
Document storage abstraction for AutoRAG-Live.

Provides a unified interface for document storage with
multiple backend support including in-memory, file-based,
and database storage.

Features:
- Unified storage interface
- Multiple backend support (memory, file, SQLite)
- CRUD operations
- Batch operations
- Metadata filtering
- Document indexing

Example usage:
    >>> store = DocumentStore(backend="sqlite", path="docs.db")
    >>> store.add(Document(id="1", content="Hello world"))
    >>> doc = store.get("1")
    >>> results = store.search({"tag": "important"})
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Union

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document for storage."""
    
    id: str
    content: str
    
    # Optional fields
    title: Optional[str] = None
    source: Optional[str] = None
    
    # Embeddings
    embedding: Optional[List[float]] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate ID from content hash."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return content_hash[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'title': self.title,
            'source': self.source,
            'embedding': self.embedding,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata,
            'tags': self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Document:
        """Create from dictionary."""
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return cls(
            id=data.get('id', ''),
            content=data.get('content', ''),
            title=data.get('title'),
            source=data.get('source'),
            embedding=data.get('embedding'),
            created_at=created_at or datetime.now(),
            updated_at=updated_at or datetime.now(),
            metadata=data.get('metadata', {}),
            tags=data.get('tags', []),
        )


@dataclass
class QueryFilter:
    """Filter for document queries."""
    
    # Exact matches
    ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    
    # Metadata filters
    metadata_equals: Optional[Dict[str, Any]] = None
    metadata_contains: Optional[Dict[str, Any]] = None
    
    # Date filters
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    
    # Content filters
    content_contains: Optional[str] = None
    
    # Limits
    limit: Optional[int] = None
    offset: int = 0


@dataclass
class StoreStats:
    """Storage statistics."""
    
    total_documents: int = 0
    total_size_bytes: int = 0
    unique_tags: int = 0
    unique_sources: int = 0
    
    # Time range
    oldest_doc: Optional[datetime] = None
    newest_doc: Optional[datetime] = None
    
    # Backend info
    backend: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def add(self, document: Document) -> str:
        """Add a document."""
        pass
    
    @abstractmethod
    def get(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        pass
    
    @abstractmethod
    def update(self, document: Document) -> bool:
        """Update a document."""
        pass
    
    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        """Delete a document."""
        pass
    
    @abstractmethod
    def list(self, filter: Optional[QueryFilter] = None) -> List[Document]:
        """List documents with optional filtering."""
        pass
    
    @abstractmethod
    def count(self, filter: Optional[QueryFilter] = None) -> int:
        """Count documents."""
        pass
    
    @abstractmethod
    def clear(self) -> int:
        """Clear all documents."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get backend name."""
        pass


class InMemoryBackend(StorageBackend):
    """In-memory storage backend."""
    
    def __init__(self):
        """Initialize in-memory backend."""
        self._documents: Dict[str, Document] = {}
    
    @property
    def name(self) -> str:
        return "memory"
    
    def add(self, document: Document) -> str:
        """Add document to memory."""
        self._documents[document.id] = document
        return document.id
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get document from memory."""
        return self._documents.get(doc_id)
    
    def update(self, document: Document) -> bool:
        """Update document in memory."""
        if document.id in self._documents:
            document.updated_at = datetime.now()
            self._documents[document.id] = document
            return True
        return False
    
    def delete(self, doc_id: str) -> bool:
        """Delete document from memory."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False
    
    def list(self, filter: Optional[QueryFilter] = None) -> List[Document]:
        """List documents from memory."""
        docs = list(self._documents.values())
        
        if filter:
            docs = self._apply_filter(docs, filter)
        
        return docs
    
    def count(self, filter: Optional[QueryFilter] = None) -> int:
        """Count documents in memory."""
        if filter:
            return len(self.list(filter))
        return len(self._documents)
    
    def clear(self) -> int:
        """Clear all documents from memory."""
        count = len(self._documents)
        self._documents.clear()
        return count
    
    def _apply_filter(
        self,
        docs: List[Document],
        filter: QueryFilter,
    ) -> List[Document]:
        """Apply filter to documents."""
        result = docs
        
        # Filter by IDs
        if filter.ids:
            result = [d for d in result if d.id in filter.ids]
        
        # Filter by tags
        if filter.tags:
            result = [
                d for d in result
                if any(t in d.tags for t in filter.tags)
            ]
        
        # Filter by sources
        if filter.sources:
            result = [d for d in result if d.source in filter.sources]
        
        # Filter by metadata
        if filter.metadata_equals:
            result = [
                d for d in result
                if all(
                    d.metadata.get(k) == v
                    for k, v in filter.metadata_equals.items()
                )
            ]
        
        if filter.metadata_contains:
            result = [
                d for d in result
                if all(
                    k in d.metadata and str(v) in str(d.metadata.get(k, ''))
                    for k, v in filter.metadata_contains.items()
                )
            ]
        
        # Filter by date
        if filter.created_after:
            result = [d for d in result if d.created_at >= filter.created_after]
        
        if filter.created_before:
            result = [d for d in result if d.created_at <= filter.created_before]
        
        # Filter by content
        if filter.content_contains:
            search_term = filter.content_contains.lower()
            result = [
                d for d in result
                if search_term in d.content.lower()
            ]
        
        # Apply offset and limit
        if filter.offset:
            result = result[filter.offset:]
        
        if filter.limit:
            result = result[:filter.limit]
        
        return result


class FileBackend(StorageBackend):
    """File-based storage backend."""
    
    def __init__(
        self,
        path: Union[str, Path],
        format: str = "json",
    ):
        """
        Initialize file backend.
        
        Args:
            path: Storage directory path
            format: File format (json, pickle)
        """
        self.path = Path(path)
        self.format = format
        self.path.mkdir(parents=True, exist_ok=True)
        
        # Index file
        self._index_path = self.path / "_index.json"
        self._index: Dict[str, str] = {}
        self._load_index()
    
    @property
    def name(self) -> str:
        return "file"
    
    def _load_index(self) -> None:
        """Load document index."""
        if self._index_path.exists():
            with open(self._index_path, 'r') as f:
                self._index = json.load(f)
    
    def _save_index(self) -> None:
        """Save document index."""
        with open(self._index_path, 'w') as f:
            json.dump(self._index, f)
    
    def _doc_path(self, doc_id: str) -> Path:
        """Get document file path."""
        ext = ".json" if self.format == "json" else ".pkl"
        return self.path / f"{doc_id}{ext}"
    
    def add(self, document: Document) -> str:
        """Add document to file."""
        doc_path = self._doc_path(document.id)
        
        if self.format == "json":
            with open(doc_path, 'w') as f:
                json.dump(document.to_dict(), f)
        else:
            with open(doc_path, 'wb') as f:
                pickle.dump(document, f)
        
        self._index[document.id] = str(doc_path)
        self._save_index()
        
        return document.id
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get document from file."""
        if doc_id not in self._index:
            return None
        
        doc_path = self._doc_path(doc_id)
        if not doc_path.exists():
            return None
        
        try:
            if self.format == "json":
                with open(doc_path, 'r') as f:
                    return Document.from_dict(json.load(f))
            else:
                with open(doc_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load document {doc_id}: {e}")
            return None
    
    def update(self, document: Document) -> bool:
        """Update document in file."""
        if document.id not in self._index:
            return False
        
        document.updated_at = datetime.now()
        self.add(document)
        return True
    
    def delete(self, doc_id: str) -> bool:
        """Delete document file."""
        if doc_id not in self._index:
            return False
        
        doc_path = self._doc_path(doc_id)
        if doc_path.exists():
            doc_path.unlink()
        
        del self._index[doc_id]
        self._save_index()
        
        return True
    
    def list(self, filter: Optional[QueryFilter] = None) -> List[Document]:
        """List documents from files."""
        docs = []
        
        for doc_id in self._index:
            doc = self.get(doc_id)
            if doc:
                docs.append(doc)
        
        if filter:
            # Use in-memory backend's filter logic
            memory = InMemoryBackend()
            for doc in docs:
                memory.add(doc)
            docs = memory.list(filter)
        
        return docs
    
    def count(self, filter: Optional[QueryFilter] = None) -> int:
        """Count documents in files."""
        if filter:
            return len(self.list(filter))
        return len(self._index)
    
    def clear(self) -> int:
        """Clear all document files."""
        count = 0
        for doc_id in list(self._index.keys()):
            if self.delete(doc_id):
                count += 1
        return count


class SQLiteBackend(StorageBackend):
    """SQLite storage backend."""
    
    def __init__(
        self,
        path: Union[str, Path] = ":memory:",
    ):
        """
        Initialize SQLite backend.
        
        Args:
            path: Database file path or ":memory:"
        """
        self.path = str(path)
        self.conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
        )
        self._create_tables()
    
    @property
    def name(self) -> str:
        return "sqlite"
    
    def _create_tables(self) -> None:
        """Create database tables."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                title TEXT,
                source TEXT,
                embedding BLOB,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT,
                tags TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source ON documents(source)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created ON documents(created_at)
        """)
        
        self.conn.commit()
    
    def add(self, document: Document) -> str:
        """Add document to SQLite."""
        cursor = self.conn.cursor()
        
        embedding_blob = None
        if document.embedding:
            embedding_blob = pickle.dumps(document.embedding)
        
        cursor.execute("""
            INSERT OR REPLACE INTO documents
            (id, content, title, source, embedding, created_at, updated_at, metadata, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            document.id,
            document.content,
            document.title,
            document.source,
            embedding_blob,
            document.created_at.isoformat(),
            document.updated_at.isoformat(),
            json.dumps(document.metadata),
            json.dumps(document.tags),
        ))
        
        self.conn.commit()
        return document.id
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get document from SQLite."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, content, title, source, embedding, created_at, updated_at, metadata, tags
            FROM documents WHERE id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return self._row_to_document(row)
    
    def update(self, document: Document) -> bool:
        """Update document in SQLite."""
        if not self.get(document.id):
            return False
        
        document.updated_at = datetime.now()
        self.add(document)
        return True
    
    def delete(self, doc_id: str) -> bool:
        """Delete document from SQLite."""
        cursor = self.conn.cursor()
        
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        
        return cursor.rowcount > 0
    
    def list(self, filter: Optional[QueryFilter] = None) -> List[Document]:
        """List documents from SQLite."""
        cursor = self.conn.cursor()
        
        query = "SELECT id, content, title, source, embedding, created_at, updated_at, metadata, tags FROM documents"
        params = []
        conditions = []
        
        if filter:
            if filter.ids:
                placeholders = ','.join('?' * len(filter.ids))
                conditions.append(f"id IN ({placeholders})")
                params.extend(filter.ids)
            
            if filter.sources:
                placeholders = ','.join('?' * len(filter.sources))
                conditions.append(f"source IN ({placeholders})")
                params.extend(filter.sources)
            
            if filter.created_after:
                conditions.append("created_at >= ?")
                params.append(filter.created_after.isoformat())
            
            if filter.created_before:
                conditions.append("created_at <= ?")
                params.append(filter.created_before.isoformat())
            
            if filter.content_contains:
                conditions.append("content LIKE ?")
                params.append(f"%{filter.content_contains}%")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC"
        
        if filter and filter.limit:
            query += f" LIMIT {filter.limit}"
            if filter.offset:
                query += f" OFFSET {filter.offset}"
        
        cursor.execute(query, params)
        
        docs = [self._row_to_document(row) for row in cursor.fetchall()]
        
        # Apply additional filters not supported by SQL
        if filter:
            if filter.tags:
                docs = [
                    d for d in docs
                    if any(t in d.tags for t in filter.tags)
                ]
            
            if filter.metadata_equals:
                docs = [
                    d for d in docs
                    if all(
                        d.metadata.get(k) == v
                        for k, v in filter.metadata_equals.items()
                    )
                ]
        
        return docs
    
    def count(self, filter: Optional[QueryFilter] = None) -> int:
        """Count documents in SQLite."""
        if filter:
            return len(self.list(filter))
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        return cursor.fetchone()[0]
    
    def clear(self) -> int:
        """Clear all documents from SQLite."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        
        cursor.execute("DELETE FROM documents")
        self.conn.commit()
        
        return count
    
    def _row_to_document(self, row: tuple) -> Document:
        """Convert database row to Document."""
        (doc_id, content, title, source, embedding_blob,
         created_at, updated_at, metadata_json, tags_json) = row
        
        embedding = None
        if embedding_blob:
            embedding = pickle.loads(embedding_blob)
        
        return Document(
            id=doc_id,
            content=content,
            title=title,
            source=source,
            embedding=embedding,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at),
            metadata=json.loads(metadata_json) if metadata_json else {},
            tags=json.loads(tags_json) if tags_json else [],
        )
    
    def close(self) -> None:
        """Close database connection."""
        self.conn.close()


class DocumentStore:
    """
    Main document store interface.
    
    Example:
        >>> # In-memory store
        >>> store = DocumentStore()
        >>> 
        >>> # File-based store
        >>> store = DocumentStore(backend="file", path="./docs")
        >>> 
        >>> # SQLite store
        >>> store = DocumentStore(backend="sqlite", path="docs.db")
        >>> 
        >>> # Add documents
        >>> doc_id = store.add(Document(id="1", content="Hello"))
        >>> 
        >>> # Get document
        >>> doc = store.get("1")
        >>> 
        >>> # Search
        >>> results = store.search(content="hello")
    """
    
    def __init__(
        self,
        backend: str = "memory",
        path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initialize document store.
        
        Args:
            backend: Backend type (memory, file, sqlite)
            path: Path for file/sqlite backends
            **kwargs: Additional backend parameters
        """
        self.backend_type = backend
        
        if backend == "memory":
            self._backend = InMemoryBackend()
        elif backend == "file":
            if not path:
                raise ValueError("Path required for file backend")
            self._backend = FileBackend(path, **kwargs)
        elif backend == "sqlite":
            self._backend = SQLiteBackend(path or ":memory:")
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    @property
    def backend(self) -> StorageBackend:
        """Get storage backend."""
        return self._backend
    
    def add(
        self,
        document: Union[Document, Dict[str, Any]],
    ) -> str:
        """
        Add a document.
        
        Args:
            document: Document or dict
            
        Returns:
            Document ID
        """
        if isinstance(document, dict):
            document = Document.from_dict(document)
        
        return self._backend.add(document)
    
    def add_many(
        self,
        documents: List[Union[Document, Dict[str, Any]]],
    ) -> List[str]:
        """
        Add multiple documents.
        
        Args:
            documents: List of documents
            
        Returns:
            List of document IDs
        """
        ids = []
        for doc in documents:
            ids.append(self.add(doc))
        return ids
    
    def get(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document or None
        """
        return self._backend.get(doc_id)
    
    def get_many(self, doc_ids: List[str]) -> List[Document]:
        """
        Get multiple documents.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            List of documents (excludes not found)
        """
        docs = []
        for doc_id in doc_ids:
            doc = self.get(doc_id)
            if doc:
                docs.append(doc)
        return docs
    
    def update(
        self,
        document: Union[Document, Dict[str, Any]],
    ) -> bool:
        """
        Update a document.
        
        Args:
            document: Document or dict
            
        Returns:
            True if updated
        """
        if isinstance(document, dict):
            document = Document.from_dict(document)
        
        return self._backend.update(document)
    
    def delete(self, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deleted
        """
        return self._backend.delete(doc_id)
    
    def delete_many(self, doc_ids: List[str]) -> int:
        """
        Delete multiple documents.
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Number of documents deleted
        """
        count = 0
        for doc_id in doc_ids:
            if self.delete(doc_id):
                count += 1
        return count
    
    def search(
        self,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Document]:
        """
        Search for documents.
        
        Args:
            content: Content search term
            tags: Filter by tags
            source: Filter by source
            metadata: Filter by metadata
            limit: Maximum results
            offset: Result offset
            
        Returns:
            List of matching documents
        """
        filter = QueryFilter(
            content_contains=content,
            tags=tags,
            sources=[source] if source else None,
            metadata_equals=metadata,
            limit=limit,
            offset=offset,
        )
        
        return self._backend.list(filter)
    
    def list(
        self,
        filter: Optional[QueryFilter] = None,
    ) -> List[Document]:
        """
        List documents with optional filtering.
        
        Args:
            filter: Query filter
            
        Returns:
            List of documents
        """
        return self._backend.list(filter)
    
    def count(
        self,
        filter: Optional[QueryFilter] = None,
    ) -> int:
        """
        Count documents.
        
        Args:
            filter: Optional filter
            
        Returns:
            Document count
        """
        return self._backend.count(filter)
    
    def clear(self) -> int:
        """
        Clear all documents.
        
        Returns:
            Number of documents cleared
        """
        return self._backend.clear()
    
    def stats(self) -> StoreStats:
        """
        Get store statistics.
        
        Returns:
            StoreStats
        """
        docs = self._backend.list()
        
        all_tags: Set[str] = set()
        all_sources: Set[str] = set()
        total_size = 0
        
        oldest = None
        newest = None
        
        for doc in docs:
            all_tags.update(doc.tags)
            if doc.source:
                all_sources.add(doc.source)
            total_size += len(doc.content)
            
            if oldest is None or doc.created_at < oldest:
                oldest = doc.created_at
            if newest is None or doc.created_at > newest:
                newest = doc.created_at
        
        return StoreStats(
            total_documents=len(docs),
            total_size_bytes=total_size,
            unique_tags=len(all_tags),
            unique_sources=len(all_sources),
            oldest_doc=oldest,
            newest_doc=newest,
            backend=self._backend.name,
        )
    
    def iter_documents(
        self,
        batch_size: int = 100,
    ) -> Iterator[Document]:
        """
        Iterate over all documents.
        
        Args:
            batch_size: Batch size for iteration
            
        Yields:
            Documents
        """
        offset = 0
        while True:
            filter = QueryFilter(limit=batch_size, offset=offset)
            batch = self._backend.list(filter)
            
            if not batch:
                break
            
            for doc in batch:
                yield doc
            
            offset += batch_size
    
    def close(self) -> None:
        """Close store and release resources."""
        if hasattr(self._backend, 'close'):
            self._backend.close()


# Convenience functions

def create_document(
    content: str,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Document:
    """
    Create a document.
    
    Args:
        content: Document content
        title: Optional title
        metadata: Optional metadata
        **kwargs: Additional document fields
        
    Returns:
        Document
    """
    return Document(
        id=kwargs.pop('id', ''),
        content=content,
        title=title,
        metadata=metadata or {},
        **kwargs,
    )


def load_documents(
    path: Union[str, Path],
    backend: str = "sqlite",
) -> DocumentStore:
    """
    Load documents from storage.
    
    Args:
        path: Storage path
        backend: Backend type
        
    Returns:
        DocumentStore
    """
    return DocumentStore(backend=backend, path=path)
