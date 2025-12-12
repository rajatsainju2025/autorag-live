"""
Document indexing pipeline for AutoRAG-Live.

Provides a complete pipeline for indexing documents:
chunking, embedding, and storage in vector stores.

Features:
- Document preprocessing
- Multiple chunking strategies
- Batch embedding generation
- Vector store integration
- Incremental updates
- Index management

Example usage:
    >>> indexer = DocumentIndexer(
    ...     embedding_model="text-embedding-3-small",
    ...     vector_store="faiss"
    ... )
    >>> await indexer.index_documents([doc1, doc2, doc3])
    >>> await indexer.search("query", top_k=5)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Document chunking strategies."""
    
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"
    RECURSIVE = "recursive"


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    
    MEMORY = "memory"
    FAISS = "faiss"
    NUMPY = "numpy"


@dataclass
class IndexConfig:
    """Configuration for document indexing."""
    
    # Chunking
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1500
    
    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100
    
    # Vector store
    vector_store_type: VectorStoreType = VectorStoreType.MEMORY
    index_path: Optional[str] = None
    
    # Processing
    max_concurrent: int = 4
    show_progress: bool = True


@dataclass
class Document:
    """Represents a document to be indexed."""
    
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional source info
    source: Optional[str] = None
    title: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if not self.id:
            self.id = hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class Chunk:
    """Represents a chunk of a document."""
    
    id: str
    content: str
    document_id: str
    
    # Position info
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Embedding
    embedding: Optional[List[float]] = None
    
    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if not self.id:
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.id = f"{self.document_id}_{self.chunk_index}_{content_hash}"


@dataclass
class SearchResult:
    """Result from vector search."""
    
    chunk: Chunk
    score: float
    rank: int = 0


@dataclass
class IndexStats:
    """Statistics about the index."""
    
    total_documents: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    embedding_dimensions: int = 0
    index_size_bytes: int = 0
    last_updated: Optional[float] = None


class Chunker(ABC):
    """Abstract base class for document chunkers."""
    
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split document into chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of Chunk objects
        """
        pass


class FixedSizeChunker(Chunker):
    """Chunk by fixed character count."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
    ):
        """
        Initialize fixed size chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into fixed-size chunks."""
        content = document.content
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            
            # Try to break at word boundary
            if end < len(content):
                # Look for space within last 50 chars
                space_pos = content.rfind(" ", end - 50, end)
                if space_pos > start:
                    end = space_pos
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunks.append(Chunk(
                    id="",
                    content=chunk_content,
                    document_id=document.id,
                    start_char=start,
                    end_char=end,
                    chunk_index=chunk_index,
                    metadata=document.metadata.copy(),
                ))
                chunk_index += 1
            
            start = end - self.overlap
            if start <= chunks[-1].start_char if chunks else False:
                start = end
        
        return chunks


class SentenceChunker(Chunker):
    """Chunk by sentences."""
    
    def __init__(
        self,
        max_sentences: int = 5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
    ):
        """
        Initialize sentence chunker.
        
        Args:
            max_sentences: Maximum sentences per chunk
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
        """
        self.max_sentences = max_sentences
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with positions."""
        import re
        
        sentences = []
        pattern = r'(?<=[.!?])\s+'
        
        last_end = 0
        for match in re.finditer(pattern, text):
            sent = text[last_end:match.start() + 1].strip()
            if sent:
                sentences.append((sent, last_end, match.start() + 1))
            last_end = match.end()
        
        # Handle last sentence
        remaining = text[last_end:].strip()
        if remaining:
            sentences.append((remaining, last_end, len(text)))
        
        return sentences
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into sentence-based chunks."""
        sentences = self._split_sentences(document.content)
        chunks = []
        current_sentences = []
        current_start = 0
        chunk_index = 0
        
        for sent_text, start, end in sentences:
            current_sentences.append(sent_text)
            current_text = " ".join(current_sentences)
            
            # Check if we should create a chunk
            should_chunk = (
                len(current_sentences) >= self.max_sentences
                or len(current_text) >= self.max_chunk_size
            )
            
            if should_chunk and len(current_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    id="",
                    content=current_text,
                    document_id=document.id,
                    start_char=current_start,
                    end_char=end,
                    chunk_index=chunk_index,
                    metadata=document.metadata.copy(),
                ))
                chunk_index += 1
                current_sentences = []
                current_start = end
        
        # Handle remaining sentences
        if current_sentences:
            chunks.append(Chunk(
                id="",
                content=" ".join(current_sentences),
                document_id=document.id,
                start_char=current_start,
                end_char=len(document.content),
                chunk_index=chunk_index,
                metadata=document.metadata.copy(),
            ))
        
        return chunks


class ParagraphChunker(Chunker):
    """Chunk by paragraphs."""
    
    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1500,
    ):
        """
        Initialize paragraph chunker.
        
        Args:
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document by paragraphs."""
        paragraphs = document.content.split("\n\n")
        chunks = []
        current_paras = []
        current_start = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            current_paras.append(para)
            current_text = "\n\n".join(current_paras)
            
            if len(current_text) >= self.max_chunk_size:
                # Save current chunk
                if len(current_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        id="",
                        content=current_text,
                        document_id=document.id,
                        start_char=current_start,
                        end_char=current_start + len(current_text),
                        chunk_index=chunk_index,
                        metadata=document.metadata.copy(),
                    ))
                    chunk_index += 1
                
                current_start = current_start + len(current_text) + 2
                current_paras = []
        
        # Handle remaining
        if current_paras:
            current_text = "\n\n".join(current_paras)
            chunks.append(Chunk(
                id="",
                content=current_text,
                document_id=document.id,
                start_char=current_start,
                end_char=len(document.content),
                chunk_index=chunk_index,
                metadata=document.metadata.copy(),
            ))
        
        return chunks


class SlidingWindowChunker(Chunker):
    """Chunk with sliding window."""
    
    def __init__(
        self,
        window_size: int = 512,
        step_size: int = 256,
    ):
        """
        Initialize sliding window chunker.
        
        Args:
            window_size: Window size in characters
            step_size: Step size between windows
        """
        self.window_size = window_size
        self.step_size = step_size
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Create overlapping chunks."""
        content = document.content
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + self.window_size, len(content))
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunks.append(Chunk(
                    id="",
                    content=chunk_content,
                    document_id=document.id,
                    start_char=start,
                    end_char=end,
                    chunk_index=chunk_index,
                    metadata=document.metadata.copy(),
                ))
                chunk_index += 1
            
            start += self.step_size
        
        return chunks


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks to the store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks by ID."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all chunks."""
        pass
    
    @property
    @abstractmethod
    def count(self) -> int:
        """Get number of chunks."""
        pass


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store."""
    
    def __init__(self):
        """Initialize in-memory store."""
        self._chunks: Dict[str, Chunk] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
    
    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks to store."""
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            
            self._chunks[chunk.id] = chunk
            self._embeddings[chunk.id] = np.array(chunk.embedding, dtype=np.float32)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search by cosine similarity."""
        if not self._embeddings:
            return []
        
        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        
        if query_norm == 0:
            return []
        
        query = query / query_norm
        
        # Compute similarities
        similarities = []
        for chunk_id, embedding in self._embeddings.items():
            emb_norm = np.linalg.norm(embedding)
            if emb_norm > 0:
                similarity = np.dot(query, embedding / emb_norm)
                similarities.append((chunk_id, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = []
        for rank, (chunk_id, score) in enumerate(similarities[:top_k]):
            results.append(SearchResult(
                chunk=self._chunks[chunk_id],
                score=score,
                rank=rank,
            ))
        
        return results
    
    def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks."""
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)
            self._embeddings.pop(chunk_id, None)
    
    def clear(self) -> None:
        """Clear store."""
        self._chunks.clear()
        self._embeddings.clear()
    
    @property
    def count(self) -> int:
        """Get chunk count."""
        return len(self._chunks)
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID."""
        return self._chunks.get(chunk_id)


class NumpyVectorStore(VectorStore):
    """NumPy-based vector store with efficient operations."""
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize NumPy store.
        
        Args:
            index_path: Path to save/load index
        """
        self.index_path = index_path
        self._chunks: List[Chunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self._id_to_idx: Dict[str, int] = {}
    
    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks to store."""
        embeddings_to_add = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            
            idx = len(self._chunks)
            self._chunks.append(chunk)
            self._id_to_idx[chunk.id] = idx
            embeddings_to_add.append(chunk.embedding)
        
        if embeddings_to_add:
            new_embeddings = np.array(embeddings_to_add, dtype=np.float32)
            
            if self._embeddings is None:
                self._embeddings = new_embeddings
            else:
                self._embeddings = np.vstack([self._embeddings, new_embeddings])
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search using matrix multiplication."""
        if self._embeddings is None or len(self._embeddings) == 0:
            return []
        
        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        
        if query_norm == 0:
            return []
        
        query = query / query_norm
        
        # Normalize embeddings
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = self._embeddings / norms
        
        # Compute similarities
        similarities = np.dot(normalized, query)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            results.append(SearchResult(
                chunk=self._chunks[idx],
                score=float(similarities[idx]),
                rank=rank,
            ))
        
        return results
    
    def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks (marks for deletion)."""
        indices_to_delete = set()
        for chunk_id in chunk_ids:
            if chunk_id in self._id_to_idx:
                indices_to_delete.add(self._id_to_idx[chunk_id])
        
        if not indices_to_delete:
            return
        
        # Rebuild without deleted indices
        new_chunks = []
        new_embeddings = []
        new_id_to_idx = {}
        
        for i, chunk in enumerate(self._chunks):
            if i not in indices_to_delete:
                new_id_to_idx[chunk.id] = len(new_chunks)
                new_chunks.append(chunk)
                if self._embeddings is not None:
                    new_embeddings.append(self._embeddings[i])
        
        self._chunks = new_chunks
        self._id_to_idx = new_id_to_idx
        self._embeddings = np.array(new_embeddings) if new_embeddings else None
    
    def clear(self) -> None:
        """Clear store."""
        self._chunks.clear()
        self._embeddings = None
        self._id_to_idx.clear()
    
    @property
    def count(self) -> int:
        """Get chunk count."""
        return len(self._chunks)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save index to disk."""
        path = path or self.index_path
        if not path:
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save embeddings
        if self._embeddings is not None:
            np.save(f"{path}.npy", self._embeddings)
        
        # Save chunks metadata
        chunks_data = [
            {
                "id": c.id,
                "content": c.content,
                "document_id": c.document_id,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        
        with open(f"{path}.json", "w") as f:
            json.dump(chunks_data, f)
    
    def load(self, path: Optional[str] = None) -> None:
        """Load index from disk."""
        path = path or self.index_path
        if not path:
            return
        
        # Load embeddings
        npy_path = f"{path}.npy"
        if os.path.exists(npy_path):
            self._embeddings = np.load(npy_path)
        
        # Load chunks
        json_path = f"{path}.json"
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                chunks_data = json.load(f)
            
            self._chunks = []
            self._id_to_idx = {}
            
            for i, data in enumerate(chunks_data):
                chunk = Chunk(
                    id=data["id"],
                    content=data["content"],
                    document_id=data["document_id"],
                    start_char=data.get("start_char", 0),
                    end_char=data.get("end_char", 0),
                    chunk_index=data.get("chunk_index", i),
                    metadata=data.get("metadata", {}),
                )
                self._chunks.append(chunk)
                self._id_to_idx[chunk.id] = i


class EmbeddingGenerator:
    """Generate embeddings for chunks."""
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
    ):
        """
        Initialize embedding generator.
        
        Args:
            model: Embedding model name
            batch_size: Batch size for embedding
        """
        self.model = model
        self.batch_size = batch_size
        self._dimensions: Optional[int] = None
    
    async def generate(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: Chunks to embed
            
        Returns:
            Chunks with embeddings
        """
        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            texts = [c.content for c in batch]
            
            embeddings = await self._embed_batch(texts)
            
            for j, embedding in enumerate(embeddings):
                batch[j].embedding = embedding
                
                if self._dimensions is None:
                    self._dimensions = len(embedding)
        
        return chunks
    
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        # Try to import and use actual embedding service
        try:
            from autorag_live.embeddings.embedding_service import get_embedding_service
            
            service = get_embedding_service(model=self.model)
            result = service.embed_batch(texts)
            return [r.embedding for r in result.results]
        except ImportError:
            # Fallback to mock embeddings for testing
            return [self._mock_embedding(t) for t in texts]
    
    def _mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding for testing."""
        # Create deterministic embedding from text
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions or 384


class DocumentIndexer:
    """
    Main document indexing pipeline.
    
    Example:
        >>> indexer = DocumentIndexer()
        >>> 
        >>> # Index documents
        >>> docs = [Document(id="1", content="...")]
        >>> await indexer.index_documents(docs)
        >>> 
        >>> # Search
        >>> results = await indexer.search("query", top_k=5)
    """
    
    def __init__(
        self,
        config: Optional[IndexConfig] = None,
        chunker: Optional[Chunker] = None,
        vector_store: Optional[VectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ):
        """
        Initialize document indexer.
        
        Args:
            config: Index configuration
            chunker: Document chunker
            vector_store: Vector store
            embedding_generator: Embedding generator
        """
        self.config = config or IndexConfig()
        
        # Initialize chunker
        if chunker:
            self._chunker = chunker
        else:
            self._chunker = self._create_chunker()
        
        # Initialize vector store
        if vector_store:
            self._store = vector_store
        else:
            self._store = self._create_store()
        
        # Initialize embedding generator
        if embedding_generator:
            self._embedder = embedding_generator
        else:
            self._embedder = EmbeddingGenerator(
                model=self.config.embedding_model,
                batch_size=self.config.embedding_batch_size,
            )
        
        # Track indexed documents
        self._document_ids: Set[str] = set()
        self._stats = IndexStats()
    
    def _create_chunker(self) -> Chunker:
        """Create chunker based on config."""
        strategy = self.config.chunking_strategy
        
        if strategy == ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunker(
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
        elif strategy == ChunkingStrategy.SENTENCE:
            return SentenceChunker(
                min_chunk_size=self.config.min_chunk_size,
                max_chunk_size=self.config.max_chunk_size,
            )
        elif strategy == ChunkingStrategy.PARAGRAPH:
            return ParagraphChunker(
                min_chunk_size=self.config.min_chunk_size,
                max_chunk_size=self.config.max_chunk_size,
            )
        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            return SlidingWindowChunker(
                window_size=self.config.chunk_size,
                step_size=self.config.chunk_size - self.config.chunk_overlap,
            )
        else:
            return FixedSizeChunker()
    
    def _create_store(self) -> VectorStore:
        """Create vector store based on config."""
        store_type = self.config.vector_store_type
        
        if store_type == VectorStoreType.NUMPY:
            return NumpyVectorStore(index_path=self.config.index_path)
        else:
            return InMemoryVectorStore()
    
    async def index_documents(
        self,
        documents: List[Document],
        update_existing: bool = True,
    ) -> int:
        """
        Index documents.
        
        Args:
            documents: Documents to index
            update_existing: Update if document exists
            
        Returns:
            Number of chunks indexed
        """
        all_chunks: List[Chunk] = []
        
        for doc in documents:
            # Skip if already indexed and not updating
            if doc.id in self._document_ids and not update_existing:
                continue
            
            # Remove existing chunks if updating
            if doc.id in self._document_ids:
                await self.delete_document(doc.id)
            
            # Chunk document
            chunks = self._chunker.chunk(doc)
            all_chunks.extend(chunks)
            
            self._document_ids.add(doc.id)
        
        if not all_chunks:
            return 0
        
        # Generate embeddings
        all_chunks = await self._embedder.generate(all_chunks)
        
        # Add to store
        self._store.add(all_chunks)
        
        # Update stats
        self._stats.total_documents = len(self._document_ids)
        self._stats.total_chunks = self._store.count
        self._stats.embedding_dimensions = self._embedder.dimensions
        self._stats.last_updated = time.time()
        
        return len(all_chunks)
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search the index.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of SearchResult
        """
        # Generate query embedding
        query_chunks = [Chunk(id="query", content=query, document_id="")]
        query_chunks = await self._embedder.generate(query_chunks)
        query_embedding = query_chunks[0].embedding
        
        # Search
        results = self._store.search(query_embedding, top_k=top_k * 2)
        
        # Apply filters
        if filters:
            results = [
                r for r in results
                if self._matches_filters(r.chunk, filters)
            ]
        
        return results[:top_k]
    
    def _matches_filters(
        self, chunk: Chunk, filters: Dict[str, Any]
    ) -> bool:
        """Check if chunk matches filters."""
        for key, value in filters.items():
            chunk_value = chunk.metadata.get(key)
            
            if isinstance(value, list):
                if chunk_value not in value:
                    return False
            elif chunk_value != value:
                return False
        
        return True
    
    async def delete_document(self, document_id: str) -> int:
        """
        Delete a document and its chunks.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            Number of chunks deleted
        """
        if document_id not in self._document_ids:
            return 0
        
        # Find chunk IDs for this document
        chunk_ids = []
        
        if isinstance(self._store, (InMemoryVectorStore, NumpyVectorStore)):
            for chunk_id, chunk in (
                self._store._chunks.items() 
                if isinstance(self._store, InMemoryVectorStore)
                else [(c.id, c) for c in self._store._chunks]
            ):
                if (isinstance(chunk, Chunk) and chunk.document_id == document_id):
                    chunk_ids.append(chunk.id if isinstance(chunk, Chunk) else chunk_id)
        
        # Delete chunks
        self._store.delete(chunk_ids)
        self._document_ids.discard(document_id)
        
        # Update stats
        self._stats.total_documents = len(self._document_ids)
        self._stats.total_chunks = self._store.count
        
        return len(chunk_ids)
    
    def clear(self) -> None:
        """Clear the entire index."""
        self._store.clear()
        self._document_ids.clear()
        self._stats = IndexStats()
    
    @property
    def stats(self) -> IndexStats:
        """Get index statistics."""
        return self._stats
    
    def save(self, path: Optional[str] = None) -> None:
        """Save index to disk."""
        if isinstance(self._store, NumpyVectorStore):
            self._store.save(path)
    
    def load(self, path: Optional[str] = None) -> None:
        """Load index from disk."""
        if isinstance(self._store, NumpyVectorStore):
            self._store.load(path)
            
            # Rebuild document ID set
            self._document_ids = set()
            for chunk in self._store._chunks:
                self._document_ids.add(chunk.document_id)


# Global indexer instance
_default_indexer: Optional[DocumentIndexer] = None


def get_indexer(
    config: Optional[IndexConfig] = None,
) -> DocumentIndexer:
    """Get or create the default indexer."""
    global _default_indexer
    if _default_indexer is None or config is not None:
        _default_indexer = DocumentIndexer(config)
    return _default_indexer


async def index_documents(
    documents: List[Document],
    **kwargs: Any,
) -> int:
    """
    Convenience function to index documents.
    
    Args:
        documents: Documents to index
        **kwargs: Additional config
        
    Returns:
        Number of chunks indexed
    """
    indexer = get_indexer()
    return await indexer.index_documents(documents, **kwargs)


async def search_index(
    query: str,
    top_k: int = 10,
) -> List[SearchResult]:
    """
    Convenience function to search the index.
    
    Args:
        query: Search query
        top_k: Number of results
        
    Returns:
        Search results
    """
    indexer = get_indexer()
    return await indexer.search(query, top_k=top_k)
