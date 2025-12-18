"""
Batch embedding module for AutoRAG-Live.

Provides efficient batch embedding with automatic batching,
caching, and multi-provider support.

Features:
- Automatic batching for efficiency
- Multiple embedding providers (OpenAI, Sentence Transformers, etc.)
- Embedding caching
- Async support
- Progress tracking
- Retry logic and error handling

Example usage:
    >>> embedder = BatchEmbedder(model="text-embedding-3-small")
    >>> embeddings = embedder.embed_batch(["text1", "text2", "text3"])
    >>> 
    >>> # With progress tracking
    >>> for batch in embedder.embed_batches(large_text_list):
    ...     print(f"Processed batch of {len(batch)} embeddings")
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


class EmbeddingProvider(Enum):
    """Supported embedding providers."""
    
    OPENAI = auto()
    SENTENCE_TRANSFORMERS = auto()
    COHERE = auto()
    HUGGINGFACE = auto()
    CUSTOM = auto()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    
    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    model: str = "text-embedding-3-small"
    
    # Batching
    batch_size: int = 100
    max_tokens_per_batch: int = 8000
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Caching
    cache_enabled: bool = True
    cache_ttl: int = 86400  # 24 hours
    
    # Performance
    parallel_batches: int = 4
    timeout: float = 60.0
    
    # Model-specific
    dimensions: Optional[int] = None
    normalize: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    
    text: str
    embedding: List[float]
    
    # Metadata
    model: str = ""
    dimensions: int = 0
    token_count: int = 0
    
    # Timing
    generation_time_ms: float = 0.0
    
    # Cache info
    from_cache: bool = False
    cache_key: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of batch embedding."""
    
    embeddings: List[EmbeddingResult]
    
    # Statistics
    total_texts: int = 0
    successful: int = 0
    failed: int = 0
    from_cache: int = 0
    
    # Timing
    total_time_ms: float = 0.0
    avg_time_per_text_ms: float = 0.0
    
    # Errors
    errors: List[Tuple[int, str]] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingCache:
    """Simple in-memory embedding cache."""
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl: int = 86400,
    ):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum cache entries
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[List[float], float]] = {}
    
    def _make_key(self, text: str, model: str) -> str:
        """Generate cache key."""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(
        self,
        text: str,
        model: str,
    ) -> Optional[List[float]]:
        """Get cached embedding."""
        key = self._make_key(text, model)
        
        if key in self._cache:
            embedding, timestamp = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp < self.ttl:
                return embedding
            else:
                del self._cache[key]
        
        return None
    
    def set(
        self,
        text: str,
        model: str,
        embedding: List[float],
    ) -> None:
        """Cache embedding."""
        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            # Remove oldest entries
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )
            for key in sorted_keys[:len(sorted_keys) // 4]:
                del self._cache[key]
        
        key = self._make_key(text, model)
        self._cache[key] = (embedding, time.time())
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
    
    @property
    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @abstractmethod
    def embed(
        self,
        texts: List[str],
        model: str,
        **kwargs,
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass
    
    async def aembed(
        self,
        texts: List[str],
        model: str,
        **kwargs,
    ) -> List[List[float]]:
        """Async embedding generation."""
        # Default: wrap sync
        return self.embed(texts, model, **kwargs)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
    
    @property
    def name(self) -> str:
        return "openai"
    
    def embed(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
        **kwargs,
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.embeddings.create(
                model=model,
                input=texts,
                **kwargs,
            )
            
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [d.embedding for d in sorted_data]
            
        except ImportError:
            raise ImportError("openai package required for OpenAI embeddings")
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Sentence Transformers embedding provider."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize Sentence Transformers provider.
        
        Args:
            device: Device to use (cpu, cuda)
        """
        self.device = device
        self._models: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        return "sentence_transformers"
    
    def _get_model(self, model_name: str) -> Any:
        """Get or load model."""
        if model_name not in self._models:
            try:
                from sentence_transformers import SentenceTransformer
                
                model = SentenceTransformer(model_name)
                if self.device:
                    model = model.to(self.device)
                
                self._models[model_name] = model
                
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required"
                )
        
        return self._models[model_name]
    
    def embed(
        self,
        texts: List[str],
        model: str = "all-MiniLM-L6-v2",
        **kwargs,
    ) -> List[List[float]]:
        """Generate embeddings using Sentence Transformers."""
        model_instance = self._get_model(model)
        
        embeddings = model_instance.encode(
            texts,
            convert_to_numpy=True,
            **kwargs,
        )
        
        return embeddings.tolist()


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock provider for testing."""
    
    def __init__(self, dimensions: int = 384):
        """
        Initialize mock provider.
        
        Args:
            dimensions: Embedding dimensions
        """
        self.dimensions = dimensions
    
    @property
    def name(self) -> str:
        return "mock"
    
    def embed(
        self,
        texts: List[str],
        model: str = "mock",
        **kwargs,
    ) -> List[List[float]]:
        """Generate mock embeddings."""
        import random
        
        embeddings = []
        for text in texts:
            # Deterministic based on text
            random.seed(hash(text) % (2**32))
            embedding = [random.random() for _ in range(self.dimensions)]
            
            # Normalize
            norm = sum(x**2 for x in embedding) ** 0.5
            embedding = [x / norm for x in embedding]
            
            embeddings.append(embedding)
        
        return embeddings


class BatchEmbedder:
    """
    Main batch embedding interface.
    
    Example:
        >>> # Basic usage
        >>> embedder = BatchEmbedder(model="text-embedding-3-small")
        >>> results = embedder.embed_batch(["Hello", "World"])
        >>> 
        >>> # With caching
        >>> embedder = BatchEmbedder(
        ...     model="all-MiniLM-L6-v2",
        ...     provider="sentence_transformers",
        ...     cache_enabled=True,
        ... )
        >>> 
        >>> # Process large dataset
        >>> texts = ["text1", "text2", ...]  # thousands of texts
        >>> for batch_result in embedder.embed_batches(texts):
        ...     print(f"Processed {batch_result.successful} texts")
    """
    
    PROVIDERS = {
        'openai': OpenAIEmbeddingProvider,
        'sentence_transformers': SentenceTransformerProvider,
        'mock': MockEmbeddingProvider,
    }
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        provider: str = "openai",
        batch_size: int = 100,
        cache_enabled: bool = True,
        max_retries: int = 3,
        **kwargs,
    ):
        """
        Initialize batch embedder.
        
        Args:
            model: Embedding model name
            provider: Provider name
            batch_size: Batch size
            cache_enabled: Enable caching
            max_retries: Maximum retries
            **kwargs: Provider-specific kwargs
        """
        self.model = model
        self.provider_name = provider
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self.max_retries = max_retries
        self.kwargs = kwargs
        
        # Initialize provider
        self._provider = self._create_provider()
        
        # Initialize cache
        self._cache = EmbeddingCache() if cache_enabled else None
    
    def _create_provider(self) -> BaseEmbeddingProvider:
        """Create embedding provider."""
        provider_name = self.provider_name.lower()
        
        if provider_name not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        provider_class = self.PROVIDERS[provider_name]
        
        # Filter kwargs for provider
        import inspect
        sig = inspect.signature(provider_class.__init__)
        valid_kwargs = {
            k: v for k, v in self.kwargs.items()
            if k in sig.parameters
        }
        
        return provider_class(**valid_kwargs)
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Simple approximation: ~4 chars per token
        return len(text) // 4
    
    def _create_batches(
        self,
        texts: List[str],
        max_tokens: int = 8000,
    ) -> Iterator[List[Tuple[int, str]]]:
        """Create batches respecting token limits."""
        current_batch = []
        current_tokens = 0
        
        for i, text in enumerate(texts):
            tokens = self._count_tokens(text)
            
            # Check if adding this text exceeds limits
            if (len(current_batch) >= self.batch_size or 
                current_tokens + tokens > max_tokens) and current_batch:
                yield current_batch
                current_batch = []
                current_tokens = 0
            
            current_batch.append((i, text))
            current_tokens += tokens
        
        if current_batch:
            yield current_batch
    
    def embed(self, text: str) -> EmbeddingResult:
        """
        Embed single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult
        """
        results = self.embed_batch([text])
        return results.embeddings[0] if results.embeddings else None
    
    def embed_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """
        Embed batch of texts.
        
        Args:
            texts: Texts to embed
            progress_callback: Optional progress callback
            
        Returns:
            BatchResult
        """
        start_time = time.time()
        
        result = BatchResult(
            embeddings=[],
            total_texts=len(texts),
        )
        
        # Check cache first
        cache_hits = {}
        texts_to_embed = []
        
        for i, text in enumerate(texts):
            if self._cache and self.cache_enabled:
                cached = self._cache.get(text, self.model)
                if cached is not None:
                    cache_hits[i] = cached
                    result.from_cache += 1
                    continue
            
            texts_to_embed.append((i, text))
        
        # Embed remaining texts in batches
        embeddings_map: Dict[int, List[float]] = {}
        
        for batch in self._create_batches(
            [t[1] for t in texts_to_embed]
        ):
            batch_indices = [texts_to_embed[i][0] for i, _ in enumerate(batch)]
            batch_texts = [t[1] for t in batch]
            
            # Retry logic
            for attempt in range(self.max_retries):
                try:
                    batch_embeddings = self._provider.embed(
                        batch_texts,
                        self.model,
                    )
                    
                    # Map results back to indices
                    for idx, emb in zip(batch_indices, batch_embeddings):
                        embeddings_map[idx] = emb
                        
                        # Cache
                        if self._cache and self.cache_enabled:
                            original_idx = next(
                                i for i, (j, _) in enumerate(texts_to_embed)
                                if j == idx
                            )
                            self._cache.set(
                                texts_to_embed[original_idx][1],
                                self.model,
                                emb,
                            )
                    
                    break
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        delay = (2 ** attempt) * 1.0  # Exponential backoff
                        time.sleep(delay)
                    else:
                        for idx in batch_indices:
                            result.errors.append((idx, str(e)))
                        result.failed += len(batch_indices)
            
            if progress_callback:
                progress_callback(
                    result.successful + result.failed + result.from_cache,
                    result.total_texts,
                )
        
        # Build final results in order
        for i in range(len(texts)):
            if i in cache_hits:
                embedding = cache_hits[i]
                from_cache = True
            elif i in embeddings_map:
                embedding = embeddings_map[i]
                from_cache = False
                result.successful += 1
            else:
                continue
            
            result.embeddings.append(EmbeddingResult(
                text=texts[i],
                embedding=embedding,
                model=self.model,
                dimensions=len(embedding),
                from_cache=from_cache,
            ))
        
        # Calculate timing
        result.total_time_ms = (time.time() - start_time) * 1000
        if result.total_texts > 0:
            result.avg_time_per_text_ms = result.total_time_ms / result.total_texts
        
        return result
    
    def embed_batches(
        self,
        texts: List[str],
        yield_every: int = 100,
    ) -> Iterator[BatchResult]:
        """
        Embed texts yielding batch results.
        
        Args:
            texts: Texts to embed
            yield_every: Yield every N embeddings
            
        Yields:
            BatchResult for each batch
        """
        for i in range(0, len(texts), yield_every):
            batch = texts[i:i + yield_every]
            yield self.embed_batch(batch)
    
    async def aembed_batch(
        self,
        texts: List[str],
    ) -> BatchResult:
        """
        Async embed batch.
        
        Args:
            texts: Texts to embed
            
        Returns:
            BatchResult
        """
        start_time = time.time()
        
        result = BatchResult(
            embeddings=[],
            total_texts=len(texts),
        )
        
        # Check cache
        cache_hits = {}
        texts_to_embed = []
        
        for i, text in enumerate(texts):
            if self._cache and self.cache_enabled:
                cached = self._cache.get(text, self.model)
                if cached is not None:
                    cache_hits[i] = cached
                    result.from_cache += 1
                    continue
            
            texts_to_embed.append((i, text))
        
        # Embed remaining
        if texts_to_embed:
            batch_texts = [t[1] for t in texts_to_embed]
            batch_indices = [t[0] for t in texts_to_embed]
            
            try:
                batch_embeddings = await self._provider.aembed(
                    batch_texts,
                    self.model,
                )
                
                for idx, emb in zip(batch_indices, batch_embeddings):
                    # Build result
                    result.embeddings.append(EmbeddingResult(
                        text=texts[idx],
                        embedding=emb,
                        model=self.model,
                        dimensions=len(emb),
                        from_cache=False,
                    ))
                    
                    # Cache
                    if self._cache and self.cache_enabled:
                        self._cache.set(texts[idx], self.model, emb)
                
                result.successful = len(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Async embedding error: {e}")
                result.failed = len(texts_to_embed)
                for idx in batch_indices:
                    result.errors.append((idx, str(e)))
        
        # Add cache hits
        for i, emb in cache_hits.items():
            result.embeddings.append(EmbeddingResult(
                text=texts[i],
                embedding=emb,
                model=self.model,
                dimensions=len(emb),
                from_cache=True,
            ))
        
        # Sort by original order
        result.embeddings.sort(
            key=lambda r: texts.index(r.text)
        )
        
        result.total_time_ms = (time.time() - start_time) * 1000
        if result.total_texts > 0:
            result.avg_time_per_text_ms = result.total_time_ms / result.total_texts
        
        return result
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self._cache:
            self._cache.clear()
    
    @property
    def cache_size(self) -> int:
        """Get cache size."""
        return self._cache.size if self._cache else 0


class EmbeddingPipeline:
    """Pipeline for embedding with preprocessing."""
    
    def __init__(
        self,
        embedder: BatchEmbedder,
        preprocessors: Optional[List[Callable[[str], str]]] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            embedder: BatchEmbedder instance
            preprocessors: Text preprocessing functions
        """
        self.embedder = embedder
        self.preprocessors = preprocessors or []
    
    def add_preprocessor(
        self,
        fn: Callable[[str], str],
    ) -> EmbeddingPipeline:
        """Add preprocessor."""
        self.preprocessors.append(fn)
        return self
    
    def _preprocess(self, text: str) -> str:
        """Apply preprocessors."""
        for fn in self.preprocessors:
            text = fn(text)
        return text
    
    def embed(
        self,
        texts: List[str],
    ) -> BatchResult:
        """Embed with preprocessing."""
        processed = [self._preprocess(t) for t in texts]
        return self.embedder.embed_batch(processed)


# Convenience functions

def embed_texts(
    texts: List[str],
    model: str = "text-embedding-3-small",
    provider: str = "openai",
) -> List[List[float]]:
    """
    Quick batch embedding.
    
    Args:
        texts: Texts to embed
        model: Model name
        provider: Provider name
        
    Returns:
        List of embeddings
    """
    embedder = BatchEmbedder(model=model, provider=provider)
    result = embedder.embed_batch(texts)
    return [r.embedding for r in result.embeddings]


def embed_text(
    text: str,
    model: str = "text-embedding-3-small",
    provider: str = "openai",
) -> List[float]:
    """
    Quick single text embedding.
    
    Args:
        text: Text to embed
        model: Model name
        provider: Provider name
        
    Returns:
        Embedding vector
    """
    embeddings = embed_texts([text], model, provider)
    return embeddings[0] if embeddings else []


def cosine_similarity(
    vec1: List[float],
    vec2: List[float],
) -> float:
    """
    Calculate cosine similarity.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a ** 2 for a in vec1) ** 0.5
    norm2 = sum(b ** 2 for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (norm1 * norm2)
