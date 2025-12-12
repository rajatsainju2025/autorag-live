"""
Unified embedding service for AutoRAG-Live.

Provides a consistent interface for text embedding across multiple providers
with built-in batching, caching, rate limiting, and error handling.

Supported providers:
- OpenAI (text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)
- Cohere (embed-english-v3.0, embed-multilingual-v3.0)
- HuggingFace Inference API
- Sentence Transformers (local)

Example usage:
    >>> service = EmbeddingService(provider="openai", model="text-embedding-3-small")
    >>> result = service.embed("Hello, world!")
    >>> print(result.embedding[:5])  # First 5 dimensions
    
    >>> # Batch embedding
    >>> results = service.embed_batch(["Text 1", "Text 2", "Text 3"])
    
    >>> # With caching enabled
    >>> service = EmbeddingService(provider="openai", enable_cache=True)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProviderType(str, Enum):
    """Supported embedding provider types."""
    
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    CUSTOM = "custom"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service."""
    
    # Provider settings
    provider: EmbeddingProviderType = EmbeddingProviderType.OPENAI
    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Batch settings
    batch_size: int = 100
    max_tokens_per_batch: int = 8191
    
    # Rate limiting
    requests_per_minute: int = 3000
    tokens_per_minute: int = 1000000
    
    # Caching
    enable_cache: bool = True
    cache_size: int = 10000
    cache_ttl: int = 3600  # seconds
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    # Normalization
    normalize: bool = True
    
    # Timeout
    timeout: float = 30.0
    
    def __post_init__(self) -> None:
        """Validate and process configuration."""
        if isinstance(self.provider, str):
            self.provider = EmbeddingProviderType(self.provider)


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    
    text: str
    embedding: List[float]
    model: str
    provider: str
    dimensions: int
    tokens_used: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    
    def to_numpy(self) -> np.ndarray:
        """Convert embedding to numpy array."""
        return np.array(self.embedding, dtype=np.float32)
    
    def similarity(self, other: "EmbeddingResult") -> float:
        """Compute cosine similarity with another embedding."""
        a = self.to_numpy()
        b = other.to_numpy()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding operation."""
    
    results: List[EmbeddingResult]
    total_tokens: int
    total_latency_ms: float
    cache_hits: int
    cache_misses: int
    
    def to_numpy(self) -> np.ndarray:
        """Convert all embeddings to numpy array."""
        return np.array([r.embedding for r in self.results], dtype=np.float32)


class EmbeddingCache:
    """LRU cache for embeddings with TTL support."""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of entries
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Tuple[List[float], float]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, text: str, model: str, provider: str) -> str:
        """Create cache key from text and model."""
        content = f"{provider}:{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(
        self, text: str, model: str, provider: str
    ) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._make_key(text, model, provider)
        
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            embedding, timestamp = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return embedding
    
    def put(
        self, text: str, model: str, provider: str, embedding: List[float]
    ) -> None:
        """Store embedding in cache."""
        key = self._make_key(text, model, provider)
        
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
                
            self._cache[key] = (embedding, time.time())
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(
        self,
        requests_per_minute: int = 3000,
        tokens_per_minute: int = 1000000,
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        
        self._request_tokens = float(requests_per_minute)
        self._token_tokens = float(tokens_per_minute)
        self._last_update = time.time()
        self._lock = Lock()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        
        # Refill at rate of X per minute = X/60 per second
        self._request_tokens = min(
            self.requests_per_minute,
            self._request_tokens + elapsed * self.requests_per_minute / 60,
        )
        self._token_tokens = min(
            self.tokens_per_minute,
            self._token_tokens + elapsed * self.tokens_per_minute / 60,
        )
        self._last_update = now
    
    def acquire(self, tokens: int = 1) -> float:
        """
        Acquire rate limit tokens.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        with self._lock:
            self._refill()
            
            # Check if we have enough tokens
            if self._request_tokens >= 1 and self._token_tokens >= tokens:
                self._request_tokens -= 1
                self._token_tokens -= tokens
                return 0.0
            
            # Calculate wait time
            request_wait = 0.0
            if self._request_tokens < 1:
                request_wait = (1 - self._request_tokens) * 60 / self.requests_per_minute
            
            token_wait = 0.0
            if self._token_tokens < tokens:
                token_wait = (tokens - self._token_tokens) * 60 / self.tokens_per_minute
            
            return max(request_wait, token_wait)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model for this provider."""
        pass
    
    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """List of supported models."""
        pass
    
    @abstractmethod
    def embed(
        self,
        texts: List[str],
        model: str,
        **kwargs: Any,
    ) -> List[Tuple[List[float], int]]:
        """
        Embed texts.
        
        Args:
            texts: List of texts to embed
            model: Model to use
            **kwargs: Additional arguments
            
        Returns:
            List of (embedding, tokens_used) tuples
        """
        pass
    
    @abstractmethod
    def get_dimensions(self, model: str) -> int:
        """Get embedding dimensions for model."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4 + 1


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            api_base: Custom API base URL
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or "https://api.openai.com/v1"
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return "openai"
    
    @property
    def default_model(self) -> str:
        return "text-embedding-3-small"
    
    @property
    def supported_models(self) -> List[str]:
        return list(self.MODEL_DIMENSIONS.keys())
    
    def get_dimensions(self, model: str) -> int:
        return self.MODEL_DIMENSIONS.get(model, 1536)
    
    def embed(
        self,
        texts: List[str],
        model: str,
        **kwargs: Any,
    ) -> List[Tuple[List[float], int]]:
        """Embed texts using OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
        )
        
        response = client.embeddings.create(
            input=texts,
            model=model,
        )
        
        results = []
        for item in response.data:
            embedding = item.embedding
            tokens = response.usage.total_tokens // len(texts)
            results.append((embedding, tokens))
        
        return results


class CohereEmbeddingProvider(EmbeddingProvider):
    """Cohere embedding provider."""
    
    MODEL_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
        "embed-english-v2.0": 4096,
        "embed-multilingual-v2.0": 768,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize Cohere provider.
        
        Args:
            api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return "cohere"
    
    @property
    def default_model(self) -> str:
        return "embed-english-v3.0"
    
    @property
    def supported_models(self) -> List[str]:
        return list(self.MODEL_DIMENSIONS.keys())
    
    def get_dimensions(self, model: str) -> int:
        return self.MODEL_DIMENSIONS.get(model, 1024)
    
    def embed(
        self,
        texts: List[str],
        model: str,
        input_type: str = "search_document",
        **kwargs: Any,
    ) -> List[Tuple[List[float], int]]:
        """Embed texts using Cohere API."""
        try:
            import cohere
        except ImportError:
            raise ImportError("cohere package required: pip install cohere")
        
        client = cohere.Client(api_key=self.api_key)
        
        response = client.embed(
            texts=texts,
            model=model,
            input_type=input_type,
        )
        
        results = []
        tokens_per_text = sum(len(t.split()) for t in texts) // len(texts)
        for embedding in response.embeddings:
            results.append((embedding, tokens_per_text))
        
        return results


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """HuggingFace Inference API embedding provider."""
    
    MODEL_DIMENSIONS = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "thenlper/gte-base": 768,
        "thenlper/gte-large": 1024,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize HuggingFace provider.
        
        Args:
            api_key: HuggingFace API key (defaults to HF_TOKEN env var)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return "huggingface"
    
    @property
    def default_model(self) -> str:
        return "sentence-transformers/all-MiniLM-L6-v2"
    
    @property
    def supported_models(self) -> List[str]:
        return list(self.MODEL_DIMENSIONS.keys())
    
    def get_dimensions(self, model: str) -> int:
        return self.MODEL_DIMENSIONS.get(model, 768)
    
    def embed(
        self,
        texts: List[str],
        model: str,
        **kwargs: Any,
    ) -> List[Tuple[List[float], int]]:
        """Embed texts using HuggingFace Inference API."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required: pip install requests")
        
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=self.timeout,
        )
        response.raise_for_status()
        
        embeddings = response.json()
        
        results = []
        for i, embedding in enumerate(embeddings):
            # Handle nested structure
            if isinstance(embedding[0], list):
                # Mean pooling for token embeddings
                embedding = np.mean(embedding, axis=0).tolist()
            tokens = len(texts[i].split())
            results.append((embedding, tokens))
        
        return results


class SentenceTransformerProvider(EmbeddingProvider):
    """Local Sentence Transformer embedding provider."""
    
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "multi-qa-MiniLM-L6-cos-v1": 384,
        "msmarco-MiniLM-L6-cos-v5": 384,
    }
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize Sentence Transformer provider.
        
        Args:
            device: Device to use (cpu, cuda, mps)
        """
        self.device = device
        self._models: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        return "sentence_transformer"
    
    @property
    def default_model(self) -> str:
        return "all-MiniLM-L6-v2"
    
    @property
    def supported_models(self) -> List[str]:
        return list(self.MODEL_DIMENSIONS.keys())
    
    def get_dimensions(self, model: str) -> int:
        return self.MODEL_DIMENSIONS.get(model, 384)
    
    def _get_model(self, model: str) -> Any:
        """Get or load model."""
        if model not in self._models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers"
                )
            
            self._models[model] = SentenceTransformer(model, device=self.device)
        
        return self._models[model]
    
    def embed(
        self,
        texts: List[str],
        model: str,
        **kwargs: Any,
    ) -> List[Tuple[List[float], int]]:
        """Embed texts using Sentence Transformers."""
        st_model = self._get_model(model)
        
        embeddings = st_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=kwargs.get("normalize", True),
        )
        
        results = []
        for i, embedding in enumerate(embeddings):
            tokens = len(texts[i].split())
            results.append((embedding.tolist(), tokens))
        
        return results


class EmbeddingService:
    """
    Unified embedding service with caching, batching, and rate limiting.
    
    Example usage:
        >>> service = EmbeddingService(provider="openai")
        >>> result = service.embed("Hello world")
        >>> print(result.dimensions)
        1536
        
        >>> # Batch embedding
        >>> results = service.embed_batch(["Text 1", "Text 2"])
        
        >>> # Custom provider
        >>> service = EmbeddingService(
        ...     provider="sentence_transformer",
        ...     model="all-MiniLM-L6-v2"
        ... )
    """
    
    PROVIDERS = {
        EmbeddingProviderType.OPENAI: OpenAIEmbeddingProvider,
        EmbeddingProviderType.COHERE: CohereEmbeddingProvider,
        EmbeddingProviderType.HUGGINGFACE: HuggingFaceEmbeddingProvider,
        EmbeddingProviderType.SENTENCE_TRANSFORMER: SentenceTransformerProvider,
    }
    
    def __init__(
        self,
        provider: Union[str, EmbeddingProviderType, EmbeddingProvider] = "openai",
        model: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize embedding service.
        
        Args:
            provider: Provider type or instance
            model: Model to use (defaults to provider's default)
            config: Configuration object
            **kwargs: Additional config overrides
        """
        # Handle config
        if config is None:
            config = EmbeddingConfig()
        
        # Apply overrides
        if isinstance(provider, str):
            config.provider = EmbeddingProviderType(provider)
        elif isinstance(provider, EmbeddingProviderType):
            config.provider = provider
        
        if model:
            config.model = model
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.config = config
        
        # Initialize provider
        if isinstance(provider, EmbeddingProvider):
            self._provider = provider
        else:
            provider_class = self.PROVIDERS.get(config.provider)
            if provider_class is None:
                raise ValueError(f"Unknown provider: {config.provider}")
            
            init_kwargs = {}
            if config.api_key:
                init_kwargs["api_key"] = config.api_key
            if config.api_base:
                init_kwargs["api_base"] = config.api_base
            if hasattr(provider_class, "__init__"):
                if "timeout" in provider_class.__init__.__code__.co_varnames:
                    init_kwargs["timeout"] = config.timeout
            
            self._provider = provider_class(**init_kwargs)
        
        # Set model
        self.model = config.model or self._provider.default_model
        
        # Initialize cache
        self._cache: Optional[EmbeddingCache] = None
        if config.enable_cache:
            self._cache = EmbeddingCache(
                max_size=config.cache_size,
                ttl=config.cache_ttl,
            )
        
        # Initialize rate limiter
        self._rate_limiter = RateLimiter(
            requests_per_minute=config.requests_per_minute,
            tokens_per_minute=config.tokens_per_minute,
        )
        
        # Statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._total_latency = 0.0
    
    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self._provider.name
    
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions for current model."""
        return self._provider.get_dimensions(self.model)
    
    def embed(
        self,
        text: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            model: Model to use (defaults to service model)
            **kwargs: Additional provider arguments
            
        Returns:
            EmbeddingResult with embedding and metadata
        """
        results = self.embed_batch([text], model=model, **kwargs)
        return results.results[0]
    
    def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> BatchEmbeddingResult:
        """
        Embed multiple texts with batching.
        
        Args:
            texts: Texts to embed
            model: Model to use
            **kwargs: Additional provider arguments
            
        Returns:
            BatchEmbeddingResult with all embeddings
        """
        model = model or self.model
        start_time = time.time()
        
        results: List[EmbeddingResult] = []
        cache_hits = 0
        cache_misses = 0
        total_tokens = 0
        
        # Check cache for each text
        texts_to_embed: List[Tuple[int, str]] = []
        for i, text in enumerate(texts):
            cached = None
            if self._cache:
                cached = self._cache.get(text, model, self._provider.name)
            
            if cached is not None:
                cache_hits += 1
                results.append(EmbeddingResult(
                    text=text,
                    embedding=cached,
                    model=model,
                    provider=self._provider.name,
                    dimensions=len(cached),
                    cached=True,
                ))
            else:
                cache_misses += 1
                texts_to_embed.append((i, text))
                results.append(None)  # Placeholder
        
        # Batch embed remaining texts
        if texts_to_embed:
            batches = self._create_batches([t for _, t in texts_to_embed])
            
            for batch in batches:
                batch_results = self._embed_with_retry(batch, model, **kwargs)
                
                for j, (embedding, tokens) in enumerate(batch_results):
                    original_idx = texts_to_embed[j][0]
                    text = texts_to_embed[j][1]
                    
                    # Normalize if configured
                    if self.config.normalize:
                        embedding = self._normalize(embedding)
                    
                    # Cache the result
                    if self._cache:
                        self._cache.put(text, model, self._provider.name, embedding)
                    
                    total_tokens += tokens
                    
                    results[original_idx] = EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        model=model,
                        provider=self._provider.name,
                        dimensions=len(embedding),
                        tokens_used=tokens,
                        cached=False,
                    )
                
                # Update texts_to_embed for next batch
                texts_to_embed = texts_to_embed[len(batch):]
        
        total_latency = (time.time() - start_time) * 1000
        
        # Update statistics
        self._total_requests += 1
        self._total_tokens += total_tokens
        self._total_latency += total_latency
        
        return BatchEmbeddingResult(
            results=results,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )
    
    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """Create batches respecting size and token limits."""
        batches = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            tokens = self._provider.count_tokens(text)
            
            # Check if adding this text would exceed limits
            if (
                len(current_batch) >= self.config.batch_size
                or current_tokens + tokens > self.config.max_tokens_per_batch
            ):
                if current_batch:
                    batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(text)
            current_tokens += tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _embed_with_retry(
        self,
        texts: List[str],
        model: str,
        **kwargs: Any,
    ) -> List[Tuple[List[float], int]]:
        """Embed with retry logic and rate limiting."""
        estimated_tokens = sum(self._provider.count_tokens(t) for t in texts)
        
        for attempt in range(self.config.max_retries):
            # Rate limiting
            wait_time = self._rate_limiter.acquire(estimated_tokens)
            if wait_time > 0:
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
            
            try:
                return self._provider.embed(texts, model, **kwargs)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                
                delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                logger.warning(
                    f"Embedding failed (attempt {attempt + 1}): {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                time.sleep(delay)
        
        raise RuntimeError("Failed to embed after all retries")
    
    def _normalize(self, embedding: List[float]) -> List[float]:
        """Normalize embedding to unit length."""
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()
    
    def similarity(
        self,
        text1: str,
        text2: str,
        model: Optional[str] = None,
    ) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            model: Model to use
            
        Returns:
            Cosine similarity score
        """
        results = self.embed_batch([text1, text2], model=model)
        return results.results[0].similarity(results.results[1])
    
    def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        threshold: float = 0.0,
        model: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Find most similar texts to query.
        
        Args:
            query: Query text
            candidates: Candidate texts to compare
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            model: Model to use
            
        Returns:
            List of (text, similarity) tuples sorted by similarity
        """
        # Embed all texts
        all_texts = [query] + candidates
        results = self.embed_batch(all_texts, model=model)
        
        query_embedding = results.results[0].to_numpy()
        
        similarities = []
        for i, result in enumerate(results.results[1:]):
            candidate_embedding = result.to_numpy()
            sim = float(np.dot(query_embedding, candidate_embedding))
            
            if sim >= threshold:
                similarities.append((candidates[i], sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._cache:
            self._cache.clear()
    
    @property
    def cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self._cache:
            return self._cache.stats
        return None
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "provider": self._provider.name,
            "model": self.model,
            "dimensions": self.dimensions,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_latency_ms": self._total_latency,
            "avg_latency_ms": (
                self._total_latency / self._total_requests
                if self._total_requests > 0
                else 0
            ),
            "cache": self.cache_stats,
        }


# Global service instance
_default_service: Optional[EmbeddingService] = None


def get_embedding_service(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> EmbeddingService:
    """
    Get or create the default embedding service.
    
    Args:
        provider: Provider to use
        model: Model to use
        **kwargs: Additional configuration
        
    Returns:
        EmbeddingService instance
    """
    global _default_service
    
    if _default_service is None or provider is not None or model is not None:
        _default_service = EmbeddingService(
            provider=provider or "openai",
            model=model,
            **kwargs,
        )
    
    return _default_service


def embed_texts(
    texts: List[str],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> BatchEmbeddingResult:
    """
    Convenience function to embed multiple texts.
    
    Args:
        texts: Texts to embed
        provider: Provider to use
        model: Model to use
        **kwargs: Additional configuration
        
    Returns:
        BatchEmbeddingResult
    """
    service = get_embedding_service(provider=provider, model=model, **kwargs)
    return service.embed_batch(texts)


def embed_text(
    text: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> EmbeddingResult:
    """
    Convenience function to embed a single text.
    
    Args:
        text: Text to embed
        provider: Provider to use
        model: Model to use
        **kwargs: Additional configuration
        
    Returns:
        EmbeddingResult
    """
    service = get_embedding_service(provider=provider, model=model, **kwargs)
    return service.embed(text)
