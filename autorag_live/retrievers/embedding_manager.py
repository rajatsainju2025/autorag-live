"""Embedding Manager for AutoRAG-Live.

Centralized embedding model management with:
- Multiple embedding model support
- Batch embedding generation
- Embedding caching
- Model lazy loading
- Dimension reduction
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModelType(Enum):
    """Types of embedding models."""
    
    OPENAI = "openai"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class EmbeddingConfig:
    """Configuration for an embedding model."""
    
    model_type: EmbeddingModelType
    model_name: str
    dimension: int
    max_tokens: int = 512
    batch_size: int = 32
    normalize: bool = True
    cache_enabled: bool = True
    api_key: str | None = None
    device: str = "cpu"
    extra_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    
    text: str
    embedding: np.ndarray
    model_name: str
    dimension: int
    tokens_used: int
    latency_ms: float
    from_cache: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingCache:
    """Cache for embeddings with persistence support."""
    
    def __init__(
        self,
        max_size: int = 10000,
        persist_path: Path | None = None,
    ) -> None:
        """Initialize embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings
            persist_path: Path for cache persistence
        """
        self.max_size = max_size
        self.persist_path = persist_path
        self._cache: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        self._access_order: list[str] = []
        
        if persist_path and persist_path.exists():
            self._load_cache()
    
    def _compute_key(self, text: str, model_name: str) -> str:
        """Compute cache key for text and model."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(
        self,
        text: str,
        model_name: str,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Get embedding from cache.
        
        Args:
            text: Text to look up
            model_name: Model name for lookup
            
        Returns:
            Tuple of (embedding, metadata) or None if not found
        """
        key = self._compute_key(text, model_name)
        
        if key in self._cache:
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        
        return None
    
    def put(
        self,
        text: str,
        model_name: str,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store embedding in cache.
        
        Args:
            text: Original text
            model_name: Model name
            embedding: Embedding vector
            metadata: Optional metadata
        """
        key = self._compute_key(text, model_name)
        
        # Evict if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)
        
        self._cache[key] = (embedding, metadata or {})
        self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self.persist_path:
            return
            
        try:
            import pickle
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            self._cache = data["cache"]
            self._access_order = data["access_order"]
            logger.info(f"Loaded {len(self._cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
    
    def save(self) -> None:
        """Save cache to disk."""
        if not self.persist_path:
            return
            
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            # Use pickle for complex dict structure
            import pickle
            with open(self.persist_path, "wb") as f:
                pickle.dump({
                    "cache": self._cache,
                    "access_order": self._access_order,
                }, f)
            logger.info(f"Saved {len(self._cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    @property
    def hit_ratio(self) -> float:
        """Get cache hit ratio (needs tracking implementation)."""
        return 0.0  # Placeholder


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize embedding model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self._model: Any = None
    
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
    
    def unload(self) -> None:
        """Unload model from memory."""
        self._model = None


class SentenceTransformerModel(BaseEmbeddingModel):
    """Sentence Transformer embedding model."""
    
    def load(self) -> None:
        """Load Sentence Transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
            logger.info(f"Loaded SentenceTransformer: {self.config.model_name}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed single text."""
        if not self.is_loaded():
            self.load()
        
        embedding = self._model.encode(
            text,
            normalize_embeddings=self.config.normalize,
        )
        return np.array(embedding)
    
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed batch of texts."""
        if not self.is_loaded():
            self.load()
        
        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return [np.array(emb) for emb in embeddings]


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI embedding model."""
    
    def load(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai
            
            self._model = openai.OpenAI(api_key=self.config.api_key)
            logger.info(f"Initialized OpenAI embeddings: {self.config.model_name}")
        except ImportError:
            raise ImportError(
                "openai not installed. Install with: pip install openai"
            )
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed single text via OpenAI."""
        if not self.is_loaded():
            self.load()
        
        response = self._model.embeddings.create(
            model=self.config.model_name,
            input=text,
        )
        return np.array(response.data[0].embedding)
    
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed batch via OpenAI."""
        if not self.is_loaded():
            self.load()
        
        response = self._model.embeddings.create(
            model=self.config.model_name,
            input=texts,
        )
        
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [np.array(item.embedding) for item in sorted_data]


class CohereEmbeddingModel(BaseEmbeddingModel):
    """Cohere embedding model."""
    
    def load(self) -> None:
        """Initialize Cohere client."""
        try:
            import cohere
            
            self._model = cohere.Client(api_key=self.config.api_key)
            logger.info(f"Initialized Cohere embeddings: {self.config.model_name}")
        except ImportError:
            raise ImportError(
                "cohere not installed. Install with: pip install cohere"
            )
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed single text via Cohere."""
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed batch via Cohere."""
        if not self.is_loaded():
            self.load()
        
        input_type = self.config.extra_config.get("input_type", "search_document")
        
        response = self._model.embed(
            texts=texts,
            model=self.config.model_name,
            input_type=input_type,
        )
        return [np.array(emb) for emb in response.embeddings]


class MockEmbeddingModel(BaseEmbeddingModel):
    """Mock embedding model for testing."""
    
    def load(self) -> None:
        """No-op load."""
        self._model = True
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding."""
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self.config.dimension)
        if self.config.normalize:
            embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate mock embeddings for batch."""
        return [self.embed_single(text) for text in texts]


class EmbeddingManager:
    """Centralized manager for embedding models and operations."""
    
    def __init__(
        self,
        default_model: str | None = None,
        cache_size: int = 10000,
        cache_path: Path | None = None,
    ) -> None:
        """Initialize embedding manager.
        
        Args:
            default_model: Default model name to use
            cache_size: Maximum cache size
            cache_path: Path for cache persistence
        """
        self.default_model = default_model
        self._models: dict[str, BaseEmbeddingModel] = {}
        self._configs: dict[str, EmbeddingConfig] = {}
        self._cache = EmbeddingCache(max_size=cache_size, persist_path=cache_path)
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics
        self._stats: dict[str, Any] = {
            "total_embeddings": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens": 0,
            "total_latency_ms": 0.0,
        }
    
    def register_model(
        self,
        name: str,
        config: EmbeddingConfig,
        set_default: bool = False,
    ) -> None:
        """Register an embedding model.
        
        Args:
            name: Unique name for the model
            config: Model configuration
            set_default: Whether to set as default model
        """
        self._configs[name] = config
        
        # Create model instance
        model = self._create_model(config)
        self._models[name] = model
        
        if set_default or self.default_model is None:
            self.default_model = name
        
        logger.info(f"Registered embedding model: {name}")
    
    def _create_model(self, config: EmbeddingConfig) -> BaseEmbeddingModel:
        """Create model instance from config."""
        model_classes: dict[EmbeddingModelType, type[BaseEmbeddingModel]] = {
            EmbeddingModelType.SENTENCE_TRANSFORMER: SentenceTransformerModel,
            EmbeddingModelType.OPENAI: OpenAIEmbeddingModel,
            EmbeddingModelType.COHERE: CohereEmbeddingModel,
            EmbeddingModelType.CUSTOM: MockEmbeddingModel,
        }
        
        model_class = model_classes.get(config.model_type, MockEmbeddingModel)
        return model_class(config)
    
    def get_model(self, name: str | None = None) -> BaseEmbeddingModel:
        """Get a registered model by name.
        
        Args:
            name: Model name (uses default if None)
            
        Returns:
            Embedding model instance
        """
        model_name = name or self.default_model
        
        if model_name is None:
            raise ValueError("No model name provided and no default model set")
        
        if model_name not in self._models:
            raise ValueError(f"Model not registered: {model_name}")
        
        return self._models[model_name]
    
    def embed(
        self,
        text: str,
        model_name: str | None = None,
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """Embed a single text.
        
        Args:
            text: Text to embed
            model_name: Model to use (default if None)
            use_cache: Whether to use cache
            
        Returns:
            Embedding result
        """
        model_name = model_name or self.default_model
        if model_name is None:
            raise ValueError("No model specified and no default set")
        
        model = self.get_model(model_name)
        config = self._configs[model_name]
        
        # Check cache
        if use_cache and config.cache_enabled:
            cached = self._cache.get(text, model_name)
            if cached is not None:
                embedding, metadata = cached
                self._stats["cache_hits"] += 1
                return EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model_name=model_name,
                    dimension=len(embedding),
                    tokens_used=0,
                    latency_ms=0.0,
                    from_cache=True,
                    metadata=metadata,
                )
        
        self._stats["cache_misses"] += 1
        
        # Generate embedding
        start_time = time.time()
        embedding = model.embed_single(text)
        latency_ms = (time.time() - start_time) * 1000
        
        # Estimate tokens (rough approximation)
        tokens_used = len(text.split())
        
        # Update stats
        self._stats["total_embeddings"] += 1
        self._stats["total_tokens"] += tokens_used
        self._stats["total_latency_ms"] += latency_ms
        
        # Cache result
        if use_cache and config.cache_enabled:
            self._cache.put(text, model_name, embedding)
        
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model_name=model_name,
            dimension=len(embedding),
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            from_cache=False,
        )
    
    def embed_batch(
        self,
        texts: list[str],
        model_name: str | None = None,
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> list[EmbeddingResult]:
        """Embed a batch of texts.
        
        Args:
            texts: Texts to embed
            model_name: Model to use
            use_cache: Whether to use cache
            show_progress: Whether to show progress
            
        Returns:
            List of embedding results
        """
        model_name = model_name or self.default_model
        if model_name is None:
            raise ValueError("No model specified and no default set")
        
        model = self.get_model(model_name)
        config = self._configs[model_name]
        
        results: list[EmbeddingResult] = []
        texts_to_embed: list[tuple[int, str]] = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if use_cache and config.cache_enabled:
                cached = self._cache.get(text, model_name)
                if cached is not None:
                    embedding, metadata = cached
                    self._stats["cache_hits"] += 1
                    results.append(EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        model_name=model_name,
                        dimension=len(embedding),
                        tokens_used=0,
                        latency_ms=0.0,
                        from_cache=True,
                        metadata=metadata,
                    ))
                    continue
            
            texts_to_embed.append((i, text))
            self._stats["cache_misses"] += 1
        
        # Embed uncached texts in batches
        if texts_to_embed:
            start_time = time.time()
            
            batch_texts = [t[1] for t in texts_to_embed]
            embeddings = model.embed_batch(batch_texts)
            
            total_latency_ms = (time.time() - start_time) * 1000
            per_text_latency = total_latency_ms / len(batch_texts)
            
            for (orig_idx, text), embedding in zip(texts_to_embed, embeddings):
                tokens_used = len(text.split())
                
                self._stats["total_embeddings"] += 1
                self._stats["total_tokens"] += tokens_used
                self._stats["total_latency_ms"] += per_text_latency
                
                # Cache result
                if use_cache and config.cache_enabled:
                    self._cache.put(text, model_name, embedding)
                
                results.append(EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model_name=model_name,
                    dimension=len(embedding),
                    tokens_used=tokens_used,
                    latency_ms=per_text_latency,
                    from_cache=False,
                ))
        
        # Sort results back to original order
        text_to_result = {r.text: r for r in results}
        return [text_to_result[text] for text in texts]
    
    async def embed_async(
        self,
        text: str,
        model_name: str | None = None,
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """Embed text asynchronously.
        
        Args:
            text: Text to embed
            model_name: Model to use
            use_cache: Whether to use cache
            
        Returns:
            Embedding result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.embed(text, model_name, use_cache),
        )
    
    async def embed_batch_async(
        self,
        texts: list[str],
        model_name: str | None = None,
        use_cache: bool = True,
    ) -> list[EmbeddingResult]:
        """Embed batch asynchronously.
        
        Args:
            texts: Texts to embed
            model_name: Model to use
            use_cache: Whether to use cache
            
        Returns:
            List of embedding results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.embed_batch(texts, model_name, use_cache),
        )
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine",
    ) -> float:
        """Compute similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric (cosine, euclidean, dot)
            
        Returns:
            Similarity score
        """
        if metric == "cosine":
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
        
        elif metric == "dot":
            return float(np.dot(embedding1, embedding2))
        
        elif metric == "euclidean":
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(1.0 / (1.0 + distance))
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: list[np.ndarray],
        top_k: int = 10,
        metric: str = "cosine",
        threshold: float | None = None,
    ) -> list[tuple[int, float]]:
        """Find most similar embeddings.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: Candidate embeddings to search
            top_k: Number of results to return
            metric: Similarity metric
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, similarity) tuples
        """
        similarities: list[tuple[int, float]] = []
        
        for i, candidate in enumerate(candidate_embeddings):
            sim = self.compute_similarity(query_embedding, candidate, metric)
            
            if threshold is None or sim >= threshold:
                similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def reduce_dimensions(
        self,
        embeddings: list[np.ndarray],
        target_dim: int,
        method: str = "pca",
    ) -> list[np.ndarray]:
        """Reduce embedding dimensions.
        
        Args:
            embeddings: Embeddings to reduce
            target_dim: Target dimension
            method: Reduction method (pca, random)
            
        Returns:
            Reduced embeddings
        """
        if not embeddings:
            return []
        
        stacked = np.vstack(embeddings)
        current_dim = stacked.shape[1]
        
        if target_dim >= current_dim:
            return embeddings
        
        if method == "pca":
            # Simple PCA using SVD
            mean = np.mean(stacked, axis=0)
            centered = stacked - mean
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            projection = vt[:target_dim].T
            reduced = np.dot(centered, projection)
            return [reduced[i] for i in range(len(embeddings))]
        
        elif method == "random":
            # Random projection
            np.random.seed(42)  # For reproducibility
            projection = np.random.randn(current_dim, target_dim)
            projection = projection / np.linalg.norm(projection, axis=0)
            reduced = np.dot(stacked, projection)
            return [reduced[i] for i in range(len(embeddings))]
        
        else:
            raise ValueError(f"Unknown reduction method: {method}")
    
    def get_stats(self) -> dict[str, Any]:
        """Get embedding statistics."""
        stats = self._stats.copy()
        
        # Calculate derived metrics
        total = stats["cache_hits"] + stats["cache_misses"]
        if total > 0:
            stats["cache_hit_ratio"] = stats["cache_hits"] / total
        else:
            stats["cache_hit_ratio"] = 0.0
        
        if stats["total_embeddings"] > 0:
            stats["avg_latency_ms"] = (
                stats["total_latency_ms"] / stats["total_embeddings"]
            )
        else:
            stats["avg_latency_ms"] = 0.0
        
        stats["cache_size"] = self._cache.size
        stats["registered_models"] = list(self._models.keys())
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()
        logger.info("Cleared embedding cache")
    
    def save_cache(self) -> None:
        """Save cache to disk."""
        self._cache.save()
    
    def unload_model(self, name: str) -> None:
        """Unload a model from memory.
        
        Args:
            name: Model name to unload
        """
        if name in self._models:
            self._models[name].unload()
            logger.info(f"Unloaded model: {name}")
    
    def unload_all(self) -> None:
        """Unload all models."""
        for name in self._models:
            self._models[name].unload()
        logger.info("Unloaded all models")


def create_default_manager(
    use_openai: bool = False,
    use_sentence_transformer: bool = True,
) -> EmbeddingManager:
    """Create embedding manager with default configuration.
    
    Args:
        use_openai: Whether to configure OpenAI model
        use_sentence_transformer: Whether to configure SentenceTransformer
        
    Returns:
        Configured EmbeddingManager
    """
    manager = EmbeddingManager()
    
    if use_sentence_transformer:
        manager.register_model(
            "default",
            EmbeddingConfig(
                model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
                model_name="all-MiniLM-L6-v2",
                dimension=384,
                max_tokens=256,
                batch_size=64,
            ),
            set_default=True,
        )
    
    if use_openai:
        import os
        manager.register_model(
            "openai",
            EmbeddingConfig(
                model_type=EmbeddingModelType.OPENAI,
                model_name="text-embedding-3-small",
                dimension=1536,
                max_tokens=8191,
                batch_size=100,
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
        )
    
    return manager


# Convenience functions
@lru_cache(maxsize=1)
def get_global_manager() -> EmbeddingManager:
    """Get or create global embedding manager."""
    return create_default_manager()


def embed_text(text: str, model: str | None = None) -> np.ndarray:
    """Convenience function to embed text.
    
    Args:
        text: Text to embed
        model: Model name to use
        
    Returns:
        Embedding vector
    """
    manager = get_global_manager()
    result = manager.embed(text, model)
    return result.embedding


def embed_texts(texts: list[str], model: str | None = None) -> list[np.ndarray]:
    """Convenience function to embed multiple texts.
    
    Args:
        texts: Texts to embed
        model: Model name to use
        
    Returns:
        List of embedding vectors
    """
    manager = get_global_manager()
    results = manager.embed_batch(texts, model)
    return [r.embedding for r in results]
