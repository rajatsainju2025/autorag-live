"""
Configuration profiles for AutoRAG-Live.

Provides pre-configured profiles for common use cases,
making it easy to optimize for speed, quality, or balance.

Features:
- Pre-built profiles (speed, quality, balanced, dev)
- Profile inheritance and composition
- Environment-based profile selection
- Dynamic profile switching
- Profile validation

Example usage:
    >>> from autorag_live.config.profiles import get_profile
    >>> 
    >>> # Get a profile
    >>> config = get_profile("speed")
    >>> print(config.retrieval.top_k)
    >>> 
    >>> # Apply to pipeline
    >>> pipeline = RAGPipeline(config=config)
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)


T = TypeVar("T")


class ProfileType(str, Enum):
    """Available profile types."""
    
    SPEED = "speed"
    QUALITY = "quality"
    BALANCED = "balanced"
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    MINIMAL = "minimal"
    CUSTOM = "custom"


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    
    # Core settings
    top_k: int = 5
    min_score: float = 0.3
    
    # Retriever settings
    retriever_type: str = "dense"
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: Optional[str] = None
    
    # Hybrid settings
    enable_hybrid: bool = False
    hybrid_alpha: float = 0.5
    
    # Performance
    batch_size: int = 32
    use_cache: bool = True
    cache_ttl: int = 3600
    
    # Advanced
    rerank: bool = False
    rerank_model: Optional[str] = None
    rerank_top_k: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "top_k": self.top_k,
            "min_score": self.min_score,
            "retriever_type": self.retriever_type,
            "dense_model": self.dense_model,
            "sparse_model": self.sparse_model,
            "enable_hybrid": self.enable_hybrid,
            "hybrid_alpha": self.hybrid_alpha,
            "batch_size": self.batch_size,
            "use_cache": self.use_cache,
            "cache_ttl": self.cache_ttl,
            "rerank": self.rerank,
            "rerank_model": self.rerank_model,
            "rerank_top_k": self.rerank_top_k,
        }


@dataclass
class GenerationConfig:
    """Generation configuration."""
    
    # Model settings
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 500
    
    # Response settings
    include_citations: bool = True
    citation_style: str = "numeric"
    response_format: str = "markdown"
    
    # Performance
    stream: bool = False
    timeout: float = 30.0
    retry_attempts: int = 3
    
    # Advanced
    system_prompt: Optional[str] = None
    stop_sequences: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "include_citations": self.include_citations,
            "citation_style": self.citation_style,
            "response_format": self.response_format,
            "stream": self.stream,
            "timeout": self.timeout,
            "retry_attempts": self.retry_attempts,
            "system_prompt": self.system_prompt,
            "stop_sequences": self.stop_sequences,
        }


@dataclass
class AgentConfig:
    """Agent configuration."""
    
    # Agent settings
    enabled: bool = False
    max_iterations: int = 5
    
    # Planning
    enable_planning: bool = False
    max_plan_steps: int = 10
    
    # Tools
    enabled_tools: List[str] = field(default_factory=lambda: ["retrieve", "search"])
    
    # Memory
    enable_memory: bool = False
    memory_size: int = 10
    
    # Behavior
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "max_iterations": self.max_iterations,
            "enable_planning": self.enable_planning,
            "max_plan_steps": self.max_plan_steps,
            "enabled_tools": self.enabled_tools,
            "enable_memory": self.enable_memory,
            "memory_size": self.memory_size,
            "verbose": self.verbose,
        }


@dataclass
class CacheConfig:
    """Cache configuration."""
    
    # Cache settings
    enabled: bool = True
    backend: str = "memory"
    
    # TTL settings
    default_ttl: int = 3600
    embedding_ttl: int = 86400
    query_ttl: int = 1800
    
    # Size limits
    max_size: int = 10000
    max_memory_mb: float = 100.0
    
    # Advanced
    eviction_policy: str = "lru"
    enable_compression: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "default_ttl": self.default_ttl,
            "embedding_ttl": self.embedding_ttl,
            "query_ttl": self.query_ttl,
            "max_size": self.max_size,
            "max_memory_mb": self.max_memory_mb,
            "eviction_policy": self.eviction_policy,
            "enable_compression": self.enable_compression,
        }


@dataclass
class ObservabilityConfig:
    """Observability configuration."""
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Telemetry
    enable_telemetry: bool = True
    telemetry_export: str = "console"
    
    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Tracing
    enable_tracing: bool = False
    trace_sample_rate: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "log_level": self.log_level,
            "log_format": self.log_format,
            "enable_telemetry": self.enable_telemetry,
            "telemetry_export": self.telemetry_export,
            "enable_metrics": self.enable_metrics,
            "metrics_port": self.metrics_port,
            "enable_tracing": self.enable_tracing,
            "trace_sample_rate": self.trace_sample_rate,
        }


@dataclass
class Profile:
    """Complete configuration profile."""
    
    name: str
    profile_type: ProfileType
    description: str = ""
    
    # Sub-configurations
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    
    # Profile metadata
    version: str = "1.0"
    parent: Optional[str] = None
    
    # Custom extensions
    extensions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "profile_type": self.profile_type.value,
            "description": self.description,
            "version": self.version,
            "parent": self.parent,
            "retrieval": self.retrieval.to_dict(),
            "generation": self.generation.to_dict(),
            "agent": self.agent.to_dict(),
            "cache": self.cache.to_dict(),
            "observability": self.observability.to_dict(),
            "extensions": self.extensions,
        }
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get nested configuration value.
        
        Args:
            path: Dot-separated path (e.g., "retrieval.top_k")
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        parts = path.split(".")
        value: Any = self
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set(self, path: str, value: Any) -> None:
        """
        Set nested configuration value.
        
        Args:
            path: Dot-separated path
            value: Value to set
        """
        parts = path.split(".")
        target = self
        
        for part in parts[:-1]:
            if hasattr(target, part):
                target = getattr(target, part)
            else:
                raise ValueError(f"Invalid path: {path}")
        
        setattr(target, parts[-1], value)
    
    def copy(self) -> "Profile":
        """Create a deep copy of the profile."""
        return copy.deepcopy(self)
    
    def merge(self, other: "Profile") -> "Profile":
        """
        Merge another profile into this one.
        
        Args:
            other: Profile to merge
            
        Returns:
            New merged profile
        """
        merged = self.copy()
        
        # Merge retrieval
        for key, value in other.retrieval.to_dict().items():
            if value is not None:
                setattr(merged.retrieval, key, value)
        
        # Merge generation
        for key, value in other.generation.to_dict().items():
            if value is not None:
                setattr(merged.generation, key, value)
        
        # Merge agent
        for key, value in other.agent.to_dict().items():
            if value is not None:
                setattr(merged.agent, key, value)
        
        # Merge cache
        for key, value in other.cache.to_dict().items():
            if value is not None:
                setattr(merged.cache, key, value)
        
        # Merge observability
        for key, value in other.observability.to_dict().items():
            if value is not None:
                setattr(merged.observability, key, value)
        
        # Merge extensions
        merged.extensions.update(other.extensions)
        
        return merged


# ============================================================
# Pre-built Profiles
# ============================================================

def create_speed_profile() -> Profile:
    """Create speed-optimized profile."""
    return Profile(
        name="speed",
        profile_type=ProfileType.SPEED,
        description="Optimized for low latency and fast responses",
        retrieval=RetrievalConfig(
            top_k=3,
            min_score=0.4,
            retriever_type="dense",
            dense_model="sentence-transformers/all-MiniLM-L6-v2",
            enable_hybrid=False,
            batch_size=64,
            use_cache=True,
            cache_ttl=7200,
            rerank=False,
        ),
        generation=GenerationConfig(
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=300,
            include_citations=False,
            stream=True,
            timeout=15.0,
            retry_attempts=2,
        ),
        agent=AgentConfig(
            enabled=False,
            max_iterations=3,
            enable_planning=False,
        ),
        cache=CacheConfig(
            enabled=True,
            backend="memory",
            default_ttl=7200,
            max_size=20000,
            eviction_policy="lru",
        ),
        observability=ObservabilityConfig(
            log_level="WARNING",
            enable_telemetry=False,
            enable_metrics=False,
            enable_tracing=False,
        ),
    )


def create_quality_profile() -> Profile:
    """Create quality-optimized profile."""
    return Profile(
        name="quality",
        profile_type=ProfileType.QUALITY,
        description="Optimized for best response quality",
        retrieval=RetrievalConfig(
            top_k=10,
            min_score=0.2,
            retriever_type="dense",
            dense_model="sentence-transformers/all-mpnet-base-v2",
            enable_hybrid=True,
            hybrid_alpha=0.6,
            batch_size=16,
            use_cache=True,
            cache_ttl=3600,
            rerank=True,
            rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_top_k=5,
        ),
        generation=GenerationConfig(
            model="gpt-4",
            temperature=0.3,
            max_tokens=1000,
            include_citations=True,
            citation_style="numeric",
            response_format="markdown",
            stream=False,
            timeout=60.0,
            retry_attempts=5,
        ),
        agent=AgentConfig(
            enabled=True,
            max_iterations=10,
            enable_planning=True,
            max_plan_steps=10,
            enabled_tools=["retrieve", "search", "verify"],
            enable_memory=True,
            memory_size=20,
        ),
        cache=CacheConfig(
            enabled=True,
            backend="memory",
            default_ttl=1800,
            embedding_ttl=172800,
            max_size=50000,
            eviction_policy="lfu",
        ),
        observability=ObservabilityConfig(
            log_level="DEBUG",
            enable_telemetry=True,
            telemetry_export="file",
            enable_metrics=True,
            enable_tracing=True,
            trace_sample_rate=1.0,
        ),
    )


def create_balanced_profile() -> Profile:
    """Create balanced profile."""
    return Profile(
        name="balanced",
        profile_type=ProfileType.BALANCED,
        description="Balanced between speed and quality",
        retrieval=RetrievalConfig(
            top_k=5,
            min_score=0.3,
            retriever_type="dense",
            dense_model="sentence-transformers/all-MiniLM-L12-v2",
            enable_hybrid=True,
            hybrid_alpha=0.5,
            batch_size=32,
            use_cache=True,
            cache_ttl=3600,
            rerank=True,
            rerank_model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            rerank_top_k=3,
        ),
        generation=GenerationConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500,
            include_citations=True,
            stream=True,
            timeout=30.0,
            retry_attempts=3,
        ),
        agent=AgentConfig(
            enabled=False,
            max_iterations=5,
            enable_planning=False,
        ),
        cache=CacheConfig(
            enabled=True,
            backend="memory",
            default_ttl=3600,
            max_size=10000,
        ),
        observability=ObservabilityConfig(
            log_level="INFO",
            enable_telemetry=True,
            enable_metrics=True,
            enable_tracing=False,
        ),
    )


def create_development_profile() -> Profile:
    """Create development profile."""
    return Profile(
        name="development",
        profile_type=ProfileType.DEVELOPMENT,
        description="Profile for local development and debugging",
        retrieval=RetrievalConfig(
            top_k=5,
            min_score=0.2,
            retriever_type="dense",
            dense_model="sentence-transformers/all-MiniLM-L6-v2",
            enable_hybrid=False,
            batch_size=8,
            use_cache=False,
            rerank=False,
        ),
        generation=GenerationConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500,
            include_citations=True,
            stream=False,
            timeout=60.0,
            retry_attempts=1,
        ),
        agent=AgentConfig(
            enabled=True,
            max_iterations=5,
            enable_planning=True,
            verbose=True,
        ),
        cache=CacheConfig(
            enabled=False,
            backend="memory",
        ),
        observability=ObservabilityConfig(
            log_level="DEBUG",
            enable_telemetry=True,
            telemetry_export="console",
            enable_metrics=True,
            enable_tracing=True,
            trace_sample_rate=1.0,
        ),
    )


def create_production_profile() -> Profile:
    """Create production profile."""
    return Profile(
        name="production",
        profile_type=ProfileType.PRODUCTION,
        description="Production-ready profile with reliability focus",
        retrieval=RetrievalConfig(
            top_k=5,
            min_score=0.3,
            retriever_type="dense",
            dense_model="sentence-transformers/all-mpnet-base-v2",
            enable_hybrid=True,
            hybrid_alpha=0.5,
            batch_size=32,
            use_cache=True,
            cache_ttl=3600,
            rerank=True,
            rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_top_k=3,
        ),
        generation=GenerationConfig(
            model="gpt-4",
            temperature=0.5,
            max_tokens=800,
            include_citations=True,
            stream=True,
            timeout=30.0,
            retry_attempts=3,
        ),
        agent=AgentConfig(
            enabled=True,
            max_iterations=5,
            enable_planning=True,
            max_plan_steps=5,
            verbose=False,
        ),
        cache=CacheConfig(
            enabled=True,
            backend="redis",
            default_ttl=3600,
            embedding_ttl=86400,
            max_size=100000,
            eviction_policy="lru",
            enable_compression=True,
        ),
        observability=ObservabilityConfig(
            log_level="INFO",
            enable_telemetry=True,
            telemetry_export="file",
            enable_metrics=True,
            metrics_port=9090,
            enable_tracing=True,
            trace_sample_rate=0.1,
        ),
    )


def create_minimal_profile() -> Profile:
    """Create minimal profile for basic usage."""
    return Profile(
        name="minimal",
        profile_type=ProfileType.MINIMAL,
        description="Minimal configuration for basic usage",
        retrieval=RetrievalConfig(
            top_k=3,
            min_score=0.5,
            retriever_type="dense",
            dense_model="sentence-transformers/all-MiniLM-L6-v2",
            enable_hybrid=False,
            use_cache=False,
            rerank=False,
        ),
        generation=GenerationConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=200,
            include_citations=False,
            stream=False,
        ),
        agent=AgentConfig(
            enabled=False,
        ),
        cache=CacheConfig(
            enabled=False,
        ),
        observability=ObservabilityConfig(
            log_level="WARNING",
            enable_telemetry=False,
            enable_metrics=False,
            enable_tracing=False,
        ),
    )


# ============================================================
# Profile Registry
# ============================================================

class ProfileRegistry:
    """Registry of available profiles."""
    
    _instance: Optional["ProfileRegistry"] = None
    
    def __new__(cls) -> "ProfileRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._profiles = {}
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize registry."""
        if not self._initialized:
            self._register_defaults()
            self._initialized = True
    
    def _register_defaults(self) -> None:
        """Register default profiles."""
        self.register("speed", create_speed_profile)
        self.register("quality", create_quality_profile)
        self.register("balanced", create_balanced_profile)
        self.register("development", create_development_profile)
        self.register("dev", create_development_profile)  # Alias
        self.register("production", create_production_profile)
        self.register("prod", create_production_profile)  # Alias
        self.register("minimal", create_minimal_profile)
    
    def register(
        self,
        name: str,
        factory: Callable[[], Profile],
    ) -> None:
        """
        Register a profile factory.
        
        Args:
            name: Profile name
            factory: Function that creates the profile
        """
        self._profiles[name.lower()] = factory
    
    def get(self, name: str) -> Profile:
        """
        Get a profile by name.
        
        Args:
            name: Profile name
            
        Returns:
            Profile instance
            
        Raises:
            ValueError: If profile not found
        """
        name = name.lower()
        if name not in self._profiles:
            available = ", ".join(self._profiles.keys())
            raise ValueError(f"Profile '{name}' not found. Available: {available}")
        
        return self._profiles[name]()
    
    def list_profiles(self) -> List[str]:
        """Get list of available profile names."""
        return list(self._profiles.keys())
    
    def exists(self, name: str) -> bool:
        """Check if profile exists."""
        return name.lower() in self._profiles


# ============================================================
# Environment-based Profile Selection
# ============================================================

def get_profile_from_env(
    env_var: str = "AUTORAG_PROFILE",
    default: str = "balanced",
) -> Profile:
    """
    Get profile based on environment variable.
    
    Args:
        env_var: Environment variable name
        default: Default profile if not set
        
    Returns:
        Profile instance
    """
    profile_name = os.environ.get(env_var, default)
    return get_profile(profile_name)


def get_profile(
    name: str = "balanced",
) -> Profile:
    """
    Get a profile by name.
    
    Args:
        name: Profile name
        
    Returns:
        Profile instance
    """
    registry = ProfileRegistry()
    return registry.get(name)


def list_profiles() -> List[str]:
    """Get list of available profiles."""
    registry = ProfileRegistry()
    return registry.list_profiles()


def register_profile(
    name: str,
    factory: Callable[[], Profile],
) -> None:
    """
    Register a custom profile.
    
    Args:
        name: Profile name
        factory: Profile factory function
    """
    registry = ProfileRegistry()
    registry.register(name, factory)


# ============================================================
# Profile Builder
# ============================================================

class ProfileBuilder:
    """
    Fluent builder for creating custom profiles.
    
    Example:
        >>> profile = (ProfileBuilder("custom")
        ...     .based_on("balanced")
        ...     .with_retrieval(top_k=10, rerank=True)
        ...     .with_generation(model="gpt-4")
        ...     .build())
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
    ):
        """
        Initialize builder.
        
        Args:
            name: Profile name
            description: Profile description
        """
        self._name = name
        self._description = description
        self._base: Optional[Profile] = None
        self._retrieval_overrides: Dict[str, Any] = {}
        self._generation_overrides: Dict[str, Any] = {}
        self._agent_overrides: Dict[str, Any] = {}
        self._cache_overrides: Dict[str, Any] = {}
        self._observability_overrides: Dict[str, Any] = {}
        self._extensions: Dict[str, Any] = {}
    
    def based_on(self, profile_name: str) -> "ProfileBuilder":
        """
        Base this profile on an existing one.
        
        Args:
            profile_name: Name of base profile
            
        Returns:
            Self for chaining
        """
        self._base = get_profile(profile_name)
        return self
    
    def with_retrieval(self, **kwargs: Any) -> "ProfileBuilder":
        """Set retrieval configuration."""
        self._retrieval_overrides.update(kwargs)
        return self
    
    def with_generation(self, **kwargs: Any) -> "ProfileBuilder":
        """Set generation configuration."""
        self._generation_overrides.update(kwargs)
        return self
    
    def with_agent(self, **kwargs: Any) -> "ProfileBuilder":
        """Set agent configuration."""
        self._agent_overrides.update(kwargs)
        return self
    
    def with_cache(self, **kwargs: Any) -> "ProfileBuilder":
        """Set cache configuration."""
        self._cache_overrides.update(kwargs)
        return self
    
    def with_observability(self, **kwargs: Any) -> "ProfileBuilder":
        """Set observability configuration."""
        self._observability_overrides.update(kwargs)
        return self
    
    def with_extension(self, name: str, config: Any) -> "ProfileBuilder":
        """Add an extension configuration."""
        self._extensions[name] = config
        return self
    
    def build(self) -> Profile:
        """Build the profile."""
        # Start with base or default
        if self._base:
            profile = self._base.copy()
            profile.name = self._name
            profile.parent = self._base.name
        else:
            profile = Profile(
                name=self._name,
                profile_type=ProfileType.CUSTOM,
            )
        
        profile.description = self._description or profile.description
        profile.profile_type = ProfileType.CUSTOM
        
        # Apply overrides
        for key, value in self._retrieval_overrides.items():
            setattr(profile.retrieval, key, value)
        
        for key, value in self._generation_overrides.items():
            setattr(profile.generation, key, value)
        
        for key, value in self._agent_overrides.items():
            setattr(profile.agent, key, value)
        
        for key, value in self._cache_overrides.items():
            setattr(profile.cache, key, value)
        
        for key, value in self._observability_overrides.items():
            setattr(profile.observability, key, value)
        
        profile.extensions.update(self._extensions)
        
        return profile


# ============================================================
# Profile Validation
# ============================================================

class ProfileValidator:
    """Validate profile configurations."""
    
    VALID_RETRIEVER_TYPES = ["dense", "sparse", "hybrid"]
    VALID_CACHE_BACKENDS = ["memory", "redis", "file"]
    VALID_EVICTION_POLICIES = ["lru", "lfu", "fifo"]
    VALID_CITATION_STYLES = ["numeric", "author_year", "inline", "footnote"]
    VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    def validate(self, profile: Profile) -> List[str]:
        """
        Validate a profile.
        
        Args:
            profile: Profile to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate retrieval
        if profile.retrieval.retriever_type not in self.VALID_RETRIEVER_TYPES:
            errors.append(f"Invalid retriever_type: {profile.retrieval.retriever_type}")
        
        if profile.retrieval.top_k < 1:
            errors.append("top_k must be at least 1")
        
        if not 0 <= profile.retrieval.min_score <= 1:
            errors.append("min_score must be between 0 and 1")
        
        if not 0 <= profile.retrieval.hybrid_alpha <= 1:
            errors.append("hybrid_alpha must be between 0 and 1")
        
        # Validate generation
        if profile.generation.max_tokens < 1:
            errors.append("max_tokens must be at least 1")
        
        if not 0 <= profile.generation.temperature <= 2:
            errors.append("temperature must be between 0 and 2")
        
        if profile.generation.citation_style not in self.VALID_CITATION_STYLES:
            errors.append(f"Invalid citation_style: {profile.generation.citation_style}")
        
        # Validate cache
        if profile.cache.backend not in self.VALID_CACHE_BACKENDS:
            errors.append(f"Invalid cache backend: {profile.cache.backend}")
        
        if profile.cache.eviction_policy not in self.VALID_EVICTION_POLICIES:
            errors.append(f"Invalid eviction_policy: {profile.cache.eviction_policy}")
        
        # Validate observability
        if profile.observability.log_level not in self.VALID_LOG_LEVELS:
            errors.append(f"Invalid log_level: {profile.observability.log_level}")
        
        if not 0 <= profile.observability.trace_sample_rate <= 1:
            errors.append("trace_sample_rate must be between 0 and 1")
        
        return errors
    
    def is_valid(self, profile: Profile) -> bool:
        """Check if profile is valid."""
        return len(self.validate(profile)) == 0


def validate_profile(profile: Profile) -> List[str]:
    """
    Validate a profile.
    
    Args:
        profile: Profile to validate
        
    Returns:
        List of validation errors
    """
    validator = ProfileValidator()
    return validator.validate(profile)
