"""
Advanced Configuration System for Agentic RAG.

This module provides a modern, composable configuration system with:
- Dependency injection container
- Composable configuration from multiple sources
- Type-safe configuration with validation
- Environment variable support
- Configuration hot-reloading
- Configuration versioning and migration
- Profiles and presets
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import yaml

# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar("T")
ConfigT = TypeVar("ConfigT", bound="BaseConfig")


class ConfigSource(Enum):
    """Sources for configuration values."""

    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"
    REMOTE = "remote"


class ComponentType(Enum):
    """Types of injectable components."""

    RETRIEVER = "retriever"
    RERANKER = "reranker"
    LLM = "llm"
    EMBEDDER = "embedder"
    MEMORY = "memory"
    CACHE = "cache"
    TOOL = "tool"
    AGENT = "agent"
    PIPELINE = "pipeline"
    EVALUATOR = "evaluator"


# =============================================================================
# Configuration Protocols
# =============================================================================


@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol for configuration providers."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        ...

    def has(self, key: str) -> bool:
        """Check if key exists."""
        ...


@runtime_checkable
class Configurable(Protocol):
    """Protocol for configurable components."""

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the component."""
        ...

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        ...


# =============================================================================
# Base Configuration Classes
# =============================================================================


@dataclass
class ConfigValue:
    """A configuration value with metadata."""

    key: str
    value: Any
    source: ConfigSource
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigSchema:
    """Schema definition for configuration."""

    key: str
    value_type: Type
    required: bool = False
    default: Any = None
    description: str = ""
    validator: Optional[Callable[[Any], bool]] = None
    choices: Optional[List[Any]] = None
    env_var: Optional[str] = None

    def validate(self, value: Any) -> bool:
        """Validate a value against the schema."""
        # Type check
        if not isinstance(value, self.value_type):
            return False

        # Choice check
        if self.choices and value not in self.choices:
            return False

        # Custom validator
        if self.validator and not self.validator(value):
            return False

        return True


class BaseConfig:
    """Base class for typed configurations."""

    _schema: Dict[str, ConfigSchema] = {}

    def __init__(self, **kwargs: Any):
        """Initialize with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {key: getattr(self, key, schema.default) for key, schema in self._schema.items()}

    @classmethod
    def from_dict(cls: Type[ConfigT], data: Dict[str, Any]) -> ConfigT:
        """Create from dictionary."""
        return cls(**data)

    def validate(self) -> List[str]:
        """Validate configuration, return list of errors."""
        errors = []
        for key, schema in self._schema.items():
            value = getattr(self, key, schema.default)
            if schema.required and value is None:
                errors.append(f"Required field '{key}' is missing")
            elif value is not None and not schema.validate(value):
                errors.append(f"Invalid value for '{key}': {value}")
        return errors

    def merge(self, other: "BaseConfig") -> "BaseConfig":
        """Merge with another config, other takes precedence."""
        merged_data = self.to_dict()
        merged_data.update(other.to_dict())
        return self.__class__.from_dict(merged_data)


# =============================================================================
# Specific Configuration Types
# =============================================================================


@dataclass
class RetrieverConfig(BaseConfig):
    """Configuration for retriever components."""

    type: str = "dense"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 10
    min_score: float = 0.0
    batch_size: int = 32
    use_cache: bool = True
    cache_ttl: int = 3600
    timeout: float = 30.0

    # Hybrid settings
    enable_hybrid: bool = False
    hybrid_alpha: float = 0.5
    sparse_weight: float = 0.3
    dense_weight: float = 0.7

    # ColBERT settings
    colbert_enabled: bool = False
    colbert_model: str = "colbert-ir/colbertv2.0"

    _schema = {
        "type": ConfigSchema("type", str, choices=["dense", "sparse", "hybrid", "colbert"]),
        "top_k": ConfigSchema("top_k", int, validator=lambda x: x > 0),
        "min_score": ConfigSchema("min_score", float, validator=lambda x: 0 <= x <= 1),
    }


@dataclass
class RerankerConfig(BaseConfig):
    """Configuration for reranker components."""

    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    batch_size: int = 16
    use_cache: bool = True

    _schema = {
        "enabled": ConfigSchema("enabled", bool),
        "top_k": ConfigSchema("top_k", int, validator=lambda x: x > 0),
    }


@dataclass
class LLMConfig(BaseConfig):
    """Configuration for LLM components."""

    provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: float = 60.0
    retry_attempts: int = 3
    stream: bool = False

    _schema = {
        "provider": ConfigSchema("provider", str, choices=["openai", "anthropic", "local"]),
        "temperature": ConfigSchema("temperature", float, validator=lambda x: 0 <= x <= 2),
        "max_tokens": ConfigSchema("max_tokens", int, validator=lambda x: x > 0),
    }


@dataclass
class MemoryConfig(BaseConfig):
    """Configuration for memory components."""

    enabled: bool = True
    working_memory_size: int = 10
    episodic_memory_enabled: bool = True
    semantic_memory_enabled: bool = True
    procedural_memory_enabled: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.7
    max_memory_items: int = 1000
    ttl_seconds: int = 86400

    _schema = {
        "working_memory_size": ConfigSchema("working_memory_size", int, validator=lambda x: x > 0),
        "similarity_threshold": ConfigSchema(
            "similarity_threshold", float, validator=lambda x: 0 <= x <= 1
        ),
    }


@dataclass
class AgentConfig(BaseConfig):
    """Configuration for agent components."""

    type: str = "react"
    max_iterations: int = 10
    max_tokens_per_step: int = 500
    enable_planning: bool = False
    enable_reflection: bool = False
    enable_tools: bool = True
    tools: List[str] = field(default_factory=lambda: ["retrieve", "search"])
    verbose: bool = False
    thought_budget: int = 5

    # Advanced reasoning
    reasoning_strategy: str = "chain_of_thought"
    tree_width: int = 3
    tree_depth: int = 3

    _schema = {
        "type": ConfigSchema("type", str, choices=["react", "plan", "reflect", "tree_of_thoughts"]),
        "max_iterations": ConfigSchema("max_iterations", int, validator=lambda x: x > 0),
    }


@dataclass
class PipelineConfig(BaseConfig):
    """Configuration for pipeline components."""

    name: str = "default"
    stages: List[str] = field(default_factory=lambda: ["retrieval", "rerank", "generation"])
    parallel_execution: bool = False
    enable_caching: bool = True
    enable_metrics: bool = True
    timeout: float = 120.0

    # Rate limiting
    rate_limit_enabled: bool = False
    requests_per_second: float = 10.0
    burst_size: int = 20

    # Circuit breaker
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: float = 30.0

    _schema = {
        "timeout": ConfigSchema("timeout", float, validator=lambda x: x > 0),
    }


# =============================================================================
# Dependency Injection Container
# =============================================================================


class ServiceLifetime(Enum):
    """Service lifetime in the DI container."""

    TRANSIENT = "transient"  # New instance every time
    SCOPED = "scoped"  # New instance per scope
    SINGLETON = "singleton"  # Single instance for app lifetime


@dataclass
class ServiceDescriptor:
    """Describes a service registration."""

    service_type: Type
    implementation: Union[Type, Callable, object]
    lifetime: ServiceLifetime
    name: Optional[str] = None


class DIContainer:
    """
    Dependency Injection container for agentic RAG components.

    Supports:
    - Constructor injection
    - Multiple implementations per interface
    - Singleton, scoped, and transient lifetimes
    - Factory functions
    - Lazy instantiation
    """

    def __init__(self):
        """Initialize DI container."""
        self._services: Dict[Type, List[ServiceDescriptor]] = {}
        self._singletons: Dict[Type, Dict[str, object]] = {}
        self._scopes: Dict[str, Dict[Type, Dict[str, object]]] = {}
        self._current_scope: Optional[str] = None
        self._lock = threading.Lock()
        self.logger = logging.getLogger("DIContainer")

    def register(
        self,
        service_type: Type[T],
        implementation: Union[Type[T], Callable[..., T], T],
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
        name: Optional[str] = None,
    ) -> "DIContainer":
        """
        Register a service.

        Args:
            service_type: The service interface/type
            implementation: Implementation class, factory, or instance
            lifetime: Service lifetime
            name: Optional name for named registrations

        Returns:
            Self for chaining
        """
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            lifetime=lifetime,
            name=name or "default",
        )

        with self._lock:
            if service_type not in self._services:
                self._services[service_type] = []
            self._services[service_type].append(descriptor)

        return self

    def register_singleton(
        self, service_type: Type[T], implementation: Union[Type[T], T], name: Optional[str] = None
    ) -> "DIContainer":
        """Register a singleton service."""
        return self.register(service_type, implementation, ServiceLifetime.SINGLETON, name)

    def register_transient(
        self, service_type: Type[T], implementation: Type[T], name: Optional[str] = None
    ) -> "DIContainer":
        """Register a transient service."""
        return self.register(service_type, implementation, ServiceLifetime.TRANSIENT, name)

    def register_scoped(
        self, service_type: Type[T], implementation: Type[T], name: Optional[str] = None
    ) -> "DIContainer":
        """Register a scoped service."""
        return self.register(service_type, implementation, ServiceLifetime.SCOPED, name)

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[..., T],
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
        name: Optional[str] = None,
    ) -> "DIContainer":
        """Register a factory function."""
        return self.register(service_type, factory, lifetime, name)

    def resolve(self, service_type: Type[T], name: Optional[str] = None) -> T:
        """
        Resolve a service.

        Args:
            service_type: The service type to resolve
            name: Optional name for named resolution

        Returns:
            Service instance
        """
        name = name or "default"

        if service_type not in self._services:
            raise KeyError(f"Service {service_type.__name__} not registered")

        descriptors = self._services[service_type]
        descriptor = next(
            (d for d in descriptors if d.name == name),
            descriptors[0] if descriptors else None,
        )

        if not descriptor:
            raise KeyError(f"Service {service_type.__name__} with name '{name}' not found")

        return self._create_instance(descriptor)

    def resolve_all(self, service_type: Type[T]) -> List[T]:
        """Resolve all registered implementations of a service type."""
        if service_type not in self._services:
            return []

        return [self._create_instance(d) for d in self._services[service_type]]

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create or retrieve service instance."""
        service_type = descriptor.service_type
        name = descriptor.name or "default"

        # Check for existing singleton
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if service_type in self._singletons:
                if name in self._singletons[service_type]:
                    return self._singletons[service_type][name]

        # Check for existing scoped instance
        if descriptor.lifetime == ServiceLifetime.SCOPED and self._current_scope:
            if self._current_scope in self._scopes:
                scope_instances = self._scopes[self._current_scope]
                if service_type in scope_instances and name in scope_instances[service_type]:
                    return scope_instances[service_type][name]

        # Create new instance
        impl = descriptor.implementation

        if isinstance(impl, type):
            # Class - instantiate with dependency injection
            instance = self._instantiate_with_injection(impl)
        elif callable(impl):
            # Factory function
            instance = impl()
        else:
            # Already an instance
            instance = impl

        # Store if singleton or scoped
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            with self._lock:
                if service_type not in self._singletons:
                    self._singletons[service_type] = {}
                self._singletons[service_type][name] = instance

        elif descriptor.lifetime == ServiceLifetime.SCOPED and self._current_scope:
            with self._lock:
                if self._current_scope not in self._scopes:
                    self._scopes[self._current_scope] = {}
                if service_type not in self._scopes[self._current_scope]:
                    self._scopes[self._current_scope][service_type] = {}
                self._scopes[self._current_scope][service_type][name] = instance

        return instance

    def _instantiate_with_injection(self, cls: Type[T]) -> T:
        """Instantiate a class with constructor injection."""
        import inspect

        sig = inspect.signature(cls.__init__)
        kwargs = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            if param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[param_name] = self.resolve(param.annotation)
                except KeyError:
                    if param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
                    else:
                        raise

        return cls(**kwargs)

    def create_scope(self, scope_id: str) -> "ScopeContext":
        """Create a new scope for scoped services."""
        return ScopeContext(self, scope_id)

    def clear_scope(self, scope_id: str) -> None:
        """Clear a scope's instances."""
        with self._lock:
            if scope_id in self._scopes:
                del self._scopes[scope_id]


class ScopeContext:
    """Context manager for DI scopes."""

    def __init__(self, container: DIContainer, scope_id: str):
        """Initialize scope context."""
        self.container = container
        self.scope_id = scope_id
        self._previous_scope: Optional[str] = None

    def __enter__(self) -> "ScopeContext":
        """Enter scope."""
        self._previous_scope = self.container._current_scope
        self.container._current_scope = self.scope_id
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit scope and cleanup."""
        self.container._current_scope = self._previous_scope
        self.container.clear_scope(self.scope_id)


# =============================================================================
# Configuration Manager
# =============================================================================


class ConfigurationManager:
    """
    Central configuration manager with multiple sources.

    Features:
    - Hierarchical configuration (file < env < runtime)
    - Hot reloading
    - Configuration validation
    - Version tracking
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        env_prefix: str = "AUTORAG",
        auto_reload: bool = False,
    ):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file
            env_prefix: Prefix for environment variables
            auto_reload: Enable automatic config reloading
        """
        self.config_path = config_path
        self.env_prefix = env_prefix
        self.auto_reload = auto_reload

        self._values: Dict[str, ConfigValue] = {}
        self._schemas: Dict[str, ConfigSchema] = {}
        self._watchers: List[Callable[[str, Any, Any], None]] = []
        self._lock = threading.Lock()
        self._file_hash: Optional[str] = None
        self._reload_thread: Optional[threading.Thread] = None

        self.logger = logging.getLogger("ConfigurationManager")

        # Load initial configuration
        self._load_defaults()
        if config_path:
            self._load_from_file(config_path)
        self._load_from_environment()

        # Start hot reload if enabled
        if auto_reload and config_path:
            self._start_hot_reload()

    def _load_defaults(self) -> None:
        """Load default configuration values."""
        defaults = {
            "retriever.type": "dense",
            "retriever.top_k": 10,
            "reranker.enabled": False,
            "llm.provider": "openai",
            "llm.temperature": 0.7,
            "agent.type": "react",
            "agent.max_iterations": 10,
            "pipeline.timeout": 120.0,
        }

        for key, value in defaults.items():
            self._values[key] = ConfigValue(key=key, value=value, source=ConfigSource.DEFAULT)

    def _load_from_file(self, path: Path) -> None:
        """Load configuration from file."""
        if not path.exists():
            self.logger.warning(f"Config file not found: {path}")
            return

        try:
            content = path.read_text()
            self._file_hash = hashlib.md5(content.encode()).hexdigest()

            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(content)
            elif path.suffix == ".json":
                data = json.loads(content)
            else:
                self.logger.warning(f"Unknown config format: {path.suffix}")
                return

            self._flatten_and_store(data, ConfigSource.FILE)

        except Exception as e:
            self.logger.error(f"Failed to load config from {path}: {e}")

    def _flatten_and_store(
        self, data: Dict[str, Any], source: ConfigSource, prefix: str = ""
    ) -> None:
        """Flatten nested dict and store values."""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._flatten_and_store(value, source, full_key)
            else:
                self._values[full_key] = ConfigValue(key=full_key, value=value, source=source)

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        for env_key, env_value in os.environ.items():
            if env_key.startswith(f"{self.env_prefix}_"):
                # Convert AUTORAG_RETRIEVER_TOP_K to retriever.top_k
                config_key = env_key[len(self.env_prefix) + 1 :].lower().replace("_", ".")

                # Try to parse the value
                value = self._parse_env_value(env_value)

                self._values[config_key] = ConfigValue(
                    key=config_key, value=value, source=ConfigSource.ENVIRONMENT
                )

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Return as string
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        if key in self._values:
            return self._values[key].value
        return default

    def get_typed(self, key: str, value_type: Type[T], default: Optional[T] = None) -> Optional[T]:
        """Get a typed configuration value."""
        value = self.get(key, default)
        if value is not None and isinstance(value, value_type):
            return value
        return default

    def set(self, key: str, value: Any) -> None:
        """Set a runtime configuration value."""
        old_value = self._values.get(key)

        with self._lock:
            self._values[key] = ConfigValue(
                key=key,
                value=value,
                source=ConfigSource.RUNTIME,
                version=(old_value.version + 1) if old_value else 1,
            )

        # Notify watchers
        if old_value:
            self._notify_watchers(key, old_value.value, value)

    def has(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._values

    def watch(self, callback: Callable[[str, Any, Any], None]) -> None:
        """Register a configuration change watcher."""
        self._watchers.append(callback)

    def _notify_watchers(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify all watchers of a change."""
        for watcher in self._watchers:
            try:
                watcher(key, old_value, new_value)
            except Exception as e:
                self.logger.error(f"Watcher error: {e}")

    def _start_hot_reload(self) -> None:
        """Start hot reload thread."""

        def reload_loop() -> None:
            while True:
                time.sleep(5)  # Check every 5 seconds
                self._check_and_reload()

        self._reload_thread = threading.Thread(target=reload_loop, daemon=True)
        self._reload_thread.start()

    def _check_and_reload(self) -> None:
        """Check for config file changes and reload if needed."""
        if not self.config_path or not self.config_path.exists():
            return

        content = self.config_path.read_text()
        new_hash = hashlib.md5(content.encode()).hexdigest()

        if new_hash != self._file_hash:
            self.logger.info("Config file changed, reloading...")
            self._load_from_file(self.config_path)

    def get_section(self, prefix: str) -> Dict[str, Any]:
        """Get all values under a prefix as a dict."""
        result: Dict[str, Any] = {}
        prefix_with_dot = f"{prefix}."

        for key, config_value in self._values.items():
            if key.startswith(prefix_with_dot):
                sub_key = key[len(prefix_with_dot) :]
                result[sub_key] = config_value.value

        return result

    def get_config(self, config_class: Type[ConfigT], prefix: str) -> ConfigT:
        """Get a typed configuration object."""
        section = self.get_section(prefix)
        return config_class.from_dict(section)


# =============================================================================
# Service Registry
# =============================================================================


class ServiceRegistry:
    """
    Registry for agentic RAG services with configuration.

    Combines DI container with configuration management.
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """Initialize service registry."""
        self.config = config_manager or ConfigurationManager()
        self.container = DIContainer()
        self._factories: Dict[ComponentType, Callable[..., Any]] = {}
        self.logger = logging.getLogger("ServiceRegistry")

    def register_factory(self, component_type: ComponentType, factory: Callable[..., Any]) -> None:
        """Register a component factory."""
        self._factories[component_type] = factory

    def get_component(
        self,
        component_type: ComponentType,
        name: Optional[str] = None,
        **overrides: Any,
    ) -> Any:
        """Get a configured component."""
        # Get configuration for this component type
        config_prefix = component_type.value
        config = self.config.get_section(config_prefix)
        config.update(overrides)

        # Check if we have a factory
        if component_type in self._factories:
            return self._factories[component_type](**config)

        # Try to resolve from DI container
        # This is a simplified lookup - in practice you'd have proper type mapping
        self.logger.warning(f"No factory registered for {component_type}")
        return None

    def configure_pipeline(self) -> Dict[str, Any]:
        """Get full pipeline configuration."""
        return {
            "retriever": self.config.get_section("retriever"),
            "reranker": self.config.get_section("reranker"),
            "llm": self.config.get_section("llm"),
            "agent": self.config.get_section("agent"),
            "memory": self.config.get_section("memory"),
            "pipeline": self.config.get_section("pipeline"),
        }


# =============================================================================
# Factory Functions
# =============================================================================


@lru_cache(maxsize=1)
def get_default_container() -> DIContainer:
    """Get the default DI container."""
    return DIContainer()


@lru_cache(maxsize=1)
def get_default_config_manager() -> ConfigurationManager:
    """Get the default configuration manager."""
    # Look for config file in common locations
    for path in [
        Path("config/config.yaml"),
        Path("config.yaml"),
        Path.home() / ".autorag" / "config.yaml",
    ]:
        if path.exists():
            return ConfigurationManager(config_path=path)

    return ConfigurationManager()


def create_service_registry(
    config_path: Optional[Path] = None,
    env_prefix: str = "AUTORAG",
) -> ServiceRegistry:
    """Create a configured service registry."""
    config_manager = ConfigurationManager(config_path=config_path, env_prefix=env_prefix)
    return ServiceRegistry(config_manager=config_manager)


def configure_from_yaml(yaml_path: Path) -> ConfigurationManager:
    """Create configuration manager from YAML file."""
    return ConfigurationManager(config_path=yaml_path)


def configure_from_env(prefix: str = "AUTORAG") -> ConfigurationManager:
    """Create configuration manager from environment variables only."""
    return ConfigurationManager(env_prefix=prefix)


# =============================================================================
# Preset Configurations
# =============================================================================


class ConfigPresets:
    """Pre-built configuration presets."""

    @staticmethod
    def development() -> Dict[str, Any]:
        """Development preset - verbose, fast iterations."""
        return {
            "retriever.type": "dense",
            "retriever.top_k": 5,
            "reranker.enabled": False,
            "llm.model_name": "gpt-3.5-turbo",
            "llm.temperature": 0.7,
            "agent.verbose": True,
            "agent.max_iterations": 5,
            "pipeline.enable_metrics": True,
            "pipeline.timeout": 60.0,
        }

    @staticmethod
    def production() -> Dict[str, Any]:
        """Production preset - optimized for quality."""
        return {
            "retriever.type": "hybrid",
            "retriever.top_k": 20,
            "retriever.enable_hybrid": True,
            "reranker.enabled": True,
            "reranker.top_k": 5,
            "llm.model_name": "gpt-4",
            "llm.temperature": 0.3,
            "agent.verbose": False,
            "agent.max_iterations": 10,
            "pipeline.enable_metrics": True,
            "pipeline.circuit_breaker_enabled": True,
            "pipeline.rate_limit_enabled": True,
        }

    @staticmethod
    def fast() -> Dict[str, Any]:
        """Fast preset - optimized for speed."""
        return {
            "retriever.type": "dense",
            "retriever.top_k": 3,
            "reranker.enabled": False,
            "llm.model_name": "gpt-3.5-turbo",
            "llm.max_tokens": 256,
            "agent.max_iterations": 3,
            "pipeline.timeout": 30.0,
        }

    @staticmethod
    def quality() -> Dict[str, Any]:
        """Quality preset - optimized for answer quality."""
        return {
            "retriever.type": "hybrid",
            "retriever.top_k": 30,
            "retriever.colbert_enabled": True,
            "reranker.enabled": True,
            "reranker.top_k": 10,
            "llm.model_name": "gpt-4",
            "llm.temperature": 0.2,
            "llm.max_tokens": 2048,
            "agent.enable_reflection": True,
            "agent.reasoning_strategy": "tree_of_thoughts",
            "pipeline.timeout": 300.0,
        }

    @staticmethod
    def apply_preset(config_manager: ConfigurationManager, preset_name: str) -> None:
        """Apply a preset to a configuration manager."""
        presets = {
            "development": ConfigPresets.development,
            "production": ConfigPresets.production,
            "fast": ConfigPresets.fast,
            "quality": ConfigPresets.quality,
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")

        preset_values = presets[preset_name]()
        for key, value in preset_values.items():
            config_manager.set(key, value)
