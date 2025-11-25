"""
Configuration management utilities for AutoRAG-Live.
"""
from pathlib import Path
from typing import Any, Optional, cast

from omegaconf import DictConfig, OmegaConf

from ..types.types import ConfigurationError
from .cache import Cache, MemoryCache
from .schema import AutoRAGConfig


class ConfigManager:
    """Central configuration manager for AutoRAG-Live with lazy loading."""

    _instance: Optional["ConfigManager"] = None
    _config: Optional[DictConfig] = None
    _config_cache: Cache = MemoryCache()
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Defer config initialization until first access
        pass

    def _initialize_config(self) -> None:
        """Initialize configuration from files (lazy loaded)."""
        if self._initialized:
            return

        try:
            # Load base config - check for test override
            import os

            from .validation import merge_with_env_vars, validate_config

            config_dir_override = os.environ.get("AUTORAG_CONFIG_DIR")
            if config_dir_override:
                config_path = Path(config_dir_override)
            else:
                config_path = Path(__file__).parent.parent.parent / "config"

            base_config = OmegaConf.load(config_path / "config.yaml")

            # Try to load component configs (might not exist in test)
            configs_to_merge = [base_config]

            for component in ["retrieval", "evaluation", "pipeline", "augmentation"]:
                component_file = config_path / component / "default.yaml"
                if component_file.exists():
                    configs_to_merge.append(OmegaConf.load(component_file))

            # Merge configs
            merged_config = cast(DictConfig, OmegaConf.merge(*configs_to_merge))

            # Apply environment variable overrides
            merged_config = merge_with_env_vars(merged_config)

            # Resolve interpolations
            OmegaConf.resolve(merged_config)

            # Ensure we have a DictConfig
            if not isinstance(merged_config, DictConfig):
                raise ConfigurationError("Configuration merge did not produce a DictConfig")

            # Validate against schema
            validate_config(merged_config, AutoRAGConfig)

            self._config = merged_config

            # Make config immutable
            OmegaConf.set_readonly(self._config, True)

            self._initialized = True

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize configuration: {e!s}")

    @property
    def config(self) -> DictConfig:
        """Get the full configuration (lazy loaded)."""
        if not self._initialized:
            self._initialize_config()
        if self._config is None:
            raise ConfigurationError("Configuration not initialized")
        if not isinstance(self._config, DictConfig):
            raise ConfigurationError("Configuration is not a DictConfig")
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key with caching.

        Args:
            key: Dot-notation path to config value
            default: Default value if key doesn't exist

        Returns:
            Configuration value
        """
        # Try to get from cache first
        cache_key = f"config_get_{key}"
        cached_value = self._config_cache.get(cache_key)
        if cached_value is not None:
            return cached_value

        # Ensure config is initialized
        if not self._initialized:
            self._initialize_config()

        try:
            value = OmegaConf.select(self.config, key, default=default)
            # Cache the result
            self._config_cache.set(cache_key, value, ttl=300.0)  # 5 minute cache
            return value
        except Exception:
            return default
        cached_value = self._config_cache.get(cache_key)
        if cached_value is not None:
            return cached_value

        try:
            if self._config is None:
                raise ConfigurationError("Configuration not initialized")

            value = OmegaConf.select(self._config, key, default=default)

            # Cache the result
            self._config_cache.set(cache_key, value)

            return value
        except Exception as e:
            raise ConfigurationError(f"Error accessing config key '{key}': {str(e)}")

    def update(self, key: str, value: Any) -> None:
        """
        Update a configuration value.

        Args:
            key: Dot-notation path to config value
            value: New value to set

        Raises:
            ConfigurationError: If update fails
        """
        if key.startswith("_"):
            raise ConfigurationError("Cannot update protected configuration fields")

        try:
            # Check current type if key exists
            current_value = self.get(key)
            if current_value is not None:
                if not isinstance(value, type(current_value)):
                    raise ConfigurationError(
                        f"Type mismatch: Cannot update '{key}' of type {type(current_value)} with value of type {type(value)}"
                    )

            # Create a mutable copy
            mutable_config = OmegaConf.create(OmegaConf.to_container(self._config))
            OmegaConf.update(mutable_config, key, value, merge=True)

            # Validate the new config with fast schema validation
            from .schema_validation import COMMON_SCHEMAS, validate_config_fast

            # Use appropriate schema based on config structure
            schema = None
            if "retriever" in str(key).lower():
                schema = COMMON_SCHEMAS["retriever_config"]
            elif "pipeline" in str(key).lower():
                schema = COMMON_SCHEMAS["pipeline_config"]
            elif "cache" in str(key).lower():
                schema = COMMON_SCHEMAS["cache_config"]

            if schema and isinstance(mutable_config, DictConfig):
                if not validate_config_fast(mutable_config, schema):
                    raise ConfigurationError(f"Configuration validation failed for key '{key}'")

            # Ensure we have a DictConfig
            if not isinstance(mutable_config, DictConfig):
                raise ConfigurationError("Configuration update did not produce a DictConfig")

            # Make immutable and update
            OmegaConf.set_readonly(mutable_config, True)
            self._config = mutable_config
            self._config_cache.clear()  # Invalidate cache after update

        except Exception as e:
            raise ConfigurationError(f"Failed to update config key '{key}': {str(e)}")

    @staticmethod
    def get_instance() -> "ConfigManager":
        """Get the singleton instance of ConfigManager."""
        return ConfigManager()
