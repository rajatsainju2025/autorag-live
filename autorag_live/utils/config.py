"""
Configuration management utilities for AutoRAG-Live.
"""
from pathlib import Path
from typing import Any, Dict, Optional, cast
from omegaconf import OmegaConf, DictConfig
import hydra

from ..types.types import ConfigurationError
from .schema import AutoRAGConfig
from .cache import Cache, MemoryCache

class ConfigManager:
    """Central configuration manager for AutoRAG-Live."""
    
    _instance: Optional["ConfigManager"] = None
    _config: Optional[DictConfig] = None
    _config = None
    _config_cache: Cache = MemoryCache()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._initialize_config()
    
    def _initialize_config(self) -> None:
        """Initialize configuration from files."""
        try:
            from .validation import validate_config, merge_with_env_vars

            # Load base config - check for test override
            import os
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
            config = cast(DictConfig, OmegaConf.merge(*configs_to_merge))

            # Apply environment variable overrides
            config = merge_with_env_vars(config)

            # Validate against schema
            validate_config(config, AutoRAGConfig)

            self._config = config
            
            # Merge configs
            if len(configs_to_merge) == 1:
                merged_config = base_config
            else:
                merged_config = OmegaConf.merge(*configs_to_merge)
            
            # Ensure we have a DictConfig
            if not isinstance(merged_config, DictConfig):
                raise ConfigurationError("Configuration merge did not produce a DictConfig")
            
            self._config = merged_config
            
            # Make config immutable
            OmegaConf.set_readonly(self._config, True)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize configuration: {str(e)}")
    
    @property
    def config(self) -> DictConfig:
        """Get the full configuration."""
        if self._config is None:
            raise ConfigurationError("Configuration not initialized")
        if not isinstance(self._config, DictConfig):
            raise ConfigurationError("Configuration is not a DictConfig")
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
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
        if key.startswith('_'):
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
            
            # Validate the new config
            # TODO: Add schema validation
            
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
    def get_instance() -> 'ConfigManager':
        """Get the singleton instance of ConfigManager."""
        return ConfigManager()