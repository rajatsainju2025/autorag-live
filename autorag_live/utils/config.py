"""
Configuration management utilities for AutoRAG-Live.
"""
from pathlib import Path
from typing import Any, Dict, Optional, cast
from omegaconf import OmegaConf, DictConfig
import hydra

from ..types.types import ConfigurationError
from .schema import AutoRAGConfig

class ConfigManager:
    """Central configuration manager for AutoRAG-Live."""
    
    _instance = None
    _config = None
    
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
            # Load base config
            config_path = Path(__file__).parent.parent.parent / "config"
            base_config = OmegaConf.load(config_path / "config.yaml")
            
            # Load component configs
            retrieval_config = OmegaConf.load(config_path / "retrieval/default.yaml")
            evaluation_config = OmegaConf.load(config_path / "evaluation/default.yaml")
            pipeline_config = OmegaConf.load(config_path / "pipeline/default.yaml")
            augmentation_config = OmegaConf.load(config_path / "augmentation/default.yaml")
            
            # Merge configs
            self._config = OmegaConf.merge(
                base_config,
                retrieval_config,
                evaluation_config,
                pipeline_config,
                augmentation_config
            )
            
            # Make config immutable
            OmegaConf.set_readonly(self._config, True)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize configuration: {str(e)}")
    
    @property
    def config(self) -> DictConfig:
        """Get the full configuration."""
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
        try:
            return OmegaConf.select(self._config, key, default=default)
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
        try:
            # Create a mutable copy
            mutable_config = OmegaConf.create(OmegaConf.to_container(self._config))
            OmegaConf.update(mutable_config, key, value, merge=True)
            
            # Validate the new config
            # TODO: Add schema validation
            
            # Make immutable and update
            OmegaConf.set_readonly(mutable_config, True)
            self._config = mutable_config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to update config key '{key}': {str(e)}")
    
    @staticmethod
    def get_instance() -> 'ConfigManager':
        """Get the singleton instance of ConfigManager."""
        return ConfigManager()