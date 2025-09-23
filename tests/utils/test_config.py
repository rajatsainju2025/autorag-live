"""
Tests for configuration management system.
"""
import pytest
from pathlib import Path
from omegaconf import OmegaConf

from autorag_live.utils.config import ConfigManager
from autorag_live.types.types import ConfigurationError


def test_config_manager_singleton(config_manager):
    """Test that ConfigManager follows singleton pattern."""
    another_instance = ConfigManager.get_instance()
    assert config_manager is another_instance
    assert ConfigManager() is config_manager


def test_config_basic_access(config_manager):
    """Test basic configuration access."""
    assert config_manager.get("name") == "autorag-test"
    assert config_manager.get("version") == "0.0.1"
    assert isinstance(config_manager.get("paths"), dict)


def test_config_nested_access(config_manager):
    """Test accessing nested configuration values."""
    data_dir = config_manager.get("paths.data_dir")
    assert isinstance(data_dir, str)
    assert Path(data_dir).name == "data"


def test_config_default_values(config_manager):
    """Test getting configuration with default values."""
    # Non-existent key with default
    assert config_manager.get("nonexistent", "default") == "default"
    
    # Nested non-existent key with default
    assert config_manager.get("some.nested.key", 123) == 123


def test_config_update(config_manager):
    """Test updating configuration values."""
    new_name = "updated-autorag"
    config_manager.update("name", new_name)
    assert config_manager.get("name") == new_name
    
    # Update nested value
    new_data_dir = "/tmp/new_data"
    config_manager.update("paths.data_dir", new_data_dir)
    assert config_manager.get("paths.data_dir") == new_data_dir


def test_invalid_config_access(config_manager):
    """Test error handling for invalid configuration access."""
    with pytest.raises(ConfigurationError):
        config_manager.get("invalid.nested.key")