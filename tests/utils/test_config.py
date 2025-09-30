"""
Tests for configuration management system.
"""
import pytest
from pathlib import Path
from omegaconf import OmegaConf

from autorag_live.utils.config import ConfigManager
from autorag_live.utils.cache import MemoryCache
from autorag_live.types.types import ConfigurationError

@pytest.fixture(autouse=True)
def reset_config_manager(monkeypatch):
    """Reset ConfigManager singleton between tests."""
    ConfigManager._instance = None
    ConfigManager._config = None
    ConfigManager._config_cache = MemoryCache()

    # Reset env vars
    monkeypatch.delenv("AUTORAG_CONFIG_DIR", raising=False)

    yield

    ConfigManager._instance = None
    ConfigManager._config = None
    ConfigManager._config_cache = MemoryCache()


def test_config_manager_singleton(config_manager):
    """Test that ConfigManager follows singleton pattern."""
    another_instance = ConfigManager.get_instance()
    assert config_manager is another_instance
    assert ConfigManager() is config_manager


def test_config_basic_access(config_manager):
    """Test basic configuration access."""
    assert config_manager.get("name") == "autorag-test"
    assert config_manager.get("version") == "0.0.1"
    paths = config_manager.get("paths")
    assert paths is not None
    assert "data_dir" in paths


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
    # Invalid keys should return None by default
    assert config_manager.get("invalid.nested.key") is None
    
    # Can provide custom default
    assert config_manager.get("invalid.nested.key", "default") == "default"


def test_config_env_override(tmp_path, monkeypatch):
    """Test configuration directory override via environment variable."""
    # Create a test config file
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    test_config = {
        "name": "env-override-test",
        "version": "1.0.0",
        "paths": {"data_dir": str(tmp_path / "data")}
    }
    config_file = config_dir / "config.yaml"
    with open(config_file, "w") as f:
        OmegaConf.save(config=test_config, f=f)
    
    # Set environment variable to override config directory
    monkeypatch.setenv("AUTORAG_CONFIG_DIR", str(config_dir))
    
    # Create new ConfigManager instance
    config_manager = ConfigManager()
    
    # Check if configuration is loaded from the override directory
    assert config_manager.get("name") == "env-override-test"
    assert config_manager.get("version") == "1.0.0"
    assert config_manager.get("paths.data_dir") == str(tmp_path / "data")


def test_config_component_merge(tmp_path, monkeypatch):
    """Test merging of component configurations."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create base config
    base_config = {
        "name": "base-config",
        "version": "1.0.0",
        "retrieval": {"k": 5}
    }
    with open(config_dir / "config.yaml", "w") as f:
        OmegaConf.save(config=base_config, f=f)
    
    # Create retrieval component config
    retrieval_dir = config_dir / "retrieval"
    retrieval_dir.mkdir()
    retrieval_config = {
        "retrieval": {
            "k": 10,
            "method": "bm25"
        }
    }
    with open(retrieval_dir / "default.yaml", "w") as f:
        OmegaConf.save(config=retrieval_config, f=f)
    
    # Set environment variable to use our test config directory
    monkeypatch.setenv("AUTORAG_CONFIG_DIR", str(config_dir))
    
    # Create new ConfigManager instance
    config_manager = ConfigManager()
    
    # Check merged configuration
    assert config_manager.get("name") == "base-config"  # From base config
    assert config_manager.get("retrieval.k") == 10  # Overridden by component config
    assert config_manager.get("retrieval.method") == "bm25"  # Added by component config


def test_config_missing_component(tmp_path, monkeypatch):
    """Test handling of missing component configuration files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create base config only
    base_config = {
        "name": "base-only",
        "version": "1.0.0"
    }
    with open(config_dir / "config.yaml", "w") as f:
        OmegaConf.save(config=base_config, f=f)
    
    # Set environment variable to use our test config directory
    monkeypatch.setenv("AUTORAG_CONFIG_DIR", str(config_dir))
    
    # Create new ConfigManager instance - should not fail
    config_manager = ConfigManager()
    
    assert config_manager.get("name") == "base-only"
    assert config_manager.get("version") == "1.0.0"


def test_config_nonexistent_dir():
    """Test error handling when config directory doesn't exist."""
    # Override with non-existent directory
    import os
    os.environ["AUTORAG_CONFIG_DIR"] = "/nonexistent/directory"
    
    with pytest.raises(ConfigurationError):
        ConfigManager()


def test_invalid_config_update(config_manager):
    """Test error handling for invalid configuration updates."""
    # Try to update a protected field
    with pytest.raises(ConfigurationError):
        config_manager.update("_protected", "value")
    
    # Try to update with invalid value type
    current_value = config_manager.get("name")
    assert isinstance(current_value, str)
    with pytest.raises(ConfigurationError):
        config_manager.update("name", {"invalid": "type"})


def test_config_caching(tmp_path, monkeypatch):
    """Test that configuration is properly cached."""
    # Create a basic config file
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    test_config = {
        "name": "cache-test",
        "version": "1.0.0"
    }
    config_file = config_dir / "config.yaml"
    with open(config_file, "w") as f:
        OmegaConf.save(config=test_config, f=f)
    
    # Set up the environment
    monkeypatch.setenv("AUTORAG_CONFIG_DIR", str(config_dir))
    
    # Create ConfigManager instance
    config_manager = ConfigManager()
    
    # First access should load from file
    val1 = config_manager.get("name")
    assert val1 == "cache-test"
    
    # Second access should use cache
    val2 = config_manager.get("name")
    assert val2 == "cache-test"
    
    # Update value should invalidate cache
    config_manager.update("name", "new-name")
    
    # Next get should reflect updated value
    val3 = config_manager.get("name")
    assert val3 == "new-name"
    
