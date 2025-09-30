"""Tests for configuration validation."""
from dataclasses import dataclass
import os
import pytest
from omegaconf import OmegaConf, DictConfig
from typing import Dict, List, Optional

from autorag_live.utils.validation import (
    validate_config,
    merge_with_env_vars,
    migrate_config,
    ConfigurationError,
)


@dataclass
class SubConfig:
    required_field: str
    optional_field: Optional[int] = None


@dataclass
class TestConfig:
    name: str
    value: float
    items: List[str]
    settings: Dict[str, int]
    sub: SubConfig
    optional: Optional[bool] = None


def test_validate_config_valid():
    """Test validation of valid configuration."""
    config = OmegaConf.create({
        "name": "test",
        "value": 1.5,
        "items": ["a", "b", "c"],
        "settings": {"x": 1, "y": 2},
        "sub": {
            "required_field": "test"
        }
    })

    # Should not raise
    validate_config(config, TestConfig)


def test_validate_config_missing_required():
    """Test validation fails with missing required field."""
    config = OmegaConf.create({
        "name": "test",
        "items": ["a", "b", "c"],
        "settings": {"x": 1},
        "sub": {
            "required_field": "test"
        }
    })

    with pytest.raises(ConfigurationError, match="Missing.*value"):
        validate_config(config, TestConfig)


def test_validate_config_wrong_type():
    """Test validation fails with wrong type."""
    config = OmegaConf.create({
        "name": "test",
        "value": "not a float",  # Wrong type
        "items": ["a", "b", "c"],
        "settings": {"x": 1},
        "sub": {
            "required_field": "test"
        }
    })

    with pytest.raises(ConfigurationError, match="invalid type"):
        validate_config(config, TestConfig)


def test_validate_config_nested_missing():
    """Test validation fails with missing nested required field."""
    config = OmegaConf.create({
        "name": "test",
        "value": 1.5,
        "items": ["a", "b", "c"],
        "settings": {"x": 1},
        "sub": {}  # Missing required_field
    })

    with pytest.raises(ConfigurationError, match="required_field"):
        validate_config(config, TestConfig)


def test_merge_env_vars(monkeypatch):
    """Test merging environment variables."""
    config = OmegaConf.create({
        "name": "test",
        "value": 1.5,
        "items": ["a"],
        "settings": {"x": 1},
        "sub": {
            "required_field": "test"
        }
    })

    # Set environment variables
    monkeypatch.setenv("AUTORAG_NAME", "from_env")
    monkeypatch.setenv("AUTORAG_VALUE", "2.5")
    monkeypatch.setenv("AUTORAG_SETTINGS__X", "10")
    monkeypatch.setenv("AUTORAG_SUB__REQUIRED_FIELD", "from_env")

    merged = merge_with_env_vars(config)

    assert merged.name == "from_env"
    assert merged.value == 2.5
    assert merged.settings.x == 10
    assert merged.sub.required_field == "from_env"


def test_migrate_config():
    """Test configuration migration."""
    config = OmegaConf.create({
        "old_field": "test",
        "nested": "value",
        "some_field": 123
    })

    migrated = migrate_config(config, "0.1.0", "0.2.0")

    assert "new_field" in migrated
    assert migrated.new_field == "test"
    assert "parent" in migrated
    assert migrated.parent.child == "value"
    assert isinstance(migrated.some_field, str)