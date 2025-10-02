"""
Shared pytest fixtures for AutoRAG-Live tests.
"""
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from omegaconf import OmegaConf

from autorag_live.types.types import DocumentText, QueryText
from autorag_live.utils.config import ConfigManager

# Test data
SAMPLE_DOCS = [
    "The sky is blue.",
    "The sun is bright.",
    "The sun in the sky is bright.",
    "We can see the shining sun, the bright sun.",
    "The quick brown fox jumps over the lazy dog.",
]

SAMPLE_QUERIES = [
    "bright sun in sky",
    "fox jumping over dog",
]


@pytest.fixture
def sample_docs() -> list[DocumentText]:
    """Sample documents for testing."""
    return SAMPLE_DOCS


@pytest.fixture
def sample_queries() -> list[QueryText]:
    """Sample queries for testing."""
    return SAMPLE_QUERIES


@pytest.fixture
def temp_config_dir() -> Generator[Path, None, None]:
    """Create a temporary config directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create config structure
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "retrieval").mkdir()
        (config_dir / "evaluation").mkdir()
        (config_dir / "pipeline").mkdir()
        (config_dir / "augmentation").mkdir()

        # Create minimal test configs
        test_config = {
            "name": "autorag-test",
            "version": "0.0.1",
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "cache_dir": str(tmp_path / ".cache"),
                "runs_dir": str(tmp_path / "runs"),
            },
        }

        OmegaConf.save(OmegaConf.create(test_config), config_dir / "config.yaml")

        yield tmp_path


@pytest.fixture
def config_manager(temp_config_dir: Path) -> Generator[ConfigManager, None, None]:
    """Get a ConfigManager instance with test configuration."""
    os.environ["AUTORAG_CONFIG_DIR"] = str(temp_config_dir / "config")
    config_manager = ConfigManager.get_instance()
    yield config_manager
    # Reset singleton for other tests
    ConfigManager._instance = None
    ConfigManager._config = None
