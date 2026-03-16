"""Tests for configuration validator."""

import pytest
from pydantic import ValidationError

from autorag_live.config.validator import (
    CacheConfig,
    ConfigValidator,
    PipelineConfig,
    RerankConfig,
    RetrieverConfig,
)


class TestRetrieverConfig:
    """Tests for RetrieverConfig."""

    def test_valid_config(self):
        """Test valid retriever configuration."""
        config = RetrieverConfig(type="qdrant")
        assert config.type == "qdrant"
        assert config.host == "localhost"
        assert config.port == 6333

    def test_custom_port(self):
        """Test retriever with custom port."""
        config = RetrieverConfig(type="qdrant", port=9200)
        assert config.port == 9200

    def test_invalid_port_too_high(self):
        """Test invalid port number."""
        with pytest.raises(ValidationError):
            RetrieverConfig(type="qdrant", port=99999)

    def test_invalid_port_too_low(self):
        """Test invalid port below range."""
        with pytest.raises(ValidationError):
            RetrieverConfig(type="qdrant", port=0)


class TestRerankConfig:
    """Tests for RerankConfig."""

    def test_valid_config(self):
        """Test valid rerank configuration."""
        config = RerankConfig(type="bge")
        assert config.type == "bge"
        assert config.model == "bge-reranker-base"
        assert config.batch_size == 100
        assert config.threshold == 0.0

    def test_custom_threshold(self):
        """Test rerank with custom threshold."""
        config = RerankConfig(type="bge", threshold=0.5)
        assert config.threshold == 0.5

    def test_invalid_threshold_negative(self):
        """Test invalid negative threshold."""
        with pytest.raises(ValidationError):
            RerankConfig(type="bge", threshold=-0.1)

    def test_invalid_threshold_too_high(self):
        """Test invalid threshold above 1."""
        with pytest.raises(ValidationError):
            RerankConfig(type="bge", threshold=1.5)


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_config(self):
        """Test default cache configuration."""
        config = CacheConfig()
        assert config.enabled is True
        assert config.backend == "memory"
        assert config.ttl_seconds == 3600
        assert config.max_size == 10000

    def test_redis_backend(self):
        """Test cache with redis backend."""
        config = CacheConfig(enabled=True, backend="redis", ttl_seconds=7200)
        assert config.backend == "redis"
        assert config.ttl_seconds == 7200


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_valid_minimal_config(self):
        """Test minimal valid pipeline config."""
        config = PipelineConfig(
            name="test-pipeline",
            retriever=RetrieverConfig(type="qdrant"),
        )
        assert config.name == "test-pipeline"
        assert config.retriever.type == "qdrant"
        assert config.reranker is None
        assert config.cache is None
        assert config.num_retrievals == 10

    def test_valid_full_config(self):
        """Test full pipeline config with all components."""
        config = PipelineConfig(
            name="full-pipeline",
            retriever=RetrieverConfig(type="elasticsearch"),
            reranker=RerankConfig(type="bge"),
            cache=CacheConfig(backend="redis"),
            num_retrievals=20,
        )
        assert config.num_retrievals == 20
        assert config.reranker is not None
        assert config.cache is not None
        assert config.cache.backend == "redis"

    def test_invalid_num_retrievals(self):
        """Test invalid number of retrievals."""
        with pytest.raises(ValidationError):
            PipelineConfig(
                name="test",
                retriever=RetrieverConfig(type="qdrant"),
                num_retrievals=0,
            )


class TestConfigValidator:
    """Tests for ConfigValidator class."""

    def test_validate_retriever_config_valid(self):
        """Test validation of valid retriever config."""
        config_dict = {
            "type": "qdrant",
        }
        config = ConfigValidator.validate_retriever_config(config_dict)
        assert isinstance(config, RetrieverConfig)
        assert config.type == "qdrant"

    def test_validate_retriever_config_invalid(self):
        """Test validation of invalid retriever config."""
        config_dict = {"type": "qdrant", "port": 99999}
        with pytest.raises(ValueError):
            ConfigValidator.validate_retriever_config(config_dict)

    def test_validate_pipeline_config_valid(self):
        """Test validation of valid pipeline config."""
        config_dict = {
            "name": "test-pipeline",
            "retriever": {"type": "elasticsearch"},
        }
        config = ConfigValidator.validate_pipeline_config(config_dict)
        assert isinstance(config, PipelineConfig)
        assert config.name == "test-pipeline"

    def test_validate_pipeline_config_invalid(self):
        """Test validation of invalid pipeline config."""
        config_dict = {
            "name": "test",
            "retriever": {"type": "qdrant", "port": -1},
        }
        with pytest.raises(ValueError):
            ConfigValidator.validate_pipeline_config(config_dict)

    def test_validate_all_configs(self):
        """Test validation of multiple configs."""
        configs = [
            {
                "name": "pipeline1",
                "retriever": {"type": "qdrant"},
            },
            {
                "name": "pipeline2",
                "retriever": {"type": "elasticsearch"},
            },
        ]
        validated = ConfigValidator.validate_all_configs(configs)
        assert len(validated) == 2
        assert all(isinstance(c, PipelineConfig) for c in validated)
