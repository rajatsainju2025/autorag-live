"""Configuration validator for AutoRAG-Live."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator


class RetrieverConfig(BaseModel):
    """Configuration for retriever components."""

    type: str
    host: str = "localhost"
    port: int = 6333
    collection: str = "documents"
    embedding_model: str = "bge-base-en-v1.5"

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class RerankConfig(BaseModel):
    """Configuration for reranking components."""

    type: str
    model: str = "bge-reranker-base"
    batch_size: int = 100
    threshold: float = 0.0

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold value."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0 and 1")
        return v


class CacheConfig(BaseModel):
    """Configuration for caching."""

    enabled: bool = True
    backend: str = "memory"
    ttl_seconds: int = 3600
    max_size: int = 10000


class PipelineConfig(BaseModel):
    """Configuration for RAG pipeline."""

    name: str
    retriever: RetrieverConfig
    reranker: Optional[RerankConfig] = None
    cache: Optional[CacheConfig] = None
    num_retrievals: int = 10
    timeout_seconds: float = 30.0

    @field_validator("num_retrievals")
    @classmethod
    def validate_num_retrievals(cls, v: int) -> int:
        """Validate number of retrievals."""
        if v < 1:
            raise ValueError("num_retrievals must be at least 1")
        return v


class ConfigValidator:
    """Validates AutoRAG-Live configuration."""

    @staticmethod
    def validate_pipeline_config(config_dict: Dict[str, Any]) -> PipelineConfig:
        """Validate pipeline configuration.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Validated PipelineConfig instance

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            return PipelineConfig(**config_dict)
        except Exception as e:
            raise ValueError(f"Invalid pipeline configuration: {e}")

    @staticmethod
    def validate_retriever_config(config_dict: Dict[str, Any]) -> RetrieverConfig:
        """Validate retriever configuration.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Validated RetrieverConfig instance

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            return RetrieverConfig(**config_dict)
        except Exception as e:
            raise ValueError(f"Invalid retriever configuration: {e}")

    @staticmethod
    def validate_all_configs(configs: List[Dict[str, Any]]) -> List[PipelineConfig]:
        """Validate multiple pipeline configurations.

        Args:
            configs: List of configuration dictionaries

        Returns:
            List of validated PipelineConfig instances

        Raises:
            ValueError: If any configuration is invalid
        """
        validated = []
        for i, config in enumerate(configs):
            try:
                validated.append(ConfigValidator.validate_pipeline_config(config))
            except ValueError as e:
                raise ValueError(f"Configuration {i} is invalid: {e}")
        return validated
