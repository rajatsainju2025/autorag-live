"""
Schema validation with intelligent caching for configuration objects.

This module provides fast schema validation with memoization and
structural optimizations for frequently validated configurations.
"""
import hashlib
import threading
from functools import lru_cache
from typing import Any, Dict, Optional, Union

try:
    import jsonschema
    from jsonschema import validate

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    jsonschema = None
    validate = None
    JSONSCHEMA_AVAILABLE = False

from omegaconf import DictConfig, OmegaConf


class SchemaCache:
    """
    Thread-safe cache for schema validation results.

    Uses structural hashing to avoid re-validating identical configurations.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, bool] = {}
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _hash_config(self, config: Union[Dict, DictConfig]) -> str:
        """Create a structural hash of the configuration."""
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config

        # Create deterministic string representation
        config_str = str(
            sorted(config_dict.items()) if isinstance(config_dict, dict) else config_dict
        )
        return hashlib.md5(config_str.encode()).hexdigest()

    def get(self, config: Union[Dict, DictConfig], schema_hash: str) -> Optional[bool]:
        """Get cached validation result."""
        with self._lock:
            config_hash = self._hash_config(config)
            cache_key = f"{schema_hash}:{config_hash}"

            if cache_key in self._cache:
                self._stats["hits"] += 1
                return self._cache[cache_key]

            self._stats["misses"] += 1
            return None

    def put(self, config: Union[Dict, DictConfig], schema_hash: str, is_valid: bool) -> None:
        """Cache validation result."""
        with self._lock:
            config_hash = self._hash_config(config)
            cache_key = f"{schema_hash}:{config_hash}"

            # Evict oldest entries if cache is full
            if len(self._cache) >= self.max_size:
                # Remove 10% of oldest entries (simple FIFO)
                remove_count = max(1, self.max_size // 10)
                for _ in range(remove_count):
                    if self._cache:
                        self._cache.pop(next(iter(self._cache)))
                        self._stats["evictions"] += 1

            self._cache[cache_key] = is_valid

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {**self._stats, "cache_size": len(self._cache)}

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._stats = {"hits": 0, "misses": 0, "evictions": 0}


class FastSchemaValidator:
    """
    High-performance schema validator with caching and optimizations.
    """

    def __init__(self, cache_size: int = 1000):
        self._schema_cache = SchemaCache(cache_size)
        self._compiled_schemas: Dict[str, Any] = {}
        self._schema_hashes: Dict[str, str] = {}

    @lru_cache(maxsize=100)
    def _compile_schema(self, schema_str: str) -> Optional[Any]:
        """Compile and cache JSON schema."""
        if not JSONSCHEMA_AVAILABLE:
            return None

        try:
            import json

            schema = json.loads(schema_str)
            # Pre-compile validator for better performance
            return jsonschema.Draft7Validator(schema)
        except Exception:
            return None

    def _get_schema_hash(self, schema: Dict[str, Any]) -> str:
        """Get hash of schema for caching."""
        schema_str = str(sorted(schema.items()))
        return hashlib.md5(schema_str.encode()).hexdigest()

    def validate_config(
        self, config: Union[Dict, DictConfig], schema: Dict[str, Any], use_cache: bool = True
    ) -> bool:
        """
        Validate configuration against schema with caching.

        Args:
            config: Configuration to validate
            schema: JSON schema
            use_cache: Whether to use validation cache

        Returns:
            True if valid, False otherwise
        """
        if not JSONSCHEMA_AVAILABLE:
            # Fallback: basic structure validation
            return self._basic_validation(config, schema)

        schema_hash = self._get_schema_hash(schema)

        # Check cache first
        if use_cache:
            cached_result = self._schema_cache.get(config, schema_hash)
            if cached_result is not None:
                return cached_result

        # Perform validation
        try:
            validator = self._compile_schema(str(schema).replace("'", '"'))
            if validator is None:
                is_valid = self._basic_validation(config, schema)
            else:
                config_dict = (
                    OmegaConf.to_container(config, resolve=True)
                    if isinstance(config, DictConfig)
                    else config
                )
                errors = list(validator.iter_errors(config_dict))
                is_valid = len(errors) == 0

            # Cache result
            if use_cache:
                self._schema_cache.put(config, schema_hash, is_valid)

            return is_valid

        except Exception:
            # Fallback to basic validation on error
            is_valid = self._basic_validation(config, schema)
            if use_cache:
                self._schema_cache.put(config, schema_hash, is_valid)
            return is_valid

    def _basic_validation(self, config: Union[Dict, DictConfig], schema: Dict[str, Any]) -> bool:
        """
        Basic validation fallback when jsonschema is not available.

        Performs simple type and required field checking.
        """
        try:
            config_dict = (
                OmegaConf.to_container(config, resolve=True)
                if isinstance(config, DictConfig)
                else config
            )

            if not isinstance(config_dict, dict):
                return False

            # Check required properties
            required = schema.get("required", [])
            for prop in required:
                if prop not in config_dict:
                    return False

            # Basic type checking for properties
            properties = schema.get("properties", {})
            for prop, prop_schema in properties.items():
                if prop in config_dict:
                    expected_type = prop_schema.get("type")
                    value = config_dict[prop]

                    if expected_type == "string" and not isinstance(value, str):
                        return False
                    elif expected_type == "integer" and not isinstance(value, int):
                        return False
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        return False
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        return False
                    elif expected_type == "array" and not isinstance(value, list):
                        return False
                    elif expected_type == "object" and not isinstance(value, dict):
                        return False

            return True

        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "cache_stats": self._schema_cache.get_stats(),
            "compiled_schemas": len(self._compiled_schemas),
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._schema_cache.clear()
        self._compiled_schemas.clear()
        self._schema_hashes.clear()


# Global validator instance
_global_validator = FastSchemaValidator()


def validate_config_fast(
    config: Union[Dict, DictConfig], schema: Dict[str, Any], use_cache: bool = True
) -> bool:
    """
    Fast configuration validation using global validator.

    Args:
        config: Configuration to validate
        schema: JSON schema
        use_cache: Whether to use validation cache

    Returns:
        True if valid, False otherwise
    """
    return _global_validator.validate_config(config, schema, use_cache)


def get_validation_stats() -> Dict[str, Any]:
    """Get global validation statistics."""
    return _global_validator.get_stats()


# Common schema patterns for reuse
COMMON_SCHEMAS = {
    "retriever_config": {
        "type": "object",
        "properties": {
            "model_name": {"type": "string"},
            "batch_size": {"type": "integer", "minimum": 1},
            "cache_embeddings": {"type": "boolean"},
            "use_fp16": {"type": "boolean"},
        },
        "required": ["model_name"],
    },
    "pipeline_config": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "retrievers": {"type": "array"},
            "rerankers": {"type": "array"},
            "evaluators": {"type": "array"},
        },
        "required": ["name", "retrievers"],
    },
    "cache_config": {
        "type": "object",
        "properties": {
            "max_size": {"type": "integer", "minimum": 1},
            "ttl_seconds": {"type": "number", "minimum": 0},
            "enabled": {"type": "boolean"},
        },
    },
}
