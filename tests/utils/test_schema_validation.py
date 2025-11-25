"""Tests for schema validation utilities."""
from omegaconf import OmegaConf

from autorag_live.utils.schema_validation import (
    COMMON_SCHEMAS,
    FastSchemaValidator,
    SchemaCache,
    get_validation_stats,
    validate_config_fast,
)


def test_schema_cache_basic():
    """Test basic schema cache functionality."""
    cache = SchemaCache(max_size=10)

    config = {"test": "value"}
    schema_hash = "test_schema"

    # Initially should return None
    assert cache.get(config, schema_hash) is None

    # Put a result
    cache.put(config, schema_hash, True)

    # Should now return the cached result
    assert cache.get(config, schema_hash) is True

    # Stats should show one hit and one miss
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["cache_size"] == 1


def test_schema_cache_eviction():
    """Test cache eviction when max size is reached."""
    cache = SchemaCache(max_size=2)

    # Fill cache to capacity
    cache.put({"a": 1}, "schema1", True)
    cache.put({"b": 2}, "schema2", False)

    assert cache.get_stats()["cache_size"] == 2

    # Add one more - should trigger eviction
    cache.put({"c": 3}, "schema3", True)

    # Should have evicted something
    assert cache.get_stats()["evictions"] > 0


def test_schema_cache_clear():
    """Test clearing the schema cache."""
    cache = SchemaCache()

    cache.put({"test": "value"}, "schema", True)
    assert cache.get_stats()["cache_size"] == 1

    cache.clear()
    stats = cache.get_stats()
    assert stats["cache_size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_fast_schema_validator_basic():
    """Test basic schema validation."""
    validator = FastSchemaValidator()

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name"],
    }

    # Valid config
    valid_config = {"name": "test", "age": 25}
    assert validator.validate_config(valid_config, schema) is True

    # Invalid config (missing required field)
    invalid_config = {"age": 25}
    assert validator.validate_config(invalid_config, schema) is False

    # Invalid config (wrong type)
    invalid_config2 = {"name": 123}
    assert validator.validate_config(invalid_config2, schema) is False


def test_fast_schema_validator_with_omegaconf():
    """Test schema validation with OmegaConf objects."""
    validator = FastSchemaValidator()

    schema = {
        "type": "object",
        "properties": {"model_name": {"type": "string"}, "batch_size": {"type": "integer"}},
        "required": ["model_name"],
    }

    # Create OmegaConf config
    config_dict = {"model_name": "test-model", "batch_size": 32}
    omega_config = OmegaConf.create(config_dict)

    assert validator.validate_config(omega_config, schema) is True


def test_fast_schema_validator_caching():
    """Test that validation results are cached."""
    validator = FastSchemaValidator()

    schema = {"type": "object", "properties": {"test": {"type": "string"}}}
    config = {"test": "value"}

    # First validation
    result1 = validator.validate_config(config, schema, use_cache=True)

    # Second validation should use cache
    result2 = validator.validate_config(config, schema, use_cache=True)

    assert result1 == result2

    # Check cache stats
    stats = validator.get_stats()
    assert stats["cache_stats"]["hits"] > 0


def test_fast_schema_validator_no_cache():
    """Test validation without caching."""
    validator = FastSchemaValidator()

    schema = {"type": "object", "properties": {"test": {"type": "string"}}}
    config = {"test": "value"}

    # Validate without caching
    result = validator.validate_config(config, schema, use_cache=False)
    assert result is True

    # Cache should be empty
    stats = validator.get_stats()
    assert stats["cache_stats"]["cache_size"] == 0


def test_global_validate_config_fast():
    """Test global validation function."""
    schema = COMMON_SCHEMAS["retriever_config"]

    # Valid config
    valid_config = {"model_name": "test-model", "batch_size": 16}
    assert validate_config_fast(valid_config, schema) is True

    # Invalid config
    invalid_config = {"batch_size": 16}  # Missing required model_name
    assert validate_config_fast(invalid_config, schema) is False


def test_common_schemas():
    """Test that common schemas are properly defined."""
    # Test retriever schema
    retriever_schema = COMMON_SCHEMAS["retriever_config"]
    assert "model_name" in retriever_schema["required"]

    # Test pipeline schema
    pipeline_schema = COMMON_SCHEMAS["pipeline_config"]
    assert "name" in pipeline_schema["required"]
    assert "retrievers" in pipeline_schema["required"]

    # Test cache schema
    cache_schema = COMMON_SCHEMAS["cache_config"]
    assert "properties" in cache_schema


def test_basic_validation_fallback():
    """Test basic validation when jsonschema is not available."""
    validator = FastSchemaValidator()

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["name"],
    }

    # Valid config
    config = {"name": "test", "count": 5}
    assert validator._basic_validation(config, schema) is True

    # Invalid - missing required
    config_invalid = {"count": 5}
    assert validator._basic_validation(config_invalid, schema) is False

    # Invalid - wrong type
    config_invalid2 = {"name": 123}
    assert validator._basic_validation(config_invalid2, schema) is False


def test_get_validation_stats():
    """Test getting global validation statistics."""
    # Perform a validation to generate stats
    schema = COMMON_SCHEMAS["cache_config"]
    config = {"max_size": 100, "enabled": True}

    validate_config_fast(config, schema)

    stats = get_validation_stats()
    assert "cache_stats" in stats
    assert isinstance(stats["cache_stats"]["hits"], int)
    assert isinstance(stats["cache_stats"]["misses"], int)
