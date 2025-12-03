"""
Configuration validation utilities.

This module provides schema validation and version migration tools for configuration management.
It includes type checking, field validation, and environment variable merging.
"""
import collections.abc
from dataclasses import Field
from typing import Any, Dict, List, Set, Type, TypeVar, Union, cast, get_type_hints

from omegaconf import DictConfig, OmegaConf

from ..types.types import ConfigurationError

T = TypeVar("T")  # Generic type for config classes

# Cache type hints to avoid repeated lookups
_TYPE_HINTS_CACHE: Dict[Type, Dict[str, Any]] = {}


def _get_cached_type_hints(schema_cls: Type) -> Dict[str, Any]:
    """
    Get type hints with caching to avoid repeated lookups.

    Args:
        schema_cls: The class to get type hints from.

    Returns:
        Dict[str, Any]: Cached type hints for the class.
    """
    if schema_cls not in _TYPE_HINTS_CACHE:
        _TYPE_HINTS_CACHE[schema_cls] = get_type_hints(schema_cls)
    return _TYPE_HINTS_CACHE[schema_cls]


def validate_config(config: DictConfig, schema_cls: Type[T]) -> None:
    """
    Validate a configuration against its schema.

    Args:
        config: Configuration to validate
        schema_cls: Schema class to validate against

    Raises:
        ConfigurationError: If validation fails
    """
    try:
        # Check if schema_cls is a dataclass
        if not hasattr(schema_cls, "__dataclass_fields__"):
            raise ConfigurationError(f"{schema_cls.__name__} is not a dataclass")

        schema_fields = cast(Dict[str, Field], getattr(schema_cls, "__dataclass_fields__"))

        # Get type hints with caching
        type_hints = _get_cached_type_hints(schema_cls)

        # Get required fields (not Optional and no default)
        required_fields: Set[str] = set()
        for field_name, field in schema_fields.items():
            # Check if field is Optional (Union with None)
            field_type = type_hints.get(field_name)
            is_optional = False
            if field_type and hasattr(field_type, "__origin__"):
                if field_type.__origin__ is Union and hasattr(field_type, "__args__"):
                    is_optional = type(None) in field_type.__args__

            # Field is required if it has no default and is not Optional
            if not is_optional and (
                field.default is field.default_factory or field.default is None
            ):
                required_fields.add(field_name)

        # Check required fields
        missing_fields = [f for f in required_fields if f not in config]
        if missing_fields:
            raise ConfigurationError(
                f"Missing required configuration fields: {', '.join(missing_fields)}"
            )

        # Validate each field
        for field_name, field_type in type_hints.items():
            if field_name not in config:
                continue

            value = config[field_name]
            validate_field(field_name, value, field_type)

    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e!s}")


def validate_field(name: str, value: Any, expected_type: Any) -> None:
    """
    Validate a single configuration field.

    Args:
        name: Field name
        value: Field value
        expected_type: Expected type annotation

    Raises:
        ConfigurationError: If validation fails
    """
    try:
        # Handle Optional types
        if getattr(expected_type, "__origin__", None) is Union:
            if type(None) in expected_type.__args__:
                if value is None:
                    return
                non_none_types = tuple(t for t in expected_type.__args__ if t is not type(None))
                expected_type = (
                    non_none_types[0] if len(non_none_types) == 1 else Union[non_none_types]
                )

        # Handle Lists - optimized with early type check
        origin = getattr(expected_type, "__origin__", None)
        if origin is list:
            if not isinstance(value, collections.abc.Sequence):
                raise ConfigurationError(f"Field '{name}' must be a list")
            # Only validate items if type args are present
            if hasattr(expected_type, "__args__") and expected_type.__args__:
                item_type = expected_type.__args__[0]
                for i, item in enumerate(value):
                    validate_field(f"{name}[{i}]", item, item_type)
            return

        # Handle Dicts (including OmegaConf DictConfig) - optimized with early check
        if origin is dict:
            if not isinstance(value, collections.abc.Mapping):
                raise ConfigurationError(f"Field '{name}' must be a dict or DictConfig")
            # Only validate nested if type args are present
            if hasattr(expected_type, "__args__") and len(expected_type.__args__) >= 2:
                key_type, value_type = expected_type.__args__
                for k, v in value.items():
                    validate_field(f"{name}.key", k, key_type)
                    validate_field(f"{name}.{k}", v, value_type)
            return

        # Handle nested configs
        if hasattr(expected_type, "__dataclass_fields__"):
            validate_config(value, expected_type)
            return

        # Skip validation for typing.Any - cached import
        from typing import Any as TypingAny

        if expected_type is TypingAny:
            return

        # Basic type checking - optimized with hasattr check
        if not isinstance(value, expected_type):
            raise ConfigurationError(
                f"Field '{name}' has invalid type. Expected {expected_type}, got {type(value)}"
            )

    except ConfigurationError:
        # Re-raise ConfigurationError as-is
        raise
    except Exception as e:
        raise ConfigurationError(f"Validation failed for field '{name}': {str(e)}")


def merge_with_env_vars(config: DictConfig, prefix: str = "AUTORAG") -> DictConfig:
    """
    Merge configuration with environment variables.

    Environment variables override config values using the format:
    {PREFIX}_{PATH}={VALUE}

    Example:
        AUTORAG_RETRIEVAL_BM25_K1=1.5

    Args:
        config: Base configuration
        prefix: Environment variable prefix

    Returns:
        Updated configuration
    """
    import os

    try:
        # Create mutable copy
        config = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(config)))

        # Get relevant env vars
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(f"{prefix}_")}

        # Update config
        for env_key, env_value in env_vars.items():
            # Convert env var name to config path
            config_path = env_key.replace(f"{prefix}_", "").lower()
            config_path = config_path.replace("__", ".")

            # Convert value to appropriate type
            try:
                # Try to interpret as number/boolean first
                if env_value.lower() in ("true", "false"):
                    typed_value = env_value.lower() == "true"
                else:
                    try:
                        typed_value = int(env_value)
                    except ValueError:
                        try:
                            typed_value = float(env_value)
                        except ValueError:
                            typed_value = env_value
            except Exception:
                typed_value = env_value

            # Update config
            OmegaConf.update(config, config_path, typed_value)

        return config

    except Exception as e:
        raise ConfigurationError(f"Failed to merge environment variables: {str(e)}")


def migrate_config(config: DictConfig, from_version: str, to_version: str) -> DictConfig:
    """
    Migrate configuration from one version to another.

    Args:
        config: Configuration to migrate
        from_version: Current version
        to_version: Target version

    Returns:
        Migrated configuration

    Raises:
        ConfigurationError: If migration fails
    """
    try:
        # Create mutable copy
        migrated_config = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(config)))

        # Get migration steps
        migration_steps = _get_migration_steps(from_version, to_version)

        # Apply migrations
        for step in migration_steps:
            migrated_config = step(migrated_config)

        return migrated_config

    except Exception as e:
        raise ConfigurationError(
            f"Failed to migrate config from v{from_version} to v{to_version}: {str(e)}"
        )


def _get_migration_steps(from_version: str, to_version: str) -> List[Any]:
    """Get ordered list of migration steps between versions."""
    # Define version migrations
    migrations = {
        "0.1.0": {
            "0.2.0": _migrate_0_1_0_to_0_2_0,
        }
    }

    try:
        return [migrations[from_version][to_version]]
    except KeyError:
        raise ConfigurationError(f"No migration path from v{from_version} to v{to_version}")


def _migrate_0_1_0_to_0_2_0(config: DictConfig) -> DictConfig:
    """Migrate from v0.1.0 to v0.2.0 format."""
    try:
        # Create mutable copy
        migrated = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(config)))

        # Example migrations:
        # 1. Rename fields
        if "old_field" in migrated:
            migrated["new_field"] = migrated.pop("old_field")

        # 2. Update nested structure
        if "nested" in migrated:
            migrated["parent"] = {"child": migrated.pop("nested")}

        # 3. Convert values
        if "some_field" in migrated:
            migrated.some_field = str(migrated.some_field)

        return migrated

    except Exception as e:
        raise ConfigurationError(f"Failed to migrate to v0.2.0: {str(e)}")
