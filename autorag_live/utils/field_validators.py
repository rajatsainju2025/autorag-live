"""
Field validation patterns for configuration system.

This module provides specialized validators for common field types like URLs,
file paths, email addresses, and other domain-specific values.
"""
import re
from pathlib import Path
from typing import Any, List, Optional, Union
from urllib.parse import urlparse

from ..types.types import ConfigurationError


def validate_url(url: str, schemes: Optional[List[str]] = None) -> None:
    """
    Validate URL format and scheme.

    Args:
        url: URL to validate
        schemes: Allowed schemes (default: ['http', 'https'])

    Raises:
        ConfigurationError: If URL is invalid
    """
    if schemes is None:
        schemes = ["http", "https"]

    try:
        parsed = urlparse(url)

        if not parsed.scheme:
            raise ConfigurationError(f"URL '{url}' missing scheme")

        if parsed.scheme not in schemes:
            raise ConfigurationError(
                f"URL scheme '{parsed.scheme}' not allowed. "
                f"Allowed schemes: {', '.join(schemes)}"
            )

        if not parsed.netloc:
            raise ConfigurationError(f"URL '{url}' missing netloc/domain")

    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(f"Invalid URL '{url}': {str(e)}")


def validate_email(email: str) -> None:
    """
    Validate email address format.

    Args:
        email: Email address to validate

    Raises:
        ConfigurationError: If email is invalid
    """
    # Basic email regex pattern
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    if not re.match(pattern, email):
        raise ConfigurationError(f"Invalid email format: '{email}'")


def validate_file_path(
    path: str,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    create_if_missing: bool = False,
) -> None:
    """
    Validate file or directory path.

    Args:
        path: Path to validate
        must_exist: Path must already exist
        must_be_file: Path must be a file (if exists)
        must_be_dir: Path must be a directory (if exists)
        create_if_missing: Create path if it doesn't exist

    Raises:
        ConfigurationError: If path validation fails
    """
    try:
        path_obj = Path(path).expanduser().resolve()

        # Check existence
        if must_exist and not path_obj.exists():
            raise ConfigurationError(f"Path does not exist: '{path}'")

        # Check type constraints
        if path_obj.exists():
            if must_be_file and not path_obj.is_file():
                raise ConfigurationError(f"Path is not a file: '{path}'")

            if must_be_dir and not path_obj.is_dir():
                raise ConfigurationError(f"Path is not a directory: '{path}'")

        # Create if requested
        if create_if_missing and not path_obj.exists():
            if must_be_dir or path.endswith("/"):
                path_obj.mkdir(parents=True, exist_ok=True)
            else:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                path_obj.touch()

    except PermissionError:
        raise ConfigurationError(f"Permission denied accessing path: '{path}'")
    except OSError as e:
        raise ConfigurationError(f"Invalid path '{path}': {str(e)}")


def validate_port(port: Union[int, str]) -> None:
    """
    Validate network port number.

    Args:
        port: Port number to validate

    Raises:
        ConfigurationError: If port is invalid
    """
    try:
        port_int = int(port)

        if not (1 <= port_int <= 65535):
            raise ConfigurationError(f"Port must be between 1-65535, got: {port}")

    except ValueError:
        raise ConfigurationError(f"Port must be a valid integer, got: '{port}'")


def validate_model_name(model_name: str) -> None:
    """
    Validate machine learning model name format.

    Args:
        model_name: Model name to validate

    Raises:
        ConfigurationError: If model name is invalid
    """
    # Check for empty or whitespace-only names
    if not model_name or not model_name.strip():
        raise ConfigurationError("Model name cannot be empty")

    # Check for valid characters (allow alphanumeric, hyphens, underscores, slashes, dots)
    if not re.match(r"^[a-zA-Z0-9._/-]+$", model_name):
        raise ConfigurationError(
            f"Model name '{model_name}' contains invalid characters. "
            "Only alphanumeric, hyphens, underscores, slashes, and dots allowed."
        )

    # Check length constraints
    if len(model_name) > 200:
        raise ConfigurationError(f"Model name too long (max 200 chars): '{model_name}'")


def validate_log_level(level: str) -> None:
    """
    Validate logging level.

    Args:
        level: Log level to validate

    Raises:
        ConfigurationError: If log level is invalid
    """
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    if level.upper() not in valid_levels:
        raise ConfigurationError(
            f"Invalid log level '{level}'. " f"Valid levels: {', '.join(valid_levels)}"
        )


def validate_device(device: str) -> None:
    """
    Validate device specification for ML models.

    Args:
        device: Device to validate ('cpu', 'cuda', 'cuda:0', 'mps', 'auto')

    Raises:
        ConfigurationError: If device specification is invalid
    """
    device = device.lower().strip()

    # Valid device patterns
    valid_patterns = [
        r"^cpu$",  # cpu
        r"^cuda$",  # cuda (default GPU)
        r"^cuda:\d+$",  # cuda:0, cuda:1, etc.
        r"^mps$",  # Apple Metal Performance Shaders
        r"^auto$",  # Automatic device selection
    ]

    if not any(re.match(pattern, device) for pattern in valid_patterns):
        raise ConfigurationError(
            f"Invalid device '{device}'. " "Valid formats: 'cpu', 'cuda', 'cuda:N', 'mps', 'auto'"
        )


def validate_batch_size(batch_size: Union[int, str]) -> None:
    """
    Validate batch size parameter.

    Args:
        batch_size: Batch size to validate

    Raises:
        ConfigurationError: If batch size is invalid
    """
    try:
        size = int(batch_size)

        if size <= 0:
            raise ConfigurationError(f"Batch size must be positive, got: {size}")

        if size > 10000:
            raise ConfigurationError(f"Batch size too large (max 10000), got: {size}")

    except ValueError:
        raise ConfigurationError(f"Batch size must be an integer, got: '{batch_size}'")


def validate_timeout(timeout: Union[int, float, str]) -> None:
    """
    Validate timeout value in seconds.

    Args:
        timeout: Timeout to validate

    Raises:
        ConfigurationError: If timeout is invalid
    """
    try:
        timeout_float = float(timeout)

        if timeout_float <= 0:
            raise ConfigurationError(f"Timeout must be positive, got: {timeout}")

        if timeout_float > 3600:  # 1 hour max
            raise ConfigurationError(f"Timeout too large (max 3600s), got: {timeout}")

    except ValueError:
        raise ConfigurationError(f"Timeout must be a number, got: '{timeout}'")


def validate_percentage(value: Union[int, float, str], field_name: str = "value") -> None:
    """
    Validate percentage value (0-100).

    Args:
        value: Percentage value to validate
        field_name: Field name for error messages

    Raises:
        ConfigurationError: If percentage is invalid
    """
    try:
        pct = float(value)

        if not (0 <= pct <= 100):
            raise ConfigurationError(f"{field_name} must be between 0-100%, got: {pct}")

    except ValueError:
        raise ConfigurationError(f"{field_name} must be a number, got: '{value}'")


def validate_probability(value: Union[int, float, str], field_name: str = "value") -> None:
    """
    Validate probability value (0.0-1.0).

    Args:
        value: Probability value to validate
        field_name: Field name for error messages

    Raises:
        ConfigurationError: If probability is invalid
    """
    try:
        prob = float(value)

        if not (0.0 <= prob <= 1.0):
            raise ConfigurationError(f"{field_name} must be between 0.0-1.0, got: {prob}")

    except ValueError:
        raise ConfigurationError(f"{field_name} must be a number, got: '{value}'")


def validate_memory_size(size: str) -> None:
    """
    Validate memory size specification (e.g., '1GB', '512MB').

    Args:
        size: Memory size string to validate

    Raises:
        ConfigurationError: If memory size is invalid
    """
    pattern = r"^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)$"
    match = re.match(pattern, size.upper().strip())

    if not match:
        raise ConfigurationError(
            f"Invalid memory size format '{size}'. " "Use format like '1GB', '512MB', '1.5TB'"
        )

    value, unit = match.groups()
    value = float(value)

    # Convert to bytes for validation
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}

    bytes_value = value * multipliers[unit]

    # Reasonable limits
    if bytes_value > 1024**4 * 10:  # 10 TB
        raise ConfigurationError(f"Memory size too large: '{size}'")


# Validation registry for automatic field validation
FIELD_VALIDATORS = {
    "url": validate_url,
    "email": validate_email,
    "file_path": validate_file_path,
    "port": validate_port,
    "model_name": validate_model_name,
    "log_level": validate_log_level,
    "device": validate_device,
    "batch_size": validate_batch_size,
    "timeout": validate_timeout,
    "percentage": validate_percentage,
    "probability": validate_probability,
    "memory_size": validate_memory_size,
}


def validate_field_by_pattern(field_name: str, value: Any, pattern: str) -> None:
    """
    Validate field using registered pattern validator.

    Args:
        field_name: Name of the field
        value: Value to validate
        pattern: Validation pattern name

    Raises:
        ConfigurationError: If validation fails
    """
    if pattern not in FIELD_VALIDATORS:
        raise ConfigurationError(f"Unknown validation pattern: '{pattern}'")

    try:
        FIELD_VALIDATORS[pattern](value)
    except ConfigurationError as e:
        raise ConfigurationError(f"Validation failed for field '{field_name}': {str(e)}")
    except Exception as e:
        raise ConfigurationError(f"Validation error for field '{field_name}': {str(e)}")
