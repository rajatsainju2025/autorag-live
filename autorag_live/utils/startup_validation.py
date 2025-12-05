"""Startup validation helpers for AutoRAG-Live.

This module provides utilities to validate system state at application startup,
ensuring all required dependencies and configurations are available before
processing begins.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class StartupValidationError(Exception):
    """Raised when startup validation fails."""

    pass


def validate_python_version(min_version: Tuple[int, int] = (3, 10)) -> None:
    """Validate Python version meets minimum requirements.

    Args:
        min_version: Minimum required version as (major, minor) tuple

    Raises:
        StartupValidationError: If Python version is too old
    """
    current = sys.version_info[:2]
    if current < min_version:
        raise StartupValidationError(
            f"Python {min_version[0]}.{min_version[1]}+ required, "
            f"but running {current[0]}.{current[1]}"
        )
    logger.info(f"Python version {current[0]}.{current[1]} OK")


def validate_config_file(config_path: Path) -> DictConfig:
    """Validate configuration file exists and is valid YAML.

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded configuration

    Raises:
        StartupValidationError: If config file is missing or invalid
    """
    if not config_path.exists():
        raise StartupValidationError(f"Configuration file not found: {config_path}")

    try:
        config = OmegaConf.load(config_path)
        if not isinstance(config, DictConfig):
            raise StartupValidationError("Configuration must be a dictionary")
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        raise StartupValidationError(f"Invalid configuration file: {e}")


def validate_required_packages(packages: List[str]) -> List[str]:
    """Check if required Python packages are available.

    Args:
        packages: List of package names to check

    Returns:
        List of missing packages (empty if all available)
    """
    missing = []
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        logger.warning(f"Missing packages: {', '.join(missing)}")
    else:
        logger.info(f"All {len(packages)} required packages available")

    return missing


def validate_cache_directory(cache_dir: Path, create: bool = True) -> None:
    """Validate cache directory exists and is writable.

    Args:
        cache_dir: Path to cache directory
        create: Whether to create directory if missing

    Raises:
        StartupValidationError: If directory cannot be created or is not writable
    """
    if not cache_dir.exists():
        if create:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created cache directory: {cache_dir}")
            except Exception as e:
                raise StartupValidationError(f"Cannot create cache directory: {e}")
        else:
            raise StartupValidationError(f"Cache directory not found: {cache_dir}")

    # Test writability
    test_file = cache_dir / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
        logger.debug(f"Cache directory {cache_dir} is writable")
    except Exception as e:
        raise StartupValidationError(f"Cache directory not writable: {e}")


def validate_config_schema(config: DictConfig, required_keys: List[str]) -> None:
    """Validate configuration contains required keys.

    Args:
        config: Configuration to validate
        required_keys: List of required top-level keys (can use dot notation)

    Raises:
        StartupValidationError: If required keys are missing
    """
    missing = []
    for key in required_keys:
        try:
            OmegaConf.select(config, key, throw_on_missing=True)
        except Exception:
            missing.append(key)

    if missing:
        raise StartupValidationError(f"Configuration missing required keys: {', '.join(missing)}")

    logger.info(f"Configuration schema validated ({len(required_keys)} keys)")


def run_startup_validation(
    config_path: Optional[Path] = None,
    required_packages: Optional[List[str]] = None,
    required_config_keys: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    min_python_version: Tuple[int, int] = (3, 10),
) -> Dict[str, Any]:
    """Run complete startup validation checks.

    Args:
        config_path: Path to configuration file (optional)
        required_packages: List of required packages (optional)
        required_config_keys: List of required config keys (optional)
        cache_dir: Path to cache directory (optional)
        min_python_version: Minimum Python version required

    Returns:
        Dictionary with validation results:
        - config: Loaded configuration (if config_path provided)
        - missing_packages: List of missing packages
        - all_checks_passed: Boolean indicating overall success

    Raises:
        StartupValidationError: If critical validation checks fail
    """
    logger.info("Running startup validation checks...")
    results: Dict[str, Any] = {
        "config": None,
        "missing_packages": [],
        "all_checks_passed": True,
    }

    try:
        # Check Python version
        validate_python_version(min_python_version)

        # Load and validate configuration
        if config_path:
            config = validate_config_file(config_path)
            results["config"] = config

            # Validate config schema
            if required_config_keys:
                validate_config_schema(config, required_config_keys)

        # Check required packages
        if required_packages:
            missing = validate_required_packages(required_packages)
            results["missing_packages"] = missing
            if missing:
                results["all_checks_passed"] = False
                logger.warning(f"Startup validation passed with {len(missing)} missing packages")

        # Validate cache directory
        if cache_dir:
            validate_cache_directory(cache_dir)

        if results["all_checks_passed"]:
            logger.info("✓ All startup validation checks passed")
        else:
            logger.warning("⚠ Startup validation completed with warnings")

        return results

    except StartupValidationError as e:
        logger.error(f"✗ Startup validation failed: {e}")
        results["all_checks_passed"] = False
        raise


__all__ = [
    "StartupValidationError",
    "validate_python_version",
    "validate_config_file",
    "validate_required_packages",
    "validate_cache_directory",
    "validate_config_schema",
    "run_startup_validation",
]
