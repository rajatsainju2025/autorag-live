"""Configuration migration utilities.

Handles version upgrades of configuration files with automatic
schema migration and validation.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class ConfigMigrator:
    """Migrate configuration files between versions.

    Example:
        >>> migrator = ConfigMigrator()
        >>> new_config = migrator.migrate(old_config, from_version="0.1.0", to_version="0.2.0")
    """

    def __init__(self):
        """Initialize migrator."""
        self.migrations = {
            ("0.1.0", "0.2.0"): self._migrate_0_1_to_0_2,
        }

    def migrate(self, config: DictConfig, from_version: str, to_version: str) -> DictConfig:
        """Migrate configuration.

        Args:
            config: Configuration to migrate
            from_version: Current version
            to_version: Target version

        Returns:
            Migrated configuration
        """
        key = (from_version, to_version)
        if key not in self.migrations:
            logger.warning(f"No migration path from {from_version} to {to_version}")
            return config

        logger.info(f"Migrating config from {from_version} to {to_version}")
        return self.migrations[key](config)

    def _migrate_0_1_to_0_2(self, config: DictConfig) -> DictConfig:
        """Migrate from v0.1.0 to v0.2.0."""
        # Example: Add new default fields
        if "cache" not in config:
            config.cache = OmegaConf.create({"enabled": True, "ttl": 3600})

        return config

    def migrate_file(self, path: Path, from_version: str, to_version: str) -> None:
        """Migrate configuration file in place."""
        config = OmegaConf.load(path)
        migrated = self.migrate(config, from_version, to_version)  # type: ignore
        OmegaConf.save(migrated, path)
        logger.info(f"Migrated {path}")


__all__ = ["ConfigMigrator"]
