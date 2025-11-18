#!/usr/bin/env python3
"""
Configuration migration CLI tool for AutoRAG-Live.

This tool helps migrate configuration files between different versions,
validate configurations, and perform configuration management tasks.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

from autorag_live.types.types import ConfigurationError
from autorag_live.utils import get_logger
from autorag_live.utils.schema import AutoRAGConfig
from autorag_live.utils.validation import migrate_config, validate_config

logger = get_logger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def migrate_command(args: argparse.Namespace) -> int:
    """
    Migrate configuration from one version to another.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else None

        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return 1

        # Load configuration
        logger.info(f"Loading configuration from {input_path}")
        config = OmegaConf.load(input_path)

        # Determine current version
        dict_config = cast(DictConfig, config)
        current_version = dict_config.get("version", "0.1.0")
        target_version = args.to_version

        logger.info(f"Migrating from v{current_version} to v{target_version}")

        # Perform migration
        migrated_config = migrate_config(
            config=dict_config, from_version=current_version, to_version=target_version
        )

        # Validate migrated configuration
        if args.validate:
            logger.info("Validating migrated configuration...")
            validate_config(migrated_config, AutoRAGConfig)
            logger.info("✓ Configuration is valid")

        # Save migrated configuration
        if output_path:
            logger.info(f"Saving migrated configuration to {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(migrated_config, output_path)
        else:
            # Log to stdout
            logger.info("Migrated configuration:")
            logger.info(OmegaConf.to_yaml(migrated_config))

        logger.info("✓ Migration completed successfully")
        return 0

    except ConfigurationError as e:
        logger.error(f"Configuration migration failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during migration: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def validate_command(args: argparse.Namespace) -> int:
    """
    Validate configuration file.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        input_path = Path(args.input)

        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return 1

        # Load and validate configuration
        logger.info(f"Validating configuration from {input_path}")
        config = OmegaConf.load(input_path)

        dict_config = cast(DictConfig, config)
        validate_config(dict_config, AutoRAGConfig)

        logger.info("✓ Configuration is valid")
        return 0

    except ConfigurationError as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def info_command(args: argparse.Namespace) -> int:
    """
    Display information about configuration file.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        input_path = Path(args.input)

        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return 1

        # Load configuration
        config = OmegaConf.load(input_path)

        # Display information
        logger.info(f"Configuration Information for: {input_path}")
        logger.info("=" * 50)

        dict_config = cast(DictConfig, config)
        version = dict_config.get("version", "unknown")
        logger.info(f"Version: {version}")

        # Count sections
        sections = [k for k in config.keys() if k != "version"]
        logger.info(f"Sections: {len(sections)}")
        for section in sections:
            logger.info(f"  - {section}")

        # Display structure
        if args.show_structure:
            logger.info("Configuration Structure:")
            logger.info(OmegaConf.to_yaml(config))

        return 0

    except Exception as e:
        print(f"ERROR: Failed to read configuration: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def init_command(args: argparse.Namespace) -> int:
    """
    Initialize a new configuration file.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        output_path = Path(args.output)

        if output_path.exists() and not args.force:
            print(f"ERROR: Output file already exists: {output_path}")
            print("Use --force to overwrite")
            return 1

        # Create default configuration
        default_config = OmegaConf.create(
            {
                "version": "0.1.0",
                "debug": False,
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "file": None,
                    "rotation": True,
                    "max_size": "10MB",
                },
                "data": {
                    "cache_dir": "${oc.env:HOME}/.autorag/cache",
                    "temp_dir": "/tmp/autorag",
                    "max_cache_size": "1GB",
                    "cleanup_on_exit": True,
                },
                "retrieval": {
                    "default_top_k": 10,
                    "timeout_seconds": 30.0,
                    "batch_size": 100,
                    "cache_embeddings": True,
                },
                "evaluation": {
                    "metrics": ["accuracy", "f1_score"],
                    "batch_size": 32,
                    "timeout_seconds": 120.0,
                },
                "pipeline": {"max_workers": 4, "chunk_size": 1000},
            }
        )

        # Save configuration
        output_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(default_config, output_path)

        print(f"✓ Created default configuration at: {output_path}")
        return 0

    except Exception as e:
        print(f"ERROR: Failed to create configuration: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoRAG-Live Configuration Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate configuration to new version
  %(prog)s migrate config.yaml --to-version 0.2.0 --output config_v2.yaml

  # Validate configuration
  %(prog)s validate config.yaml

  # Show configuration information
  %(prog)s info config.yaml --show-structure

  # Initialize new configuration
  %(prog)s init --output config.yaml
        """,
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate configuration between versions")
    migrate_parser.add_argument("input", help="Input configuration file")
    migrate_parser.add_argument("--to-version", required=True, help="Target version to migrate to")
    migrate_parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    migrate_parser.add_argument(
        "--validate", action="store_true", default=True, help="Validate migrated configuration"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration file")
    validate_parser.add_argument("input", help="Configuration file to validate")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show configuration information")
    info_parser.add_argument("input", help="Configuration file to analyze")
    info_parser.add_argument(
        "--show-structure", action="store_true", help="Show complete configuration structure"
    )

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize new configuration file")
    init_parser.add_argument(
        "--output",
        "-o",
        default="config.yaml",
        help="Output configuration file (default: config.yaml)",
    )
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing file")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Execute command
    if args.command == "migrate":
        return migrate_command(args)
    elif args.command == "validate":
        return validate_command(args)
    elif args.command == "info":
        return info_command(args)
    elif args.command == "init":
        return init_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
