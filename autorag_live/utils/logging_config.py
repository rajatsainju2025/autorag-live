"""
Logging configuration for AutoRAG-Live.

This module provides centralized logging configuration with structured logging,
appropriate log levels, and consistent formatting across all modules.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class AutoRAGLogger:
    """Centralized logger configuration for AutoRAG-Live."""

    # Default logging configuration
    DEFAULT_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(name)s - %(message)s'
            },
            'json': {
                'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}',
                'datefmt': '%Y-%m-%dT%H:%M:%S%z'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'logs/autorag_live.log',
                'maxBytes': 10 * 1024 * 1024,  # 10MB
                'backupCount': 5
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        },
        'loggers': {
            'autorag_live': {
                'level': 'DEBUG',
                'handlers': ['console'],
                'propagate': False
            },
            'autorag_live.cli': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'autorag_live.retrievers': {
                'level': 'DEBUG',
                'handlers': ['console'],
                'propagate': False
            },
            'autorag_live.evals': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'autorag_live.pipeline': {
                'level': 'DEBUG',
                'handlers': ['console'],
                'propagate': False
            }
        }
    }

    _configured = False

    @classmethod
    def configure(
        cls,
        level: str = 'INFO',
        log_file: Optional[str] = None,
        json_format: bool = False,
        config_override: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Configure logging for AutoRAG-Live.

        Args:
            level: Root logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (if None, only console logging)
            json_format: Whether to use JSON format for logs
            config_override: Custom logging configuration to override defaults
        """
        if cls._configured:
            return

        config = cls.DEFAULT_CONFIG.copy()

        # Apply configuration overrides
        if config_override:
            cls._deep_update(config, config_override)

        # Set root level
        config['root']['level'] = level

        # Configure log file if specified
        if log_file:
            config['handlers']['file']['filename'] = log_file
            # Ensure logs directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Add file handler to root if not already present
            if 'file' not in config['root']['handlers']:
                config['root']['handlers'].append('file')
        else:
            # Remove file handler entirely if no log file
            if 'file' in config['handlers']:
                del config['handlers']['file']
            # Remove file handler from all loggers if no log file
            for logger_config in config['loggers'].values():
                if 'file' in logger_config['handlers']:
                    logger_config['handlers'].remove('file')
            # Also remove from root if present
            if 'file' in config['root']['handlers']:
                config['root']['handlers'].remove('file')

        # Set JSON format if requested
        if json_format:
            for handler_config in config['handlers'].values():
                if 'formatter' in handler_config:
                    handler_config['formatter'] = 'json'

        # Apply configuration
        logging.config.dictConfig(config)
        cls._configured = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a configured logger for the given name.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Configured logger instance
        """
        if not cls._configured:
            cls.configure()

        return logging.getLogger(name)

    @classmethod
    def _deep_update(cls, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update a dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                cls._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a configured logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return AutoRAGLogger.get_logger(name)


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    json_format: bool = False
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Root logging level
        log_file: Path to log file
        json_format: Whether to use JSON format
    """
    AutoRAGLogger.configure(level=level, log_file=log_file, json_format=json_format)