"""
Logging configuration for AutoRAG-Live.

This module provides centralized logging configuration with structured logging,
appropriate log levels, and consistent formatting across all modules.
"""

import json
import logging
import logging.config
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

# Context variables for structured logging
request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
operation_id: ContextVar[Optional[str]] = ContextVar("operation_id", default=None)


class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name

    def _get_context(self) -> Dict[str, Any]:
        """Get current logging context."""
        context = {}
        if request_id.get():
            context["request_id"] = request_id.get()
        if user_id.get():
            context["user_id"] = user_id.get()
        if session_id.get():
            context["session_id"] = session_id.get()
        if operation_id.get():
            context["operation_id"] = operation_id.get()
        return context

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with structured data."""
        self._log("INFO", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with structured data."""
        self._log("ERROR", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with structured data."""
        self._log("WARNING", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with structured data."""
        self._log("DEBUG", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with structured data."""
        self._log("CRITICAL", message, **kwargs)

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """Internal logging method."""
        context = self._get_context()
        if kwargs:
            context.update(kwargs)

        if context:
            # Add structured context to message
            extra = {"structured_data": context}
            getattr(self.logger, level.lower())(f"{message} | {json.dumps(context)}", extra=extra)
        else:
            getattr(self.logger, level.lower())(message)


class PerformanceLogger:
    """Logger for performance monitoring and metrics."""

    def __init__(self, name: str = "autorag_live.performance"):
        self.logger = StructuredLogger(name)
        self._timers: Dict[str, float] = {}
        self._counters: Dict[str, int] = {}

    @contextmanager
    def timer(self, operation: str, **context: Any) -> Generator[None, None, None]:
        """Context manager for timing operations."""
        start_time = time.time()
        operation_id.set(operation)  # Set operation context

        try:
            yield
        finally:
            duration = time.time() - start_time
            self.logger.info(
                f"Operation completed: {operation}",
                operation=operation,
                duration_ms=duration * 1000,
                **context,
            )
            operation_id.set(None)  # Clear context

    def increment_counter(self, name: str, value: int = 1, **context: Any) -> None:
        """Increment a performance counter."""
        if name not in self._counters:
            self._counters[name] = 0
        self._counters[name] += value

        self.logger.info(
            f"Counter incremented: {name}",
            counter_name=name,
            counter_value=self._counters[name],
            increment=value,
            **context,
        )

    def gauge(self, name: str, value: Union[int, float], **context: Any) -> None:
        """Record a gauge metric."""
        self.logger.info(
            f"Gauge recorded: {name}={value}", gauge_name=name, gauge_value=value, **context
        )


class AuditLogger:
    """Logger for security and audit events."""

    def __init__(self, name: str = "autorag_live.audit"):
        self.logger = StructuredLogger(name)

    def log_access(
        self, resource: str, action: str, user: Optional[str] = None, **context: Any
    ) -> None:
        """Log resource access events."""
        self.logger.info(
            f"Access: {action} on {resource}",
            event_type="access",
            resource=resource,
            action=action,
            user=user,
            **context,
        )

    def log_security_event(self, event: str, severity: str = "INFO", **context: Any) -> None:
        """Log security-related events."""
        self.logger.info(
            f"Security event: {event}", event_type="security", severity=severity, **context
        )

    def log_config_change(
        self, component: str, old_value: Any, new_value: Any, user: Optional[str] = None
    ) -> None:
        """Log configuration changes."""
        self.logger.warning(
            f"Configuration changed: {component}",
            event_type="config_change",
            component=component,
            old_value=str(old_value),
            new_value=str(new_value),
            user=user,
        )


class AutoRAGLogger:
    """Centralized logger configuration for AutoRAG-Live."""

    # Environment-specific configurations
    ENV_CONFIGS = {
        "development": {
            "level": "DEBUG",
            "handlers": ["console"],
            "json_format": False,
        },
        "testing": {
            "level": "DEBUG",
            "handlers": ["console"],
            "json_format": False,
        },
        "staging": {
            "level": "INFO",
            "handlers": ["console", "file"],
            "json_format": True,
        },
        "production": {
            "level": "WARNING",
            "handlers": ["file", "syslog"],
            "json_format": True,
        },
    }

    # Default logging configuration
    DEFAULT_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {"format": "%(levelname)s - %(name)s - %(message)s"},
            "json": {
                "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s", "structured_data": %(structured_data)s}',
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
            "performance": {
                "format": "%(asctime)s - PERFORMANCE - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "audit": {
                "format": "%(asctime)s - AUDIT - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "logs/autorag_live.log",
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
            },
            "performance_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "performance",
                "filename": "logs/performance.log",
                "maxBytes": 50 * 1024 * 1024,  # 50MB
                "backupCount": 3,
            },
            "audit_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "audit",
                "filename": "logs/audit.log",
                "maxBytes": 100 * 1024 * 1024,  # 100MB
                "backupCount": 10,
            },
            "syslog": {
                "class": "logging.handlers.SysLogHandler",
                "level": "WARNING",
                "formatter": "json",
                "address": "/dev/log" if sys.platform != "win32" else ("localhost", 514),
            },
        },
        "root": {"level": "INFO", "handlers": ["console"]},
        "loggers": {
            "autorag_live": {"level": "DEBUG", "handlers": ["console"], "propagate": False},
            "autorag_live.cli": {"level": "INFO", "handlers": ["console"], "propagate": False},
            "autorag_live.retrievers": {
                "level": "DEBUG",
                "handlers": ["console"],
                "propagate": False,
            },
            "autorag_live.evals": {"level": "INFO", "handlers": ["console"], "propagate": False},
            "autorag_live.pipeline": {
                "level": "DEBUG",
                "handlers": ["console"],
                "propagate": False,
            },
            "autorag_live.performance": {
                "level": "INFO",
                "handlers": ["performance_file"],
                "propagate": False,
            },
            "autorag_live.audit": {
                "level": "INFO",
                "handlers": ["audit_file"],
                "propagate": False,
            },
        },
    }

    _configured = False

    @classmethod
    def configure_for_environment(cls, environment: str = "development") -> None:
        """Configure logging based on environment."""
        if environment not in cls.ENV_CONFIGS:
            raise ValueError(f"Unknown environment: {environment}")

        config = cls.ENV_CONFIGS[environment]
        cls.configure(**config)

    @classmethod
    def configure(
        cls,
        level: str = "INFO",
        log_file: Optional[str] = None,
        json_format: bool = False,
        config_override: Optional[Dict[str, Any]] = None,
        enable_performance_logging: bool = True,
        enable_audit_logging: bool = True,
    ) -> None:
        """
        Configure logging for AutoRAG-Live.

        Args:
            level: Root logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (if None, only console logging)
            json_format: Whether to use JSON format for logs
            config_override: Custom logging configuration to override defaults
            enable_performance_logging: Whether to enable performance logging
            enable_audit_logging: Whether to enable audit logging
        """
        if cls._configured:
            return

        config = cls.DEFAULT_CONFIG.copy()

        # Apply configuration overrides
        if config_override:
            cls._deep_update(config, config_override)

        # Set root level
        config["root"]["level"] = level

        # Configure log file if specified
        if log_file:
            config["handlers"]["file"]["filename"] = log_file
            # Ensure logs directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Add file handler to root if not already present
            if "file" not in config["root"]["handlers"]:
                config["root"]["handlers"].append("file")
        else:
            # Remove file handler entirely if no log file
            if "file" in config["handlers"]:
                del config["handlers"]["file"]
            # Remove file handler from all loggers if no log file
            for logger_config in config["loggers"].values():
                if "file" in logger_config["handlers"]:
                    logger_config["handlers"].remove("file")
            # Also remove from root if present
            if "file" in config["root"]["handlers"]:
                config["root"]["handlers"].remove("file")

        # Ensure log directories exist for all file handlers
        for handler_name, handler_config in config["handlers"].items():
            if handler_config.get("class") == "logging.handlers.RotatingFileHandler":
                log_path = Path(handler_config["filename"])
                log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure performance and audit logging
        if not enable_performance_logging:
            if "performance_file" in config["handlers"]:
                del config["handlers"]["performance_file"]
            if "autorag_live.performance" in config["loggers"]:
                del config["loggers"]["autorag_live.performance"]

        if not enable_audit_logging:
            if "audit_file" in config["handlers"]:
                del config["handlers"]["audit_file"]
            if "autorag_live.audit" in config["loggers"]:
                del config["loggers"]["autorag_live.audit"]

        # Remove syslog handler if not available
        if "syslog" in config["handlers"]:
            try:
                import logging.handlers

                # Test if syslog is available
                if not hasattr(logging.handlers, "SysLogHandler"):
                    del config["handlers"]["syslog"]
                else:
                    # Test creating syslog handler
                    test_handler = logging.handlers.SysLogHandler(
                        address=config["handlers"]["syslog"]["address"]
                    )
                    test_handler.close()
            except (OSError, AttributeError):
                del config["handlers"]["syslog"]

        # Remove syslog from any logger handlers if syslog was removed
        if "syslog" not in config["handlers"]:
            for logger_config in config["loggers"].values():
                if "syslog" in logger_config["handlers"]:
                    logger_config["handlers"].remove("syslog")
            if "syslog" in config["root"]["handlers"]:
                config["root"]["handlers"].remove("syslog")

        # Set JSON format if requested
        if json_format:
            for handler_config in config["handlers"].values():
                if "formatter" in handler_config:
                    handler_config["formatter"] = "json"

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
    def get_structured_logger(cls, name: str) -> StructuredLogger:
        """Get a structured logger instance."""
        if not cls._configured:
            cls.configure()
        return StructuredLogger(name)

    @classmethod
    def get_performance_logger(cls) -> PerformanceLogger:
        """Get the performance logger instance."""
        if not cls._configured:
            cls.configure()
        return PerformanceLogger()

    @classmethod
    def get_audit_logger(cls) -> AuditLogger:
        """Get the audit logger instance."""
        if not cls._configured:
            cls.configure()
        return AuditLogger()

    @classmethod
    def _deep_update(cls, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Recursively update a dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                cls._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Context managers for logging context
@contextmanager
def logging_context(**kwargs):
    """Context manager for setting logging context variables."""
    tokens = {}
    for key, value in kwargs.items():
        context_var = globals().get(key)
        if context_var:
            tokens[key] = context_var.set(value)

    try:
        yield
    finally:
        for key, token in tokens.items():
            context_var = globals().get(key)
            if context_var:
                context_var.reset(token)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a configured logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return AutoRAGLogger.get_logger(name)


def get_structured_logger(name: str) -> StructuredLogger:
    """Convenience function to get a structured logger."""
    return AutoRAGLogger.get_structured_logger(name)


def get_performance_logger() -> PerformanceLogger:
    """Convenience function to get the performance logger."""
    return AutoRAGLogger.get_performance_logger()


def get_audit_logger() -> AuditLogger:
    """Convenience function to get the audit logger."""
    return AutoRAGLogger.get_audit_logger()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    environment: Optional[str] = None,
    enable_performance_logging: bool = True,
    enable_audit_logging: bool = True,
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Root logging level
        log_file: Path to log file
        json_format: Whether to use JSON format
        environment: Environment name (development, testing, staging, production)
        enable_performance_logging: Whether to enable performance logging
        enable_audit_logging: Whether to enable audit logging
    """
    if environment:
        AutoRAGLogger.configure_for_environment(environment)
    else:
        AutoRAGLogger.configure(
            level=level,
            log_file=log_file,
            json_format=json_format,
            enable_performance_logging=enable_performance_logging,
            enable_audit_logging=enable_audit_logging,
        )
