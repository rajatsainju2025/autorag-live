"""Structured logging configuration with context variables."""

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any, Dict, Optional

# Context variables for request tracing
request_id: ContextVar[str] = ContextVar("request_id", default="")
user_id: ContextVar[str] = ContextVar("user_id", default="")
operation: ContextVar[str] = ContextVar("operation", default="")


class ContextFilter(logging.Filter):
    """Inject context variables into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to record."""
        record.request_id = request_id.get("")
        record.user_id = user_id.get("")
        record.operation = operation.get("")
        return True


class JsonFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format record as JSON."""
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context if available
        req_id = getattr(record, "request_id", "")
        if req_id:
            log_obj["request_id"] = req_id
        uid = getattr(record, "user_id", "")
        if uid:
            log_obj["user_id"] = uid
        op = getattr(record, "operation", "")
        if op:
            log_obj["operation"] = op

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def configure_logging(
    level: int = logging.INFO,
    json_format: bool = True,
    file_path: Optional[str] = None,
) -> None:
    """Configure structured logging.

    Args:
        level: logging level (default: INFO)
        json_format: use JSON formatter (default: True)
        file_path: optional file path for file logging
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.addFilter(ContextFilter())

    if json_format:
        console_handler.setFormatter(JsonFormatter())
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(console_handler)

    # File handler (optional)
    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.addFilter(ContextFilter())
        if json_format:
            file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)
