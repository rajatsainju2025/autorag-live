"""Tests for structured logging and tracing."""

import json
import logging
from io import StringIO

import pytest

from autorag_live.core.logging_config import ContextFilter, JsonFormatter, request_id, user_id
from autorag_live.core.tracing import NoOpTracer, traced


def test_json_formatter():
    """Test JSON formatting of log records."""
    formatter = JsonFormatter()

    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-123"
    record.user_id = "user-456"

    output = formatter.format(record)
    obj = json.loads(output)

    assert obj["level"] == "INFO"
    assert obj["logger"] == "test.logger"
    assert obj["message"] == "Test message"
    assert obj["request_id"] == "req-123"
    assert obj["user_id"] == "user-456"


def test_context_filter():
    """Test context filter injects variables."""
    filter_obj = ContextFilter()

    # Set context
    token1 = request_id.set("req-789")
    token2 = user_id.set("user-999")

    try:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="msg",
            args=(),
            exc_info=None,
        )

        # Filter should inject context
        result = filter_obj.filter(record)

        assert result is True
        assert getattr(record, "request_id") == "req-789"
        assert getattr(record, "user_id") == "user-999"
    finally:
        request_id.reset(token1)
        user_id.reset(token2)


def test_configure_logging_json():
    """Test logging configuration with JSON format."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger("test_json")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("Test log")

    output = stream.getvalue()
    obj = json.loads(output.strip())

    assert obj["message"] == "Test log"
    assert obj["level"] == "INFO"


@pytest.mark.asyncio
async def test_traced_async_decorator():
    """Test @traced decorator on async function."""

    @traced("test_operation")
    async def async_func():
        return "result"

    result = await async_func()
    assert result == "result"


def test_traced_sync_decorator():
    """Test @traced decorator on sync function."""

    @traced("sync_operation")
    def sync_func(x: int):
        return x * 2

    result = sync_func(5)
    assert result == 10


def test_noop_tracer():
    """Test NoOpTracer when OTel not available."""
    tracer = NoOpTracer()

    with tracer.start_as_current_span("test"):
        pass  # Should not raise
