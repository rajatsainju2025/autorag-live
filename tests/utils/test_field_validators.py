"""Tests for field validation utilities."""

from unittest.mock import patch

import pytest

from autorag_live.types.types import ConfigurationError
from autorag_live.utils.field_validators import (
    FIELD_VALIDATORS,
    validate_batch_size,
    validate_device,
    validate_email,
    validate_field_by_pattern,
    validate_file_path,
    validate_log_level,
    validate_memory_size,
    validate_model_name,
    validate_percentage,
    validate_port,
    validate_probability,
    validate_timeout,
    validate_url,
)


class TestValidateURL:
    """Test URL validation functionality."""

    def test_valid_http_url(self):
        """Test valid HTTP URL."""
        validate_url("http://example.com")
        validate_url("https://example.com/path")

    def test_valid_custom_schemes(self):
        """Test URLs with custom allowed schemes."""
        validate_url("ftp://example.com", schemes=["ftp", "http"])
        # Note: file:// URLs don't have netloc, so we need to allow that
        validate_url("ftp://example.com/path", schemes=["ftp"])

    def test_invalid_url_missing_scheme(self):
        """Test URL without scheme."""
        with pytest.raises(ConfigurationError, match="missing scheme"):
            validate_url("example.com")

    def test_invalid_url_wrong_scheme(self):
        """Test URL with disallowed scheme."""
        with pytest.raises(ConfigurationError, match="scheme.*not allowed"):
            validate_url("ftp://example.com")

    def test_invalid_url_missing_netloc(self):
        """Test URL without netloc/domain."""
        with pytest.raises(ConfigurationError, match="missing netloc"):
            validate_url("http://")

    def test_malformed_url(self):
        """Test malformed URL."""
        with pytest.raises(ConfigurationError, match="missing scheme"):
            validate_url("not-a-url")


class TestValidateEmail:
    """Test email validation functionality."""

    def test_valid_emails(self):
        """Test valid email addresses."""
        validate_email("user@example.com")
        validate_email("test.email+tag@domain.co.uk")
        validate_email("user_name@sub.domain.org")

    def test_invalid_emails(self):
        """Test invalid email addresses."""
        invalid_emails = [
            "invalid",  # No @ symbol
            "user@",  # Missing domain
            "@example.com",  # Missing local part
            "user@.com",  # Domain starts with dot
            "user@",  # Missing domain (duplicate)
            "@domain.com",  # Missing local part (duplicate)
            "user@domain",  # Missing TLD
            "user name@example.com",  # Space in local part
            "user@exam ple.com",  # Space in domain
        ]

        for email in invalid_emails:
            with pytest.raises(ConfigurationError, match="Invalid email format"):
                validate_email(email)


class TestValidateFilePath:
    """Test file path validation functionality."""

    def test_valid_existing_file(self, tmp_path):
        """Test validation of existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        validate_file_path(str(test_file), must_exist=True, must_be_file=True)

    def test_valid_existing_directory(self, tmp_path):
        """Test validation of existing directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        validate_file_path(str(test_dir), must_exist=True, must_be_dir=True)

    def test_create_missing_file(self, tmp_path):
        """Test creating missing file."""
        test_file = tmp_path / "new_file.txt"

        validate_file_path(str(test_file), create_if_missing=True)

        assert test_file.exists()

    def test_create_missing_directory(self, tmp_path):
        """Test creating missing directory."""
        test_dir = tmp_path / "new_dir"

        validate_file_path(str(test_dir), create_if_missing=True, must_be_dir=True)

        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_nonexistent_file_without_create(self):
        """Test nonexistent file without create flag."""
        with pytest.raises(ConfigurationError, match="does not exist"):
            validate_file_path("/nonexistent/file.txt", must_exist=True)

    def test_file_when_directory_required(self, tmp_path):
        """Test file when directory is required."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ConfigurationError, match="not a directory"):
            validate_file_path(str(test_file), must_be_dir=True)

    def test_directory_when_file_required(self, tmp_path):
        """Test directory when file is required."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        with pytest.raises(ConfigurationError, match="not a file"):
            validate_file_path(str(test_dir), must_be_file=True)

    def test_permission_error(self):
        """Test permission error handling."""
        with patch("pathlib.Path.exists", side_effect=PermissionError):
            with pytest.raises(ConfigurationError, match="Permission denied"):
                validate_file_path("/restricted/path")


class TestValidatePort:
    """Test port validation functionality."""

    def test_valid_ports(self):
        """Test valid port numbers."""
        validate_port(80)
        validate_port("443")
        validate_port(1)
        validate_port(65535)

    def test_invalid_ports(self):
        """Test invalid port numbers."""
        invalid_ports = [0, -1, 65536, "abc", "80.5"]

        for port in invalid_ports:
            with pytest.raises(ConfigurationError):
                validate_port(port)


class TestValidateModelName:
    """Test model name validation functionality."""

    def test_valid_model_names(self):
        """Test valid model names."""
        validate_model_name("bert-base-uncased")
        validate_model_name("gpt-3.5-turbo")
        validate_model_name("sentence-transformers/all-MiniLM-L6-v2")
        validate_model_name("custom/model_v1.0")

    def test_invalid_model_names(self):
        """Test invalid model names."""
        invalid_names = [
            "",
            "   ",
            "model with spaces",
            "model@invalid",
            "model#hash",
            "a" * 201,  # Too long
        ]

        for name in invalid_names:
            with pytest.raises(ConfigurationError):
                validate_model_name(name)


class TestValidateLogLevel:
    """Test log level validation functionality."""

    def test_valid_log_levels(self):
        """Test valid log levels."""
        validate_log_level("DEBUG")
        validate_log_level("info")
        validate_log_level("WARNING")
        validate_log_level("Error")
        validate_log_level("critical")

    def test_invalid_log_levels(self):
        """Test invalid log levels."""
        invalid_levels = ["trace", "fatal", "verbose", ""]

        for level in invalid_levels:
            with pytest.raises(ConfigurationError, match="Invalid log level"):
                validate_log_level(level)


class TestValidateDevice:
    """Test device validation functionality."""

    def test_valid_devices(self):
        """Test valid device specifications."""
        validate_device("cpu")
        validate_device("CUDA")
        validate_device("cuda:0")
        validate_device("cuda:1")
        validate_device("mps")
        validate_device("auto")

    def test_invalid_devices(self):
        """Test invalid device specifications."""
        invalid_devices = ["gpu", "cuda:abc", "tpu", ""]

        for device in invalid_devices:
            with pytest.raises(ConfigurationError, match="Invalid device"):
                validate_device(device)


class TestValidateBatchSize:
    """Test batch size validation functionality."""

    def test_valid_batch_sizes(self):
        """Test valid batch sizes."""
        validate_batch_size(1)
        validate_batch_size("32")
        validate_batch_size(10000)

    def test_invalid_batch_sizes(self):
        """Test invalid batch sizes."""
        invalid_sizes = [0, -1, "abc", 10001]

        for size in invalid_sizes:
            with pytest.raises(ConfigurationError):
                validate_batch_size(size)


class TestValidateTimeout:
    """Test timeout validation functionality."""

    def test_valid_timeouts(self):
        """Test valid timeout values."""
        validate_timeout(1)
        validate_timeout("30.5")
        validate_timeout(3600)

    def test_invalid_timeouts(self):
        """Test invalid timeout values."""
        invalid_timeouts = [0, -1, "abc", 3601]

        for timeout in invalid_timeouts:
            with pytest.raises(ConfigurationError):
                validate_timeout(timeout)


class TestValidatePercentage:
    """Test percentage validation functionality."""

    def test_valid_percentages(self):
        """Test valid percentage values."""
        validate_percentage(0)
        validate_percentage("50.5")
        validate_percentage(100)

    def test_invalid_percentages(self):
        """Test invalid percentage values."""
        invalid_percentages = [-1, 101, "abc"]

        for pct in invalid_percentages:
            with pytest.raises(ConfigurationError):
                validate_percentage(pct)

    def test_custom_field_name(self):
        """Test custom field name in error messages."""
        with pytest.raises(ConfigurationError, match="Custom field"):
            validate_percentage(150, field_name="Custom field")


class TestValidateProbability:
    """Test probability validation functionality."""

    def test_valid_probabilities(self):
        """Test valid probability values."""
        validate_probability(0.0)
        validate_probability("0.5")
        validate_probability(1.0)

    def test_invalid_probabilities(self):
        """Test invalid probability values."""
        invalid_probabilities = [-0.1, 1.1, "abc"]

        for prob in invalid_probabilities:
            with pytest.raises(ConfigurationError):
                validate_probability(prob)


class TestValidateMemorySize:
    """Test memory size validation functionality."""

    def test_valid_memory_sizes(self):
        """Test valid memory size specifications."""
        validate_memory_size("1GB")
        validate_memory_size("512MB")
        validate_memory_size("1.5TB")
        validate_memory_size("1024KB")
        validate_memory_size("100B")

    def test_invalid_memory_sizes(self):
        """Test invalid memory size specifications."""
        invalid_sizes = ["1", "GB", "1XB", "abc", "11PB"]

        for size in invalid_sizes:
            with pytest.raises(ConfigurationError, match="Invalid memory size"):
                validate_memory_size(size)


class TestValidateFieldByPattern:
    """Test field validation by pattern functionality."""

    def test_valid_field_validation(self):
        """Test successful field validation."""
        validate_field_by_pattern("test_url", "http://example.com", "url")
        validate_field_by_pattern("test_email", "user@example.com", "email")

    def test_invalid_pattern(self):
        """Test unknown validation pattern."""
        with pytest.raises(ConfigurationError, match="Unknown validation pattern"):
            validate_field_by_pattern("test_field", "value", "unknown_pattern")

    def test_validation_failure(self):
        """Test validation failure propagation."""
        with pytest.raises(ConfigurationError, match="Validation failed for field"):
            validate_field_by_pattern("test_url", "invalid-url", "url")


class TestFieldValidatorsRegistry:
    """Test field validators registry."""

    def test_all_validators_registered(self):
        """Test that all expected validators are registered."""
        expected_validators = [
            "url",
            "email",
            "file_path",
            "port",
            "model_name",
            "log_level",
            "device",
            "batch_size",
            "timeout",
            "percentage",
            "probability",
            "memory_size",
        ]

        for validator_name in expected_validators:
            assert validator_name in FIELD_VALIDATORS
            assert callable(FIELD_VALIDATORS[validator_name])

    def test_registry_integrity(self):
        """Test that registry contains valid functions."""
        for name, validator in FIELD_VALIDATORS.items():
            assert callable(validator)
            assert name == name.lower()  # All names should be lowercase
