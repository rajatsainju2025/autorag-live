"""
Configuration Hot-Reload System for Agentic RAG.

Provides runtime configuration management with validation,
hot-reload capability, and change notification support.
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

import yaml

T = TypeVar("T")


class ConfigFormat(str, Enum):
    """Supported configuration file formats."""

    YAML = "yaml"
    JSON = "json"
    ENV = "env"


class ValidationLevel(str, Enum):
    """Configuration validation levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_info(self, message: str) -> None:
        """Add an info message."""
        self.info.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        if not other.is_valid:
            self.is_valid = False


@dataclass
class ConfigChange:
    """Represents a configuration change."""

    path: str
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_addition(self) -> bool:
        """Check if this is an addition."""
        return self.old_value is None

    @property
    def is_removal(self) -> bool:
        """Check if this is a removal."""
        return self.new_value is None


@dataclass
class ConfigSnapshot:
    """Snapshot of configuration at a point in time."""

    config: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    checksum: str = ""

    def __post_init__(self):
        """Calculate checksum if not provided."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate config checksum."""
        content = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()


class ConfigValidator:
    """Validates configuration against a schema."""

    def __init__(self):
        """Initialize validator."""
        self._rules: list[Callable[[dict[str, Any]], ValidationResult]] = []
        self._type_validators: dict[str, Callable[[Any], bool]] = {
            "str": lambda x: isinstance(x, str),
            "int": lambda x: isinstance(x, int),
            "float": lambda x: isinstance(x, (int, float)),
            "bool": lambda x: isinstance(x, bool),
            "list": lambda x: isinstance(x, list),
            "dict": lambda x: isinstance(x, dict),
        }

    def add_rule(self, rule: Callable[[dict[str, Any]], ValidationResult]) -> None:
        """Add a validation rule."""
        self._rules.append(rule)

    def add_required_field(self, field_path: str, field_type: Optional[str] = None) -> None:
        """Add a required field rule."""

        def rule(config: dict[str, Any]) -> ValidationResult:
            result = ValidationResult(is_valid=True)
            value = self._get_nested(config, field_path)

            if value is None:
                result.add_error(f"Required field '{field_path}' is missing")
            elif field_type and field_type in self._type_validators:
                if not self._type_validators[field_type](value):
                    result.add_error(f"Field '{field_path}' must be of type {field_type}")

            return result

        self._rules.append(rule)

    def add_range_check(
        self,
        field_path: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> None:
        """Add a numeric range validation rule."""

        def rule(config: dict[str, Any]) -> ValidationResult:
            result = ValidationResult(is_valid=True)
            value = self._get_nested(config, field_path)

            if value is not None:
                if min_value is not None and value < min_value:
                    result.add_error(f"Field '{field_path}' must be >= {min_value}, got {value}")
                if max_value is not None and value > max_value:
                    result.add_error(f"Field '{field_path}' must be <= {max_value}, got {value}")

            return result

        self._rules.append(rule)

    def add_enum_check(self, field_path: str, allowed_values: list[Any]) -> None:
        """Add an enum validation rule."""

        def rule(config: dict[str, Any]) -> ValidationResult:
            result = ValidationResult(is_valid=True)
            value = self._get_nested(config, field_path)

            if value is not None and value not in allowed_values:
                result.add_error(
                    f"Field '{field_path}' must be one of {allowed_values}, got {value}"
                )

            return result

        self._rules.append(rule)

    def validate(self, config: dict[str, Any]) -> ValidationResult:
        """Validate configuration against all rules."""
        result = ValidationResult(is_valid=True)

        for rule in self._rules:
            rule_result = rule(config)
            result.merge(rule_result)

        return result

    def _get_nested(self, config: dict[str, Any], path: str) -> Any:
        """Get nested value from config using dot notation."""
        keys = path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value


class ConfigLoader:
    """Loads configuration from various sources."""

    def __init__(self):
        """Initialize loader."""
        self._loaders: dict[ConfigFormat, Callable[[str], dict[str, Any]]] = {
            ConfigFormat.YAML: self._load_yaml,
            ConfigFormat.JSON: self._load_json,
            ConfigFormat.ENV: self._load_env,
        }

    def load(
        self, source: Union[str, Path], format: Optional[ConfigFormat] = None
    ) -> dict[str, Any]:
        """Load configuration from source."""
        source_path = Path(source) if isinstance(source, str) else source

        if format is None:
            format = self._detect_format(source_path)

        loader = self._loaders.get(format)
        if not loader:
            raise ValueError(f"Unsupported config format: {format}")

        return loader(str(source_path))

    def _detect_format(self, path: Path) -> ConfigFormat:
        """Detect configuration format from file extension."""
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return ConfigFormat.YAML
        elif suffix == ".json":
            return ConfigFormat.JSON
        elif suffix == ".env":
            return ConfigFormat.ENV
        else:
            return ConfigFormat.YAML

    def _load_yaml(self, path: str) -> dict[str, Any]:
        """Load YAML configuration."""
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def _load_json(self, path: str) -> dict[str, Any]:
        """Load JSON configuration."""
        with open(path) as f:
            return json.load(f)

    def _load_env(self, path: str) -> dict[str, Any]:
        """Load environment-style configuration."""
        config: dict[str, Any] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = self._parse_env_value(value.strip())
        return config

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value."""
        value = value.strip("'\"")

        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            pass

        return value


class ConfigWatcher:
    """Watches configuration files for changes."""

    def __init__(
        self,
        config_path: Union[str, Path],
        callback: Callable[[dict[str, Any]], None],
        check_interval: float = 5.0,
    ):
        """Initialize watcher."""
        self.config_path = Path(config_path)
        self.callback = callback
        self.check_interval = check_interval

        self._loader = ConfigLoader()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_mtime = 0.0
        self._last_checksum = ""

    def start(self) -> None:
        """Start watching for changes."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop watching for changes."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.check_interval * 2)

    def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._running:
            try:
                self._check_for_changes()
            except Exception:
                pass
            time.sleep(self.check_interval)

    def _check_for_changes(self) -> None:
        """Check if configuration file has changed."""
        if not self.config_path.exists():
            return

        mtime = self.config_path.stat().st_mtime
        if mtime == self._last_mtime:
            return

        self._last_mtime = mtime

        config = self._loader.load(self.config_path)
        checksum = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

        if checksum != self._last_checksum:
            self._last_checksum = checksum
            self.callback(config)


class ConfigManager:
    """Central configuration manager with hot-reload support."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        validator: Optional[ConfigValidator] = None,
        auto_reload: bool = True,
        reload_interval: float = 5.0,
    ):
        """Initialize configuration manager."""
        self.config_path = Path(config_path) if config_path else None
        self.validator = validator
        self.auto_reload = auto_reload
        self.reload_interval = reload_interval

        self._config: dict[str, Any] = {}
        self._loader = ConfigLoader()
        self._watcher: Optional[ConfigWatcher] = None
        self._lock = threading.RLock()

        self._snapshots: list[ConfigSnapshot] = []
        self._max_snapshots = 10

        self._change_callbacks: list[Callable[[list[ConfigChange]], None]] = []

        if self.config_path and self.config_path.exists():
            self.load()

        if self.auto_reload and self.config_path:
            self._start_watcher()

    def load(self, source: Optional[Union[str, Path]] = None) -> ValidationResult:
        """Load configuration from source."""
        path = Path(source) if source else self.config_path
        if not path:
            return ValidationResult(is_valid=False, errors=["No config path specified"])

        with self._lock:
            new_config = self._loader.load(path)

            if self.validator:
                result = self.validator.validate(new_config)
                if not result.is_valid:
                    return result
            else:
                result = ValidationResult(is_valid=True)

            old_config = self._config.copy()
            self._config = new_config

            snapshot = ConfigSnapshot(
                config=new_config.copy(),
                source=str(path),
            )
            self._add_snapshot(snapshot)

            changes = self._compute_changes(old_config, new_config)
            if changes:
                self._notify_changes(changes)

            return result

    def reload(self) -> ValidationResult:
        """Reload configuration from original source."""
        return self.load()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        with self._lock:
            keys = key.split(".")
            value = self._config

            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default

            return value

    def set(self, key: str, value: Any, persist: bool = False) -> None:
        """Set configuration value."""
        with self._lock:
            old_value = self.get(key)

            keys = key.split(".")
            config = self._config

            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            config[keys[-1]] = value

            change = ConfigChange(path=key, old_value=old_value, new_value=value)
            self._notify_changes([change])

            if persist and self.config_path:
                self._save_config()

    def _save_config(self) -> None:
        """Save current configuration to file."""
        if not self.config_path:
            return

        with open(self.config_path, "w") as f:
            if self.config_path.suffix in (".yaml", ".yml"):
                yaml.dump(self._config, f, default_flow_style=False)
            else:
                json.dump(self._config, f, indent=2)

    def on_change(self, callback: Callable[[list[ConfigChange]], None]) -> None:
        """Register a callback for configuration changes."""
        self._change_callbacks.append(callback)

    def _notify_changes(self, changes: list[ConfigChange]) -> None:
        """Notify all registered callbacks of changes."""
        for callback in self._change_callbacks:
            try:
                callback(changes)
            except Exception:
                pass

    def _compute_changes(
        self, old: dict[str, Any], new: dict[str, Any], prefix: str = ""
    ) -> list[ConfigChange]:
        """Compute changes between two configurations."""
        changes = []

        all_keys = set(old.keys()) | set(new.keys())

        for key in all_keys:
            path = f"{prefix}.{key}" if prefix else key
            old_val = old.get(key)
            new_val = new.get(key)

            if old_val != new_val:
                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    changes.extend(self._compute_changes(old_val, new_val, path))
                else:
                    changes.append(ConfigChange(path=path, old_value=old_val, new_value=new_val))

        return changes

    def _add_snapshot(self, snapshot: ConfigSnapshot) -> None:
        """Add a configuration snapshot."""
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots :]

    def rollback(self, steps: int = 1) -> bool:
        """Rollback to a previous configuration."""
        with self._lock:
            if len(self._snapshots) <= steps:
                return False

            target_idx = -(steps + 1)
            target_snapshot = self._snapshots[target_idx]

            old_config = self._config.copy()
            self._config = target_snapshot.config.copy()

            changes = self._compute_changes(old_config, self._config)
            if changes:
                self._notify_changes(changes)

            return True

    def get_snapshot_history(self) -> list[dict[str, Any]]:
        """Get configuration snapshot history."""
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "source": s.source,
                "checksum": s.checksum,
            }
            for s in self._snapshots
        ]

    def _start_watcher(self) -> None:
        """Start configuration file watcher."""
        if not self.config_path:
            return

        def on_file_change(new_config: dict[str, Any]) -> None:
            with self._lock:
                if self.validator:
                    result = self.validator.validate(new_config)
                    if not result.is_valid:
                        return

                old_config = self._config.copy()
                self._config = new_config

                snapshot = ConfigSnapshot(
                    config=new_config.copy(),
                    source=str(self.config_path),
                )
                self._add_snapshot(snapshot)

                changes = self._compute_changes(old_config, new_config)
                if changes:
                    self._notify_changes(changes)

        self._watcher = ConfigWatcher(
            self.config_path,
            on_file_change,
            self.reload_interval,
        )
        self._watcher.start()

    def stop(self) -> None:
        """Stop configuration manager."""
        if self._watcher:
            self._watcher.stop()

    @property
    def config(self) -> dict[str, Any]:
        """Get current configuration."""
        with self._lock:
            return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None


def create_rag_validator() -> ConfigValidator:
    """Create a validator for RAG pipeline configuration."""
    validator = ConfigValidator()

    validator.add_required_field("llm.provider", "str")
    validator.add_required_field("llm.model", "str")
    validator.add_range_check("llm.temperature", 0.0, 2.0)
    validator.add_range_check("llm.max_tokens", 1, 100000)

    validator.add_enum_check("llm.provider", ["openai", "anthropic", "ollama", "local"])

    validator.add_range_check("retrieval.top_k", 1, 100)

    return validator


__all__ = [
    "ConfigFormat",
    "ValidationLevel",
    "ValidationResult",
    "ConfigChange",
    "ConfigSnapshot",
    "ConfigValidator",
    "ConfigLoader",
    "ConfigWatcher",
    "ConfigManager",
    "create_rag_validator",
]
