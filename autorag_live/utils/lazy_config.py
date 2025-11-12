"""Lazy configuration loading to avoid parsing all options upfront."""

from typing import Any, Dict, Optional

import yaml


class LazyConfig:
    """Config that lazy-loads sections on demand."""

    def __init__(self, config_path: str):
        """Initialize lazy config."""
        self._path = config_path
        self._data: Optional[Dict] = None
        self._sections: Dict[str, Any] = {}

    def _load_file(self) -> Dict:
        """Load config file."""
        if self._data is None:
            with open(self._path) as f:
                self._data = yaml.safe_load(f) or {}
        return self._data or {}

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get config value with lazy loading.

        Args:
            section: Top-level section name
            key: Optional nested key
            default: Default value if not found

        Returns:
            Config value
        """
        if section not in self._sections:
            data = self._load_file()
            self._sections[section] = data.get(section, {})

        section_data = self._sections[section]

        if key is None:
            return section_data or default

        return section_data.get(key, default)

    def reload(self):
        """Force reload of config."""
        self._data = None
        self._sections.clear()

    def get_all(self) -> Dict:
        """Get all config data."""
        return self._load_file()
