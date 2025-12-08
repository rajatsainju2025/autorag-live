"""
Flexible observation and tool response handling.

Supports structured, unstructured, and streaming observations with automatic parsing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ObservationType(Enum):
    """Types of observations from tool execution."""

    STRUCTURED = "structured"  # JSON or dict
    UNSTRUCTURED = "unstructured"  # Free text
    STREAMING = "streaming"  # Streamed content
    ERROR = "error"
    EMPTY = "empty"


@dataclass
class ParsedObservation:
    """Parsed observation from tool output."""

    observation_type: ObservationType
    raw_content: str
    parsed_content: Any
    confidence: float  # 0-1 confidence in parsing
    metadata: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        """Set default metadata."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Export observation as dictionary."""
        return {
            "type": self.observation_type.value,
            "raw": self.raw_content,
            "parsed": self.parsed_content,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class ObservationParser:
    """Parses tool outputs into structured observations."""

    def __init__(self):
        """Initialize observation parser."""
        self.logger = logging.getLogger("ObservationParser")
        self.parsers: Dict[str, Callable] = {
            "json": self._parse_json,
            "csv": self._parse_csv,
            "list": self._parse_list,
            "dict": self._parse_dict,
        }

    def parse(self, content: str, hint: Optional[str] = None) -> ParsedObservation:
        """
        Parse tool output into structured observation.

        Args:
            content: Raw tool output
            hint: Optional hint about content type

        Returns:
            Parsed observation
        """
        if not content or not content.strip():
            return ParsedObservation(
                observation_type=ObservationType.EMPTY,
                raw_content=content,
                parsed_content=None,
                confidence=1.0,
            )

        # Try JSON parsing first
        parsed = self._try_parse_json(content)
        if parsed:
            return ParsedObservation(
                observation_type=ObservationType.STRUCTURED,
                raw_content=content,
                parsed_content=parsed,
                confidence=0.95,
            )

        # Try dict-like parsing
        parsed = self._try_parse_dict_like(content)
        if parsed:
            return ParsedObservation(
                observation_type=ObservationType.STRUCTURED,
                raw_content=content,
                parsed_content=parsed,
                confidence=0.8,
            )

        # Try list parsing
        parsed = self._try_parse_list(content)
        if parsed:
            return ParsedObservation(
                observation_type=ObservationType.STRUCTURED,
                raw_content=content,
                parsed_content=parsed,
                confidence=0.7,
            )

        # Fall back to unstructured
        return ParsedObservation(
            observation_type=ObservationType.UNSTRUCTURED,
            raw_content=content,
            parsed_content=content,
            confidence=0.5,
        )

    def _try_parse_json(self, content: str) -> Optional[Any]:
        """Try to parse as JSON."""
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return None

    def _try_parse_dict_like(self, content: str) -> Optional[Dict]:
        """Try to parse dict-like format."""
        # Match key: value patterns
        pattern = r"(\w+):\s*([^,\n]+)"
        matches = re.findall(pattern, content)
        if matches:
            return {key: value.strip() for key, value in matches}
        return None

    def _try_parse_list(self, content: str) -> Optional[List]:
        """Try to parse as list."""
        lines = content.strip().split("\n")
        # If multiple lines that look like items
        if len(lines) > 2:
            cleaned_lines = [line.strip().lstrip("- • *").strip() for line in lines if line.strip()]
            if all(cleaned_lines):
                return cleaned_lines
        return None

    def _parse_json(self, content: str) -> Optional[Any]:
        """Parse JSON format."""
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return None

    def _parse_csv(self, content: str) -> Optional[List[Dict]]:
        """Parse CSV format."""
        lines = content.strip().split("\n")
        if len(lines) < 2:
            return None

        headers = lines[0].split(",")
        rows = []
        for line in lines[1:]:
            values = line.split(",")
            if len(values) == len(headers):
                rows.append(dict(zip(headers, values)))

        return rows if rows else None

    def _parse_list(self, content: str) -> Optional[List[str]]:
        """Parse list format."""
        lines = content.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

    def _parse_dict(self, content: str) -> Optional[Dict]:
        """Parse dict format."""
        try:
            # Try eval (dangerous, use carefully)
            result = eval(content)
            return result if isinstance(result, dict) else None
        except Exception:
            return None


class ObservationPostProcessor:
    """Post-processes observations for use in reasoning."""

    def __init__(self):
        """Initialize post-processor."""
        self.logger = logging.getLogger("ObservationPostProcessor")

    def extract_key_information(self, observation: ParsedObservation) -> Dict[str, Any]:
        """
        Extract key information from observation.

        Args:
            observation: Parsed observation

        Returns:
            Extracted key information
        """
        key_info = {
            "type": observation.observation_type.value,
            "confidence": observation.confidence,
        }

        if observation.observation_type == ObservationType.STRUCTURED:
            if isinstance(observation.parsed_content, dict):
                key_info["keys"] = list(observation.parsed_content.keys())
                key_info["summary"] = f"Dictionary with {len(observation.parsed_content)} keys"
            elif isinstance(observation.parsed_content, list):
                key_info["length"] = len(observation.parsed_content)
                key_info["summary"] = f"List with {len(observation.parsed_content)} items"
                if observation.parsed_content:
                    key_info["first_item"] = str(observation.parsed_content[0])[:100]
            else:
                key_info["summary"] = str(observation.parsed_content)[:200]

        elif observation.observation_type == ObservationType.UNSTRUCTURED:
            key_info["summary"] = observation.raw_content[:200]
            key_info["length"] = len(observation.raw_content)

        return key_info

    def validate_observation(self, observation: ParsedObservation) -> Tuple[bool, Optional[str]]:
        """
        Validate observation quality.

        Args:
            observation: Observation to validate

        Returns:
            (is_valid, error_message if any)
        """
        if observation.observation_type == ObservationType.EMPTY:
            return False, "Empty observation"

        if observation.confidence < 0.3:
            return False, "Very low confidence in parsing"

        if observation.observation_type == ObservationType.ERROR:
            return False, "Tool returned error"

        return True, None

    def summarize_observation(self, observation: ParsedObservation, max_length: int = 200) -> str:
        """
        Create summary of observation for inclusion in context.

        Args:
            observation: Observation to summarize
            max_length: Max summary length

        Returns:
            Summary text
        """
        if observation.observation_type == ObservationType.STRUCTURED:
            summary = f"[Structured] {str(observation.parsed_content)[:max_length]}"
        elif observation.observation_type == ObservationType.UNSTRUCTURED:
            summary = f"[Text] {observation.raw_content[:max_length]}"
        elif observation.observation_type == ObservationType.ERROR:
            summary = f"[Error] {observation.raw_content[:max_length]}"
        else:
            summary = "[Empty]"

        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return summary


class ObservationBuffer:
    """Buffers observations from multiple tool executions."""

    def __init__(self, max_observations: int = 100):
        """
        Initialize observation buffer.

        Args:
            max_observations: Max observations to buffer
        """
        self.max_observations = max_observations
        self.observations: List[ParsedObservation] = []
        self.parser = ObservationParser()
        self.post_processor = ObservationPostProcessor()
        self.logger = logging.getLogger("ObservationBuffer")

    def add_observation(self, content: str, tool_name: str = "") -> ParsedObservation:
        """
        Add parsed observation to buffer.

        Args:
            content: Raw observation content
            tool_name: Name of tool that produced observation

        Returns:
            Parsed observation
        """
        parsed = self.parser.parse(content)
        if parsed.metadata is None:
            parsed.metadata = {}
        parsed.metadata["tool"] = tool_name
        parsed.metadata["sequence"] = len(self.observations)

        self.observations.append(parsed)

        # Trim if too many
        if len(self.observations) > self.max_observations:
            self.observations = self.observations[-self.max_observations :]

        return parsed

    def get_last_observation(self) -> Optional[ParsedObservation]:
        """Get most recent observation."""
        return self.observations[-1] if self.observations else None

    def get_observations_by_tool(self, tool_name: str) -> List[ParsedObservation]:
        """Get all observations from specific tool."""
        return [o for o in self.observations if o.metadata and o.metadata.get("tool") == tool_name]

    def get_context_summary(self, max_items: int = 5) -> str:
        """
        Get summary of recent observations for context.

        Args:
            max_items: Max observations to include

        Returns:
            Context summary
        """
        recent = self.observations[-max_items:]
        summary_lines = []

        for obs in recent:
            summary = self.post_processor.summarize_observation(obs)
            summary_lines.append(summary)

        return "\n".join(summary_lines) if summary_lines else "No observations yet."

    def clear(self) -> None:
        """Clear all observations."""
        self.observations.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export buffer as dictionary."""
        return {
            "total_observations": len(self.observations),
            "observations": [o.to_dict() for o in self.observations[-10:]],  # Last 10
        }


class ToolResponseHandler:
    """Handles tool responses and converts to observations."""

    def __init__(self):
        """Initialize tool response handler."""
        self.buffer = ObservationBuffer()
        self.parser = ObservationParser()
        self.post_processor = ObservationPostProcessor()
        self.logger = logging.getLogger("ToolResponseHandler")

    def handle_response(
        self,
        tool_name: str,
        response: Any,
        expected_type: Optional[str] = None,
    ) -> Tuple[ParsedObservation, bool]:
        """
        Handle tool response and convert to observation.

        Args:
            tool_name: Name of tool
            response: Tool response
            expected_type: Expected type hint

        Returns:
            (parsed_observation, is_valid)
        """
        # Convert response to string if needed
        if isinstance(response, str):
            content = response
        elif isinstance(response, dict):
            content = json.dumps(response, indent=2)
        elif isinstance(response, list):
            content = json.dumps(response, indent=2)
        else:
            content = str(response)

        # Parse observation
        observation = self.buffer.add_observation(content, tool_name=tool_name)

        # Validate
        is_valid, error = self.post_processor.validate_observation(observation)

        if not is_valid:
            self.logger.warning(f"Invalid observation from {tool_name}: {error}")

        return observation, is_valid

    def get_buffer_summary(self) -> str:
        """Get summary of buffered observations."""
        return self.buffer.get_context_summary()
