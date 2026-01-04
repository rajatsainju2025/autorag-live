"""
Structured Output Parser for LLM Responses.

Provides robust parsing and validation of LLM outputs using Pydantic models,
with automatic retry on parse failures and schema-guided generation.

Key Features:
1. Pydantic model-based output validation
2. Automatic JSON schema generation for prompts
3. Retry with feedback on parse errors
4. Multiple extraction strategies (JSON, XML, regex)
5. Partial parsing for streaming responses

Example:
    >>> class Answer(BaseModel):
    ...     answer: str
    ...     confidence: float
    ...     sources: List[str]
    ...
    >>> parser = StructuredOutputParser(Answer)
    >>> result = await parser.parse(llm, "What is Python?")
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from autorag_live.core.protocols import BaseLLM, Message

logger = logging.getLogger(__name__)

T = TypeVar("T")
ModelT = TypeVar("ModelT")


# =============================================================================
# Schema Generation
# =============================================================================


def python_type_to_json_schema(
    python_type: Any,
    definitions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert Python type to JSON Schema.

    Args:
        python_type: Python type or type hint
        definitions: Shared definitions for recursive types

    Returns:
        JSON Schema dictionary
    """
    definitions = definitions or {}

    # Handle None
    if python_type is type(None):
        return {"type": "null"}

    # Handle basic types
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        bytes: {"type": "string", "format": "byte"},
    }
    if python_type in type_map:
        return type_map[python_type]

    # Handle origin types (generics)
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle Optional (Union with None)
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return python_type_to_json_schema(non_none[0], definitions)
        return {"anyOf": [python_type_to_json_schema(a, definitions) for a in args]}

    # Handle List
    if origin in (list, List):
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": python_type_to_json_schema(item_type, definitions),
        }

    # Handle Dict
    if origin in (dict, Dict):
        value_type = args[1] if len(args) > 1 else Any
        return {
            "type": "object",
            "additionalProperties": python_type_to_json_schema(value_type, definitions),
        }

    # Handle Enum
    if isinstance(python_type, type) and issubclass(python_type, Enum):
        return {
            "type": "string",
            "enum": [e.value for e in python_type],
        }

    # Handle Pydantic models
    try:
        from pydantic import BaseModel

        if isinstance(python_type, type) and issubclass(python_type, BaseModel):
            return model_to_json_schema(python_type)
    except ImportError:
        pass

    # Handle dataclasses
    import dataclasses

    if dataclasses.is_dataclass(python_type):
        return dataclass_to_json_schema(python_type)

    # Default
    return {"type": "string"}


def dataclass_to_json_schema(cls: type) -> Dict[str, Any]:
    """Convert dataclass to JSON Schema."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")

    properties: Dict[str, Any] = {}
    required: List[str] = []

    hints = get_type_hints(cls)
    for f in dataclasses.fields(cls):
        prop_schema = python_type_to_json_schema(hints.get(f.name, str))

        # Add description from field metadata
        if f.metadata.get("description"):
            prop_schema["description"] = f.metadata["description"]

        properties[f.name] = prop_schema

        # Check if required
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            required.append(f.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def model_to_json_schema(model_cls: type) -> Dict[str, Any]:
    """
    Convert Pydantic model to JSON Schema.

    Args:
        model_cls: Pydantic model class

    Returns:
        JSON Schema dictionary
    """
    try:
        # Pydantic v2
        return model_cls.model_json_schema()
    except AttributeError:
        pass

    try:
        # Pydantic v1
        return model_cls.schema()
    except AttributeError:
        pass

    # Manual conversion
    return dataclass_to_json_schema(model_cls)


def schema_to_prompt_description(schema: Dict[str, Any], indent: int = 0) -> str:
    """
    Convert JSON Schema to human-readable description.

    Args:
        schema: JSON Schema
        indent: Indentation level

    Returns:
        Formatted description string
    """
    prefix = "  " * indent
    lines = []

    schema_type = schema.get("type", "object")

    if schema_type == "object":
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for name, prop in properties.items():
            req_mark = "*" if name in required else ""
            prop_type = prop.get("type", "any")
            description = prop.get("description", "")

            if prop_type == "array":
                items = prop.get("items", {})
                item_type = items.get("type", "any")
                lines.append(f"{prefix}- {name}{req_mark}: List[{item_type}]")
            elif prop_type == "object":
                lines.append(f"{prefix}- {name}{req_mark}: object")
                lines.append(schema_to_prompt_description(prop, indent + 1))
            else:
                lines.append(f"{prefix}- {name}{req_mark}: {prop_type}")

            if description:
                lines.append(f"{prefix}    {description}")

    return "\n".join(lines)


# =============================================================================
# Parsing Strategies
# =============================================================================


class ParseStrategy(ABC):
    """Abstract base for parsing strategies."""

    @abstractmethod
    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to parse text into a dictionary.

        Args:
            text: Raw text to parse

        Returns:
            Parsed dictionary or None if parsing fails
        """
        ...


class JSONParseStrategy(ParseStrategy):
    """Parse JSON from text."""

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from text."""
        # Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match:
            try:
                return json.loads(obj_match.group())
            except json.JSONDecodeError:
                pass

        # Try to find JSON array
        arr_match = re.search(r"\[[\s\S]*\]", text)
        if arr_match:
            try:
                result = json.loads(arr_match.group())
                if isinstance(result, list):
                    return {"items": result}
            except json.JSONDecodeError:
                pass

        return None


class XMLParseStrategy(ParseStrategy):
    """Parse XML-style tags from text."""

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract values from XML-style tags."""
        result: Dict[str, Any] = {}

        # Find all <tag>content</tag> patterns
        pattern = r"<(\w+)>([\s\S]*?)</\1>"
        matches = re.findall(pattern, text, re.MULTILINE)

        for tag, content in matches:
            # Try to parse content as JSON
            try:
                result[tag] = json.loads(content.strip())
            except (json.JSONDecodeError, TypeError):
                result[tag] = content.strip()

        return result if result else None


class KeyValueParseStrategy(ParseStrategy):
    """Parse key: value pairs from text."""

    def __init__(self, keys: Optional[List[str]] = None):
        """Initialize with expected keys."""
        self.keys = keys or []

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract key-value pairs."""
        result: Dict[str, Any] = {}

        # Pattern for "Key: Value" or "Key = Value"
        if self.keys:
            for key in self.keys:
                pattern = rf"{key}\s*[:=]\s*(.+?)(?=\n|$)"
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    # Try to parse as JSON
                    try:
                        result[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        result[key] = value
        else:
            # Generic pattern
            pattern = r"(\w+)\s*[:=]\s*(.+?)(?=\n|$)"
            for match in re.finditer(pattern, text):
                key = match.group(1)
                value = match.group(2).strip()
                try:
                    result[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    result[key] = value

        return result if result else None


class RegexParseStrategy(ParseStrategy):
    """Parse using custom regex patterns."""

    def __init__(self, patterns: Dict[str, str]):
        """
        Initialize with field patterns.

        Args:
            patterns: Dict mapping field names to regex patterns
        """
        self.patterns = patterns

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract values using patterns."""
        result: Dict[str, Any] = {}

        for field_name, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                # Use first captured group or full match
                value = match.group(1) if match.groups() else match.group()
                result[field_name] = value.strip()

        return result if result else None


# =============================================================================
# Output Parser
# =============================================================================


@dataclass
class ParseResult(Generic[T]):
    """
    Result of parsing attempt.

    Attributes:
        success: Whether parsing succeeded
        value: Parsed value if successful
        raw_text: Original text
        error: Error message if failed
        attempts: Number of parse attempts made
    """

    success: bool
    value: Optional[T] = None
    raw_text: str = ""
    error: Optional[str] = None
    attempts: int = 1

    def unwrap(self) -> T:
        """Get value or raise error."""
        if self.success and self.value is not None:
            return self.value
        raise ValueError(self.error or "Parse failed")


class StructuredOutputParser(Generic[ModelT]):
    """
    Parses LLM outputs into structured Pydantic models.

    Features:
    - Multiple parsing strategies
    - Automatic retry with error feedback
    - Schema-guided prompting
    - Partial parsing support

    Example:
        >>> class Response(BaseModel):
        ...     answer: str
        ...     confidence: float
        ...
        >>> parser = StructuredOutputParser(Response)
        >>> result = await parser.parse_response(llm, messages)
    """

    def __init__(
        self,
        model_cls: Type[ModelT],
        *,
        max_retries: int = 3,
        strategies: Optional[List[ParseStrategy]] = None,
        include_schema_in_prompt: bool = True,
    ):
        """
        Initialize parser.

        Args:
            model_cls: Pydantic model class for output
            max_retries: Maximum retry attempts
            strategies: Custom parsing strategies
            include_schema_in_prompt: Add schema to prompts
        """
        self.model_cls = model_cls
        self.max_retries = max_retries
        self.include_schema_in_prompt = include_schema_in_prompt

        # Default strategies
        self.strategies = strategies or [
            JSONParseStrategy(),
            XMLParseStrategy(),
            KeyValueParseStrategy(list(self._get_field_names())),
        ]

        # Cache schema
        self._schema = model_to_json_schema(model_cls)

    def _get_field_names(self) -> List[str]:
        """Get field names from model."""
        try:
            # Pydantic v2
            return list(self.model_cls.model_fields.keys())
        except AttributeError:
            pass

        try:
            # Pydantic v1
            return list(self.model_cls.__fields__.keys())
        except AttributeError:
            pass

        # Dataclass
        import dataclasses

        if dataclasses.is_dataclass(self.model_cls):
            return [f.name for f in dataclasses.fields(self.model_cls)]

        return []

    def get_format_instructions(self) -> str:
        """Get formatting instructions for LLM."""
        schema_desc = schema_to_prompt_description(self._schema)

        return f"""Output your response as JSON with the following structure:

Fields (* = required):
{schema_desc}

Example format:
```json
{json.dumps(self._get_example_output(), indent=2)}
```

Respond with ONLY the JSON object, no additional text."""

    def _get_example_output(self) -> Dict[str, Any]:
        """Generate example output for schema."""
        example: Dict[str, Any] = {}
        properties = self._schema.get("properties", {})

        for name, prop in properties.items():
            prop_type = prop.get("type", "string")

            if prop_type == "string":
                example[name] = f"<{name}>"
            elif prop_type == "integer":
                example[name] = 0
            elif prop_type == "number":
                example[name] = 0.0
            elif prop_type == "boolean":
                example[name] = True
            elif prop_type == "array":
                example[name] = []
            elif prop_type == "object":
                example[name] = {}
            else:
                example[name] = None

        return example

    def _try_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Try all parsing strategies."""
        for strategy in self.strategies:
            try:
                result = strategy.parse(text)
                if result:
                    return result
            except Exception as e:
                logger.debug(f"Strategy {type(strategy).__name__} failed: {e}")
        return None

    def _validate(self, data: Dict[str, Any]) -> ModelT:
        """Validate data against model."""
        try:
            # Pydantic v2
            return self.model_cls.model_validate(data)
        except AttributeError:
            pass

        try:
            # Pydantic v1
            return self.model_cls.parse_obj(data)
        except AttributeError:
            pass

        # Direct instantiation
        return self.model_cls(**data)

    def parse_text(self, text: str) -> ParseResult[ModelT]:
        """
        Parse text into model.

        Args:
            text: Raw text to parse

        Returns:
            ParseResult with model instance or error
        """
        # Try parsing
        data = self._try_parse(text)

        if data is None:
            return ParseResult(
                success=False,
                raw_text=text,
                error="Failed to extract structured data from text",
            )

        # Validate
        try:
            instance = self._validate(data)
            return ParseResult(
                success=True,
                value=instance,
                raw_text=text,
            )
        except Exception as e:
            return ParseResult(
                success=False,
                raw_text=text,
                error=f"Validation error: {str(e)}",
            )

    async def parse_response(
        self,
        llm: BaseLLM,
        messages: List[Message],
        *,
        include_format_instructions: bool = True,
        temperature: float = 0.0,
    ) -> ParseResult[ModelT]:
        """
        Generate and parse structured response from LLM.

        Args:
            llm: Language model
            messages: Conversation messages
            include_format_instructions: Add format instructions
            temperature: LLM temperature

        Returns:
            ParseResult with parsed model
        """
        # Add format instructions
        if include_format_instructions and self.include_schema_in_prompt:
            format_msg = Message.system(self.get_format_instructions())
            messages = [format_msg] + messages

        last_error = ""

        for attempt in range(self.max_retries):
            # Add error feedback for retries
            if attempt > 0 and last_error:
                messages.append(
                    Message.user(
                        f"The previous response had a parsing error: {last_error}\n"
                        "Please respond with valid JSON matching the schema."
                    )
                )

            # Generate
            result = await llm.generate(
                messages,
                temperature=temperature,
            )

            # Parse
            parse_result = self.parse_text(result.content)

            if parse_result.success:
                parse_result.attempts = attempt + 1
                return parse_result

            last_error = parse_result.error or "Unknown error"
            logger.warning(f"Parse attempt {attempt + 1} failed: {last_error}")

        return ParseResult(
            success=False,
            raw_text=result.content if "result" in dir() else "",
            error=f"Failed after {self.max_retries} attempts: {last_error}",
            attempts=self.max_retries,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_parser(model_cls: Type[ModelT], **kwargs: Any) -> StructuredOutputParser[ModelT]:
    """Create a parser for a model class."""
    return StructuredOutputParser(model_cls, **kwargs)


async def parse_as(
    llm: BaseLLM,
    prompt: str,
    model_cls: Type[ModelT],
    **kwargs: Any,
) -> ModelT:
    """
    Parse LLM response as structured model.

    Args:
        llm: Language model
        prompt: User prompt
        model_cls: Target model class
        **kwargs: Additional parser options

    Returns:
        Parsed model instance

    Raises:
        ValueError: If parsing fails
    """
    parser = StructuredOutputParser(model_cls, **kwargs)
    result = await parser.parse_response(
        llm,
        [Message.user(prompt)],
    )
    return result.unwrap()


# =============================================================================
# Common Output Models
# =============================================================================


@dataclass
class AgentThought:
    """Structured agent thought."""

    reasoning: str
    next_action: str
    confidence: float = 1.0


@dataclass
class ToolSelection:
    """Structured tool selection."""

    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class AnswerWithSources:
    """Answer with source citations."""

    answer: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ClassificationResult:
    """Classification result."""

    label: str
    confidence: float
    reasoning: str = ""


@dataclass
class ExtractionResult:
    """Entity extraction result."""

    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)


# Pre-built parsers for common patterns
def get_thought_parser() -> StructuredOutputParser[AgentThought]:
    """Get parser for agent thoughts."""
    return StructuredOutputParser(AgentThought)


def get_tool_parser() -> StructuredOutputParser[ToolSelection]:
    """Get parser for tool selections."""
    return StructuredOutputParser(ToolSelection)


def get_answer_parser() -> StructuredOutputParser[AnswerWithSources]:
    """Get parser for answers with sources."""
    return StructuredOutputParser(AnswerWithSources)
