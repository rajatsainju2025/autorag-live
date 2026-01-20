"""
Prompt building and engineering utilities for AutoRAG-Live.

Provides tools for constructing, templating, and optimizing
prompts for RAG systems.

Features:
- Prompt templates with variable substitution
- Context formatting for different LLMs
- Prompt optimization utilities
- Few-shot example management
- System prompt construction

Example usage:
    >>> builder = PromptBuilder()
    >>> prompt = builder.build_rag_prompt(
    ...     query="What is machine learning?",
    ...     contexts=["ML is a subset of AI...", "Deep learning..."],
    ... )
"""

from __future__ import annotations

import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from string import Template
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, Union

logger = logging.getLogger(__name__)

# TypeVar for generic operations
T = TypeVar("T")


class PromptFormat(str, Enum):
    """Supported prompt formats."""

    CHAT = "chat"
    COMPLETION = "completion"
    INSTRUCT = "instruct"


class LLMProvider(str, Enum):
    """LLM providers with different formatting needs."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LLAMA = "llama"
    MISTRAL = "mistral"
    GENERIC = "generic"


@dataclass
class PromptTemplate:
    """A reusable prompt template."""

    name: str
    template: str
    description: str = ""

    # Template variables
    required_vars: List[str] = field(default_factory=list)
    optional_vars: List[str] = field(default_factory=list)
    default_values: Dict[str, str] = field(default_factory=dict)

    # Metadata
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)

    def render(self, **kwargs: Any) -> str:
        """
        Render template with variables.

        Args:
            **kwargs: Variable values

        Returns:
            Rendered prompt
        """
        # Apply defaults
        values = {**self.default_values, **kwargs}

        # Check required variables
        missing = [v for v in self.required_vars if v not in values]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Render using string.Template
        tmpl = Template(self.template)
        return tmpl.safe_substitute(values)

    def validate(self) -> List[str]:
        """Validate template."""
        issues = []

        # Check that required vars appear in template
        for var in self.required_vars:
            if f"${var}" not in self.template and f"${{{var}}}" not in self.template:
                issues.append(f"Required variable '{var}' not found in template")

        return issues


@dataclass
class FewShotExample:
    """A few-shot example for prompts."""

    input_text: str
    output_text: str

    # Optional metadata
    explanation: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def format(self, input_prefix: str = "Input:", output_prefix: str = "Output:") -> str:
        """Format example for prompt inclusion."""
        parts = [f"{input_prefix} {self.input_text}"]
        if self.explanation:
            parts.append(f"Reasoning: {self.explanation}")
        parts.append(f"{output_prefix} {self.output_text}")
        return "\n".join(parts)


class ContextFormatter:
    """Format context documents for prompts."""

    def __init__(
        self,
        max_context_length: int = 4000,
        separator: str = "\n\n---\n\n",
        enumeration: bool = True,
        truncate_strategy: str = "end",
    ):
        """
        Initialize context formatter.

        Args:
            max_context_length: Maximum total context length
            separator: Separator between contexts
            enumeration: Whether to number contexts
            truncate_strategy: How to truncate ('end', 'middle', 'smart')
        """
        self.max_context_length = max_context_length
        self.separator = separator
        self.enumeration = enumeration
        self.truncate_strategy = truncate_strategy

    def format(
        self,
        contexts: List[str],
        scores: Optional[List[float]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Format contexts for prompt inclusion.

        Args:
            contexts: List of context passages
            scores: Optional relevance scores
            metadata: Optional metadata per context

        Returns:
            Formatted context string
        """
        if not contexts:
            return "No context available."

        formatted_parts = []

        for i, context in enumerate(contexts):
            parts = []

            # Add enumeration
            if self.enumeration:
                parts.append(f"[{i + 1}]")

            # Add score if available
            if scores and i < len(scores):
                parts.append(f"(score: {scores[i]:.3f})")

            # Add metadata if available
            if metadata and i < len(metadata):
                meta = metadata[i]
                if "source" in meta:
                    parts.append(f"Source: {meta['source']}")

            # Add context text
            if parts:
                header = " ".join(parts)
                formatted_parts.append(f"{header}\n{context}")
            else:
                formatted_parts.append(context)

        # Join and truncate
        result = self.separator.join(formatted_parts)
        return self._truncate(result)

    def _truncate(self, text: str) -> str:
        """Truncate text to max length."""
        if len(text) <= self.max_context_length:
            return text

        if self.truncate_strategy == "end":
            return text[: self.max_context_length] + "..."

        elif self.truncate_strategy == "middle":
            half = self.max_context_length // 2
            return text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]

        elif self.truncate_strategy == "smart":
            # Try to truncate at sentence boundaries
            truncated = text[: self.max_context_length]
            last_period = truncated.rfind(".")
            if last_period > self.max_context_length * 0.8:
                return truncated[: last_period + 1]
            return truncated + "..."

        return text[: self.max_context_length]


class SystemPromptBuilder:
    """Build system prompts for different use cases."""

    # Default system prompts
    DEFAULT_RAG = """You are a helpful assistant that answers questions based on the provided context.
Use the context to answer the user's question accurately and concisely.
If the answer cannot be found in the context, say so clearly.
Always cite your sources when possible using [1], [2], etc."""

    DEFAULT_CONVERSATIONAL = """You are a helpful conversational assistant.
You maintain context from the conversation history and provide relevant, helpful responses.
Be concise but thorough in your answers."""

    DEFAULT_ANALYTICAL = """You are an analytical assistant that carefully examines information.
Break down complex problems into smaller parts.
Provide structured, well-reasoned responses with clear logic.
Acknowledge uncertainty when appropriate."""

    def __init__(self):
        """Initialize system prompt builder."""
        self._templates: Dict[str, str] = {
            "rag": self.DEFAULT_RAG,
            "conversational": self.DEFAULT_CONVERSATIONAL,
            "analytical": self.DEFAULT_ANALYTICAL,
        }

    def build(
        self,
        template: str = "rag",
        custom_instructions: Optional[str] = None,
        persona: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        output_format: Optional[str] = None,
    ) -> str:
        """
        Build a system prompt.

        Args:
            template: Base template name
            custom_instructions: Additional instructions
            persona: Custom persona description
            constraints: List of constraints
            output_format: Desired output format

        Returns:
            System prompt string
        """
        parts = []

        # Start with persona or template
        if persona:
            parts.append(persona)
        elif template in self._templates:
            parts.append(self._templates[template])
        else:
            parts.append(self.DEFAULT_RAG)

        # Add custom instructions
        if custom_instructions:
            parts.append(f"\n{custom_instructions}")

        # Add constraints
        if constraints:
            constraint_text = "\n".join(f"- {c}" for c in constraints)
            parts.append(f"\nConstraints:\n{constraint_text}")

        # Add output format
        if output_format:
            parts.append(f"\nOutput Format: {output_format}")

        return "\n".join(parts)

    def register_template(self, name: str, template: str) -> None:
        """Register a custom template."""
        self._templates[name] = template

    def get_template(self, name: str) -> Optional[str]:
        """Get a template by name."""
        return self._templates.get(name)


class PromptBuilder:
    """
    Main prompt building interface.

    Example:
        >>> builder = PromptBuilder()
        >>>
        >>> # Build RAG prompt
        >>> prompt = builder.build_rag_prompt(
        ...     query="What is Python?",
        ...     contexts=["Python is a programming language..."],
        ...     system_prompt="You are a helpful assistant."
        ... )
        >>>
        >>> # Build with few-shot examples
        >>> examples = [
        ...     FewShotExample("What is 2+2?", "4"),
        ...     FewShotExample("What is 3*3?", "9"),
        ... ]
        >>> prompt = builder.build_with_examples(
        ...     query="What is 5*5?",
        ...     examples=examples
        ... )
    """

    # Default templates
    RAG_TEMPLATE = """Context:
$context

Question: $query

Answer the question based on the context above. If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Answer:"""

    QA_TEMPLATE = """Question: $query

Please provide a clear and concise answer.

Answer:"""

    SUMMARY_TEMPLATE = """Please summarize the following text:

$text

Summary:"""

    def __init__(
        self,
        context_formatter: Optional[ContextFormatter] = None,
        system_builder: Optional[SystemPromptBuilder] = None,
        default_provider: LLMProvider = LLMProvider.GENERIC,
    ):
        """
        Initialize prompt builder.

        Args:
            context_formatter: Context formatting helper
            system_builder: System prompt builder
            default_provider: Default LLM provider
        """
        self.context_formatter = context_formatter or ContextFormatter()
        self.system_builder = system_builder or SystemPromptBuilder()
        self.default_provider = default_provider

        # Template registry
        self._templates: Dict[str, PromptTemplate] = {}
        self._register_default_templates()

    def _register_default_templates(self) -> None:
        """Register default templates."""
        self._templates["rag"] = PromptTemplate(
            name="rag",
            template=self.RAG_TEMPLATE,
            required_vars=["query", "context"],
            description="Standard RAG prompt",
        )

        self._templates["qa"] = PromptTemplate(
            name="qa",
            template=self.QA_TEMPLATE,
            required_vars=["query"],
            description="Simple Q&A prompt",
        )

        self._templates["summary"] = PromptTemplate(
            name="summary",
            template=self.SUMMARY_TEMPLATE,
            required_vars=["text"],
            description="Summarization prompt",
        )

    def build_rag_prompt(
        self,
        query: str,
        contexts: List[str],
        system_prompt: Optional[str] = None,
        template: Optional[str] = None,
        scores: Optional[List[float]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a RAG prompt.

        Args:
            query: User query
            contexts: Retrieved context passages
            system_prompt: Optional system prompt
            template: Template name or custom template
            scores: Relevance scores for contexts
            metadata: Metadata for contexts

        Returns:
            Complete prompt string
        """
        # Format contexts
        formatted_context = self.context_formatter.format(
            contexts, scores=scores, metadata=metadata
        )

        # Get template
        if template and template in self._templates:
            tmpl = self._templates[template]
            prompt = tmpl.render(query=query, context=formatted_context)
        else:
            prompt = self.RAG_TEMPLATE.replace("$query", query).replace(
                "$context", formatted_context
            )

        # Add system prompt if provided
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"

        return prompt

    def build_with_examples(
        self,
        query: str,
        examples: List[FewShotExample],
        system_prompt: Optional[str] = None,
        input_prefix: str = "Input:",
        output_prefix: str = "Output:",
    ) -> str:
        """
        Build prompt with few-shot examples.

        Args:
            query: User query
            examples: Few-shot examples
            system_prompt: Optional system prompt
            input_prefix: Prefix for inputs
            output_prefix: Prefix for outputs

        Returns:
            Prompt with examples
        """
        parts = []

        # Add system prompt
        if system_prompt:
            parts.append(system_prompt)

        # Add examples
        if examples:
            parts.append("Examples:")
            for i, example in enumerate(examples, 1):
                parts.append(f"\nExample {i}:")
                parts.append(example.format(input_prefix, output_prefix))

        # Add query
        parts.append(f"\n{input_prefix} {query}")
        parts.append(f"{output_prefix}")

        return "\n".join(parts)

    def build_chat_messages(
        self,
        query: str,
        contexts: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        provider: Optional[LLMProvider] = None,
    ) -> List[Dict[str, str]]:
        """
        Build chat-format messages.

        Args:
            query: User query
            contexts: Retrieved contexts
            system_prompt: System prompt
            conversation_history: Previous messages
            provider: LLM provider for formatting

        Returns:
            List of message dictionaries
        """
        provider = provider or self.default_provider
        messages = []

        # Add system message
        system = system_prompt or self.system_builder.build()

        if contexts:
            formatted_context = self.context_formatter.format(contexts)
            system = f"{system}\n\nContext:\n{formatted_context}"

        messages.append({"role": "system", "content": system})

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # Add user query
        messages.append({"role": "user", "content": query})

        return messages

    def build_chain_of_thought(
        self,
        query: str,
        contexts: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build chain-of-thought prompt.

        Args:
            query: User query
            contexts: Optional contexts
            system_prompt: Optional system prompt

        Returns:
            CoT prompt
        """
        parts = []

        # System instruction
        system = system_prompt or "You are a helpful assistant that thinks step by step."
        parts.append(system)

        # Add contexts if provided
        if contexts:
            formatted = self.context_formatter.format(contexts)
            parts.append(f"\nContext:\n{formatted}")

        # Add query with CoT instruction
        parts.append(f"\nQuestion: {query}")
        parts.append("\nLet's think through this step by step:")
        parts.append("1.")

        return "\n".join(parts)

    def register_template(self, template: PromptTemplate) -> None:
        """Register a custom template."""
        issues = template.validate()
        if issues:
            logger.warning(f"Template '{template.name}' has issues: {issues}")
        self._templates[template.name] = template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        """List available templates."""
        return list(self._templates.keys())


class PromptOptimizer:
    """
    Optimize prompts for better results.

    Example:
        >>> optimizer = PromptOptimizer()
        >>> optimized = optimizer.optimize(
        ...     prompt="Tell me about Python",
        ...     optimization_type="clarity"
        ... )
    """

    def __init__(self):
        """Initialize prompt optimizer."""
        pass

    def optimize(
        self,
        prompt: str,
        optimization_type: str = "clarity",
    ) -> str:
        """
        Optimize a prompt.

        Args:
            prompt: Original prompt
            optimization_type: Type of optimization

        Returns:
            Optimized prompt
        """
        if optimization_type == "clarity":
            return self._optimize_clarity(prompt)
        elif optimization_type == "conciseness":
            return self._optimize_conciseness(prompt)
        elif optimization_type == "specificity":
            return self._optimize_specificity(prompt)
        else:
            return prompt

    def _optimize_clarity(self, prompt: str) -> str:
        """Improve prompt clarity."""
        # Remove redundant whitespace
        prompt = " ".join(prompt.split())

        # Ensure ends with clear instruction
        if not prompt.rstrip().endswith(("?", ".", ":", "!")):
            prompt = prompt.rstrip() + "."

        return prompt

    def _optimize_conciseness(self, prompt: str) -> str:
        """Make prompt more concise."""
        # Remove filler phrases
        fillers = [
            "I would like you to",
            "Could you please",
            "I want you to",
            "Please help me",
            "Can you help me",
        ]

        result = prompt
        for filler in fillers:
            result = result.replace(filler, "")

        return " ".join(result.split())

    def _optimize_specificity(self, prompt: str) -> str:
        """Add specificity markers."""
        # Check for vague terms
        vague_terms = {
            "this": "the specified",
            "it": "the topic",
            "stuff": "items",
            "things": "elements",
        }

        result = prompt
        for vague, specific in vague_terms.items():
            pattern = rf"\b{vague}\b"
            result = re.sub(pattern, specific, result, flags=re.IGNORECASE)

        return result

    def estimate_tokens(self, prompt: str) -> int:
        """Estimate token count."""
        # Simple estimation: ~4 chars per token
        return len(prompt) // 4


class PromptLibrary:
    """
    Library of pre-built prompts for common tasks.

    Example:
        >>> library = PromptLibrary()
        >>> prompt = library.get("summarization")
        >>> rendered = prompt.render(text="Long document...")
    """

    def __init__(self):
        """Initialize prompt library."""
        self._prompts: Dict[str, PromptTemplate] = {}
        self._load_default_prompts()

    def _load_default_prompts(self) -> None:
        """Load default prompts."""
        # Summarization
        self._prompts["summarization"] = PromptTemplate(
            name="summarization",
            template="""Summarize the following text in a concise manner:

$text

Summary:""",
            required_vars=["text"],
            tags=["summarization", "text-processing"],
        )

        # Entity extraction
        self._prompts["entity_extraction"] = PromptTemplate(
            name="entity_extraction",
            template="""Extract all named entities from the following text.
Return them as a comma-separated list.

Text: $text

Entities:""",
            required_vars=["text"],
            tags=["ner", "extraction"],
        )

        # Sentiment analysis
        self._prompts["sentiment"] = PromptTemplate(
            name="sentiment",
            template="""Analyze the sentiment of the following text.
Respond with: POSITIVE, NEGATIVE, or NEUTRAL

Text: $text

Sentiment:""",
            required_vars=["text"],
            tags=["sentiment", "classification"],
        )

        # Question answering
        self._prompts["qa_extractive"] = PromptTemplate(
            name="qa_extractive",
            template="""Based on the context, answer the question.
Only use information from the context. If the answer is not in the context, say "Not found."

Context: $context

Question: $question

Answer:""",
            required_vars=["context", "question"],
            tags=["qa", "extractive"],
        )

        # Rewriting
        self._prompts["rewrite"] = PromptTemplate(
            name="rewrite",
            template="""Rewrite the following text to be more $style:

Original: $text

Rewritten:""",
            required_vars=["text", "style"],
            tags=["rewriting", "text-processing"],
        )

        # Classification
        self._prompts["classification"] = PromptTemplate(
            name="classification",
            template="""Classify the following text into one of these categories: $categories

Text: $text

Category:""",
            required_vars=["text", "categories"],
            tags=["classification"],
        )

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt by name."""
        return self._prompts.get(name)

    def add(self, prompt: PromptTemplate) -> None:
        """Add a prompt to the library."""
        self._prompts[prompt.name] = prompt

    def list_prompts(self, tag: Optional[str] = None) -> List[str]:
        """List available prompts, optionally filtered by tag."""
        if tag:
            return [name for name, prompt in self._prompts.items() if tag in prompt.tags]
        return list(self._prompts.keys())

    def search(self, query: str) -> List[PromptTemplate]:
        """Search prompts by name or description."""
        query_lower = query.lower()
        results = []
        for prompt in self._prompts.values():
            if (
                query_lower in prompt.name.lower()
                or query_lower in prompt.description.lower()
                or any(query_lower in tag.lower() for tag in prompt.tags)
            ):
                results.append(prompt)
        return results


# Convenience functions


def build_rag_prompt(
    query: str,
    contexts: List[str],
    system_prompt: Optional[str] = None,
) -> str:
    """
    Build a RAG prompt.

    Args:
        query: User query
        contexts: Retrieved contexts
        system_prompt: Optional system prompt

    Returns:
        Complete prompt
    """
    builder = PromptBuilder()
    return builder.build_rag_prompt(query, contexts, system_prompt)


def build_chat_messages(
    query: str,
    contexts: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build chat messages.

    Args:
        query: User query
        contexts: Retrieved contexts
        system_prompt: System prompt

    Returns:
        List of messages
    """
    builder = PromptBuilder()
    return builder.build_chat_messages(query, contexts, system_prompt)


def format_contexts(
    contexts: List[str],
    max_length: int = 4000,
) -> str:
    """
    Format contexts for prompt inclusion.

    Args:
        contexts: Context passages
        max_length: Maximum length

    Returns:
        Formatted context string
    """
    formatter = ContextFormatter(max_context_length=max_length)
    return formatter.format(contexts)


# =============================================================================
# OPTIMIZATION 6: DSPy-Style Prompt Optimization for Agentic RAG
# Based on: "DSPy: Compiling Declarative Language Model Calls into
# Self-Improving Pipelines" (Khattab et al., 2023)
#
# Implements automatic prompt optimization through:
# 1. Declarative signature-based prompt definition
# 2. Automatic few-shot example bootstrapping
# 3. Prompt instruction optimization via LLM feedback
# 4. Module composition for complex RAG pipelines
# =============================================================================


class LLMProtocol(Protocol):
    """Protocol for LLM interactions."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response from prompt."""
        ...


@dataclass
class InputField:
    """Input field definition for DSPy-style signatures."""

    name: str
    description: str = ""
    prefix: str = ""
    format_fn: Optional[Callable[[Any], str]] = None


@dataclass
class OutputField:
    """Output field definition for DSPy-style signatures."""

    name: str
    description: str = ""
    prefix: str = ""
    parse_fn: Optional[Callable[[str], Any]] = None


@dataclass
class Example:
    """A training/demo example for few-shot prompting."""

    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"inputs": self.inputs, "outputs": self.outputs}


@dataclass
class Prediction:
    """A prediction from a DSPy module."""

    outputs: Dict[str, Any]
    completions: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0


class Signature:
    """
    DSPy-style signature for declarative prompt definition.

    Signatures define the input/output contract for LLM modules,
    enabling automatic prompt optimization and composition.

    Example:
        >>> sig = Signature(
        ...     name="AnswerQuestion",
        ...     instructions="Answer questions based on context.",
        ...     inputs=[InputField("context"), InputField("question")],
        ...     outputs=[OutputField("answer")],
        ... )
    """

    def __init__(
        self,
        name: str,
        instructions: str = "",
        inputs: Optional[List[InputField]] = None,
        outputs: Optional[List[OutputField]] = None,
    ):
        """Initialize signature."""
        self.name = name
        self.instructions = instructions
        self.inputs = inputs or []
        self.outputs = outputs or []

        # Build field maps
        self._input_map = {f.name: f for f in self.inputs}
        self._output_map = {f.name: f for f in self.outputs}

    def format_input(self, **kwargs: Any) -> str:
        """Format input fields into prompt section."""
        parts = []
        for input_field in self.inputs:
            if input_field.name in kwargs:
                value = kwargs[input_field.name]
                if input_field.format_fn:
                    value = input_field.format_fn(value)
                prefix = input_field.prefix or f"{input_field.name.title()}:"
                parts.append(f"{prefix} {value}")
        return "\n".join(parts)

    def format_output_schema(self) -> str:
        """Format expected output schema."""
        parts = []
        for out_field in self.outputs:
            prefix = out_field.prefix or f"{out_field.name.title()}:"
            desc = f" ({out_field.description})" if out_field.description else ""
            parts.append(f"{prefix}{desc}")
        return "\n".join(parts)

    def parse_output(self, text: str) -> Dict[str, Any]:
        """Parse output text into structured fields."""
        result: Dict[str, Any] = {}
        for out_field in self.outputs:
            prefix = out_field.prefix or f"{out_field.name.title()}:"
            # Simple extraction - look for prefix
            pattern = rf"{re.escape(prefix)}\s*(.+?)(?=\n[A-Z]|\Z)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if out_field.parse_fn:
                    value = out_field.parse_fn(value)
                result[out_field.name] = value
        return result

    def __repr__(self) -> str:
        inputs = ", ".join(f.name for f in self.inputs)
        outputs = ", ".join(f.name for f in self.outputs)
        return f"Signature({self.name}: {inputs} -> {outputs})"


class DSPyModule(ABC):
    """
    Base class for DSPy-style modules.

    Modules are composable units that perform specific tasks
    using LLMs with automatic prompt optimization.
    """

    def __init__(self, signature: Signature, llm: Optional[LLMProtocol] = None):
        """Initialize module."""
        self.signature = signature
        self.llm = llm
        self.demos: List[Example] = []
        self._trace: List[Dict[str, Any]] = []

    @abstractmethod
    async def forward(self, **kwargs: Any) -> Prediction:
        """Execute the module."""
        pass

    def add_demo(self, example: Example) -> None:
        """Add a demonstration example."""
        self.demos.append(example)

    def clear_demos(self) -> None:
        """Clear all demos."""
        self.demos.clear()


class Predict(DSPyModule):
    """
    Basic prediction module using signature.

    Formats inputs according to signature, calls LLM,
    and parses outputs.
    """

    def __init__(
        self,
        signature: Signature,
        llm: Optional[LLMProtocol] = None,
        n_demos: int = 3,
        temperature: float = 0.7,
    ):
        """Initialize Predict module."""
        super().__init__(signature, llm)
        self.n_demos = n_demos
        self.temperature = temperature

    async def forward(self, **kwargs: Any) -> Prediction:
        """Execute prediction."""
        if not self.llm:
            return Prediction(outputs={})

        # Build prompt
        prompt = self._build_prompt(**kwargs)

        # Generate
        response = await self.llm.generate(prompt, temperature=self.temperature)

        # Parse output
        outputs = self.signature.parse_output(response)

        return Prediction(
            outputs=outputs,
            completions=[response],
            trace=[{"prompt": prompt, "response": response}],
        )

    def _build_prompt(self, **kwargs: Any) -> str:
        """Build the complete prompt."""
        parts = []

        # Instructions
        if self.signature.instructions:
            parts.append(self.signature.instructions)
            parts.append("")

        # Few-shot examples
        if self.demos:
            demos_to_use = self.demos[: self.n_demos]
            for i, demo in enumerate(demos_to_use, 1):
                parts.append(f"Example {i}:")
                parts.append(self.signature.format_input(**demo.inputs))
                for out_field in self.signature.outputs:
                    if out_field.name in demo.outputs:
                        prefix = out_field.prefix or f"{out_field.name.title()}:"
                        parts.append(f"{prefix} {demo.outputs[out_field.name]}")
                parts.append("")

        # Current input
        parts.append("Now answer:")
        parts.append(self.signature.format_input(**kwargs))
        parts.append(self.signature.format_output_schema())

        return "\n".join(parts)


class ChainOfThought(DSPyModule):
    """
    Chain-of-thought prediction module.

    Adds a rationale step before the final answer.
    """

    def __init__(
        self,
        signature: Signature,
        llm: Optional[LLMProtocol] = None,
        rationale_field: str = "rationale",
    ):
        """Initialize ChainOfThought module."""
        # Add rationale to outputs
        extended_outputs = [
            OutputField(
                name=rationale_field,
                description="Step-by-step reasoning",
                prefix="Reasoning:",
            ),
            *signature.outputs,
        ]
        extended_sig = Signature(
            name=f"{signature.name}_CoT",
            instructions=signature.instructions + "\nThink step by step before answering.",
            inputs=signature.inputs,
            outputs=extended_outputs,
        )
        super().__init__(extended_sig, llm)
        self.rationale_field = rationale_field
        self.inner = Predict(extended_sig, llm)

    async def forward(self, **kwargs: Any) -> Prediction:
        """Execute with chain of thought."""
        self.inner.demos = self.demos
        return await self.inner.forward(**kwargs)


class RAGModule(DSPyModule):
    """
    RAG-specific DSPy module with retrieval integration.

    Combines retrieval with LLM generation using optimizable prompts.
    """

    # Default RAG signature
    DEFAULT_SIGNATURE = Signature(
        name="RAGAnswer",
        instructions="Answer the question based on the provided context. Be concise and accurate.",
        inputs=[
            InputField(
                name="context",
                description="Retrieved passages",
                prefix="Context:",
            ),
            InputField(
                name="question",
                description="User question",
                prefix="Question:",
            ),
        ],
        outputs=[
            OutputField(
                name="answer",
                description="The answer",
                prefix="Answer:",
            ),
        ],
    )

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        signature: Optional[Signature] = None,
        use_chain_of_thought: bool = False,
    ):
        """Initialize RAG module."""
        sig = signature or self.DEFAULT_SIGNATURE
        super().__init__(sig, llm)
        self.use_cot = use_chain_of_thought

        if use_chain_of_thought:
            self.predictor = ChainOfThought(sig, llm)
        else:
            self.predictor = Predict(sig, llm)

    async def forward(
        self,
        question: str,
        context: Union[str, List[str]],
        **kwargs: Any,
    ) -> Prediction:
        """Execute RAG prediction."""
        # Format context
        if isinstance(context, list):
            context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context))

        self.predictor.demos = self.demos
        return await self.predictor.forward(
            question=question,
            context=context,
            **kwargs,
        )


class DSPyPromptOptimizer:
    """
    DSPy-style prompt optimizer.

    Optimizes prompts through:
    1. Bootstrap few-shot: Generate and filter examples
    2. Instruction optimization: Refine instructions via feedback
    3. Example selection: Choose best demos for each query
    """

    def __init__(
        self,
        llm: Optional[LLMProtocol] = None,
        metric_fn: Optional[Callable[[Prediction, Example], float]] = None,
    ):
        """Initialize optimizer."""
        self.llm = llm
        self.metric_fn = metric_fn or self._default_metric

    @staticmethod
    def _default_metric(pred: Prediction, example: Example) -> float:
        """Default metric: exact match on first output field."""
        for key in example.outputs:
            if key in pred.outputs:
                pred_val = str(pred.outputs[key]).lower().strip()
                true_val = str(example.outputs[key]).lower().strip()
                return 1.0 if pred_val == true_val else 0.0
        return 0.0

    async def bootstrap_fewshot(
        self,
        module: DSPyModule,
        train_examples: List[Example],
        num_candidates: int = 10,
        max_demos: int = 5,
    ) -> List[Example]:
        """
        Bootstrap few-shot examples by generating and filtering.

        Args:
            module: Module to optimize
            train_examples: Training examples
            num_candidates: Candidates to generate per example
            max_demos: Maximum demos to keep

        Returns:
            Filtered list of high-quality demos
        """
        scored_demos: List[tuple[float, Example]] = []

        for example in train_examples:
            # Generate prediction
            pred = await module.forward(**example.inputs)

            # Score it
            score = self.metric_fn(pred, example)

            if score > 0.5:  # Keep high-quality examples
                scored_demos.append((score, example))

        # Sort by score and take top
        scored_demos.sort(key=lambda x: x[0], reverse=True)
        return [demo for _, demo in scored_demos[:max_demos]]

    async def optimize_instructions(
        self,
        module: DSPyModule,
        train_examples: List[Example],
        num_iterations: int = 3,
    ) -> str:
        """
        Optimize instructions using LLM feedback.

        Args:
            module: Module to optimize
            train_examples: Training examples
            num_iterations: Optimization iterations

        Returns:
            Optimized instructions
        """
        if not self.llm:
            return module.signature.instructions

        current_instructions = module.signature.instructions
        best_score = 0.0
        best_instructions = current_instructions

        for iteration in range(num_iterations):
            # Evaluate current instructions
            total_score = 0.0
            sample = random.sample(train_examples, min(5, len(train_examples)))

            for example in sample:
                pred = await module.forward(**example.inputs)
                total_score += self.metric_fn(pred, example)

            avg_score = total_score / len(sample)

            if avg_score > best_score:
                best_score = avg_score
                best_instructions = current_instructions

            # Generate improved instructions
            if iteration < num_iterations - 1:
                improve_prompt = f"""Improve these instructions for a RAG system.

Current instructions:
{current_instructions}

Current performance: {avg_score:.2%}

Example task:
Input: {sample[0].inputs if sample else 'N/A'}
Expected: {sample[0].outputs if sample else 'N/A'}

Provide improved instructions that would help the model perform better.
Keep instructions concise but clear.

Improved instructions:"""

                current_instructions = await self.llm.generate(improve_prompt, temperature=0.7)

        return best_instructions


class SignatureOptimizer:
    """
    Optimizes entire signatures including field definitions.
    """

    def __init__(self, llm: Optional[LLMProtocol] = None):
        """Initialize signature optimizer."""
        self.llm = llm

    async def optimize_field_prefixes(
        self,
        signature: Signature,
        examples: List[Example],
    ) -> Signature:
        """
        Optimize field prefixes for better LLM understanding.

        Args:
            signature: Signature to optimize
            examples: Training examples

        Returns:
            Signature with optimized prefixes
        """
        if not self.llm:
            return signature

        # Generate candidate prefixes
        prompt = f"""Suggest better field prefixes for this prompt signature.

Current signature: {signature}
Instructions: {signature.instructions}

Input fields:
{chr(10).join(f'- {f.name}: {f.prefix or f.name.title()}' for f in signature.inputs)}

Output fields:
{chr(10).join(f'- {f.name}: {f.prefix or f.name.title()}' for f in signature.outputs)}

Suggest clearer, more descriptive prefixes for each field.
Format: field_name: new_prefix

Better prefixes:"""

        # Generate suggestions (would parse and apply in full implementation)
        _ = await self.llm.generate(prompt, temperature=0.5)

        # Parse and apply (simplified - would parse response properly)
        return signature


def create_rag_signature(
    include_citations: bool = False,
    include_confidence: bool = False,
) -> Signature:
    """
    Create a RAG signature with optional fields.

    Args:
        include_citations: Add citation field
        include_confidence: Add confidence field

    Returns:
        Configured Signature
    """
    inputs = [
        InputField(
            name="context",
            description="Retrieved passages",
            prefix="Context:",
        ),
        InputField(
            name="question",
            description="User question",
            prefix="Question:",
        ),
    ]

    outputs = [
        OutputField(
            name="answer",
            description="The answer",
            prefix="Answer:",
        ),
    ]

    if include_citations:
        outputs.append(
            OutputField(
                name="citations",
                description="Source citations",
                prefix="Citations:",
            )
        )

    if include_confidence:
        outputs.append(
            OutputField(
                name="confidence",
                description="Confidence level (high/medium/low)",
                prefix="Confidence:",
            )
        )

    return Signature(
        name="RAGWithOptions",
        instructions="Answer the question based on the provided context. Be concise and accurate.",
        inputs=inputs,
        outputs=outputs,
    )


def create_multi_hop_signature() -> Signature:
    """Create a signature for multi-hop reasoning."""
    return Signature(
        name="MultiHopRAG",
        instructions="Answer complex questions by reasoning across multiple passages.",
        inputs=[
            InputField(
                name="context",
                description="Multiple retrieved passages",
                prefix="Passages:",
            ),
            InputField(
                name="question",
                description="Multi-hop question",
                prefix="Question:",
            ),
        ],
        outputs=[
            OutputField(
                name="reasoning",
                description="Step-by-step reasoning",
                prefix="Reasoning:",
            ),
            OutputField(
                name="answer",
                description="Final answer",
                prefix="Answer:",
            ),
        ],
    )
