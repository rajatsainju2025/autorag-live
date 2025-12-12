"""Conversation Templates for AutoRAG-Live.

Provides structured prompt templates for various RAG scenarios:
- Question answering with context
- Multi-turn conversations
- Document summarization
- Fact verification
- Comparative analysis
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TemplateType(Enum):
    """Types of conversation templates."""

    QUESTION_ANSWERING = "qa"
    MULTI_TURN = "multi_turn"
    SUMMARIZATION = "summarization"
    FACT_VERIFICATION = "fact_verification"
    COMPARISON = "comparison"
    EXTRACTION = "extraction"
    EXPLANATION = "explanation"
    CREATIVE = "creative"


@dataclass
class Message:
    """Represents a conversation message."""

    role: str  # system, user, assistant
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        return {"role": self.role, "content": self.content}


@dataclass
class ConversationContext:
    """Context for conversation templates."""

    query: str
    documents: list[str] = field(default_factory=list)
    history: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    max_context_length: int = 4000
    include_sources: bool = True


class BaseTemplate(ABC):
    """Abstract base class for conversation templates."""

    template_type: TemplateType

    @abstractmethod
    def format(self, context: ConversationContext) -> list[Message]:
        """Format template with context.

        Args:
            context: Conversation context

        Returns:
            List of formatted messages
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this template."""
        pass

    def _truncate_documents(
        self,
        documents: list[str],
        max_length: int,
    ) -> str:
        """Truncate documents to fit within max length."""
        combined = "\n\n---\n\n".join(documents)
        if len(combined) <= max_length:
            return combined

        # Truncate proportionally
        per_doc_limit = max_length // len(documents) if documents else max_length

        truncated_docs = []
        for doc in documents:
            if len(doc) > per_doc_limit:
                truncated_docs.append(doc[:per_doc_limit] + "...")
            else:
                truncated_docs.append(doc)

        return "\n\n---\n\n".join(truncated_docs)

    def _format_sources(self, documents: list[str]) -> str:
        """Format documents as numbered sources."""
        if not documents:
            return ""

        formatted = []
        for i, doc in enumerate(documents, 1):
            formatted.append(f"[Source {i}]\n{doc}")

        return "\n\n".join(formatted)


class QuestionAnsweringTemplate(BaseTemplate):
    """Template for question answering with retrieved context."""

    template_type = TemplateType.QUESTION_ANSWERING

    def __init__(
        self,
        require_citations: bool = True,
        allow_uncertainty: bool = True,
        style: str = "concise",
    ) -> None:
        """Initialize QA template.

        Args:
            require_citations: Require citations to sources
            allow_uncertainty: Allow "I don't know" responses
            style: Response style (concise, detailed, technical)
        """
        self.require_citations = require_citations
        self.allow_uncertainty = allow_uncertainty
        self.style = style

    def get_system_prompt(self) -> str:
        """Get system prompt."""
        base = """You are a helpful assistant that answers questions based on provided context.

Your task is to:
1. Read the provided context carefully
2. Answer the user's question accurately using ONLY information from the context
3. Be {style} in your responses"""

        if self.require_citations:
            base += """
4. Cite your sources by referencing [Source N] when making claims"""

        if self.allow_uncertainty:
            base += """
5. If the context doesn't contain enough information, say "I don't have enough information to answer this question." """

        base += """

Important rules:
- Do NOT make up information not present in the context
- Do NOT use prior knowledge outside the provided context
- Stay focused on answering the specific question asked"""

        return base.format(style=self.style)

    def format(self, context: ConversationContext) -> list[Message]:
        """Format QA template."""
        messages = [Message(role="system", content=self.get_system_prompt())]

        # Add conversation history
        for msg in context.history:
            messages.append(msg)

        # Format context documents
        if context.documents:
            docs_text = self._truncate_documents(
                context.documents, context.max_context_length
            )
            if context.include_sources:
                docs_text = self._format_sources(context.documents)

            user_content = f"""Context:
{docs_text}

Question: {context.query}"""
        else:
            user_content = f"Question: {context.query}"

        messages.append(Message(role="user", content=user_content))

        return messages


class MultiTurnTemplate(BaseTemplate):
    """Template for multi-turn conversations with context."""

    template_type = TemplateType.MULTI_TURN

    def __init__(
        self,
        max_history_turns: int = 5,
        summarize_history: bool = False,
    ) -> None:
        """Initialize multi-turn template.

        Args:
            max_history_turns: Maximum conversation turns to include
            summarize_history: Whether to summarize older history
        """
        self.max_history_turns = max_history_turns
        self.summarize_history = summarize_history

    def get_system_prompt(self) -> str:
        """Get system prompt."""
        return """You are a helpful conversational assistant with access to relevant context.

Your task is to:
1. Maintain context across the conversation
2. Reference previous exchanges when relevant
3. Use the provided documents to inform your answers
4. Ask clarifying questions if needed
5. Be natural and conversational while staying accurate

Important:
- Stay consistent with previous responses
- Acknowledge when the conversation topic changes
- Use context to provide more detailed answers"""

    def format(self, context: ConversationContext) -> list[Message]:
        """Format multi-turn template."""
        messages = [Message(role="system", content=self.get_system_prompt())]

        # Add limited history
        history = context.history[-self.max_history_turns * 2 :]
        for msg in history:
            messages.append(msg)

        # Format current turn with context
        if context.documents:
            docs_text = self._truncate_documents(
                context.documents,
                context.max_context_length,
            )
            user_content = f"""[Relevant Context]
{docs_text}

[Current Question]
{context.query}"""
        else:
            user_content = context.query

        messages.append(Message(role="user", content=user_content))

        return messages


class SummarizationTemplate(BaseTemplate):
    """Template for document summarization."""

    template_type = TemplateType.SUMMARIZATION

    def __init__(
        self,
        summary_style: str = "bullet",
        max_points: int = 5,
        include_key_quotes: bool = True,
    ) -> None:
        """Initialize summarization template.

        Args:
            summary_style: Style of summary (bullet, paragraph, executive)
            max_points: Maximum number of points/paragraphs
            include_key_quotes: Include relevant quotes
        """
        self.summary_style = summary_style
        self.max_points = max_points
        self.include_key_quotes = include_key_quotes

    def get_system_prompt(self) -> str:
        """Get system prompt."""
        style_instructions = {
            "bullet": f"Create a bullet-point summary with up to {self.max_points} key points.",
            "paragraph": f"Write a {self.max_points}-paragraph summary covering the main ideas.",
            "executive": "Write an executive summary suitable for quick review.",
        }

        prompt = f"""You are an expert summarizer.

Your task is to summarize the provided content:
- {style_instructions.get(self.summary_style, style_instructions['bullet'])}
- Focus on the most important information
- Maintain accuracy and objectivity
- Use clear, concise language"""

        if self.include_key_quotes:
            prompt += """
- Include 1-2 key quotes that capture important points"""

        return prompt

    def format(self, context: ConversationContext) -> list[Message]:
        """Format summarization template."""
        messages = [Message(role="system", content=self.get_system_prompt())]

        # Combine all documents for summarization
        if context.documents:
            docs_text = "\n\n---\n\n".join(context.documents)
            user_content = f"""Please summarize the following content:

{docs_text}

{context.query if context.query else "Provide a comprehensive summary."}"""
        else:
            user_content = context.query

        messages.append(Message(role="user", content=user_content))

        return messages


class FactVerificationTemplate(BaseTemplate):
    """Template for fact-checking and verification."""

    template_type = TemplateType.FACT_VERIFICATION

    def __init__(
        self,
        strict_mode: bool = True,
        require_evidence: bool = True,
    ) -> None:
        """Initialize fact verification template.

        Args:
            strict_mode: Strict fact-checking mode
            require_evidence: Require evidence for claims
        """
        self.strict_mode = strict_mode
        self.require_evidence = require_evidence

    def get_system_prompt(self) -> str:
        """Get system prompt."""
        prompt = """You are a fact-checker assistant. Your task is to verify claims against provided evidence.

For each claim, provide:
1. **Verdict**: SUPPORTED, REFUTED, or INSUFFICIENT EVIDENCE
2. **Confidence**: HIGH, MEDIUM, or LOW
3. **Evidence**: Quote the relevant passages that support your verdict
4. **Reasoning**: Explain your reasoning"""

        if self.strict_mode:
            prompt += """

Important rules:
- Only use the provided context as evidence
- Do not use external knowledge
- Be conservative - if unsure, say "INSUFFICIENT EVIDENCE"
- Flag any contradictory information in the sources"""

        return prompt

    def format(self, context: ConversationContext) -> list[Message]:
        """Format fact verification template."""
        messages = [Message(role="system", content=self.get_system_prompt())]

        docs_text = self._format_sources(context.documents)

        user_content = f"""Evidence Documents:
{docs_text}

Claim to verify:
{context.query}

Please verify this claim using ONLY the evidence provided above."""

        messages.append(Message(role="user", content=user_content))

        return messages


class ComparisonTemplate(BaseTemplate):
    """Template for comparative analysis."""

    template_type = TemplateType.COMPARISON

    def __init__(
        self,
        comparison_aspects: list[str] | None = None,
        use_table_format: bool = True,
    ) -> None:
        """Initialize comparison template.

        Args:
            comparison_aspects: Specific aspects to compare
            use_table_format: Use table format for comparison
        """
        self.comparison_aspects = comparison_aspects or []
        self.use_table_format = use_table_format

    def get_system_prompt(self) -> str:
        """Get system prompt."""
        prompt = """You are an analyst specializing in comparative analysis.

Your task is to compare items based on provided information:
1. Identify key similarities and differences
2. Be objective and balanced
3. Use specific evidence from the sources
4. Highlight notable findings"""

        if self.use_table_format:
            prompt += """
5. Present comparisons in a clear table format when appropriate"""

        if self.comparison_aspects:
            aspects_str = ", ".join(self.comparison_aspects)
            prompt += f"""

Focus your comparison on these aspects: {aspects_str}"""

        return prompt

    def format(self, context: ConversationContext) -> list[Message]:
        """Format comparison template."""
        messages = [Message(role="system", content=self.get_system_prompt())]

        docs_text = self._format_sources(context.documents)

        user_content = f"""Information to compare:
{docs_text}

Comparison request:
{context.query}"""

        messages.append(Message(role="user", content=user_content))

        return messages


class ExtractionTemplate(BaseTemplate):
    """Template for structured information extraction."""

    template_type = TemplateType.EXTRACTION

    def __init__(
        self,
        schema: dict[str, Any] | None = None,
        output_format: str = "json",
    ) -> None:
        """Initialize extraction template.

        Args:
            schema: Expected output schema
            output_format: Output format (json, yaml, markdown)
        """
        self.schema = schema
        self.output_format = output_format

    def get_system_prompt(self) -> str:
        """Get system prompt."""
        prompt = f"""You are an information extraction specialist.

Your task is to extract structured information from the provided text:
1. Identify all relevant entities and relationships
2. Extract information accurately - do not infer
3. Format output as {self.output_format.upper()}
4. Mark missing information as null/empty rather than guessing"""

        if self.schema:
            import json

            schema_str = json.dumps(self.schema, indent=2)
            prompt += f"""

Expected output schema:
```json
{schema_str}
```"""

        return prompt

    def format(self, context: ConversationContext) -> list[Message]:
        """Format extraction template."""
        messages = [Message(role="system", content=self.get_system_prompt())]

        docs_text = "\n\n".join(context.documents)

        user_content = f"""Source text:
{docs_text}

{context.query if context.query else "Extract all relevant information."}"""

        messages.append(Message(role="user", content=user_content))

        return messages


class ExplanationTemplate(BaseTemplate):
    """Template for explaining concepts with context."""

    template_type = TemplateType.EXPLANATION

    def __init__(
        self,
        audience_level: str = "general",
        use_analogies: bool = True,
        include_examples: bool = True,
    ) -> None:
        """Initialize explanation template.

        Args:
            audience_level: Target audience (beginner, general, expert)
            use_analogies: Use analogies to explain concepts
            include_examples: Include examples in explanations
        """
        self.audience_level = audience_level
        self.use_analogies = use_analogies
        self.include_examples = include_examples

    def get_system_prompt(self) -> str:
        """Get system prompt."""
        level_descriptions = {
            "beginner": "simple terms suitable for someone new to the topic",
            "general": "clear language accessible to a general audience",
            "expert": "technical depth appropriate for domain experts",
        }

        level_desc = level_descriptions.get(
            self.audience_level, level_descriptions["general"]
        )

        prompt = f"""You are an expert educator skilled at explaining complex topics.

Your task is to explain concepts using {level_desc}:
1. Break down complex ideas into understandable parts
2. Build from foundational concepts to advanced ones
3. Use the provided context as your knowledge base"""

        if self.use_analogies:
            prompt += """
4. Use relatable analogies when helpful"""

        if self.include_examples:
            prompt += """
5. Provide concrete examples to illustrate points"""

        prompt += """

Important:
- Ensure accuracy based on the provided context
- Acknowledge limitations in the available information
- Structure your explanation logically"""

        return prompt

    def format(self, context: ConversationContext) -> list[Message]:
        """Format explanation template."""
        messages = [Message(role="system", content=self.get_system_prompt())]

        if context.documents:
            docs_text = self._truncate_documents(
                context.documents, context.max_context_length
            )
            user_content = f"""Reference information:
{docs_text}

Please explain: {context.query}"""
        else:
            user_content = f"Please explain: {context.query}"

        messages.append(Message(role="user", content=user_content))

        return messages


class TemplateRegistry:
    """Registry for conversation templates."""

    def __init__(self) -> None:
        """Initialize with default templates."""
        self._templates: dict[str, BaseTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default templates."""
        self._templates = {
            "qa": QuestionAnsweringTemplate(),
            "qa_detailed": QuestionAnsweringTemplate(style="detailed"),
            "qa_no_citations": QuestionAnsweringTemplate(require_citations=False),
            "multi_turn": MultiTurnTemplate(),
            "summarize_bullet": SummarizationTemplate(summary_style="bullet"),
            "summarize_paragraph": SummarizationTemplate(summary_style="paragraph"),
            "summarize_executive": SummarizationTemplate(summary_style="executive"),
            "fact_check": FactVerificationTemplate(),
            "fact_check_lenient": FactVerificationTemplate(strict_mode=False),
            "compare": ComparisonTemplate(),
            "extract_json": ExtractionTemplate(output_format="json"),
            "extract_yaml": ExtractionTemplate(output_format="yaml"),
            "explain_beginner": ExplanationTemplate(audience_level="beginner"),
            "explain_general": ExplanationTemplate(audience_level="general"),
            "explain_expert": ExplanationTemplate(audience_level="expert"),
        }

    def register(self, name: str, template: BaseTemplate) -> None:
        """Register a template.

        Args:
            name: Template name
            template: Template instance
        """
        self._templates[name] = template

    def get(self, name: str) -> BaseTemplate:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            Template instance
        """
        if name not in self._templates:
            raise KeyError(f"Template not found: {name}")
        return self._templates[name]

    def list_templates(self) -> list[str]:
        """List all registered templates."""
        return list(self._templates.keys())

    def format(
        self,
        name: str,
        context: ConversationContext,
    ) -> list[Message]:
        """Format using a named template.

        Args:
            name: Template name
            context: Conversation context

        Returns:
            Formatted messages
        """
        template = self.get(name)
        return template.format(context)


# Global registry
_registry: TemplateRegistry | None = None


def get_template_registry() -> TemplateRegistry:
    """Get global template registry."""
    global _registry
    if _registry is None:
        _registry = TemplateRegistry()
    return _registry


def format_rag_prompt(
    query: str,
    documents: list[str],
    template: str = "qa",
    history: list[Message] | None = None,
) -> list[dict[str, str]]:
    """Convenience function to format RAG prompts.

    Args:
        query: User query
        documents: Retrieved documents
        template: Template name
        history: Conversation history

    Returns:
        List of message dictionaries
    """
    context = ConversationContext(
        query=query,
        documents=documents,
        history=history or [],
    )

    messages = get_template_registry().format(template, context)
    return [msg.to_dict() for msg in messages]


def create_qa_messages(
    query: str,
    context_docs: list[str],
    require_citations: bool = True,
) -> list[dict[str, str]]:
    """Create question-answering messages.

    Args:
        query: User question
        context_docs: Context documents
        require_citations: Require source citations

    Returns:
        Formatted messages
    """
    template_name = "qa" if require_citations else "qa_no_citations"
    return format_rag_prompt(query, context_docs, template_name)


def create_summary_messages(
    documents: list[str],
    style: str = "bullet",
    query: str | None = None,
) -> list[dict[str, str]]:
    """Create summarization messages.

    Args:
        documents: Documents to summarize
        style: Summary style
        query: Optional specific summarization request

    Returns:
        Formatted messages
    """
    template_name = f"summarize_{style}"
    return format_rag_prompt(query or "", documents, template_name)


def create_fact_check_messages(
    claim: str,
    evidence: list[str],
    strict: bool = True,
) -> list[dict[str, str]]:
    """Create fact-checking messages.

    Args:
        claim: Claim to verify
        evidence: Evidence documents
        strict: Use strict mode

    Returns:
        Formatted messages
    """
    template_name = "fact_check" if strict else "fact_check_lenient"
    return format_rag_prompt(claim, evidence, template_name)
