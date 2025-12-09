"""
Prompt engineering and templating system for AutoRAG-Live.

Provides reusable prompt templates with variable injection, few-shot examples,
and dynamic instruction generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PromptTemplate(Enum):
    """Built-in prompt templates."""

    REASONING = "reasoning"
    SYNTHESIS = "synthesis"
    QUERY_REFINEMENT = "query_refinement"
    FACT_CHECKING = "fact_checking"
    CLASSIFICATION = "classification"
    SUMMARY = "summary"


@dataclass
class FewShotExample:
    """Few-shot example for in-context learning."""

    input: str
    output: str
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {
            "input": self.input,
            "output": self.output,
            "explanation": self.explanation or "",
        }


@dataclass
class PromptMetrics:
    """Metrics tracking for prompt performance."""

    template_name: str
    total_uses: int = 0
    avg_response_length: float = 0.0
    avg_latency: float = 0.0
    success_rate: float = 1.0
    quality_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        response_length: int,
        latency: float,
        success: bool,
        quality: float = 0.5,
    ) -> None:
        """Update metrics with new usage data."""
        self.total_uses += 1

        # Update rolling average for response length
        self.avg_response_length = (
            self.avg_response_length * (self.total_uses - 1) + response_length
        ) / self.total_uses

        # Update rolling average for latency
        self.avg_latency = (self.avg_latency * (self.total_uses - 1) + latency) / self.total_uses

        # Update success rate
        self.success_rate = (
            self.success_rate * (self.total_uses - 1) + (1.0 if success else 0.0)
        ) / self.total_uses

        # Update quality score
        self.quality_score = quality


class PromptTemplateManager:
    """
    Manages prompt templates with variable injection and few-shot learning.

    Provides reusable, composable prompts for RAG operations with
    performance tracking and optimization.
    """

    def __init__(self):
        """Initialize prompt template manager."""
        self.logger = logging.getLogger("PromptTemplateManager")
        self.templates = self._load_default_templates()
        self.metrics: Dict[str, PromptMetrics] = {}
        self.custom_examples: Dict[str, List[FewShotExample]] = {}

    def _load_default_templates(self) -> Dict[str, str]:
        """Load default prompt templates."""
        return {
            PromptTemplate.REASONING.value: """You are a step-by-step reasoning engine.

Given the query: "{query}"

Break down your reasoning into clear steps:
1. What are the key concepts in this query?
2. What information is needed to answer?
3. How should we approach finding this information?
4. What potential challenges might we face?

Provide your reasoning chain:""",
            PromptTemplate.SYNTHESIS.value: """You are an expert at synthesizing answers from multiple sources.

Query: "{query}"

Retrieved information:
{sources}

Please synthesize a comprehensive answer that:
- Directly addresses the query
- Cites specific sources
- Maintains factual accuracy
- Explains any uncertainties

Answer:""",
            PromptTemplate.QUERY_REFINEMENT.value: """You are a query optimization expert.

Original query: "{query}"

Refine this query to:
1. Be more specific
2. Include important context
3. Clarify any ambiguities
4. Include relevant keywords for retrieval

Refined query:""",
            PromptTemplate.FACT_CHECKING.value: """You are a fact-checking expert.

Statement to verify: "{statement}"

Sources:
{sources}

Please evaluate:
1. Is this statement supported by the sources?
2. What is the confidence level (0-100%)?
3. What evidence supports or refutes it?
4. Are there any caveats or nuances?

Fact-check result:""",
            PromptTemplate.CLASSIFICATION.value: """You are a text classification expert.

Text to classify: "{text}"

Classify this text into ONE of these categories: {categories}

Provide:
1. The chosen category
2. Confidence score (0-100%)
3. Brief reasoning

Classification:""",
            PromptTemplate.SUMMARY.value: """You are a summarization expert.

Text to summarize:
{content}

Create a concise summary that:
- Captures key points
- Maintains important details
- Is {max_length} words or less
- Is written in clear language

Summary:""",
        }

    def get_template(self, template_name: str, **variables: Any) -> str:
        """
        Get and render prompt template.

        Args:
            template_name: Name of template
            **variables: Variables to inject into template

        Returns:
            Rendered prompt
        """
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")

        template = self.templates[template_name]

        # Inject variables
        try:
            prompt = template.format(**variables)
        except KeyError as e:
            self.logger.warning(f"Missing variable {e} for template {template_name}")
            prompt = template

        # Add few-shot examples if available
        if template_name in self.custom_examples:
            examples = self.custom_examples[template_name]
            prompt = self._add_few_shots(prompt, examples)

        # Initialize metrics if needed
        if template_name not in self.metrics:
            self.metrics[template_name] = PromptMetrics(template_name)

        return prompt

    def add_custom_template(self, name: str, template: str) -> None:
        """Add custom prompt template."""
        self.templates[name] = template
        self.logger.info(f"Added custom template: {name}")

    def add_few_shot_examples(
        self,
        template_name: str,
        examples: List[FewShotExample],
        replace: bool = False,
    ) -> None:
        """
        Add few-shot examples for a template.

        Args:
            template_name: Template to enhance
            examples: List of examples
            replace: Whether to replace existing examples
        """
        if replace or template_name not in self.custom_examples:
            self.custom_examples[template_name] = examples
        else:
            self.custom_examples[template_name].extend(examples)

        self.logger.info(f"Added {len(examples)} examples to {template_name}")

    def _add_few_shots(self, prompt: str, examples: List[FewShotExample]) -> str:
        """Add few-shot examples to prompt."""
        if not examples:
            return prompt

        example_text = "Examples:\n"
        for i, example in enumerate(examples[:3], 1):
            example_text += f"\nExample {i}:\nInput: {example.input}\n"
            example_text += f"Output: {example.output}"
            if example.explanation:
                example_text += f"\nExplanation: {example.explanation}"

        return example_text + "\n\n" + prompt

    def generate_reasoning_prompt(
        self,
        query: str,
        context: Optional[str] = None,
        chain_of_thought: bool = True,
    ) -> str:
        """Generate prompt for reasoning about a query."""
        variables = {"query": query}

        prompt = self.get_template(PromptTemplate.REASONING.value, **variables)

        if context:
            prompt += f"\n\nAdditional context:\n{context}"

        if chain_of_thought:
            prompt += "\n\nThink carefully through each step before " "providing your reasoning."

        return prompt

    def generate_synthesis_prompt(
        self,
        query: str,
        sources: List[str],
        include_confidence: bool = True,
    ) -> str:
        """Generate prompt for synthesizing answer from sources."""
        sources_text = "\n".join([f"{i + 1}. {source}" for i, source in enumerate(sources)])

        variables = {"query": query, "sources": sources_text}

        prompt = self.get_template(PromptTemplate.SYNTHESIS.value, **variables)

        if include_confidence:
            prompt += "\n\nAlso provide your confidence level (0-100%) " "in this answer."

        return prompt

    def generate_refinement_prompt(self, query: str) -> str:
        """Generate prompt for query refinement."""
        return self.get_template(PromptTemplate.QUERY_REFINEMENT.value, query=query)

    def generate_classification_prompt(
        self,
        text: str,
        categories: List[str],
    ) -> str:
        """Generate prompt for text classification."""
        categories_str = ", ".join(categories)

        return self.get_template(
            PromptTemplate.CLASSIFICATION.value,
            text=text,
            categories=categories_str,
        )

    def generate_summary_prompt(
        self,
        content: str,
        max_length: int = 100,
    ) -> str:
        """Generate prompt for summarization."""
        return self.get_template(
            PromptTemplate.SUMMARY.value,
            content=content,
            max_length=str(max_length),
        )

    def record_usage(
        self,
        template_name: str,
        response_length: int,
        latency: float,
        success: bool = True,
        quality: float = 0.5,
    ) -> None:
        """Record template usage metrics."""
        if template_name not in self.metrics:
            self.metrics[template_name] = PromptMetrics(template_name)

        self.metrics[template_name].update(response_length, latency, success, quality)

    def get_best_templates(self) -> List[str]:
        """Get templates ranked by performance."""
        if not self.metrics:
            return list(self.templates.keys())

        sorted_metrics = sorted(
            self.metrics.items(),
            key=lambda x: x[1].quality_score * x[1].success_rate,
            reverse=True,
        )

        return [name for name, _ in sorted_metrics]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of template metrics."""
        return {
            name: {
                "uses": metric.total_uses,
                "success_rate": round(metric.success_rate, 3),
                "quality_score": round(metric.quality_score, 3),
                "avg_latency": round(metric.avg_latency, 3),
            }
            for name, metric in self.metrics.items()
        }


# Global instance for easy access
_default_manager: Optional[PromptTemplateManager] = None


def get_template_manager() -> PromptTemplateManager:
    """Get or create global template manager."""
    global _default_manager

    if _default_manager is None:
        _default_manager = PromptTemplateManager()

    return _default_manager
