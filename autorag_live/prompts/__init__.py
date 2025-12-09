"""
Prompts package initialization.
"""

from .templates import (
    FewShotExample,
    PromptMetrics,
    PromptTemplate,
    PromptTemplateManager,
    get_template_manager,
)

__all__ = [
    "PromptTemplateManager",
    "PromptTemplate",
    "FewShotExample",
    "PromptMetrics",
    "get_template_manager",
]
