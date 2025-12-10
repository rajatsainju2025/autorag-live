"""
LLM integration module for AutoRAG-Live.

Provides modular provider abstraction for multiple LLM backends.
"""

from .async_providers import (
    AsyncAnthropicProvider,
    AsyncLLMPool,
    AsyncLLMProvider,
    AsyncLLMResponse,
    AsyncOllamaProvider,
    AsyncOpenAIProvider,
    create_async_provider,
)
from .fallback import (
    AsyncProviderFallbackChain,
    FallbackConfig,
    ProviderFallbackChain,
    ProviderHealth,
    ProviderStatus,
    create_async_fallback_chain,
    create_fallback_chain,
)
from .providers import (
    AnthropicProvider,
    CostTracker,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    ModelProvider,
    OllamaProvider,
    OpenAIProvider,
    TokenCounter,
    create_provider,
)

__all__ = [
    # Sync providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "LLMConfig",
    "LLMResponse",
    "TokenCounter",
    "CostTracker",
    "ModelProvider",
    "create_provider",
    # Async providers
    "AsyncLLMProvider",
    "AsyncOpenAIProvider",
    "AsyncAnthropicProvider",
    "AsyncOllamaProvider",
    "AsyncLLMResponse",
    "AsyncLLMPool",
    "create_async_provider",
    # Fallback chain
    "ProviderFallbackChain",
    "AsyncProviderFallbackChain",
    "FallbackConfig",
    "ProviderHealth",
    "ProviderStatus",
    "create_fallback_chain",
    "create_async_fallback_chain",
]
