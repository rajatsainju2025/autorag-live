"""
LLM Provider Factory — pluggable, real LLM backends for agentic RAG.

Bridges the existing ``BaseLLM`` protocol in ``core.protocols`` with
concrete provider implementations that can actually make API calls.

The critical gap this fills: the project had ``BaseLLM`` as an abstract
protocol and ``providers.py`` with ABC stubs, but no actual wiring to
OpenAI/Anthropic/Ollama SDKs.  This module provides:

    1. ``LLMProviderFactory`` — create providers by name
    2. ``MockLLMProvider``    — deterministic, no-API-call provider for tests
    3. ``OpenAILLMProvider``  — real OpenAI SDK integration (optional dep)
    4. ``AnthropicLLMProvider`` — real Anthropic SDK integration (optional dep)
    5. ``OllamaLLMProvider``  — local Ollama HTTP integration (optional dep)

All providers implement ``BaseLLM`` from ``core.protocols`` so they can be
used interchangeably anywhere the protocol is expected.

Example:
    >>> from autorag_live.llm.provider_factory import LLMProviderFactory
    >>>
    >>> llm = LLMProviderFactory.create("mock")  # for testing
    >>> result = await llm.generate([Message.user("Hello")])
    >>>
    >>> llm = LLMProviderFactory.create("openai", model="gpt-4o", api_key="sk-...")
    >>> result = await llm.generate([Message.user("What is RLHF?")])
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from autorag_live.core.protocols import GenerationResult, Message, MessageRole

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified config
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    """Configuration shared across all LLM providers."""

    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base provider (satisfies BaseLLM protocol)
# ---------------------------------------------------------------------------


class BaseLLMProvider:
    """
    Base class for LLM providers.

    Subclasses must implement ``_call_api()`` which does the actual HTTP/SDK call.
    The public ``generate()`` method handles timing, error wrapping, and logging.
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._call_count = 0
        self._total_tokens = 0

    async def generate(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response from the LLM (satisfies BaseLLM protocol)."""
        start = time.perf_counter()
        try:
            result = await self._call_api(messages, **kwargs)
            latency = (time.perf_counter() - start) * 1000
            self._call_count += 1
            self._total_tokens += result.usage.get("total_tokens", 0) if result.usage else 0
            logger.debug(
                "LLM call #%d: model=%s latency=%.0fms tokens=%s",
                self._call_count,
                self.config.model,
                latency,
                result.usage,
            )
            return result
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            logger.error("LLM call failed after %.0fms: %s", latency, exc)
            raise

    async def _call_api(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> GenerationResult:
        """Override in subclasses to perform the actual API call."""
        raise NotImplementedError

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.config.model,
            "calls": self._call_count,
            "total_tokens": self._total_tokens,
        }


# ---------------------------------------------------------------------------
# MockLLMProvider — deterministic, no API calls
# ---------------------------------------------------------------------------


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic mock provider for testing.

    Returns a canned response based on the last user message.
    """

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        *,
        default_response: str = "This is a mock LLM response.",
        responses: Optional[Dict[str, str]] = None,
    ):
        super().__init__(config or ProviderConfig(model="mock"))
        self.default_response = default_response
        self.responses = responses or {}
        self.call_history: List[List[Message]] = []

    async def _call_api(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> GenerationResult:
        self.call_history.append(messages)
        # Find last user message
        user_msg = ""
        for m in reversed(messages):
            if m.role == MessageRole.USER:
                user_msg = m.content
                break

        response = self.responses.get(user_msg, self.default_response)
        return GenerationResult(
            content=response,
            model=self.config.model,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
        )


# ---------------------------------------------------------------------------
# OpenAILLMProvider — real OpenAI SDK
# ---------------------------------------------------------------------------


class OpenAILLMProvider(BaseLLMProvider):
    """
    Real OpenAI API provider using the ``openai`` SDK.

    Requires: ``pip install openai``
    """

    def __init__(self, config: Optional[ProviderConfig] = None, **kwargs: Any):
        cfg = config or ProviderConfig(**kwargs)
        if not cfg.api_key:
            cfg.api_key = os.environ.get("OPENAI_API_KEY")
        super().__init__(cfg)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError("OpenAI SDK not installed. Run: pip install openai") from exc
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        return self._client

    async def _call_api(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> GenerationResult:
        client = self._get_client()
        api_messages = [m.to_dict() for m in messages]

        response = await client.chat.completions.create(
            model=self.config.model,
            messages=api_messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
        )

        choice = response.choices[0]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        return GenerationResult(
            content=choice.message.content or "",
            model=response.model,
            usage=usage,
            finish_reason=choice.finish_reason or "stop",
        )


# ---------------------------------------------------------------------------
# AnthropicLLMProvider — real Anthropic SDK
# ---------------------------------------------------------------------------


class AnthropicLLMProvider(BaseLLMProvider):
    """
    Real Anthropic API provider using the ``anthropic`` SDK.

    Requires: ``pip install anthropic``
    """

    def __init__(self, config: Optional[ProviderConfig] = None, **kwargs: Any):
        cfg = config or ProviderConfig(model="claude-sonnet-4-20250514", **kwargs)
        if not cfg.api_key:
            cfg.api_key = os.environ.get("ANTHROPIC_API_KEY")
        super().__init__(cfg)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise ImportError(
                    "Anthropic SDK not installed. Run: pip install anthropic"
                ) from exc
            self._client = AsyncAnthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
        return self._client

    async def _call_api(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> GenerationResult:
        client = self._get_client()

        # Anthropic requires system message separate
        system_msg = ""
        api_messages = []
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                system_msg = m.content
            else:
                api_messages.append({"role": m.role.value, "content": m.content})

        create_kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": api_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        if system_msg:
            create_kwargs["system"] = system_msg

        response = await client.messages.create(**create_kwargs)

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        usage = {
            "prompt_tokens": response.usage.input_tokens if response.usage else 0,
            "completion_tokens": response.usage.output_tokens if response.usage else 0,
            "total_tokens": (
                (response.usage.input_tokens + response.usage.output_tokens)
                if response.usage
                else 0
            ),
        }
        return GenerationResult(
            content=content,
            model=self.config.model,
            usage=usage,
            finish_reason=response.stop_reason or "stop",
        )


# ---------------------------------------------------------------------------
# OllamaLLMProvider — local Ollama HTTP API
# ---------------------------------------------------------------------------


class OllamaLLMProvider(BaseLLMProvider):
    """
    Local Ollama provider using HTTP API.

    No extra SDK needed — uses stdlib ``urllib`` or ``httpx`` if available.
    """

    def __init__(self, config: Optional[ProviderConfig] = None, **kwargs: Any):
        cfg = config or ProviderConfig(model="llama3.1", **kwargs)
        if not cfg.base_url:
            cfg.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        super().__init__(cfg)

    async def _call_api(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> GenerationResult:
        import urllib.request

        api_messages = [{"role": m.role.value, "content": m.content} for m in messages]
        payload = json.dumps(
            {
                "model": self.config.model,
                "messages": api_messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                },
            }
        ).encode()

        url = f"{self.config.base_url}/api/chat"
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )

        # Run blocking HTTP in thread pool
        loop = asyncio.get_event_loop()
        response_data = await loop.run_in_executor(None, self._do_request, req)

        content = response_data.get("message", {}).get("content", "")
        return GenerationResult(
            content=content,
            model=self.config.model,
            usage=response_data.get("usage", {}),
            finish_reason="stop",
        )

    @staticmethod
    def _do_request(req: Any) -> Dict[str, Any]:
        import urllib.request

        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {
    "mock": MockLLMProvider,
    "openai": OpenAILLMProvider,
    "anthropic": AnthropicLLMProvider,
    "ollama": OllamaLLMProvider,
}


class LLMProviderFactory:
    """
    Factory for creating LLM providers by name.

    >>> llm = LLMProviderFactory.create("mock")
    >>> llm = LLMProviderFactory.create("openai", model="gpt-4o")
    """

    @staticmethod
    def create(
        provider_name: str,
        *,
        config: Optional[ProviderConfig] = None,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """Create an LLM provider by name."""
        name = provider_name.lower().strip()
        cls = _PROVIDERS.get(name)
        if cls is None:
            available = ", ".join(sorted(_PROVIDERS.keys()))
            raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")
        return cls(config=config, **kwargs)

    @staticmethod
    def register(name: str, provider_cls: Type[BaseLLMProvider]) -> None:
        """Register a custom provider."""
        _PROVIDERS[name.lower()] = provider_cls

    @staticmethod
    def available_providers() -> List[str]:
        """List registered provider names."""
        return sorted(_PROVIDERS.keys())
