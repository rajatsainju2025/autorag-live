"""
Modular LLM provider abstraction for AutoRAG-Live.

Supports OpenAI, Anthropic, Ollama, and local models with streaming,
token counting, and cost tracking.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


class ModelProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""

    provider: ModelProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    timeout: int = 60
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenCounter:
    """Tracks token usage for cost calculation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Add tokens to counter."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens

    def get_summary(self) -> Dict[str, int]:
        """Get token usage summary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class CostTracker:
    """Tracks API costs across provider calls."""

    provider: ModelProvider
    model_name: str
    prompt_cost_per_1k: float = 0.0
    completion_cost_per_1k: float = 0.0
    total_cost: float = 0.0
    call_count: int = 0

    # Provider-specific pricing (as of Dec 2024)
    PRICING = {
        "gpt-4-turbo": {
            "prompt": 0.01,
            "completion": 0.03,
        },
        "gpt-4": {
            "prompt": 0.03,
            "completion": 0.06,
        },
        "gpt-3.5-turbo": {
            "prompt": 0.0005,
            "completion": 0.0015,
        },
        "claude-3-opus": {
            "prompt": 0.015,
            "completion": 0.075,
        },
        "claude-3-sonnet": {
            "prompt": 0.003,
            "completion": 0.015,
        },
        "claude-3-haiku": {
            "prompt": 0.00025,
            "completion": 0.00125,
        },
    }

    def __post_init__(self) -> None:
        """Initialize pricing based on model name."""
        if self.model_name in self.PRICING:
            pricing = self.PRICING[self.model_name]
            self.prompt_cost_per_1k = pricing["prompt"]
            self.completion_cost_per_1k = pricing["completion"]

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for token usage."""
        prompt_cost = (prompt_tokens / 1000) * self.prompt_cost_per_1k
        completion_cost = (completion_tokens / 1000) * self.completion_cost_per_1k
        return prompt_cost + completion_cost

    def add_call(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record API call with token usage."""
        cost = self.calculate_cost(prompt_tokens, completion_tokens)
        self.total_cost += cost
        self.call_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "total_cost": round(self.total_cost, 6),
            "call_count": self.call_count,
            "avg_cost_per_call": round(
                self.total_cost / self.call_count if self.call_count > 0 else 0,
                6,
            ),
        }


@dataclass
class LLMResponse:
    """Response from LLM provider."""

    content: str
    model: str
    provider: ModelProvider
    prompt_tokens: int
    completion_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        """Initialize provider with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"LLMProvider.{config.provider.value}")
        self.token_counter = TokenCounter()
        self.cost_tracker = CostTracker(config.provider, config.model_name)

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response from prompt."""
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream response tokens."""
        pass

    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs: Any) -> List[LLMResponse]:
        """Generate responses for multiple prompts."""
        pass

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (model-specific)."""
        # Simple estimation: ~4 chars per token on average
        return len(text) // 4

    def get_token_summary(self) -> Dict[str, int]:
        """Get token usage summary."""
        return self.token_counter.get_summary()

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return self.cost_tracker.get_summary()

    def reset_counters(self) -> None:
        """Reset token and cost counters."""
        self.token_counter = TokenCounter()
        self.cost_tracker = CostTracker(self.config.provider, self.config.model_name)


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider implementation."""

    def __init__(self, config: LLMConfig):
        """Initialize OpenAI provider."""
        super().__init__(config)
        try:
            import openai

            self.client = openai.OpenAI(api_key=config.api_key)
        except ImportError:
            self.logger.warning("OpenAI library not installed. Install with: pip install openai")
            self.client = None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                **kwargs,
            )

            content = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            self.token_counter.add_tokens(prompt_tokens, completion_tokens)
            self.cost_tracker.add_call(prompt_tokens, completion_tokens)

            return LLMResponse(
                content=content,
                model=self.config.model_name,
                provider=ModelProvider.OPENAI,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream response from OpenAI API."""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                stream=True,
                **kwargs,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self.logger.error(f"OpenAI streaming error: {str(e)}")
            raise

    def batch_generate(self, prompts: List[str], **kwargs: Any) -> List[LLMResponse]:
        """Generate responses for multiple prompts."""
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        return responses


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider implementation."""

    def __init__(self, config: LLMConfig):
        """Initialize Anthropic provider."""
        super().__init__(config)
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=config.api_key)
        except ImportError:
            self.logger.warning(
                "Anthropic library not installed. Install with: pip install anthropic"
            )
            self.client = None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")

        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                **kwargs,
            )

            content = response.content[0].text if response.content else ""
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens

            self.token_counter.add_tokens(prompt_tokens, completion_tokens)
            self.cost_tracker.add_call(prompt_tokens, completion_tokens)

            return LLMResponse(
                content=content,
                model=self.config.model_name,
                provider=ModelProvider.ANTHROPIC,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            raise

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream response from Anthropic API."""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")

        try:
            with self.client.messages.stream(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                **kwargs,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            self.logger.error(f"Anthropic streaming error: {str(e)}")
            raise

    def batch_generate(self, prompts: List[str], **kwargs: Any) -> List[LLMResponse]:
        """Generate responses for multiple prompts."""
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        return responses


class OllamaProvider(LLMProvider):
    """Ollama local model provider implementation."""

    def __init__(self, config: LLMConfig):
        """Initialize Ollama provider."""
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response using Ollama."""
        try:
            import requests

            endpoint = f"{self.base_url}/api/generate"

            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": self.config.temperature,
                **kwargs,
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(endpoint, json=payload, timeout=self.config.timeout)
            response.raise_for_status()

            result = response.json()
            content = result.get("response", "")

            # Estimate tokens
            prompt_tokens = self.estimate_tokens(prompt)
            completion_tokens = self.estimate_tokens(content)

            self.token_counter.add_tokens(prompt_tokens, completion_tokens)

            return LLMResponse(
                content=content,
                model=self.config.model_name,
                provider=ModelProvider.OLLAMA,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception as e:
            self.logger.error(f"Ollama error: {str(e)}")
            raise

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream response from Ollama."""
        try:
            import requests

            endpoint = f"{self.base_url}/api/generate"

            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": True,
                "temperature": self.config.temperature,
                **kwargs,
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(
                endpoint, json=payload, timeout=self.config.timeout, stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
        except Exception as e:
            self.logger.error(f"Ollama streaming error: {str(e)}")
            raise

    def batch_generate(self, prompts: List[str], **kwargs: Any) -> List[LLMResponse]:
        """Generate responses for multiple prompts."""
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        return responses


def create_provider(config: LLMConfig) -> LLMProvider:
    """Factory function to create LLM provider."""
    if config.provider == ModelProvider.OPENAI:
        return OpenAIProvider(config)
    elif config.provider == ModelProvider.ANTHROPIC:
        return AnthropicProvider(config)
    elif config.provider == ModelProvider.OLLAMA:
        return OllamaProvider(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
