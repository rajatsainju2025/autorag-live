"""
Async LLM provider implementations for parallel processing.

Enables concurrent LLM requests for improved throughput in batch operations
and multi-agent scenarios.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from autorag_live.llm.providers import CostTracker, LLMConfig, ModelProvider, TokenCounter


@dataclass
class AsyncLLMResponse:
    """Async response from LLM provider."""

    content: str
    model: str
    provider: ModelProvider
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncLLMProvider(ABC):
    """Abstract base class for async LLM providers."""

    def __init__(self, config: LLMConfig):
        """Initialize async provider with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"AsyncLLMProvider.{config.provider.value}")
        self.token_counter = TokenCounter()
        self.cost_tracker = CostTracker(config.provider, config.model_name)
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._max_concurrent = 10  # Default max concurrent requests

    def set_max_concurrent(self, max_concurrent: int) -> None:
        """Set maximum concurrent requests."""
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncLLMResponse:
        """Generate response asynchronously."""
        pass

    @abstractmethod
    async def stream_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response tokens asynchronously."""
        pass

    async def batch_generate_async(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> List[AsyncLLMResponse]:
        """
        Generate responses for multiple prompts concurrently.

        Args:
            prompts: List of prompts to process
            system_prompt: Optional system prompt for all requests
            **kwargs: Additional arguments

        Returns:
            List of responses in same order as prompts
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)

        async def limited_generate(prompt: str) -> AsyncLLMResponse:
            async with self._semaphore:
                return await self.generate_async(prompt, system_prompt, **kwargs)

        tasks = [limited_generate(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def parallel_generate(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[AsyncLLMResponse]:
        """
        Process multiple requests with different parameters concurrently.

        Args:
            requests: List of request dicts with 'prompt', optional 'system_prompt', and kwargs

        Returns:
            List of responses
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)

        async def process_request(request: Dict[str, Any]) -> AsyncLLMResponse:
            async with self._semaphore:
                prompt = request.pop("prompt")
                system_prompt = request.pop("system_prompt", None)
                return await self.generate_async(prompt, system_prompt, **request)

        tasks = [process_request(req.copy()) for req in requests]
        return await asyncio.gather(*tasks)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4


class AsyncOpenAIProvider(AsyncLLMProvider):
    """Async OpenAI provider implementation."""

    def __init__(self, config: LLMConfig):
        """Initialize async OpenAI provider."""
        super().__init__(config)
        self.client = None
        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=config.api_key)
        except ImportError:
            self.logger.warning("OpenAI library not installed. Install with: pip install openai")

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncLLMResponse:
        """Generate response asynchronously using OpenAI API."""
        if not self.client:
            raise RuntimeError("Async OpenAI client not initialized")

        import time

        start_time = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                **kwargs,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            content = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            self.token_counter.add_tokens(prompt_tokens, completion_tokens)
            self.cost_tracker.add_call(prompt_tokens, completion_tokens)

            return AsyncLLMResponse(
                content=content,
                model=self.config.model_name,
                provider=ModelProvider.OPENAI,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
        except Exception as e:
            self.logger.error(f"Async OpenAI API error: {str(e)}")
            raise

    async def stream_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response asynchronously from OpenAI API."""
        if not self.client:
            raise RuntimeError("Async OpenAI client not initialized")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self.logger.error(f"Async OpenAI streaming error: {str(e)}")
            raise


class AsyncAnthropicProvider(AsyncLLMProvider):
    """Async Anthropic provider implementation."""

    def __init__(self, config: LLMConfig):
        """Initialize async Anthropic provider."""
        super().__init__(config)
        self.client = None
        try:
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(api_key=config.api_key)
        except ImportError:
            self.logger.warning(
                "Anthropic library not installed. Install with: pip install anthropic"
            )

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncLLMResponse:
        """Generate response asynchronously using Anthropic API."""
        if not self.client:
            raise RuntimeError("Async Anthropic client not initialized")

        import time

        start_time = time.perf_counter()

        try:
            response = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                **kwargs,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            content = response.content[0].text if response.content else ""
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens

            self.token_counter.add_tokens(prompt_tokens, completion_tokens)
            self.cost_tracker.add_call(prompt_tokens, completion_tokens)

            return AsyncLLMResponse(
                content=content,
                model=self.config.model_name,
                provider=ModelProvider.ANTHROPIC,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
        except Exception as e:
            self.logger.error(f"Async Anthropic API error: {str(e)}")
            raise

    async def stream_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response asynchronously from Anthropic API."""
        if not self.client:
            raise RuntimeError("Async Anthropic client not initialized")

        try:
            async with self.client.messages.stream(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                **kwargs,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            self.logger.error(f"Async Anthropic streaming error: {str(e)}")
            raise


class AsyncOllamaProvider(AsyncLLMProvider):
    """Async Ollama local model provider implementation."""

    def __init__(self, config: LLMConfig):
        """Initialize async Ollama provider."""
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncLLMResponse:
        """Generate response asynchronously using Ollama."""
        import time

        start_time = time.perf_counter()

        try:
            import aiohttp

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

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

            latency_ms = (time.perf_counter() - start_time) * 1000
            content = result.get("response", "")

            prompt_tokens = self.estimate_tokens(prompt)
            completion_tokens = self.estimate_tokens(content)

            self.token_counter.add_tokens(prompt_tokens, completion_tokens)

            return AsyncLLMResponse(
                content=content,
                model=self.config.model_name,
                provider=ModelProvider.OLLAMA,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
        except Exception as e:
            self.logger.error(f"Async Ollama error: {str(e)}")
            raise

    async def stream_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response asynchronously from Ollama."""
        try:
            import json

            import aiohttp

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

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
        except Exception as e:
            self.logger.error(f"Async Ollama streaming error: {str(e)}")
            raise


def create_async_provider(config: LLMConfig) -> AsyncLLMProvider:
    """Factory function to create async LLM provider."""
    if config.provider == ModelProvider.OPENAI:
        return AsyncOpenAIProvider(config)
    elif config.provider == ModelProvider.ANTHROPIC:
        return AsyncAnthropicProvider(config)
    elif config.provider == ModelProvider.OLLAMA:
        return AsyncOllamaProvider(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


class AsyncLLMPool:
    """
    Pool of async LLM providers for load balancing and failover.

    Distributes requests across multiple providers for improved reliability.
    """

    def __init__(self, providers: List[AsyncLLMProvider]):
        """Initialize provider pool."""
        self.providers = providers
        self.logger = logging.getLogger("AsyncLLMPool")
        self._current_index = 0
        self._lock = asyncio.Lock()

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncLLMResponse:
        """Generate using round-robin provider selection."""
        async with self._lock:
            provider = self.providers[self._current_index]
            self._current_index = (self._current_index + 1) % len(self.providers)

        return await provider.generate_async(prompt, system_prompt, **kwargs)

    async def batch_generate_distributed(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> List[AsyncLLMResponse]:
        """
        Distribute batch requests across all providers.

        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt
            **kwargs: Additional arguments

        Returns:
            List of responses
        """
        # Distribute prompts across providers
        chunks = [[] for _ in self.providers]
        indices = [[] for _ in self.providers]

        for i, prompt in enumerate(prompts):
            provider_idx = i % len(self.providers)
            chunks[provider_idx].append(prompt)
            indices[provider_idx].append(i)

        # Process in parallel across providers
        async def process_chunk(
            provider: AsyncLLMProvider, chunk: List[str]
        ) -> List[AsyncLLMResponse]:
            return await provider.batch_generate_async(chunk, system_prompt, **kwargs)

        results_by_provider = await asyncio.gather(
            *[process_chunk(p, c) for p, c in zip(self.providers, chunks)]
        )

        # Reconstruct ordered results
        final_results: List[Optional[AsyncLLMResponse]] = [None] * len(prompts)
        for provider_idx, provider_results in enumerate(results_by_provider):
            for result_idx, original_idx in enumerate(indices[provider_idx]):
                final_results[original_idx] = provider_results[result_idx]

        return [r for r in final_results if r is not None]
