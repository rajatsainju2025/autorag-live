"""
LLM provider fallback chain with automatic retry and circuit breaking.

Provides resilient LLM access through provider chains, health monitoring,
and intelligent failover strategies.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar

from autorag_live.llm.async_providers import AsyncLLMProvider, AsyncLLMResponse
from autorag_live.llm.providers import LLMConfig, LLMProvider, LLMResponse

T = TypeVar("T")


class ProviderStatus(Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class ProviderHealth:
    """Health status for a provider."""

    provider_id: str
    status: ProviderStatus = ProviderStatus.HEALTHY
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    avg_latency_ms: float = 0.0
    last_failure_time: Optional[float] = None
    circuit_open_until: Optional[float] = None
    error_messages: List[str] = field(default_factory=list)

    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    def record_success(self, latency_ms: float) -> None:
        """Record successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_failures = 0
        # Exponential moving average for latency
        alpha = 0.3
        self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms
        self._update_status()

    def record_failure(self, error_msg: str) -> None:
        """Record failed request."""
        self.total_requests += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        self.error_messages.append(error_msg)
        if len(self.error_messages) > 10:
            self.error_messages.pop(0)
        self._update_status()

    def _update_status(self) -> None:
        """Update health status based on metrics."""
        if self.consecutive_failures >= 5:
            self.status = ProviderStatus.CIRCUIT_OPEN
            self.circuit_open_until = time.time() + 60  # 60s circuit breaker
        elif self.consecutive_failures >= 3:
            self.status = ProviderStatus.UNHEALTHY
        elif self.success_rate() < 0.9:
            self.status = ProviderStatus.DEGRADED
        else:
            self.status = ProviderStatus.HEALTHY

    def is_available(self) -> bool:
        """Check if provider is available for requests."""
        if self.status == ProviderStatus.CIRCUIT_OPEN:
            if self.circuit_open_until and time.time() > self.circuit_open_until:
                # Half-open: allow one request to test
                self.status = ProviderStatus.DEGRADED
                return True
            return False
        return True


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""

    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    timeout_seconds: float = 30.0
    prefer_healthy_providers: bool = True


class ProviderFallbackChain:
    """
    Chain of LLM providers with automatic failover.

    Maintains health status for each provider and routes requests
    to the healthiest available provider with automatic retry.
    """

    def __init__(
        self,
        providers: List[LLMProvider],
        config: Optional[FallbackConfig] = None,
    ):
        """
        Initialize fallback chain.

        Args:
            providers: Ordered list of providers (primary first)
            config: Fallback configuration
        """
        self.providers = providers
        self.config = config or FallbackConfig()
        self.logger = logging.getLogger("ProviderFallbackChain")

        # Initialize health tracking
        self.health: Dict[str, ProviderHealth] = {
            self._provider_id(p): ProviderHealth(provider_id=self._provider_id(p))
            for p in providers
        }

    def _provider_id(self, provider: LLMProvider) -> str:
        """Get unique identifier for provider."""
        return f"{provider.config.provider.value}:{provider.config.model_name}"

    def _get_ordered_providers(self) -> List[LLMProvider]:
        """Get providers ordered by health status."""
        if not self.config.prefer_healthy_providers:
            return self.providers

        def health_score(provider: LLMProvider) -> float:
            health = self.health[self._provider_id(provider)]
            if not health.is_available():
                return -1
            status_scores = {
                ProviderStatus.HEALTHY: 1.0,
                ProviderStatus.DEGRADED: 0.5,
                ProviderStatus.UNHEALTHY: 0.1,
                ProviderStatus.CIRCUIT_OPEN: -1,
            }
            return status_scores.get(health.status, 0)

        return sorted(self.providers, key=health_score, reverse=True)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate response with automatic failover.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional provider arguments

        Returns:
            LLMResponse from first successful provider

        Raises:
            RuntimeError: If all providers fail
        """
        ordered_providers = self._get_ordered_providers()
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries):
            for provider in ordered_providers:
                provider_id = self._provider_id(provider)
                health = self.health[provider_id]

                if not health.is_available():
                    continue

                try:
                    start_time = time.perf_counter()
                    response = provider.generate(prompt, system_prompt, **kwargs)
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    health.record_success(latency_ms)
                    self.logger.debug(f"Success with {provider_id} in {latency_ms:.1f}ms")
                    return response

                except Exception as e:
                    last_error = e
                    health.record_failure(str(e))
                    self.logger.warning(f"Provider {provider_id} failed: {e}")

            # Apply retry delay with exponential backoff
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay_seconds
                if self.config.exponential_backoff:
                    delay *= 2**attempt
                time.sleep(delay)

        raise RuntimeError(
            f"All providers failed after {self.config.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Stream response with automatic failover.

        Yields tokens from first successful provider.
        """
        ordered_providers = self._get_ordered_providers()

        for provider in ordered_providers:
            provider_id = self._provider_id(provider)
            health = self.health[provider_id]

            if not health.is_available():
                continue

            try:
                start_time = time.perf_counter()
                for token in provider.stream(prompt, system_prompt, **kwargs):
                    yield token
                latency_ms = (time.perf_counter() - start_time) * 1000
                health.record_success(latency_ms)
                return  # Success, don't try other providers
            except Exception as e:
                health.record_failure(str(e))
                self.logger.warning(f"Stream from {provider_id} failed: {e}")

        raise RuntimeError("All providers failed for streaming")

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all providers."""
        return {
            pid: {
                "status": h.status.value,
                "success_rate": round(h.success_rate(), 3),
                "avg_latency_ms": round(h.avg_latency_ms, 1),
                "consecutive_failures": h.consecutive_failures,
                "total_requests": h.total_requests,
            }
            for pid, h in self.health.items()
        }

    def reset_health(self, provider_id: Optional[str] = None) -> None:
        """Reset health status for provider(s)."""
        if provider_id:
            if provider_id in self.health:
                self.health[provider_id] = ProviderHealth(provider_id=provider_id)
        else:
            for pid in self.health:
                self.health[pid] = ProviderHealth(provider_id=pid)


class AsyncProviderFallbackChain:
    """Async version of provider fallback chain."""

    def __init__(
        self,
        providers: List[AsyncLLMProvider],
        config: Optional[FallbackConfig] = None,
    ):
        """Initialize async fallback chain."""
        self.providers = providers
        self.config = config or FallbackConfig()
        self.logger = logging.getLogger("AsyncProviderFallbackChain")

        self.health: Dict[str, ProviderHealth] = {
            self._provider_id(p): ProviderHealth(provider_id=self._provider_id(p))
            for p in providers
        }
        self._lock = asyncio.Lock()

    def _provider_id(self, provider: AsyncLLMProvider) -> str:
        """Get unique identifier for provider."""
        return f"{provider.config.provider.value}:{provider.config.model_name}"

    def _get_ordered_providers(self) -> List[AsyncLLMProvider]:
        """Get providers ordered by health status."""
        if not self.config.prefer_healthy_providers:
            return self.providers

        def health_score(provider: AsyncLLMProvider) -> float:
            health = self.health[self._provider_id(provider)]
            if not health.is_available():
                return -1
            status_scores = {
                ProviderStatus.HEALTHY: 1.0,
                ProviderStatus.DEGRADED: 0.5,
                ProviderStatus.UNHEALTHY: 0.1,
                ProviderStatus.CIRCUIT_OPEN: -1,
            }
            return status_scores.get(health.status, 0)

        return sorted(self.providers, key=health_score, reverse=True)

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncLLMResponse:
        """Generate response asynchronously with automatic failover."""
        ordered_providers = self._get_ordered_providers()
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries):
            for provider in ordered_providers:
                provider_id = self._provider_id(provider)

                async with self._lock:
                    health = self.health[provider_id]
                    if not health.is_available():
                        continue

                try:
                    response = await provider.generate_async(prompt, system_prompt, **kwargs)

                    async with self._lock:
                        health.record_success(response.latency_ms)

                    self.logger.debug(f"Success with {provider_id} in {response.latency_ms:.1f}ms")
                    return response

                except Exception as e:
                    last_error = e
                    async with self._lock:
                        health.record_failure(str(e))
                    self.logger.warning(f"Provider {provider_id} failed: {e}")

            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay_seconds
                if self.config.exponential_backoff:
                    delay *= 2**attempt
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"All providers failed after {self.config.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def batch_generate_with_fallback(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> List[AsyncLLMResponse]:
        """Generate responses for multiple prompts with per-request fallback."""
        tasks = [self.generate_async(prompt, system_prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all providers."""
        return {
            pid: {
                "status": h.status.value,
                "success_rate": round(h.success_rate(), 3),
                "avg_latency_ms": round(h.avg_latency_ms, 1),
                "consecutive_failures": h.consecutive_failures,
                "total_requests": h.total_requests,
            }
            for pid, h in self.health.items()
        }


def create_fallback_chain(
    configs: List[LLMConfig],
    fallback_config: Optional[FallbackConfig] = None,
) -> ProviderFallbackChain:
    """
    Create a fallback chain from configurations.

    Args:
        configs: List of LLM configurations (primary first)
        fallback_config: Optional fallback configuration

    Returns:
        Configured fallback chain
    """
    from autorag_live.llm.providers import create_provider

    providers = [create_provider(config) for config in configs]
    return ProviderFallbackChain(providers, fallback_config)


def create_async_fallback_chain(
    configs: List[LLMConfig],
    fallback_config: Optional[FallbackConfig] = None,
) -> AsyncProviderFallbackChain:
    """
    Create an async fallback chain from configurations.

    Args:
        configs: List of LLM configurations (primary first)
        fallback_config: Optional fallback configuration

    Returns:
        Configured async fallback chain
    """
    from autorag_live.llm.async_providers import create_async_provider

    providers = [create_async_provider(config) for config in configs]
    return AsyncProviderFallbackChain(providers, fallback_config)
